#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import rclpy
from rclpy.client import Client
from rclpy.node import Node
from rclpy.task import Future

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


@dataclass(frozen=True)
class _Target:
    node_name: str

    @property
    def set_parameters_service(self) -> str:
        return f'{self.node_name}/set_parameters'


def _normalize_node_name(node_name: str) -> str:
    node_name = node_name.strip()
    if not node_name:
        raise ValueError('target node name is empty')
    return node_name if node_name.startswith('/') else f'/{node_name}'


def _build_use_sim_time_request(value: bool) -> SetParameters.Request:
    req = SetParameters.Request()
    req.parameters = [
        Parameter(
            name='use_sim_time',
            value=ParameterValue(
                type=ParameterType.PARAMETER_BOOL,
                bool_value=value,
            ),
        ),
    ]
    return req


class SimTimeEnforcer(Node):
    def __init__(self) -> None:
        super().__init__('sim_time_enforcer')

        self.declare_parameter(
            'target_nodes',
            [
                '/simple_kf_odometry_node',
                '/fisheye_of_downsampled_minimal_node',
            ],
        )
        self.declare_parameter('use_sim_time_value', True)
        self.declare_parameter('retry_period_s', 1.0)

        target_nodes: List[str] = list(self.get_parameter('target_nodes').value)
        use_sim_time_value: bool = bool(self.get_parameter('use_sim_time_value').value)
        retry_period_s: float = float(self.get_parameter('retry_period_s').value)

        targets: List[_Target] = []
        for raw_name in target_nodes:
            try:
                targets.append(_Target(node_name=_normalize_node_name(str(raw_name))))
            except ValueError as exc:
                self.get_logger().warn(f'Ignoring invalid target node name {raw_name!r}: {exc}')

        self._targets: List[_Target] = targets
        self._request = _build_use_sim_time_request(use_sim_time_value)

        # rclpy.Node already uses `_clients`
        self._param_clients: Dict[str, Client] = {}
        self._inflight: Dict[Future, str] = {}
        self._done: Set[str] = set()

        if retry_period_s <= 0.0:
            raise ValueError('retry_period_s must be > 0')

        self._timer = self.create_timer(retry_period_s, self._tick)
        self.get_logger().info(
            f'SimTimeEnforcer started. Enforcing use_sim_time={use_sim_time_value} for: '
            f'{", ".join(t.node_name for t in self._targets) if self._targets else "(none)"}'
        )

    def _get_client(self, target: _Target) -> Client:
        client = self._param_clients.get(target.node_name)
        if client is None:
            client = self.create_client(SetParameters, target.set_parameters_service)
            self._param_clients[target.node_name] = client
        return client

    def _tick(self) -> None:
        if not self._targets:
            return

        for target in self._targets:
            if target.node_name in self._done:
                client = self._param_clients.get(target.node_name)
                if client is not None and not client.service_is_ready():
                    self._done.discard(target.node_name)
                continue

            if target.node_name in self._inflight.values():
                continue

            client = self._get_client(target)
            if not client.service_is_ready():
                continue

            future = client.call_async(self._request)
            self._inflight[future] = target.node_name
            future.add_done_callback(self._on_set_parameters_done)

    def _on_set_parameters_done(self, future: Future) -> None:
        node_name: Optional[str] = self._inflight.pop(future, None)
        if node_name is None:
            return

        try:
            response: SetParameters.Response = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Failed to set use_sim_time on {node_name}: {exc}')
            return

        if not response.results:
            self.get_logger().warn(f'SetParameters returned empty results for {node_name}')
            return

        result = response.results[0]
        if result.successful:
            self._done.add(node_name)
            self.get_logger().info(f'Set use_sim_time successfully on {node_name}')
        else:
            self.get_logger().warn(f'Failed to set use_sim_time on {node_name}: {result.reason}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimTimeEnforcer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
