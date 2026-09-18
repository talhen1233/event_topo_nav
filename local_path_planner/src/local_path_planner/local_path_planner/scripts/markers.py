from __future__ import annotations

import math
from typing import Optional

import numpy as np

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


def _publish_delete_marker(pub, header, namespace: str, marker_id: int = 0) -> None:
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.action = Marker.DELETE
    pub.publish(marker)


class CmdVelMarkers:
    """Publishes `/cmd_vel_markers`: yaw arc only."""

    def __init__(self, pub) -> None:
        self._pub = pub

    def publish(self, odom: Optional[Odometry], tw: Twist, *, yaw_rate_max: float) -> None:
        if odom is None:
            return

        header = odom.header
        pos = odom.pose.pose.position
        _publish_delete_marker(self._pub, header, "cmd_linear")

        wz = float(tw.angular.z)
        yaw_speed = abs(wz)
        if yaw_speed > 0.05 and yaw_rate_max > 1e-6:
            radius_factor = min(1.0, yaw_speed / float(yaw_rate_max))
            radius = 0.2 + 0.2 * radius_factor
            center_z_offset = 0.2

            m_yaw = Marker()
            m_yaw.header = header
            m_yaw.ns = "cmd_yaw"
            m_yaw.id = 0
            m_yaw.type = Marker.LINE_STRIP
            m_yaw.action = Marker.ADD
            m_yaw.pose.orientation.w = 1.0
            m_yaw.scale.x = 0.01

            if wz >= 0.0:
                m_yaw.color.r = 0.1
                m_yaw.color.g = 1.0
                m_yaw.color.b = 0.1
            else:
                m_yaw.color.r = 1.0
                m_yaw.color.g = 0.1
                m_yaw.color.b = 0.1
            m_yaw.color.a = 1.0

            num_arc_points = 24
            angle_span = 1.75 * math.pi
            start_angle = 0.0
            end_angle = angle_span if wz >= 0.0 else -angle_span

            for i in range(num_arc_points):
                t = float(i) / float(num_arc_points - 1)
                ang = start_angle + (end_angle - start_angle) * t
                m_yaw.points.append(
                    Point(
                        x=float(pos.x + radius * math.cos(ang)),
                        y=float(pos.y + radius * math.sin(ang)),
                        z=float(pos.z + center_z_offset),
                    )
                )

            head_angle = end_angle
            head = Point(
                x=float(pos.x + radius * math.cos(head_angle)),
                y=float(pos.y + radius * math.sin(head_angle)),
                z=float(pos.z + center_z_offset),
            )
            m_yaw.points.append(head)

            head_len = 0.15 * radius
            wing_offset = 0.25 * math.pi
            for sign in (1.0, -1.0):
                wing_ang = head_angle + sign * wing_offset
                m_yaw.points.append(
                    Point(
                        x=float(head.x + head_len * math.cos(wing_ang)),
                        y=float(head.y + head_len * math.sin(wing_ang)),
                        z=float(head.z),
                    )
                )
                m_yaw.points.append(head)

            self._pub.publish(m_yaw)
        else:
            _publish_delete_marker(self._pub, header, "cmd_yaw")


class DebugMarkers:
    """Publishes `/local_planner/debug_markers`: graph overlay markers."""

    def __init__(self, pub) -> None:
        self._pub = pub

    def publish_graph_overlay(
        self,
        odom: Optional[Odometry],
        *,
        tracked_edge_pts: Optional[np.ndarray] = None,
        junction_xy: Optional[np.ndarray] = None,
        speculative_arm_angle: Optional[float] = None,
    ) -> None:
        if odom is None:
            return

        header = odom.header
        z = float(odom.pose.pose.position.z)

        if tracked_edge_pts is not None and tracked_edge_pts.shape[0] >= 2:
            m = Marker()
            m.header = header
            m.ns = "graph_tracked_edge"
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.04
            m.color.r = 0.2
            m.color.g = 1.0
            m.color.b = 0.6
            m.color.a = 0.8
            for pt in tracked_edge_pts:
                m.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=z))
            self._pub.publish(m)
        else:
            _publish_delete_marker(self._pub, header, "graph_tracked_edge")

        _publish_delete_marker(self._pub, header, "graph_tangent")
        _publish_delete_marker(self._pub, header, "graph_yaw_intent")

        if junction_xy is not None:
            m = Marker()
            m.header = header
            m.ns = "graph_junction_zone"
            m.id = 0
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(junction_xy[0])
            m.pose.position.y = float(junction_xy[1])
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = 0.6
            m.scale.y = 0.6
            m.scale.z = 0.05
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 0.4
            self._pub.publish(m)
        else:
            _publish_delete_marker(self._pub, header, "graph_junction_zone")

        if junction_xy is not None and speculative_arm_angle is not None:
            m = Marker()
            m.header = header
            m.ns = "graph_speculative_sector"
            m.id = 0
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.025
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 1.0
            m.color.a = 0.95

            center_x = float(junction_xy[0])
            center_y = float(junction_xy[1])
            z_sector = z + 0.03
            outer_radius = 0.72
            inner_radius = 0.22
            half_span = math.radians(18.0)
            start_angle = float(speculative_arm_angle - half_span)
            end_angle = float(speculative_arm_angle + half_span)
            dash_count = 6
            dash_fraction = 0.55

            for dash_idx in range(dash_count):
                t0 = float(dash_idx) / float(dash_count)
                t1 = min(1.0, t0 + dash_fraction / float(dash_count))
                ang0 = start_angle + (end_angle - start_angle) * t0
                ang1 = start_angle + (end_angle - start_angle) * t1
                m.points.extend(
                    [
                        Point(
                            x=center_x + outer_radius * math.cos(ang0),
                            y=center_y + outer_radius * math.sin(ang0),
                            z=z_sector,
                        ),
                        Point(
                            x=center_x + outer_radius * math.cos(ang1),
                            y=center_y + outer_radius * math.sin(ang1),
                            z=z_sector,
                        ),
                    ]
                )

            for boundary_angle in (start_angle, end_angle):
                m.points.extend(
                    [
                        Point(
                            x=center_x + inner_radius * math.cos(boundary_angle),
                            y=center_y + inner_radius * math.sin(boundary_angle),
                            z=z_sector,
                        ),
                        Point(
                            x=center_x + outer_radius * math.cos(boundary_angle),
                            y=center_y + outer_radius * math.sin(boundary_angle),
                            z=z_sector,
                        ),
                    ]
                )
            self._pub.publish(m)
        else:
            _publish_delete_marker(self._pub, header, "graph_speculative_sector")

        _publish_delete_marker(self._pub, header, "graph_speculative_arm")
        _publish_delete_marker(self._pub, header, "planner_status")
