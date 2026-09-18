#!/usr/bin/env python3
"""Pose, geometry, junction, and stored-event data structures."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

try:  # pragma: no cover - allow running module directly
    from .event_types import EventType, Decision, JunctionType
except ImportError:  # pragma: no cover
    from core.event_types import EventType, Decision, JunctionType

try:  # pragma: no cover
    from .radial_descriptor import RadialDescriptor
except ImportError:  # pragma: no cover
    from core.radial_descriptor import RadialDescriptor


@dataclass
class Pose3D:
    """3D pose with yaw-only heading."""
    x: float
    y: float
    z: float
    yaw: float = 0.0

    def distance_to(self, other: 'Pose3D') -> float:
        """Euclidean distance to another pose."""
        return np.linalg.norm([self.x - other.x, self.y - other.y, self.z - other.z])


@dataclass
class TunnelGeometry:
    """Tunnel cross-section at an event location."""
    width: float
    height: float
    shape_descriptor: str
    cross_section_points: Optional[np.ndarray] = None


@dataclass
class JunctionConfiguration:
    """Paths at a junction; path_angles are world-frame radians from the skeleton graph."""
    junction_type: JunctionType
    num_paths: int
    path_angles: List[float]
    selected_path_index: int = -1
    dead_end_paths: List[int] = field(default_factory=list)
    path_widths: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'junction_type': self.junction_type.name,
            'num_paths': self.num_paths,
            'path_angles': self.path_angles,
            'selected_path_index': self.selected_path_index,
            'dead_end_paths': list(self.dead_end_paths),
            'path_widths': list(self.path_widths),
        }


@dataclass
class NavigationEvent:
    """Stored navigation event used for recognition and junction decisions."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    decision: Decision
    pose: Pose3D
    pose_uncertainty: np.ndarray
    odometry_distance: float
    tunnel_geometry: TunnelGeometry
    junction_config: Optional[JunctionConfiguration] = None
    radial_descriptor: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON topics; include optional entry_angle/previous_event_id when present."""
        entry_angle = getattr(self, 'entry_angle', None)
        prev_event_id = getattr(self, 'previous_event_id', None)
        if isinstance(entry_angle, (int, float)):
            entry_angle_value: Optional[float] = float(entry_angle)
        else:
            entry_angle_value = None
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'timestamp': self.timestamp.isoformat(),
            'decision': self.decision.name,
            'pose': {
                'x': self.pose.x,
                'y': self.pose.y,
                'z': self.pose.z,
                'yaw': self.pose.yaw
            },
            'pose_uncertainty': self.pose_uncertainty.tolist(),
            'odometry_distance': self.odometry_distance,
            'tunnel_geometry': {
                'width': self.tunnel_geometry.width,
                'height': self.tunnel_geometry.height,
                'shape_descriptor': self.tunnel_geometry.shape_descriptor
            },
            'junction_config': self.junction_config.to_dict() if self.junction_config else None,
            'radial_descriptor': self.radial_descriptor.to_dict() if self.radial_descriptor is not None else None,
            'entry_angle': entry_angle_value,
            'previous_event_id': prev_event_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NavigationEvent':
        """Reconstruct a NavigationEvent from to_dict() output."""
        event = cls(
            event_id=data['event_id'],
            event_type=EventType[data['event_type']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            decision=Decision[data['decision']],
            pose=Pose3D(**data['pose']),
            pose_uncertainty=np.array(data['pose_uncertainty']),
            odometry_distance=data['odometry_distance'],
            tunnel_geometry=TunnelGeometry(**data['tunnel_geometry'])
        )

        if data.get('junction_config'):
            jc = data['junction_config']
            event.junction_config = JunctionConfiguration(
                junction_type=JunctionType[jc['junction_type']],
                num_paths=jc['num_paths'],
                path_angles=jc['path_angles'],
                selected_path_index=jc.get('selected_path_index', -1),
                dead_end_paths=jc.get('dead_end_paths', []),
                path_widths=jc.get('path_widths', []),
            )

        if data.get('radial_descriptor'):
            event.radial_descriptor = RadialDescriptor.from_dict(data['radial_descriptor'])

        entry_angle = data.get('entry_angle', None)
        if entry_angle is not None:
            if isinstance(entry_angle, (int, float)):
                event.entry_angle = float(entry_angle)
            else:
                try:
                    event.entry_angle = float(entry_angle)
                except (TypeError, ValueError):
                    pass

        prev_event_id = data.get('previous_event_id', None)
        if prev_event_id is not None:
            event.previous_event_id = str(prev_event_id)

        return event
