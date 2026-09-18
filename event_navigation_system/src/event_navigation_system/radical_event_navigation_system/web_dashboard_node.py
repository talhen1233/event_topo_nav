#!/usr/bin/env python3
"""HTTP dashboard for navigation telemetry, mission controls, and manual cmd_vel."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, String


@dataclass
class _ActivityEntry:
    timestamp: float
    message: str
    event_type: str


@dataclass
class _RadialProfileData:
    """Radial profile arrays for the matching visualization."""
    angles: List[float] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    mean_radius: float = 0.0


@dataclass
class _EventMapEntry:
    """Map-marker fields for a stored event."""
    x: float
    y: float
    event_id: str
    short_id: str
    event_type: str
    arms_count: int = 0
    entry_angle: Optional[float] = None
    selected_path_index: Optional[int] = None
    path_angles: Optional[List[float]] = None
    dead_end_paths: List[int] = field(default_factory=list)
    matched_during_return: bool = False


@dataclass
class _Telemetry:
    t_mode: float = 0.0
    t_expected_event: float = 0.0
    t_return_viz: float = 0.0
    t_graph_target: float = 0.0
    t_match_conf: float = 0.0
    t_match_found: float = 0.0
    t_odom: float = 0.0
    t_imu: float = 0.0
    t_battery: float = 0.0
    t_local_planner_state: float = 0.0
    t_local_planner_debug: float = 0.0

    mode: str = ""
    expected_event_id: str = ""
    return_target_event_id: str = ""
    return_status: str = ""
    return_progress: Dict[str, Any] = field(default_factory=dict)
    pose_correction_policy: Dict[str, Any] = field(default_factory=dict)

    match_confidence: Optional[float] = None
    last_match_event_id: str = ""

    speed_m_s: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    yaw_heading_deg: Optional[float] = None

    battery_percent: Optional[float] = None
    local_planner_state: str = ""
    graph_target: Dict[str, Any] = field(default_factory=dict)
    planner_debug: Dict[str, Any] = field(default_factory=dict)

    total_events_discovered: int = 0
    junction_events_discovered: int = 0
    dead_ends_encountered: int = 0
    events_remaining_to_home: int = 0

    event_id_map: Dict[str, int] = field(default_factory=dict)
    next_event_number: int = 1

    activity_log: deque = field(default_factory=lambda: deque(maxlen=10))

    current_profile: Optional[_RadialProfileData] = None
    stored_profile: Optional[_RadialProfileData] = None
    match_distance_m: float = 0.0
    total_distance_m: float = 0.0

    drone_x: float = 0.0
    drone_y: float = 0.0
    drone_yaw_rad: float = 0.0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=500))
    event_map: Dict[str, _EventMapEntry] = field(default_factory=dict)
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    matched_event_ids: set = field(default_factory=set)
    mission_complete_time: Optional[float] = None


class _ThreadedHTTPServer(HTTPServer):
    """HTTPServer that can be stopped from another thread."""
    timeout = 1.0

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self._shutdown_requested = threading.Event()

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        while not self._shutdown_requested.is_set():
            try:
                self.handle_request()
            except Exception:
                pass

    def shutdown_server(self) -> None:
        self._shutdown_requested.set()
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.2)
            sock.connect(self.server_address)
            sock.close()
        except OSError:
            pass


def _html_page(*, poll_hz: float) -> bytes:
    poll_ms = int(max(100, round(1000.0 / max(0.1, float(poll_hz)))))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Navigation Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-start: #0f0f23;
      --bg-end: #1a1a2e;
      --bg: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%);
      --card: rgba(255, 255, 255, 0.03);
      --card-border: rgba(255, 255, 255, 0.08);
      --card-glow: rgba(34, 211, 238, 0.05);
      --text: #f1f5f9;
      --text-dim: #94a3b8;
      --text-muted: #64748b;
      --accent: #06b6d4;
      --accent-glow: rgba(6, 182, 212, 0.4);
      --cyan: #22d3ee;
      --teal: #2dd4bf;
      --blue: #38bdf8;
      --green: #34d399;
      --yellow: #fbbf24;
      --orange: #fb923c;
      --red: #f87171;
      --purple: #a78bfa;
      --pink: #f472b6;
    }}
    [hidden] {{ display: none !important; }}
    
    [data-theme="light"] {{
      --bg-start: #f0f9ff;
      --bg-end: #e0f2fe;
      --bg: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%);
      --card: rgba(255, 255, 255, 0.7);
      --card-border: rgba(0, 0, 0, 0.08);
      --card-glow: rgba(6, 182, 212, 0.1);
      --text: #0f172a;
      --text-dim: #475569;
      --text-muted: #94a3b8;
      --accent: #0891b2;
      --accent-glow: rgba(8, 145, 178, 0.3);
      --cyan: #06b6d4;
      --teal: #14b8a6;
      --blue: #0ea5e9;
      --green: #10b981;
      --yellow: #f59e0b;
      --orange: #f97316;
      --red: #ef4444;
      --purple: #8b5cf6;
      --pink: #ec4899;
    }}
    
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    body {{
      min-height: 100vh;
      background: var(--bg);
      background-attachment: fixed;
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      padding: 10px;
      transition: all 0.4s ease;
    }}
    
    /* Animated background mesh */
    body::before {{
      content: '';
      position: fixed;
      inset: 0;
      background: 
        radial-gradient(ellipse at 20% 20%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(45, 212, 191, 0.04) 0%, transparent 60%);
      pointer-events: none;
      z-index: -1;
    }}
    
    /* Top bar - compact */
    .status-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 14px;
      background: var(--card);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      margin-bottom: 10px;
      font-size: 13px;
    }}
    .status-left {{ display: flex; align-items: center; gap: 10px; }}
    .status-dot {{
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: glow-pulse 2s infinite;
    }}
    .status-dot.offline {{ background: var(--red); box-shadow: 0 0 8px var(--red); animation: none; }}
    @keyframes glow-pulse {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.7; transform: scale(0.9); }}
    }}
    
    .theme-toggle {{
      padding: 5px 10px;
      background: var(--card);
      backdrop-filter: blur(10px);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      color: var(--text);
      transition: all 0.2s;
    }}
    .theme-toggle:hover {{ 
      border-color: var(--accent);
    }}
    
    /* Compact layout for half-screen */
    .main-layout {{
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 12px;
      min-height: calc(100vh - 90px);
    }}
    @media (max-width: 900px) {{
      .main-layout {{ grid-template-columns: 1fr; }}
    }}
    
    /* Sidebar */
    .sidebar {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    
    /* Main content */
    .main-content {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    
    /* Glass cards - compact */
    .card {{
      background: var(--card);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      overflow: hidden;
      transition: all 0.3s ease;
    }}
    .card:hover {{
      border-color: rgba(6, 182, 212, 0.2);
      box-shadow: 0 4px 20px var(--card-glow);
    }}
    .card-header {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--card-border);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .card-body {{ padding: 12px; }}
    
    /* Mode hero - compact */
    .mode-hero {{
      text-align: center;
      padding: 12px 0;
    }}
    .mode-value {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.02em;
      text-shadow: 0 0 20px var(--accent-glow);
    }}
    .mode-value.explore {{ color: var(--cyan); text-shadow: 0 0 20px rgba(34, 211, 238, 0.5); }}
    .mode-value.backtrack {{ color: var(--orange); text-shadow: 0 0 20px rgba(251, 146, 60, 0.5); }}
    .mode-value.return {{ color: var(--pink); text-shadow: 0 0 20px rgba(244, 114, 182, 0.5); }}
    .mode-value.idle {{ color: var(--text-muted); text-shadow: none; }}
    .mode-sub {{
      font-size: 13px;
      color: var(--text-dim);
      margin-top: 4px;
    }}
    
    /* Stats grid - compact */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--card-border);
    }}
    .stat {{
      text-align: center;
      padding: 8px 4px;
      background: linear-gradient(145deg, rgba(255,255,255,0.02), transparent);
      border-radius: 8px;
      border: 1px solid transparent;
    }}
    .stat-value {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 20px;
      font-weight: 600;
      line-height: 1;
    }}
    .stat-value.cyan {{ color: var(--cyan); }}
    .stat-value.orange {{ color: var(--orange); }}
    .stat-value.red {{ color: var(--red); }}
    .stat-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    
    /* Flight instruments row - compact */
    .instruments-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    .instrument {{
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .instrument-canvas {{
      width: 100px;
      height: 100px;
      border-radius: 10px;
      display: block;
    }}
    .instrument-label {{
      text-align: center;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-top: 4px;
      font-weight: 500;
    }}
    
    /* Profile matching - compact */
    .match-layout {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .match-canvas {{
      width: 90px;
      height: 90px;
      border-radius: 10px;
      flex-shrink: 0;
    }}
    .match-info {{
      flex: 1;
    }}
    .match-score {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 32px;
      font-weight: 700;
      line-height: 1;
    }}
    .match-score.high {{ color: var(--green); text-shadow: 0 0 15px rgba(52, 211, 153, 0.4); }}
    .match-score.medium {{ color: var(--yellow); text-shadow: 0 0 15px rgba(251, 191, 36, 0.4); }}
    .match-score.low {{ color: var(--text-muted); }}
    .match-label {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-top: 2px;
    }}
    .match-details {{
      margin-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .match-detail {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
    }}
    .match-detail-label {{ color: var(--text-muted); }}
    .match-detail-value {{ color: var(--cyan); font-family: 'JetBrains Mono', monospace; font-weight: 500; }}
    
    /* Activity log - compact */
    .activity-list {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      max-height: 120px;
      overflow-y: auto;
    }}
    .activity-list::-webkit-scrollbar {{ width: 3px; }}
    .activity-list::-webkit-scrollbar-track {{ background: transparent; }}
    .activity-list::-webkit-scrollbar-thumb {{ background: var(--card-border); border-radius: 2px; }}
    .activity-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      background: linear-gradient(90deg, rgba(255,255,255,0.02), transparent);
      border-radius: 6px;
      font-size: 13px;
      border-left: 2px solid transparent;
    }}
    .activity-item.info {{ border-left-color: var(--blue); }}
    .activity-item.success {{ border-left-color: var(--green); }}
    .activity-item.warn {{ border-left-color: var(--yellow); }}
    .activity-item.error {{ border-left-color: var(--red); }}
    .activity-time {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--text-muted);
      min-width: 38px;
    }}
    .activity-msg {{ color: var(--text); flex: 1; }}
    
    
    /* Mission control - compact */
    .mission-info {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .mission-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 10px;
      background: linear-gradient(90deg, rgba(255,255,255,0.02), transparent);
      border-radius: 8px;
    }}
    .mission-label {{ font-size: 13px; color: var(--text-dim); }}
    .mission-value {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      font-weight: 600;
      color: var(--cyan);
    }}
    .mission-value-stack {{
      display: flex;
      min-width: 0;
      flex-direction: column;
      align-items: flex-end;
      gap: 2px;
      text-align: right;
    }}
    .mission-value-line {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      font-weight: 600;
      color: var(--cyan);
    }}
    .mission-meta {{
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      font-weight: 500;
      line-height: 1.25;
    }}
    .mission-state {{
      padding: 2px 5px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 999px;
      color: var(--text-dim);
      font-family: 'Outfit', sans-serif;
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .mission-state.open {{
      color: #6ee7b7;
      border-color: rgba(52, 211, 153, 0.28);
      background: rgba(52, 211, 153, 0.09);
    }}
    .mission-state.approach {{
      color: #fbbf24;
      border-color: rgba(251, 191, 36, 0.25);
      background: rgba(251, 191, 36, 0.08);
    }}
    .mission-state.closed {{
      color: #fda4af;
      border-color: rgba(248, 113, 113, 0.25);
      background: rgba(248, 113, 113, 0.08);
    }}
    
    .btn-row {{
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }}
    .btn {{
      flex: 1;
      padding: 8px 12px;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      background: var(--card);
      backdrop-filter: blur(10px);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn:hover {{ 
      border-color: var(--accent);
      box-shadow: 0 2px 12px var(--accent-glow);
    }}
    .btn-primary {{
      background: linear-gradient(135deg, var(--cyan) 0%, var(--teal) 100%);
      border-color: transparent;
      color: #0f172a;
      font-weight: 700;
    }}
    .btn-primary:hover {{ 
      box-shadow: 0 2px 16px rgba(34, 211, 238, 0.4);
    }}
    .btn-danger {{ 
      border-color: rgba(248, 113, 113, 0.3);
      color: var(--red);
    }}
    .btn-danger:hover {{ 
      background: rgba(248, 113, 113, 0.1);
      border-color: var(--red);
    }}
    .btn-warning {{
      border-color: rgba(251, 191, 36, 0.3);
      color: var(--yellow);
    }}
    .btn-warning:hover {{
      background: rgba(251, 191, 36, 0.1);
      border-color: var(--yellow);
      box-shadow: 0 2px 12px rgba(251, 191, 36, 0.3);
    }}
    
    /* Battery - compact */
    .battery-widget {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 2px 0;
    }}
    .battery-bar {{
      flex: 1;
      height: 16px;
      background: rgba(255,255,255,0.05);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--card-border);
    }}
    .battery-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--teal), var(--cyan));
      border-radius: 7px;
      transition: width 0.5s ease;
      box-shadow: 0 0 10px rgba(34, 211, 238, 0.3);
    }}
    .battery-fill.low {{ background: linear-gradient(90deg, var(--red), var(--orange)); box-shadow: 0 0 10px rgba(248, 113, 113, 0.3); }}
    .battery-fill.medium {{ background: linear-gradient(90deg, var(--yellow), var(--orange)); box-shadow: 0 0 10px rgba(251, 191, 36, 0.3); }}
    .battery-text {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      font-weight: 600;
      min-width: 40px;
      text-align: right;
      color: var(--text);
    }}
    
    /* Event Map - compact */
    .map-wrapper {{
      flex: 1;
      min-height: 180px;
    }}
    .map-container {{
      position: relative;
      width: 100%;
      height: 100%;
      min-height: 160px;
      border-radius: 12px;
      overflow: hidden;
    }}
    .map-canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .map-legend {{
      position: absolute;
      bottom: 12px;
      left: 12px;
      display: flex;
      gap: 16px;
      padding: 8px 14px;
      background: rgba(0,0,0,0.5);
      backdrop-filter: blur(10px);
      border-radius: 10px;
      font-size: 13px;
      color: var(--text-dim);
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      box-shadow: 0 0 8px currentColor;
    }}
    .legend-line {{
      width: 20px;
      height: 3px;
      border-radius: 2px;
    }}
    
    /* Joystick manual control */
    .joystick-row {{
      display: flex;
      gap: 12px;
      justify-content: center;
      align-items: center;
    }}
    .joystick-col {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }}
    .joystick-canvas {{
      width: 90px;
      height: 90px;
      border-radius: 50%;
      cursor: grab;
      touch-action: none;
    }}
    .joystick-canvas:active {{ cursor: grabbing; }}
    .joystick-label {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 500;
    }}
    .joystick-status {{
      text-align: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    .joystick-status.active {{ color: var(--cyan); }}

    /* Dashboard cleanup overrides */
    .main-layout {{
      grid-template-columns: minmax(220px, 240px) minmax(0, 1fr);
      gap: 10px;
      min-height: calc(100vh - 84px);
    }}
    @media (max-width: 1200px) {{
      .main-layout {{ grid-template-columns: 1fr; }}
    }}
    .main-content {{
      min-width: 0;
    }}
    .map-wrapper {{
      display: flex;
      flex-direction: column;
      flex: 0 0 auto;
      height: clamp(320px, 46vh, 520px);
      min-height: 320px;
    }}
    .map-card-body {{
      padding: 12px;
      display: flex;
      flex: 1 1 auto;
      min-height: 0;
      height: auto;
    }}
    .map-container {{
      flex: 1 1 auto;
      min-height: 0;
      height: auto;
    }}
    .motion-card {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .speed-hero {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .speed-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
    }}
    .speed-value {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 34px;
      font-weight: 700;
      line-height: 0.95;
      color: var(--text);
    }}
    .speed-caption {{
      margin-top: 4px;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-muted);
    }}
    .speed-badge {{
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(248, 113, 113, 0.2);
      background: rgba(248, 113, 113, 0.08);
      color: #fda4af;
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .speed-track-shell {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .speed-track {{
      position: relative;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(148, 163, 184, 0.12);
      border: 1px solid rgba(148, 163, 184, 0.16);
    }}
    .speed-fill {{
      position: absolute;
      inset: 0 auto 0 0;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--teal));
      border-radius: inherit;
      transition: width 0.25s ease;
    }}
    .speed-limit-dot {{
      position: absolute;
      top: 50%;
      left: 0%;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #f87171;
      border: 2px solid rgba(15, 23, 42, 0.9);
      transform: translate(-50%, -50%);
      box-shadow: 0 0 14px rgba(248, 113, 113, 0.45);
    }}
    .speed-track-scale {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }}
    .command-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .command-card {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 9px 10px;
      border-radius: 10px;
      background: linear-gradient(145deg, rgba(255, 255, 255, 0.03), transparent);
      border: 1px solid rgba(148, 163, 184, 0.12);
    }}
    .command-card-top {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
    }}
    .command-name {{
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-muted);
    }}
    .command-reading {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
    }}
    .command-meter {{
      position: relative;
      height: 8px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.10);
      overflow: hidden;
    }}
    .command-meter-center {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 1px;
      background: rgba(241, 245, 249, 0.35);
    }}
    .command-meter-fill {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 0%;
      border-radius: 999px;
      transition: left 0.2s ease, width 0.2s ease;
    }}
    .command-meter-fill.positive {{
      background: linear-gradient(90deg, rgba(45, 212, 191, 0.25), rgba(34, 211, 238, 0.95));
    }}
    .command-meter-fill.negative {{
      background: linear-gradient(90deg, rgba(248, 113, 113, 0.95), rgba(248, 113, 113, 0.22));
    }}
    .map-toolbar {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .map-toolbar-main {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-width: 0;
    }}
    .map-toolbar-side {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .map-summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .summary-chip {{
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(148, 163, 184, 0.14);
      color: var(--text-dim);
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .summary-chip.good {{
      color: #6ee7b7;
      border-color: rgba(52, 211, 153, 0.24);
      background: rgba(52, 211, 153, 0.08);
    }}
    .summary-chip.warn {{
      color: #fbbf24;
      border-color: rgba(251, 191, 36, 0.22);
      background: rgba(251, 191, 36, 0.08);
    }}
    .summary-chip.danger {{
      color: #fda4af;
      border-color: rgba(248, 113, 113, 0.24);
      background: rgba(248, 113, 113, 0.08);
    }}
    .map-info {{
      color: var(--text-muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .map-save-btn {{
      padding: 4px 10px;
      font-size: 12px;
    }}
    .map-legend {{
      position: absolute;
      left: 12px;
      bottom: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      padding: 7px 10px;
      border-radius: 10px;
      background: rgba(0, 0, 0, 0.52);
      color: var(--text-dim);
      font-size: 10px;
      backdrop-filter: blur(10px);
    }}
    .map-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
    }}
    .map-legend-mark {{
      width: 14px;
      height: 3px;
      border-radius: 2px;
      background: #94a3b8;
    }}
    .map-legend-mark.selected {{ background: #fbbf24; }}
    .map-legend-mark.rejected {{ background: #f87171; }}
    .map-legend-x {{
      color: #f87171;
      font-size: 14px;
      font-weight: 800;
      line-height: 10px;
    }}
    .clearance-panel {{
      position: absolute;
      top: 12px;
      right: 12px;
      padding: 10px;
      min-width: 132px;
      border-radius: 12px;
      background: rgba(0, 0, 0, 0.48);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }}
    .clearance-title {{
      margin-bottom: 8px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
    }}
    .clearance-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      align-items: center;
    }}
    .clearance-spacer {{
      min-height: 28px;
    }}
    .clearance-chip {{
      min-height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 4px 6px;
      border-radius: 8px;
      background: rgba(148, 163, 184, 0.10);
      color: var(--text-dim);
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      border: 1px solid transparent;
    }}
    .clearance-chip.safe {{
      color: #67e8f9;
      border-color: rgba(34, 211, 238, 0.24);
    }}
    .clearance-chip.warn {{
      color: #fbbf24;
      border-color: rgba(251, 191, 36, 0.24);
    }}
    .clearance-chip.danger {{
      color: #fda4af;
      border-color: rgba(248, 113, 113, 0.24);
    }}
    .bottom-panels {{
      display: grid;
      grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
      gap: 12px;
    }}
    .collapsible-card > summary {{
      list-style: none;
      cursor: pointer;
      user-select: none;
    }}
    .collapsible-card > summary::-webkit-details-marker {{
      display: none;
    }}
    .collapsible-card:not([open]) {{
      overflow: hidden;
    }}
    @media (max-width: 1200px) {{
      .bottom-panels {{
        grid-template-columns: 1fr;
      }}
      .map-toolbar-side {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <!-- Top Bar -->
  <div class="status-bar">
    <div class="status-left">
      <div class="status-dot" id="conn-dot"></div>
      <span id="conn-status">Connecting...</span>
      <span style="color: var(--text-muted); margin-left: 8px;" id="uptime">—</span>
    </div>
    <button class="theme-toggle" onclick="toggleTheme()">
      <span id="theme-icon">☀️</span> <span id="theme-text">Light</span>
    </button>
  </div>

  <!-- Main 2-Column Layout -->
  <div class="main-layout">
    
    <!-- Left Sidebar -->
    <div class="sidebar">
      
      <!-- Navigation Status -->
    <div class="card">
        <div class="card-body">
          <div class="mode-hero">
            <div class="mode-value idle" id="nav-mode">IDLE</div>
            <div class="mode-sub" id="nav-status">Waiting for mission start</div>
      </div>
          <div class="stats-grid">
            <div class="stat">
              <div class="stat-value cyan" id="events-found">0</div>
              <div class="stat-label">Events</div>
            </div>
            <div class="stat">
              <div class="stat-value orange" id="events-remaining">—</div>
              <div class="stat-label">To Exit</div>
            </div>
            <div class="stat">
              <div class="stat-value red" id="dead-ends">0</div>
              <div class="stat-label">Dead Ends</div>
            </div>
          </div>
      </div>
    </div>

      <!-- Motion -->
      <div class="card">
        <div class="card-header">Motion</div>
        <div class="card-body motion-card">
          <div class="speed-hero">
            <div class="speed-header">
              <div>
                <div class="speed-value"><span id="speed-total">—</span> <span style="font-size: 15px; color: var(--text-muted);">m/s</span></div>
                <div class="speed-caption">Total speed</div>
              </div>
              <div class="speed-badge" id="speed-limit-label">Obstacle limit —</div>
            </div>
            <div class="speed-track-shell">
              <div class="speed-track">
                <div class="speed-fill" id="speed-fill"></div>
                <div class="speed-limit-dot" id="speed-limit-dot"></div>
              </div>
              <div class="speed-track-scale">
                <span>0</span>
                <span id="speed-track-max">1.0 m/s scale</span>
              </div>
            </div>
          </div>

          <div class="command-grid">
            <div class="command-card">
              <div class="command-card-top">
                <span class="command-name">X</span>
                <span class="command-reading" id="cmd-vx">—</span>
              </div>
              <div class="command-meter">
                <div class="command-meter-center"></div>
                <div class="command-meter-fill" id="cmd-vx-bar"></div>
              </div>
            </div>
            <div class="command-card">
              <div class="command-card-top">
                <span class="command-name">Y</span>
                <span class="command-reading" id="cmd-vy">—</span>
              </div>
              <div class="command-meter">
                <div class="command-meter-center"></div>
                <div class="command-meter-fill" id="cmd-vy-bar"></div>
              </div>
            </div>
            <div class="command-card">
              <div class="command-card-top">
                <span class="command-name">Z</span>
                <span class="command-reading" id="cmd-vz">—</span>
              </div>
              <div class="command-meter">
                <div class="command-meter-center"></div>
                <div class="command-meter-fill" id="cmd-vz-bar"></div>
              </div>
            </div>
            <div class="command-card">
              <div class="command-card-top">
                <span class="command-name">Yaw</span>
                <span class="command-reading" id="cmd-yaw">—</span>
              </div>
              <div class="command-meter">
                <div class="command-meter-center"></div>
                <div class="command-meter-fill" id="cmd-yaw-bar"></div>
              </div>
            </div>
          </div>

          <div class="battery-widget">
            <div class="battery-bar">
              <div class="battery-fill" id="battery-fill" style="width: 0%"></div>
            </div>
            <div class="battery-text" id="battery-text">—%</div>
          </div>
        </div>
      </div>

      <!-- Mission Control -->
      <div class="card">
        <div class="card-header">Mission Control</div>
        <div class="card-body">
          <div class="mission-info">
            <div class="mission-row">
              <span class="mission-label">Expected</span>
              <span class="mission-value" id="expected-event">—</span>
            </div>
            <div class="mission-row">
              <span class="mission-label">Return Target</span>
              <span class="mission-value" id="return-target">—</span>
            </div>
            <div class="mission-row">
              <span class="mission-label">Route Progress</span>
              <span class="mission-value" id="return-progress">—</span>
            </div>
            <div class="mission-row" id="return-gate-row" hidden
                 title="The route-distance window in which event matching is enabled.">
              <span class="mission-label">Match Gate</span>
              <div class="mission-value-stack">
                <div class="mission-value-line">
                  <span id="return-gate-range">—</span>
                  <span class="mission-state" id="return-gate-state">—</span>
                </div>
                <span class="mission-meta" id="return-gate-meta">—</span>
              </div>
            </div>
            <div class="mission-row" id="pose-guard-row" hidden
                 title="Maximum pose correction currently allowed from the last trusted localization anchor.">
              <span class="mission-label">Pose Guard</span>
              <div class="mission-value-stack">
                <div class="mission-value-line" id="pose-guard-limit">—</div>
                <span class="mission-meta" id="pose-guard-meta">—</span>
              </div>
            </div>
          </div>
          <div class="btn-row">
            <button class="btn btn-primary" onclick="post('/api/mission/start')">Start</button>
            <button class="btn btn-danger" onclick="post('/api/mission/stop')">Stop</button>
          </div>
          <div class="btn-row" style="margin-top: 4px;">
            <button class="btn btn-warning" onclick="post('/api/return_home')">Return Home</button>
          </div>
        </div>
      </div>

    </div>
    
    <!-- Main Content -->
    <div class="main-content">
      
      <!-- Live Event Map -->
      <div class="card map-wrapper">
        <div class="card-header map-toolbar">
          <div class="map-toolbar-main">
            <span>Navigation Event Map · Odometry Frame</span>
            <div class="map-summary">
              <span class="summary-chip" id="planner">Planner: —</span>
              <span class="summary-chip" id="planner-intent">Intent: —</span>
              <span class="summary-chip" id="planner-safety">Safety: —</span>
              <span class="summary-chip" id="planner-vx-gate">VX: —</span>
              <span class="summary-chip" id="graph-target">Graph: —</span>
            </div>
          </div>
          <div class="map-toolbar-side">
            <span id="map-info" class="map-info">—</span>
            <button class="btn map-save-btn" onclick="saveMapAsPng()">Save PNG</button>
          </div>
        </div>
        <div class="card-body map-card-body">
          <div class="map-container">
            <canvas class="map-canvas" id="event-map-canvas"></canvas>
            <div class="map-legend">
              <span class="map-legend-item"><i class="map-legend-mark"></i>Available arm</span>
              <span class="map-legend-item"><i class="map-legend-mark selected"></i>Selected arm</span>
              <span class="map-legend-item"><i class="map-legend-mark rejected"></i>Rejected arm</span>
              <span class="map-legend-item"><b class="map-legend-x">×</b>Dead end</span>
            </div>
            <div class="clearance-panel">
              <div class="clearance-title">Obstacle clearance</div>
              <div class="clearance-grid">
                <div class="clearance-spacer"></div>
                <div class="clearance-chip" id="clearance-up">U —</div>
                <div class="clearance-spacer"></div>
                <div class="clearance-chip" id="clearance-left">L —</div>
                <div class="clearance-chip" id="clearance-front">F —</div>
                <div class="clearance-chip" id="clearance-right">R —</div>
                <div class="clearance-spacer"></div>
                <div class="clearance-chip" id="clearance-back">B —</div>
                <div class="clearance-spacer"></div>
                <div class="clearance-spacer"></div>
                <div class="clearance-chip" id="clearance-down">D —</div>
                <div class="clearance-spacer"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Bottom Row: Profile Matching + Activity -->
      <div class="bottom-panels">
        
        <!-- Profile Matching -->
    <div class="card">
          <div class="card-header">
            <span>Profile Matching</span>
            <span id="match-status" style="color: var(--text-muted)">—</span>
      </div>
          <div class="card-body" style="display: flex; gap: 12px; align-items: center;">
            <canvas id="profile-canvas" width="120" height="120" style="width: 120px; height: 120px; border-radius: 8px;"></canvas>
            <div style="display: flex; flex-direction: column; gap: 8px; flex: 1;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <div class="match-score low" id="match-score" style="font-size: 24px;">—</div>
                <span style="color: var(--text-muted); font-size: 13px;">confidence</span>
              </div>
              <div style="font-size: 12px; display: flex; flex-direction: column; gap: 4px;">
                <div><span style="color: var(--text-muted);">Target:</span> <span id="match-target" style="font-weight: 600;">—</span></div>
                <div><span style="color: var(--text-muted);">Last:</span> <span id="match-last" style="font-weight: 600;">—</span></div>
              </div>
              <div style="display: flex; gap: 12px; font-size: 12px; color: var(--text-muted);">
                <span>● <span style="color: #22d3ee;">Current</span></span>
                <span>● <span style="color: #38bdf8;">Stored</span></span>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Activity Log -->
        <div class="card">
          <div class="card-header">Activity</div>
          <div class="card-body">
            <div class="activity-list" id="activity-list">
              <div class="activity-item info">
                <span class="activity-time">0:00</span>
                <span class="activity-msg">Waiting for activity...</span>
              </div>
            </div>
          </div>
        </div>
        
      </div>

      <details class="card collapsible-card">
        <summary class="card-header">Manual Control</summary>
        <div class="card-body">
          <div class="joystick-row">
            <div class="joystick-col">
              <canvas class="joystick-canvas" id="joy-left" width="180" height="180"></canvas>
              <div class="joystick-label">Move</div>
            </div>
            <div class="joystick-col">
              <canvas class="joystick-canvas" id="joy-right" width="180" height="180"></canvas>
              <div class="joystick-label">Alt / Yaw</div>
            </div>
          </div>
          <div class="joystick-status" id="joy-status">Idle</div>
        </div>
      </details>
    </div>
  </div>
  
  <script>
    const POLL_MS = {poll_ms};
    
    // Theme management
    function toggleTheme() {{
      const html = document.documentElement;
      const current = html.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateThemeButton(next);
    }}
    
    function updateThemeButton(theme) {{
      const icon = document.getElementById('theme-icon');
      const text = document.getElementById('theme-text');
      if (theme === 'dark') {{
        if (icon) icon.textContent = '☀️';
        if (text) text.textContent = 'Light';
      }} else {{
        if (icon) icon.textContent = '🌙';
        if (text) text.textContent = 'Dark';
      }}
    }}
    
    // Load saved theme
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeButton(saved);
    
    function fmt(x, d=1) {{
      if (x === null || x === undefined || Number.isNaN(x)) return '—';
      return Number(x).toFixed(d);
    }}

    function fmtAngleRad(rad, d=0) {{
      if (rad === null || rad === undefined || Number.isNaN(rad)) return '—';
      return Number(rad * 180 / Math.PI).toFixed(d) + '°';
    }}

    function formatGraphTarget(gt) {{
      if (!gt || typeof gt !== 'object') return '—';
      const kind = String(gt.kind || '').toLowerCase();
      if (!kind) return '—';
      if (kind === 'commit') {{
        const shortId = gt.event_short || '—';
        const mode = gt.mode ? `/${{gt.mode}}` : '';
        const arm = fmtAngleRad(gt.selected_arm_angle_rad, 0);
        const gate = gt.gate_radius_m !== null && gt.gate_radius_m !== undefined ? `${{fmt(gt.gate_radius_m, 2)}}m` : '—';
        return `COMMIT ${{shortId}}${{mode}} @ ${{arm}} r=${{gate}}`;
      }}
      if (kind === 'reverse') {{
        return `REVERSE${{gt.mode ? `/${{gt.mode}}` : ''}}`;
      }}
      if (kind === 'clear') {{
        return 'CLEAR';
      }}
      return kind.toUpperCase();
    }}

    function formatPlannerIntent(debug) {{
      if (!debug || typeof debug !== 'object') return '—';
      const source = String(debug.intent_source || '').toLowerCase();
      if (source === 'committed') {{
        const committed = debug.committed || null;
        if (committed && committed.event_short) {{
          return `Committed -> ${{committed.event_short}}${{committed.mode ? `/${{committed.mode}}` : ''}}`;
        }}
        return 'Committed';
      }}
      if (source === 'reverse_graph') {{
        const mode = debug.reverse_mode ? `/${{String(debug.reverse_mode)}}` : '';
        return `Reverse graph${{mode}}`;
      }}
      if (source === 'speculative') return 'Approach junction';
      if (source === 'track_edge') return 'Track edge';
      return 'None';
    }}

    function formatVxGate(debug) {{
      if (!debug || typeof debug !== 'object') return '—';
      const reason = String(debug.vx_gate_reason || '').toLowerCase();
      if (!reason) return '—';
      if (reason === 'free') return 'Free';
      if (reason === 'obstacle') {{
        return debug.front_distance_m !== null && debug.front_distance_m !== undefined
          ? `Obstacle (${{fmt(debug.front_distance_m, 2)}}m)`
          : 'Obstacle';
      }}
      if (reason === 'yaw') {{
        return debug.yaw_error_rad !== null && debug.yaw_error_rad !== undefined
          ? `Yaw (${{fmtAngleRad(debug.yaw_error_rad, 0)}} err)`
          : 'Yaw';
      }}
      if (reason === 'obstacle+yaw') return 'Obstacle + yaw';
      if (reason === 'no_graph') return 'No graph';
      if (reason === 'waiting_odom') return 'Waiting odom';
      if (reason === 'inactive') return 'Inactive';
      return reason;
    }}
    
    function setModeClass(mode) {{
      const el = document.getElementById('nav-mode');
      if (!el) return;
      el.className = 'mode-value';
      const m = (mode || '').toLowerCase();
      if (m.includes('explore')) el.classList.add('explore');
      else if (m.includes('backtrack')) el.classList.add('backtrack');
      else if (m.includes('return')) el.classList.add('return');
      else el.classList.add('idle');
    }}

    function fmtSigned(x, d=2) {{
      if (x === null || x === undefined || Number.isNaN(x)) return '—';
      const value = Number(x);
      const sign = value > 0 ? '+' : '';
      return sign + value.toFixed(d);
    }}

    function updateSummaryChip(id, text, tone='') {{
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = text;
      el.className = 'summary-chip';
      if (tone) el.classList.add(tone);
    }}

    function safetyTone(mode) {{
      const value = String(mode || '').toUpperCase();
      if (!value || value === 'NONE') return 'good';
      if (value === 'BLOCKED' || value === 'DEAD_END') return 'danger';
      return 'warn';
    }}

    function vxGateTone(debug) {{
      const reason = String(debug?.vx_gate_reason || '').toLowerCase();
      if (!reason || reason === 'free') return 'good';
      if (reason === 'obstacle' || reason === 'obstacle+yaw' || reason === 'yaw') return 'warn';
      return '';
    }}

    function updateSpeedPanel(speed, debug) {{
      const totalEl = document.getElementById('speed-total');
      const limitLabel = document.getElementById('speed-limit-label');
      const fillEl = document.getElementById('speed-fill');
      const dotEl = document.getElementById('speed-limit-dot');
      const scaleEl = document.getElementById('speed-track-max');
      const speedVal = speed === null || speed === undefined ? 0.0 : Math.max(0.0, Number(speed));
      const rawLimit = debug && debug.obstacle_speed_limit_mps !== null && debug.obstacle_speed_limit_mps !== undefined
        ? Number(debug.obstacle_speed_limit_mps)
        : null;
      const obstacleLimit = rawLimit !== null && !Number.isNaN(rawLimit) ? Math.max(0.0, rawLimit) : null;
      const cmdVx = debug && debug.cmd_vx_mps !== null && debug.cmd_vx_mps !== undefined
        ? Math.abs(Number(debug.cmd_vx_mps))
        : 0.0;
      const displayMax = Math.max(1.0, speedVal, obstacleLimit || 0.0, cmdVx, 0.8);

      if (totalEl) totalEl.textContent = fmt(speedVal, 2);
      if (limitLabel) {{
        limitLabel.textContent = obstacleLimit === null
          ? 'Obstacle limit —'
          : `Obstacle limit ${{fmt(obstacleLimit, 2)}} m/s`;
      }}
      if (fillEl) {{
        fillEl.style.width = `${{Math.max(0, Math.min(100, 100 * speedVal / displayMax))}}%`;
      }}
      if (dotEl) {{
        if (obstacleLimit === null) {{
          dotEl.style.display = 'none';
        }} else {{
          dotEl.style.display = 'block';
          dotEl.style.left = `${{Math.max(0, Math.min(100, 100 * obstacleLimit / displayMax))}}%`;
        }}
      }}
      if (scaleEl) scaleEl.textContent = `${{fmt(displayMax, 1)}} m/s scale`;
    }}

    function updateCommandMetric(readingId, barId, value, unit, maxAbs) {{
      const readingEl = document.getElementById(readingId);
      const barEl = document.getElementById(barId);
      const numeric = value === null || value === undefined || Number.isNaN(Number(value))
        ? null
        : Number(value);
      if (readingEl) {{
        readingEl.textContent = numeric === null ? '—' : `${{fmtSigned(numeric, 2)}} ${{unit}}`;
      }}
      if (!barEl) return;
      const ratio = numeric === null ? 0.0 : Math.max(-1.0, Math.min(1.0, numeric / maxAbs));
      if (ratio >= 0.0) {{
        barEl.style.left = '50%';
        barEl.style.width = `${{ratio * 50.0}}%`;
      }} else {{
        barEl.style.left = `${{50.0 + ratio * 50.0}}%`;
        barEl.style.width = `${{Math.abs(ratio) * 50.0}}%`;
      }}
      barEl.className = 'command-meter-fill ' + (ratio < 0.0 ? 'negative' : 'positive');
    }}

    function updateCommandPanel(debug) {{
      const data = debug && typeof debug === 'object' ? debug : {{}};
      updateCommandMetric('cmd-vx', 'cmd-vx-bar', data.cmd_vx_mps, 'm/s', 1.0);
      updateCommandMetric('cmd-vy', 'cmd-vy-bar', data.cmd_vy_mps, 'm/s', 1.0);
      updateCommandMetric('cmd-vz', 'cmd-vz-bar', data.cmd_vz_mps, 'm/s', 1.0);
      updateCommandMetric('cmd-yaw', 'cmd-yaw-bar', data.cmd_yaw_rate_rps, 'rad/s', 0.8);
    }}

    function updateClearanceChip(id, label, value) {{
      const el = document.getElementById(id);
      if (!el) return;
      const numeric = value === null || value === undefined || Number.isNaN(Number(value))
        ? null
        : Number(value);
      el.className = 'clearance-chip';
      el.textContent = `${{label}} ${{numeric === null ? '—' : fmt(numeric, 1) + 'm'}}`;
      if (numeric === null) return;
      if (numeric < 0.5) el.classList.add('danger');
      else if (numeric < 1.0) el.classList.add('warn');
      else el.classList.add('safe');
    }}

    function updateObstacleClearances(clearances) {{
      const data = clearances && typeof clearances === 'object' ? clearances : {{}};
      updateClearanceChip('clearance-front', 'F', data.front);
      updateClearanceChip('clearance-back', 'B', data.back);
      updateClearanceChip('clearance-left', 'L', data.left);
      updateClearanceChip('clearance-right', 'R', data.right);
      updateClearanceChip('clearance-up', 'U', data.up);
      updateClearanceChip('clearance-down', 'D', data.down);
    }}
    
    function updateBattery(pct) {{
      const fill = document.getElementById('battery-fill');
      const text = document.getElementById('battery-text');
      if (pct === null || pct === undefined) {{
        if (fill) fill.style.width = '0%';
        if (text) text.textContent = '—%';
        return;
      }}
      const p = Math.max(0, Math.min(100, pct));
      if (fill) {{
        fill.style.width = p + '%';
        fill.classList.remove('low', 'medium');
        if (p < 20) fill.classList.add('low');
        else if (p < 50) fill.classList.add('medium');
      }}
      if (text) text.textContent = fmt(p, 0) + '%';
    }}
    
    function updateMatchScore(conf) {{
      const el = document.getElementById('match-score');
      if (!el) return;
      if (conf === null || conf === undefined) {{
        el.textContent = '—';
        el.className = 'match-score low';
        return;
      }}
      const pct = Math.max(0, Math.min(100, conf * 100));
      el.textContent = fmt(pct, 0) + '%';
      el.className = 'match-score';
      if (pct >= 80) el.classList.add('high');
      else if (pct >= 50) el.classList.add('medium');
      else el.classList.add('low');
    }}
    
    // ========== Profile Matching ==========
    function drawProfile(canvas, current, stored) {{
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const w = canvas.width, h = canvas.height;
      const cx = w / 2, cy = h / 2;
      const r = Math.min(w, h) * 0.42;
      
      ctx.clearRect(0, 0, w, h);
      
      // Background
      ctx.fillStyle = 'rgba(15, 15, 35, 0.6)';
      ctx.beginPath();
      ctx.arc(cx, cy, r + 5, 0, Math.PI * 2);
      ctx.fill();
      
      // Grid rings
      ctx.strokeStyle = 'rgba(100, 150, 200, 0.15)';
      ctx.lineWidth = 1;
      for (let i = 0.5; i <= 1; i += 0.5) {{
        ctx.beginPath();
        ctx.arc(cx, cy, r * i, 0, Math.PI * 2);
        ctx.stroke();
      }}
      
      // Draw stored profile (blue, dashed)
      if (stored && stored.angles && stored.distances && stored.angles.length > 2) {{
        drawProfileCurve(ctx, cx, cy, r, stored, '#38bdf8', true);
      }}
      
      // Draw current profile (cyan, solid)
      if (current && current.angles && current.distances && current.angles.length > 2) {{
        drawProfileCurve(ctx, cx, cy, r, current, '#22d3ee', false);
      }}
      
      // Center dot
      ctx.fillStyle = '#22d3ee';
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
      
      // Show "waiting" text if no profiles
      if ((!current || !current.angles || current.angles.length < 3) && 
          (!stored || !stored.angles || stored.angles.length < 3)) {{
        ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
        ctx.font = '12px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting...', cx, cy);
      }}
    }}
    
    function drawProfileCurve(ctx, cx, cy, maxR, profile, color, dashed) {{
      const angles = profile.angles || [];
      const dists = profile.distances || [];
      if (angles.length === 0 || dists.length === 0) return;
      
      let maxDist = 0.1;
      for (const d of dists) if (d > maxDist) maxDist = d;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) ctx.setLineDash([3, 3]);
      else ctx.setLineDash([]);
      
      ctx.beginPath();
      for (let i = 0; i <= angles.length; i++) {{
        const idx = i % angles.length;
        // Rotate 180 degrees clockwise to match the rotated map view
        const ang = angles[idx] * Math.PI / 180 + Math.PI;
        const r = (dists[idx] / maxDist) * maxR * 0.9;
        const x = cx + Math.cos(ang) * r;
        const y = cy - Math.sin(ang) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }}
    
    function updateActivity(log) {{
      const el = document.getElementById('activity-list');
      if (!el || !log || !log.length) return;
      
      el.innerHTML = log.map(item => `
        <div class="activity-item ${{item.event_type}}">
          <span class="activity-time">${{item.time}}</span>
          <span class="activity-msg">${{item.message}}</span>
        </div>
      `).join('');
    }}
    
    // ========== Event Map ==========
    
    // Compute navigation direction relative to entry
    // Returns {{ dir: 'F'|'B'|'L'|'R', count: number }} or null
    function getNavDirection(entryAngle, selectedIdx, pathAngles) {{
      if (selectedIdx === null || selectedIdx === undefined || !pathAngles || pathAngles.length === 0) return null;
      if (selectedIdx < 0 || selectedIdx >= pathAngles.length) return null;
      
      const selectedAngle = pathAngles[selectedIdx];
      // Relative angle from entry (entry points backward, so forward is entry + PI)
      const forwardAngle = entryAngle + Math.PI;
      let relAngle = selectedAngle - forwardAngle;
      // Normalize to [-PI, PI]
      while (relAngle > Math.PI) relAngle -= 2 * Math.PI;
      while (relAngle < -Math.PI) relAngle += 2 * Math.PI;
      
      // Determine quadrant
      let dir;
      if (Math.abs(relAngle) < Math.PI / 4) dir = 'F';  // Forward
      else if (Math.abs(relAngle) > 3 * Math.PI / 4) dir = 'B';  // Back
      else if (relAngle > 0) dir = 'L';  // Left
      else dir = 'R';  // Right
      
      // Count how many paths are in the same quadrant
      let count = 0;
      let myRank = 0;
      for (let i = 0; i < pathAngles.length; i++) {{
        let ra = pathAngles[i] - forwardAngle;
        while (ra > Math.PI) ra -= 2 * Math.PI;
        while (ra < -Math.PI) ra += 2 * Math.PI;
        
        let d;
        if (Math.abs(ra) < Math.PI / 4) d = 'F';
        else if (Math.abs(ra) > 3 * Math.PI / 4) d = 'B';
        else if (ra > 0) d = 'L';
        else d = 'R';
        
        if (d === dir) {{
          count++;
          if (i < selectedIdx) myRank++;
        }}
      }}
      
      return {{ dir, count, rank: myRank + 1 }};
    }}
    
    function angularDistance(a, b) {{
      return Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
    }}

    function drawMapArrow(ctx, x0, y0, x1, y1, color, width) {{
      const angle = Math.atan2(y1 - y0, x1 - x0);
      const headLen = 7;
      const headAngle = 0.48;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.moveTo(x1, y1);
      ctx.lineTo(
        x1 - headLen * Math.cos(angle - headAngle),
        y1 - headLen * Math.sin(angle - headAngle)
      );
      ctx.moveTo(x1, y1);
      ctx.lineTo(
        x1 - headLen * Math.cos(angle + headAngle),
        y1 - headLen * Math.sin(angle + headAngle)
      );
      ctx.stroke();
    }}

    function drawMapCross(ctx, x, y, radius, color, width) {{
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x - radius, y - radius);
      ctx.lineTo(x + radius, y + radius);
      ctx.moveTo(x + radius, y - radius);
      ctx.lineTo(x - radius, y + radius);
      ctx.stroke();
    }}

    function drawEventMap(canvas, trajectory, events, dronePos, targetEventId, expectedEventId, startPos, graphTarget) {{
      try {{
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      // Resize canvas to container
      const container = canvas.parentElement;
      if (!container) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      if (rect.width < 10 || rect.height < 10) return;  // Skip if too small
      
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.scale(dpr, dpr);
      
      const w = rect.width, h = rect.height;
      
      // Background gradient
      const bgGrad = ctx.createLinearGradient(0, 0, w, h);
      bgGrad.addColorStop(0, 'rgba(15, 15, 35, 0.95)');
      bgGrad.addColorStop(1, 'rgba(26, 26, 46, 0.95)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);
      
      // Calculate bounds
      let minX = dronePos.x, maxX = dronePos.x;
      let minY = dronePos.y, maxY = dronePos.y;
      
      // Include starting position in bounds
      if (startPos && startPos.x !== null && startPos.y !== null) {{
        minX = Math.min(minX, startPos.x);
        maxX = Math.max(maxX, startPos.x);
        minY = Math.min(minY, startPos.y);
        maxY = Math.max(maxY, startPos.y);
      }}
      
      if (trajectory && trajectory.length > 0) {{
        for (const p of trajectory) {{
          minX = Math.min(minX, p.x);
          maxX = Math.max(maxX, p.x);
          minY = Math.min(minY, p.y);
          maxY = Math.max(maxY, p.y);
        }}
      }}
      
      if (events && events.length > 0) {{
        for (const e of events) {{
          minX = Math.min(minX, e.x);
          maxX = Math.max(maxX, e.x);
          minY = Math.min(minY, e.y);
          maxY = Math.max(maxY, e.y);
        }}
      }}

      if (graphTarget && Array.isArray(graphTarget.event_center_xy) && graphTarget.event_center_xy.length >= 2) {{
        minX = Math.min(minX, graphTarget.event_center_xy[0]);
        maxX = Math.max(maxX, graphTarget.event_center_xy[0]);
        minY = Math.min(minY, graphTarget.event_center_xy[1]);
        maxY = Math.max(maxY, graphTarget.event_center_xy[1]);
      }}

      // Add padding
      const pad = 3.0;
      minX -= pad; maxX += pad;
      minY -= pad; maxY += pad;
      
      // Scale to fit
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const scale = Math.min((w - 80) / rangeX, (h - 60) / rangeY);
      const offsetX = (w - rangeX * scale) / 2;
      const offsetY = (h - rangeY * scale) / 2;
      
      // Transform functions (rotated 180 degrees for intuitive map view)
      const tx = x => w - offsetX - (x - minX) * scale;
      const ty = y => offsetY + (y - minY) * scale;
      
      // Draw subtle grid
      ctx.strokeStyle = 'rgba(100, 150, 200, 0.08)';
      ctx.lineWidth = 1;
      const gridStep = Math.pow(10, Math.floor(Math.log10(Math.max(rangeX, rangeY)))) || 1;
      for (let gx = Math.ceil(minX / gridStep) * gridStep; gx <= maxX; gx += gridStep) {{
        const sx = tx(gx);
        ctx.beginPath();
        ctx.moveTo(sx, 10);
        ctx.lineTo(sx, h - 10);
        ctx.stroke();
      }}
      for (let gy = Math.ceil(minY / gridStep) * gridStep; gy <= maxY; gy += gridStep) {{
        const sy = ty(gy);
        ctx.beginPath();
        ctx.moveTo(10, sy);
        ctx.lineTo(w - 10, sy);
        ctx.stroke();
      }}
      
      // Draw starting position marker (if no START event exists)
      const hasStartEvent = events && events.some(e => e.type === 'start');
      if (!hasStartEvent && startPos && startPos.x !== null && startPos.y !== null) {{
        const sx = tx(startPos.x);
        const sy = ty(startPos.y);
        
        // Draw "origin" marker
        ctx.shadowColor = 'rgba(139, 92, 246, 0.6)';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(sx, sy, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#8b5cf6';
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Label
        ctx.font = 'bold 12px JetBrains Mono';
        ctx.fillStyle = '#f1f5f9';
        ctx.textAlign = 'center';
        ctx.fillText('Origin', sx, sy - 14);
      }}
      
      // Draw trajectory with glow
      if (trajectory && trajectory.length > 1) {{
        // Glow layer
        ctx.shadowColor = 'rgba(34, 211, 238, 0.5)';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.4)';
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        for (let i = 0; i < trajectory.length; i++) {{
          const p = trajectory[i];
          if (i === 0) ctx.moveTo(tx(p.x), ty(p.y));
          else ctx.lineTo(tx(p.x), ty(p.y));
        }}
        ctx.stroke();
        ctx.shadowBlur = 0;
        
        // Main line with gradient
        const traj0 = trajectory[0], trajN = trajectory[trajectory.length - 1];
        const grad = ctx.createLinearGradient(tx(traj0.x), ty(traj0.y), tx(trajN.x), ty(trajN.y));
        grad.addColorStop(0, 'rgba(34, 211, 238, 0.3)');
        grad.addColorStop(0.5, 'rgba(34, 211, 238, 0.7)');
        grad.addColorStop(1, 'rgba(34, 211, 238, 1)');
        
        ctx.beginPath();
        ctx.strokeStyle = grad;
        ctx.lineWidth = 3;
        for (let i = 0; i < trajectory.length; i++) {{
          const p = trajectory[i];
          if (i === 0) ctx.moveTo(tx(p.x), ty(p.y));
          else ctx.lineTo(tx(p.x), ty(p.y));
        }}
        ctx.stroke();
      }}
      
      // Draw events
      if (events && events.length > 0) {{
        for (const e of events) {{
          const ex = tx(e.x);
          const ey = ty(e.y);
          // Check both full ID and if one contains the other (for partial matches)
          const isTarget = (targetEventId && e.id === targetEventId) || 
                          (expectedEventId && e.id === expectedEventId) ||
                          (targetEventId && e.id && (e.id.includes(targetEventId) || targetEventId.includes(e.id))) ||
                          (expectedEventId && e.id && (e.id.includes(expectedEventId) || expectedEventId.includes(e.id)));
          
          // Check if event was matched during return (passed through)
          const wasMatched = e.matched === true;
          
          let color = '#34d399';  // green for junctions
          let glowColor = 'rgba(52, 211, 153, 0.5)';
          let size = 8;
          
          if (e.type === 'start') {{
            color = '#60a5fa';  // blue for HOME/START
            glowColor = 'rgba(96, 165, 250, 0.5)';
            size = 10;
          }} else if (e.type === 'dead_end') {{
            color = '#f87171';
            glowColor = 'rgba(248, 113, 113, 0.5)';
            size = 7;
          }}
          
          // Override color for matched events (passed during return)
          if (wasMatched && !isTarget) {{
            color = '#a78bfa';  // Purple for matched/passed events
            glowColor = 'rgba(167, 139, 250, 0.5)';
          }}
          
          if (isTarget) {{
            color = '#fbbf24';
            glowColor = 'rgba(251, 191, 36, 0.6)';
            size = 12;
            
            // Animated ring effect
            const t = Date.now() / 1000;
            const pulseSize = size + 8 + Math.sin(t * 3) * 4;
            ctx.beginPath();
            ctx.arc(ex, ey, pulseSize, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(251, 191, 36, ${{0.2 + Math.sin(t * 3) * 0.1}})`;
            ctx.lineWidth = 2;
            ctx.stroke();
          }}
          
          // Glow
          ctx.shadowColor = glowColor;
          ctx.shadowBlur = 15;
          
          // Event circle
          ctx.beginPath();
          ctx.arc(ex, ey, size, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.shadowBlur = 0;
          
          // Border (checkmark pattern for matched events)
          if (wasMatched) {{
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 3;
            ctx.stroke();
          }} else {{
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
          }}
          
          // Calculate distance from drone to this event
          const distToEvent = Math.sqrt(Math.pow(e.x - dronePos.x, 2) + Math.pow(e.y - dronePos.y, 2));
          
          // Label with background - two lines for cleaner display
          const label = (e.type === 'start') ? 'S' : e.short_id;
          const showDist = isTarget && distToEvent > 0.3;
          
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          
          if (showDist) {{
            // Two-line label: ID on top, distance below
            ctx.font = 'bold 13px JetBrains Mono';
            const labelWidth = ctx.measureText(label).width;
            ctx.font = '12px JetBrains Mono';
            const distText = `${{distToEvent.toFixed(1)}}m`;
            const distWidth = ctx.measureText(distText).width;
            const boxWidth = Math.max(labelWidth, distWidth) + 10;
            
            // Background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            ctx.fillRect(ex - boxWidth/2, ey - size - 32, boxWidth, 28);
            
            // Event ID (top line)
            ctx.font = 'bold 13px JetBrains Mono';
            ctx.fillStyle = wasMatched ? '#a78bfa' : '#f1f5f9';
            ctx.fillText(label, ex, ey - size - 22);
            
            // Distance (bottom line, smaller, yellow for target)
            ctx.font = '12px JetBrains Mono';
            ctx.fillStyle = '#fbbf24';
            ctx.fillText(distText, ex, ey - size - 10);
          }} else {{
            // Single-line label
            ctx.font = 'bold 13px JetBrains Mono';
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(ex - textWidth/2 - 4, ey - size - 22, textWidth + 8, 16);
            
            ctx.fillStyle = wasMatched ? '#a78bfa' : '#f1f5f9';
            ctx.fillText(label, ex, ey - size - 14);
          }}
          
          // Draw every known junction arm. Rejected knowledge takes priority
          // over the historical selection so failed choices remain explicit.
          if (Array.isArray(e.path_angles) && e.type !== 'dead_end') {{
            const rejected = new Set(
              Array.isArray(e.dead_end_paths) ? e.dead_end_paths : []
            );
            e.path_angles.forEach((worldAngle, armIndex) => {{
              const isRejected = rejected.has(armIndex);
              const isSelected = armIndex === e.selected_idx;
              const isEntry = e.entry_angle !== undefined &&
                angularDistance(worldAngle, e.entry_angle) < 0.18;
              let armColor = isEntry ? '#64748b' : '#94a3b8';
              let armWidth = 2;
              if (isSelected) {{
                armColor = '#fbbf24';
                armWidth = 3;
              }}
              if (isRejected) {{
                armColor = '#f87171';
                armWidth = 3;
              }}

              const screenAngle = Math.PI - worldAngle;
              const armStart = size + 7;
              const armEnd = armStart + 24;
              const x0 = ex + Math.cos(screenAngle) * armStart;
              const y0 = ey + Math.sin(screenAngle) * armStart;
              const x1 = ex + Math.cos(screenAngle) * armEnd;
              const y1 = ey + Math.sin(screenAngle) * armEnd;
              drawMapArrow(
                ctx,
                x0,
                y0,
                x1,
                y1,
                armColor,
                armWidth
              );
              if (isRejected) {{
                drawMapCross(ctx, x1, y1, 4, '#f87171', 2);
              }}
            }});
          }}

          if (e.type === 'dead_end') {{
            drawMapCross(ctx, ex, ey, size + 3, '#fee2e2', 3);
          }}
        }}
      }}

      // Active graph_target overlay: source-of-truth committed intent
      if (
        graphTarget &&
        graphTarget.kind === 'commit' &&
        Array.isArray(graphTarget.event_center_xy) &&
        graphTarget.event_center_xy.length >= 2
      ) {{
        const gx = tx(graphTarget.event_center_xy[0]);
        const gy = ty(graphTarget.event_center_xy[1]);
        const gatePx = Math.max(14, (Number(graphTarget.gate_radius_m || 0) * scale));
        const armAngle = graphTarget.selected_arm_angle_rad;

        ctx.save();
        ctx.setLineDash([8, 6]);
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.95)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(gx, gy, gatePx, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (armAngle !== null && armAngle !== undefined) {{
          const arrowLen = 26;
          const arrowAng = Math.PI - armAngle;
          const ax = gx + Math.cos(arrowAng) * 8;
          const ay = gy + Math.sin(arrowAng) * 8;
          const aex = gx + Math.cos(arrowAng) * (8 + arrowLen);
          const aey = gy + Math.sin(arrowAng) * (8 + arrowLen);
          ctx.strokeStyle = '#ec4899';
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(aex, aey);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(aex, aey);
          ctx.lineTo(aex - 8 * Math.cos(arrowAng - 0.5), aey - 8 * Math.sin(arrowAng - 0.5));
          ctx.moveTo(aex, aey);
          ctx.lineTo(aex - 8 * Math.cos(arrowAng + 0.5), aey - 8 * Math.sin(arrowAng + 0.5));
          ctx.stroke();
        }}

        ctx.font = 'bold 12px JetBrains Mono';
        ctx.fillStyle = '#ec4899';
        ctx.textAlign = 'center';
        ctx.fillText('GT', gx, gy + gatePx + 14);
        ctx.restore();
      }}

      // Draw drone with glow
      const dx = tx(dronePos.x);
      const dy = ty(dronePos.y);
      // For 180-rotated map: drone rotation = 3PI/2 - yaw (so drone points in screen direction PI - yaw)
      const droneYaw = 3 * Math.PI / 2 - (dronePos.yaw || 0);
      
      ctx.save();
      ctx.translate(dx, dy);
      ctx.rotate(droneYaw);
      
      // Drone glow
      ctx.shadowColor = 'rgba(56, 189, 248, 0.8)';
      ctx.shadowBlur = 20;
      
      // Drone shape
      ctx.beginPath();
      ctx.moveTo(0, -14);
      ctx.lineTo(-9, 10);
      ctx.lineTo(0, 5);
      ctx.lineTo(9, 10);
      ctx.closePath();
      
      const droneGrad = ctx.createLinearGradient(0, -14, 0, 10);
      droneGrad.addColorStop(0, '#38bdf8');
      droneGrad.addColorStop(1, '#0ea5e9');
      ctx.fillStyle = droneGrad;
      ctx.fill();
      
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      ctx.restore();
      
      // Scale indicator
      const scaleLen = 60;
      const realLen = scaleLen / scale;
      const scaleX = w - 80;
      const scaleY = h - 25;
      
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(scaleX, scaleY);
      ctx.lineTo(scaleX + scaleLen, scaleY);
      ctx.moveTo(scaleX, scaleY - 4);
      ctx.lineTo(scaleX, scaleY + 4);
      ctx.moveTo(scaleX + scaleLen, scaleY - 4);
      ctx.lineTo(scaleX + scaleLen, scaleY + 4);
      ctx.stroke();
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.font = '13px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(fmt(realLen, 1) + 'm', scaleX + scaleLen/2, scaleY + 16);
      }} catch (err) {{ console.warn('drawEventMap error:', err); }}
    }}
    
    // Save map as PNG
    function saveMapAsPng() {{
      const canvas = document.getElementById('event-map-canvas');
      if (!canvas) return;
      const link = document.createElement('a');
      link.download = 'navigation_map_' + new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-') + '.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }}
    
    function post(path) {{
      fetch(path, {{method: 'POST'}}).then(r => r.json()).catch(() => {{}});
    }}
    
    async function tick() {{
      const dot = document.getElementById('conn-dot');
      const status = document.getElementById('conn-status');
      let j = null;
      
      try {{
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        const r = await fetch('/api/state', {{cache: 'no-store', signal: controller.signal}});
        clearTimeout(timeoutId);
        if (!r.ok) throw new Error('fetch failed');
        const text = await r.text();
        j = JSON.parse(text);
        if (!j || typeof j !== 'object') throw new Error('invalid json');
        
        if (dot) dot.classList.remove('offline');
        if (status) status.textContent = 'Connected';
        
        document.getElementById('uptime').textContent = fmt(j.server_age_s, 0) + 's';
        
        // Mode
        const mode = j.mode || 'IDLE';
        document.getElementById('nav-mode').textContent = mode;
        setModeClass(mode);
        
        // Navigation status text
        const navStatus = document.getElementById('nav-status');
        const missionCompleteElapsed = j.mission_complete_elapsed_s;
        if (navStatus) {{
          // Show mission complete message for 5 seconds, then return to IDLE
          if (missionCompleteElapsed !== null && missionCompleteElapsed !== undefined && missionCompleteElapsed < 5.0) {{
            navStatus.textContent = 'MISSION COMPLETE! Returned to base';
            navStatus.style.color = 'var(--green)';
            navStatus.style.fontWeight = '600';
          }} else {{
            navStatus.style.color = '';
            navStatus.style.fontWeight = '';
            const m = mode.toLowerCase();
            const planner = String(j.local_planner_state || '').toUpperCase();
            const rstat = String(j.return_status || '').toLowerCase();
            const expected = String(j.expected_event_id || '');

            if (m.includes('explore')) {{
              navStatus.textContent = 'Exploration: scanning for events';
            }} else if (m.includes('backtrack')) {{
              navStatus.textContent = 'Dead end: backtracking to junction';
            }} else if (m.includes('return')) {{
              if (rstat === 'returning' && expected) {{
                navStatus.textContent = 'RTB: searching for expected event';
              }} else if (rstat === 'success') {{
                navStatus.textContent = 'RTB: match confirmed';
              }} else {{
                navStatus.textContent = 'RTB: returning via event chain';
              }}
            }} else {{
              if (planner === 'DEAD_END') {{
                navStatus.textContent = 'Idle (planner reports DEAD_END)';
              }} else {{
                navStatus.textContent = 'Idle';
              }}
            }}
          }}
        }}
        
        // Event counters
        document.getElementById('events-found').textContent = j.junction_events || 0;
        document.getElementById('dead-ends').textContent = j.dead_ends || 0;
        
        const remaining = j.events_remaining;
        document.getElementById('events-remaining').textContent = 
          (remaining !== null && remaining !== undefined) ? remaining : '—';
        
        // Match info
        updateMatchScore(j.match_confidence);
        document.getElementById('match-target').textContent = j.expected_short || '—';
        document.getElementById('match-last').textContent = j.last_match_short || '—';
        
        const matchStatus = document.getElementById('match-status');
        if (matchStatus) {{
          if (j.return_status === 'returning') {{
            matchStatus.textContent = j.return_progress?.search_active
              ? 'Searching...'
              : 'Approaching distance window...';
          }}
          else if (j.return_status === 'success') matchStatus.textContent = 'Matched!';
          else matchStatus.textContent = '—';
        }}
        
        // Compact motion + planner summary
        updateBattery(j.battery_percent);
        const graphTarget = j.graph_target || {{}};
        const plannerDebug = j.planner_debug || {{}};
        updateSpeedPanel(j.speed_m_s, plannerDebug);
        updateCommandPanel(plannerDebug);
        updateObstacleClearances(plannerDebug.obstacle_clearance_m);
        updateSummaryChip('planner', `Planner: ${{plannerDebug.planner_mode || j.local_planner_state || '—'}}`);
        updateSummaryChip('planner-intent', `Intent: ${{formatPlannerIntent(plannerDebug)}}`);
        updateSummaryChip(
          'planner-safety',
          `Safety: ${{plannerDebug.safety_mode || '—'}}`,
          safetyTone(plannerDebug.safety_mode),
        );
        updateSummaryChip(
          'planner-vx-gate',
          `VX: ${{formatVxGate(plannerDebug)}}`,
          vxGateTone(plannerDebug),
        );
        updateSummaryChip('graph-target', `Graph: ${{formatGraphTarget(graphTarget)}}`);
        
        // Expected/target
        document.getElementById('expected-event').textContent = j.expected_short || '—';
        document.getElementById('return-target').textContent = j.return_target_short || '—';
        const returnProgress = j.return_progress || {{}};
        const progressNow = Number(returnProgress.progress_m);
        const progressExpected = Number(returnProgress.expected_distance_m);
        document.getElementById('return-progress').textContent =
          Number.isFinite(progressNow) && Number.isFinite(progressExpected)
            ? `${{fmt(progressNow, 1)}} / ${{fmt(progressExpected, 1)}} m`
            : '—';
        const routeSigma = Number(returnProgress.sigma_m);
        const gateStart = Number(returnProgress.window_start_m);
        const gateEnd = Number(returnProgress.window_end_m);
        let gateHalfWidth = Number(returnProgress.gate_outer_radius_m);
        if (!Number.isFinite(gateHalfWidth) && Number.isFinite(progressExpected)) {{
          gateHalfWidth = Math.max(
            Number.isFinite(gateStart) ? progressExpected - gateStart : 0,
            Number.isFinite(gateEnd) ? gateEnd - progressExpected : 0,
          );
        }}
        const locMode = String(returnProgress.localization_mode || '')
          .replace(/_/g, ' ');
        const returnActive = String(j.return_status || '').toLowerCase() === 'returning';
        const gateAvailable = returnActive
          && Number.isFinite(gateStart)
          && Number.isFinite(gateEnd);
        const gateRow = document.getElementById('return-gate-row');
        gateRow.hidden = !gateAvailable;
        if (gateAvailable) {{
          document.getElementById('return-gate-range').textContent =
            `${{fmt(gateStart, 1)}}–${{fmt(gateEnd, 1)}} m`;
          const gateOpen = Boolean(returnProgress.search_active);
          const approaching = Number.isFinite(progressNow) && progressNow < gateStart;
          const gateState = document.getElementById('return-gate-state');
          gateState.textContent = gateOpen ? 'OPEN' : (approaching ? 'APPROACH' : 'CLOSED');
          gateState.className = `mission-state ${{gateOpen ? 'open' : (approaching ? 'approach' : 'closed')}}`;
          const sigmaMultiple = Number.isFinite(routeSigma) && routeSigma > 1e-9
            && Number.isFinite(gateHalfWidth)
            ? gateHalfWidth / routeSigma
            : NaN;
          const gateMeta = [];
          if (Number.isFinite(gateHalfWidth)) gateMeta.push(`±${{fmt(gateHalfWidth, 2)}} m`);
          if (Number.isFinite(sigmaMultiple)) gateMeta.push(`${{fmt(sigmaMultiple, 1)}}σ`);
          if (locMode) gateMeta.push(locMode);
          document.getElementById('return-gate-meta').textContent = gateMeta.join(' · ') || '—';
        }}

        const correctionPolicy = j.pose_correction_policy || {{}};
        const anchorDistance = Number(returnProgress.distance_since_anchor_m);
        const correctionBase = Number(correctionPolicy.base_m);
        const correctionGrowth = Number(correctionPolicy.growth_per_meter);
        const correctionCap = Number(correctionPolicy.hard_cap_m);
        let correctionLimit = correctionBase + correctionGrowth * anchorDistance;
        if (Number.isFinite(correctionCap) && correctionCap > 0) {{
          correctionLimit = Math.min(correctionLimit, correctionCap);
        }}
        const guardAvailable = returnActive
          && correctionPolicy.enabled === true
          && Number.isFinite(anchorDistance)
          && Number.isFinite(correctionLimit);
        const guardRow = document.getElementById('pose-guard-row');
        guardRow.hidden = !guardAvailable;
        if (guardAvailable) {{
          document.getElementById('pose-guard-limit').textContent =
            `≤ ${{fmt(correctionLimit, 2)}} m`;
          const anchorKind = String(returnProgress.distance_anchor_kind || '');
          const anchorShort = String(returnProgress.distance_anchor_short || '');
          let anchorLabel = 'from anchor';
          if (anchorKind === 'mission_start') anchorLabel = 'since mission start';
          else if (anchorKind === 'match') {{
            anchorLabel = anchorShort && anchorShort !== '—'
              ? `since match ${{anchorShort}}`
              : 'since last match';
          }}
          document.getElementById('pose-guard-meta').textContent =
            `${{fmt(anchorDistance, 1)}} m ${{anchorLabel}}`;
        }}
        
        // Activity log
        if (j.activity_log) {{
          updateActivity(j.activity_log);
        }}
        
        
        // Profile matching
        try {{
          const profileCanvas = document.getElementById('profile-canvas');
          if (profileCanvas) {{
            drawProfile(profileCanvas, j.current_profile, j.stored_profile);
          }}
        }} catch (e) {{}}
        
        // Event map
        try {{
          const mapCanvas = document.getElementById('event-map-canvas');
          if (mapCanvas) {{
            const dronePos = {{
              x: j.drone_x || 0,
              y: j.drone_y || 0,
              yaw: j.drone_yaw_rad || 0
            }};
            const startPos = {{
              x: j.start_x,
              y: j.start_y
            }};
            const targetId = j.return_target_event_id || j.expected_event_id || '';
            const targetShort = (j.graph_target && j.graph_target.event_short) || j.return_target_short || j.expected_short || '';
            drawEventMap(mapCanvas, j.trajectory || [], j.events || [], dronePos, 
                         j.return_target_event_id || '', j.expected_event_id || '', startPos,
                         j.graph_target || null);
            
            // Update map info with target debug
            const mapInfo = document.getElementById('map-info');
            if (mapInfo) {{
              const nEvents = (j.events || []).length;
              const nTraj = (j.trajectory || []).length;
              const matchedCount = (j.events || []).filter(e => e.matched).length;
              if (targetId) {{
                mapInfo.textContent = `${{nEvents}} events | Target: ${{targetShort || targetId.slice(-8)}}`;
              }} else if (matchedCount > 0) {{
                mapInfo.textContent = `${{nEvents}} events (${{matchedCount}} passed)`;
              }} else {{
                mapInfo.textContent = `${{nEvents}} events, ${{nTraj}} pts`;
              }}
            }}
          }}
        }} catch (mapErr) {{ console.warn('Map draw error:', mapErr); }}
        
      }} catch (e) {{
        console.warn('tick error:', e);
        if (dot) dot.classList.add('offline');
        if (status) status.textContent = 'Offline';
      }} finally {{
        // Always schedule next tick
      setTimeout(tick, POLL_MS);
    }}
    }}
    
    // ========== Virtual Joysticks ==========
    function createJoystick(canvasId) {{
      const canvas = document.getElementById(canvasId);
      if (!canvas) return null;
      const state = {{ x: 0, y: 0, active: false }};
      const dpr = window.devicePixelRatio || 1;

      function drawStick() {{
        const ctx = canvas.getContext('2d');
        const w = canvas.width / dpr, h = canvas.height / dpr;
        const cx = w / 2, cy = h / 2;
        const r = Math.min(w, h) * 0.44;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        // Base ring
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(15, 15, 35, 0.5)';
        ctx.fill();
        ctx.strokeStyle = state.active ? 'rgba(34, 211, 238, 0.5)' : 'rgba(100, 150, 200, 0.2)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Crosshair
        ctx.strokeStyle = 'rgba(100, 150, 200, 0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy);
        ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r);
        ctx.stroke();

        // Thumb knob
        const knobR = r * 0.32;
        const kx = cx + state.x * r * 0.7;
        const ky = cy - state.y * r * 0.7;

        if (state.active) {{
          ctx.shadowColor = 'rgba(34, 211, 238, 0.6)';
          ctx.shadowBlur = 12;
        }}
        ctx.beginPath();
        ctx.arc(kx, ky, knobR, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(kx, ky, 0, kx, ky, knobR);
        grad.addColorStop(0, state.active ? '#38bdf8' : '#475569');
        grad.addColorStop(1, state.active ? '#0ea5e9' : '#334155');
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }}

      function getPos(e) {{
        const rect = canvas.getBoundingClientRect();
        const r = Math.min(rect.width, rect.height) * 0.44;
        const cx = rect.width / 2, cy = rect.height / 2;
        let clientX, clientY;
        if (e.touches) {{
          clientX = e.touches[0].clientX;
          clientY = e.touches[0].clientY;
        }} else {{
          clientX = e.clientX;
          clientY = e.clientY;
        }}
        let dx = (clientX - rect.left - cx) / (r * 0.7);
        let dy = -(clientY - rect.top - cy) / (r * 0.7);
        const mag = Math.sqrt(dx * dx + dy * dy);
        if (mag > 1) {{ dx /= mag; dy /= mag; }}
        return {{ x: dx, y: dy }};
      }}

      function onStart(e) {{
        e.preventDefault();
        state.active = true;
        const p = getPos(e);
        state.x = p.x; state.y = p.y;
        drawStick();
      }}
      function onMove(e) {{
        if (!state.active) return;
        e.preventDefault();
        const p = getPos(e);
        state.x = p.x; state.y = p.y;
        drawStick();
      }}
      function onEnd(e) {{
        e.preventDefault();
        state.active = false;
        state.x = 0; state.y = 0;
        drawStick();
      }}

      canvas.addEventListener('mousedown', onStart);
      canvas.addEventListener('touchstart', onStart, {{ passive: false }});
      window.addEventListener('mousemove', (e) => {{ if (state.active) onMove(e); }});
      window.addEventListener('touchmove', (e) => {{ if (state.active) onMove(e); }}, {{ passive: false }});
      window.addEventListener('mouseup', (e) => {{ if (state.active) onEnd(e); }});
      window.addEventListener('touchend', (e) => {{ if (state.active) onEnd(e); }});

      // Initial draw
      canvas.width = 180 * dpr;
      canvas.height = 180 * dpr;
      canvas.style.width = '90px';
      canvas.style.height = '90px';
      drawStick();
      return state;
    }}

    const joyL = createJoystick('joy-left');
    const joyR = createJoystick('joy-right');

    let joySendTimer = null;
    function joySendLoop() {{
      const statusEl = document.getElementById('joy-status');
      const anyActive = (joyL && joyL.active) || (joyR && joyR.active);
      const anyNonzero = joyL && joyR &&
        (Math.abs(joyL.x) > 0.02 || Math.abs(joyL.y) > 0.02 ||
         Math.abs(joyR.x) > 0.02 || Math.abs(joyR.y) > 0.02);

      if (anyActive || anyNonzero) {{
        const body = {{
          vx:  joyL ? joyL.y : 0,
          vy:  joyL ? joyL.x : 0,
          vz:  joyR ? joyR.y : 0,
          yaw_rate: joyR ? -joyR.x : 0
        }};
        fetch('/api/manual_cmd', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(body)
        }}).catch(() => {{}});
        if (statusEl) {{
          statusEl.textContent =
            `vx:${{(body.vx).toFixed(2)}} vy:${{(body.vy).toFixed(2)}} vz:${{(body.vz).toFixed(2)}} yaw:${{(body.yaw_rate).toFixed(2)}}`;
          statusEl.classList.add('active');
        }}
      }} else {{
        if (statusEl) {{
          statusEl.textContent = 'Idle';
          statusEl.classList.remove('active');
        }}
      }}
    }}
    joySendTimer = setInterval(joySendLoop, 80);

    tick();
  </script>
</body>
</html>
"""
    return html.encode("utf-8")


class WebDashboardNode(Node):
    def __init__(self) -> None:
        super().__init__("web_dashboard")

        self._qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.declare_parameters(
            namespace="",
            parameters=[
                ("host", "0.0.0.0"),
                ("port", 8000),
                ("client_poll_hz", 5.0),
                ("enable_mission_controls", True),
                ("clear_match_after_distance_m", 2.0),
                ("battery_sim.enabled", True),
                ("battery_sim.start_percent", 100.0),
                ("battery_sim.drain_rate_percent_per_s", 0.05),
                ("battery_sim.publish_hz", 2.0),
                ("battery_sim.return_home_level", 55.0),
                ("topic_nav_mode", "/navigation/mode"),
                ("topic_expected_event", "/navigation/expected_event"),
                ("topic_return_viz", "/navigation_events/return_viz"),
                (
                    "topic_pose_correction_policy",
                    "/navigation_events/pose_correction_policy",
                ),
                ("topic_match_confidence", "/navigation_events/match_confidence"),
                ("topic_match_found", "/navigation_events/match_found"),
                ("topic_odometry", "/drone/state_estimate"),
                ("topic_imu", "/crazyflie/imu/acc_derived"),
                ("topic_battery", "/battery_level"),
                ("topic_local_planner_state", "/local_planner/state"),
                ("topic_local_planner_debug", "/local_planner/debug_state"),
                ("topic_graph_target", "/navigation/graph_target"),
                ("topic_event_stored", "/navigation_events/stored"),
                ("topic_event_detected", "/navigation_events/detected"),
                ("topic_event_deleted", "/navigation_events/deleted"),
                ("topic_current_descriptor", "/radial_descriptor/current"),
            ],
        )

        self._telemetry = _Telemetry()
        self._lock = threading.RLock()
        self._start_time_wall_s = time.time()
        self._start_time_mono_s = time.monotonic()
        # Count unique stored ids so merge-update messages are not treated as new events.
        self._known_stored_junction_ids: set[str] = set()

        self._enable_mission_controls = bool(self.get_parameter("enable_mission_controls").value)
        self._clear_match_after_distance_m = float(self.get_parameter("clear_match_after_distance_m").value)
        self._start_pub = self.create_publisher(Bool, "/mission_control/start", self._qos)
        self._stop_pub = self.create_publisher(Bool, "/mission_control/stop", self._qos)
        self._reset_pub = self.create_publisher(Bool, "/mission_control/reset", self._qos)

        self._battery_sim_enabled = bool(self.get_parameter("battery_sim.enabled").value)
        self._battery_sim_level = float(self.get_parameter("battery_sim.start_percent").value)
        self._battery_drain_rate = float(self.get_parameter("battery_sim.drain_rate_percent_per_s").value)
        self._battery_publish_hz = float(self.get_parameter("battery_sim.publish_hz").value)
        self._battery_sim_pub: Optional[Any] = None
        self._battery_sim_timer: Optional[Any] = None
        
        if self._battery_sim_enabled:
            self._battery_sim_pub = self.create_publisher(Float32, "/battery_level", self._qos)
            self._start_battery_simulation()

        self._manual_cmd_vel_pub = self.create_publisher(
            Twist, "/crazyflie/cmd_vel_user", 1,
        )
        self._manual_max_vx = 1.5
        self._manual_max_vy = 0.2
        self._manual_max_vz = 1.0
        self._manual_max_yaw_rate = 1.0
        self._manual_active = False
        self._manual_last_cmd_time: float = 0.0
        self._manual_timeout_s = 0.25
        self._manual_watchdog_timer = self.create_timer(0.05, self._manual_watchdog)

        self.create_subscription(String, str(self.get_parameter("topic_nav_mode").value), 
                                 self._on_nav_mode, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_expected_event").value), 
                                 self._on_expected_event, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_return_viz").value), 
                                 self._on_return_viz, self._qos)
        self.create_subscription(
            String,
            str(self.get_parameter("topic_pose_correction_policy").value),
            self._on_pose_correction_policy,
            self._qos_latched,
        )
        self.create_subscription(Float32, str(self.get_parameter("topic_match_confidence").value), 
                                 self._on_match_confidence, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_match_found").value), 
                                 self._on_match_found, self._qos)
        self.create_subscription(Odometry, str(self.get_parameter("topic_odometry").value), 
                                 self._on_odometry, self._qos)
        self.create_subscription(Imu, str(self.get_parameter("topic_imu").value), 
                                 self._on_imu, self._qos)
        self.create_subscription(Float32, str(self.get_parameter("topic_battery").value), 
                                 self._on_battery, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_local_planner_state").value), 
                                 self._on_local_planner_state, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_local_planner_debug").value),
                                 self._on_local_planner_debug, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_graph_target").value),
                                 self._on_graph_target, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_event_stored").value), 
                                 self._on_event_stored, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_event_detected").value), 
                                 self._on_event_detected, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_event_deleted").value), 
                                 self._on_event_deleted, self._qos)
        self.create_subscription(String, str(self.get_parameter("topic_current_descriptor").value), 
                                 self._on_current_descriptor, self._qos)

        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        poll_hz = float(self.get_parameter("client_poll_hz").value)

        handler = self._make_handler(poll_hz=poll_hz)
        self._httpd = _ThreadedHTTPServer((host, port), handler)
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._http_thread.start()
        self.get_logger().info(f"Web dashboard running on http://{host}:{port}")

    def _get_short_id(self, event_id: str) -> str:
        """Get or create a short ID for an event."""
        if not event_id:
            return "—"
        with self._lock:
            if event_id not in self._telemetry.event_id_map:
                self._telemetry.event_id_map[event_id] = self._telemetry.next_event_number
                self._telemetry.next_event_number += 1
            num = self._telemetry.event_id_map[event_id]
            return f"E{num}"

    def _add_activity(self, message: str, event_type: str = "info") -> None:
        """Add an entry to the activity log."""
        now = time.monotonic() - self._start_time_mono_s
        entry = _ActivityEntry(timestamp=now, message=message, event_type=event_type)
        with self._lock:
            self._telemetry.activity_log.append(entry)

    def _start_battery_simulation(self) -> None:
        if self._battery_sim_timer is not None:
            self._battery_sim_timer.cancel()
        
        hz = max(0.1, self._battery_publish_hz)
        dt = 1.0 / hz
        self._publish_battery_level()
        self.get_logger().info(
            f"Battery simulation started: {self._battery_sim_level:.1f}%, "
            f"drain rate: {self._battery_drain_rate:.4f}%/s, publish rate: {hz:.1f}Hz"
        )
        self._add_activity(f"Battery sim started at {self._battery_sim_level:.0f}%", "info")
        
        self._battery_sim_timer = self.create_timer(dt, self._battery_tick)

    def _battery_tick(self) -> None:
        dt = 1.0 / max(0.1, self._battery_publish_hz)
        self._battery_sim_level = max(0.0, self._battery_sim_level - self._battery_drain_rate * dt)
        self._publish_battery_level()

    def _publish_battery_level(self) -> None:
        if self._battery_sim_pub is None:
            return
        msg = Float32()
        msg.data = float(self._battery_sim_level)
        self._battery_sim_pub.publish(msg)

    def _trigger_return_home(self) -> bool:
        """Simulate low battery to trigger return-to-home behavior."""
        if not self._battery_sim_enabled or self._battery_sim_pub is None:
            self.get_logger().warning(
                "Ignoring return-home battery injection because battery simulation is disabled."
            )
            self._add_activity("Return Home battery simulation is disabled", "warn")
            return False
        return_home_level = float(self.get_parameter("battery_sim.return_home_level").value)
        self._battery_sim_level = return_home_level
        self._publish_battery_level()
        self.get_logger().info(f"Return home triggered: battery set to {return_home_level:.1f}%")
        self._add_activity(f"Return home triggered (battery: {return_home_level:.0f}%)", "warn")
        return True

    def _on_nav_mode(self, msg: String) -> None:
        with self._lock:
            old_mode = self._telemetry.mode
            new_mode = (msg.data or "").strip()
            self._telemetry.mode = new_mode
            self._telemetry.t_mode = time.monotonic()

        if old_mode != new_mode and new_mode:
            if "EXPLORE" in new_mode.upper():
                self._add_activity("Started exploration", "info")
            elif "BACKTRACK" in new_mode.upper():
                self._add_activity("Backtracking to previous junction", "warn")
            elif "RETURN" in new_mode.upper():
                self._add_activity("Returning to base", "warn")

    def _on_expected_event(self, msg: String) -> None:
        with self._lock:
            self._telemetry.expected_event_id = (msg.data or "").strip()
            self._telemetry.t_expected_event = time.monotonic()

    def _on_return_viz(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        target = ""
        status = ""
        progress: Dict[str, Any] = {}
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    target = str(data.get("target_event_id") or "")
                    status = str(data.get("status") or "")
                    for key in (
                        "search_active",
                        "expected_distance_m",
                        "progress_m",
                        "remaining_m",
                        "window_start_m",
                        "window_end_m",
                        "sigma_m",
                        "gate_inner_radius_m",
                        "gate_outer_radius_m",
                        "integration_variance_m2",
                        "distance_since_anchor_m",
                        "distance_anchor_kind",
                        "distance_anchor_event_id",
                        "localization_mode",
                        "handover_from_event_id",
                    ):
                        if key in data:
                            progress[key] = data[key]
            except json.JSONDecodeError:
                pass
        with self._lock:
            old_status = self._telemetry.return_status
            self._telemetry.return_target_event_id = target
            self._telemetry.return_status = status
            self._telemetry.return_progress = progress
            self._telemetry.t_return_viz = time.monotonic()

        if status != old_status:
            if status == "success":
                short_id = self._get_short_id(target)
                is_home_event = False
                with self._lock:
                    if target in self._telemetry.event_map:
                        ev = self._telemetry.event_map[target]
                        if ev.event_type == "start":
                            is_home_event = True
                            self._telemetry.mission_complete_time = time.monotonic()

                if is_home_event:
                    self._add_activity("MISSION COMPLETE! Returned to base!", "success")
                else:
                    self._add_activity(f"Matched event {short_id}", "success")
            elif status == "returning" and target:
                short_id = self._get_short_id(target)
                self._add_activity(f"Searching for {short_id}", "info")

    def _on_pose_correction_policy(self, msg: String) -> None:
        """Cache the event matcher's live correction-limit policy."""
        try:
            data = json.loads((msg.data or "").strip())
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return

        policy: Dict[str, Any] = {
            "enabled": bool(data.get("enabled", False)),
        }
        for key in ("base_m", "growth_per_meter", "hard_cap_m"):
            try:
                value = float(data[key])
            except (KeyError, TypeError, ValueError):
                return
            if not math.isfinite(value) or value < 0.0:
                return
            policy[key] = value
        with self._lock:
            self._telemetry.pose_correction_policy = policy

    def _on_match_confidence(self, msg: Float32) -> None:
        with self._lock:
            self._telemetry.match_confidence = float(msg.data)
            self._telemetry.t_match_conf = time.monotonic()

    def _on_match_found(self, msg: String) -> None:
        """Cache match id and stored profile for visualization."""
        raw = (msg.data or "").strip()
        if not raw:
            return
        event_id = ""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                event_id = str(data.get("event_id", ""))
                stored_desc = data.get("stored_descriptor")
                if stored_desc and isinstance(stored_desc, dict):
                    angles = stored_desc.get("angles", [])
                    dists = stored_desc.get("filtered_distances", stored_desc.get("raw_distances", []))
                    if isinstance(angles, list) and isinstance(dists, list) and len(angles) > 2:
                        max_pts = 72
                        angles = angles[:max_pts] if len(angles) > max_pts else angles
                        dists = dists[:max_pts] if len(dists) > max_pts else dists
                        profile = _RadialProfileData(
                            angles=angles,
                            distances=dists,
                            mean_radius=float(stored_desc.get("mean_radius", 0) or 0),
                        )
                        with self._lock:
                            self._telemetry.stored_profile = profile
                            self._telemetry.match_distance_m = self._telemetry.total_distance_m
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        if event_id:
            short_id = self._get_short_id(event_id)
            self._add_activity(f"Match found: {short_id}", "success")
            with self._lock:
                mode_upper = (self._telemetry.mode or "").upper()
                if "RETURN" in mode_upper or "BACKTRACK" in mode_upper:
                    self._telemetry.matched_event_ids.add(event_id)
                    if event_id in self._telemetry.event_map:
                        self._telemetry.event_map[event_id].matched_during_return = True
        with self._lock:
            self._telemetry.last_match_event_id = event_id
            self._telemetry.t_match_found = time.monotonic()

    def _on_odometry(self, msg: Odometry) -> None:
        v = msg.twist.twist.linear
        speed = float(math.sqrt(float(v.x) ** 2 + float(v.y) ** 2 + float(v.z) ** 2))
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        with self._lock:
            self._telemetry.speed_m_s = speed
            self._telemetry.drone_x = px
            self._telemetry.drone_y = py
            self._telemetry.drone_yaw_rad = yaw
            self._telemetry.t_odom = time.monotonic()
            if self._telemetry.start_x is None:
                self._telemetry.start_x = px
                self._telemetry.start_y = py
            traj = self._telemetry.trajectory
            if len(traj) == 0:
                traj.append({"x": px, "y": py})
            else:
                last = traj[-1]
                dist = math.sqrt((px - last["x"])**2 + (py - last["y"])**2)
                if dist > 0.15:
                    traj.append({"x": px, "y": py})
                    self._telemetry.total_distance_m += dist
            if (self._telemetry.stored_profile is not None and 
                self._clear_match_after_distance_m > 0 and
                self._telemetry.total_distance_m - self._telemetry.match_distance_m > self._clear_match_after_distance_m):
                self._telemetry.stored_profile = None

    def _on_imu(self, msg: Imu) -> None:
        """Extract pitch and yaw from the IMU quaternion."""
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        with self._lock:
            self._telemetry.pitch_deg = math.degrees(pitch)
            self._telemetry.yaw_deg = math.degrees(yaw)
            hdg = float((math.degrees(yaw) + 360.0) % 360.0)
            self._telemetry.yaw_heading_deg = hdg
            self._telemetry.t_imu = time.monotonic()

    def _on_battery(self, msg: Float32) -> None:
        with self._lock:
            self._telemetry.battery_percent = float(msg.data)
            self._telemetry.t_battery = time.monotonic()

    def _on_local_planner_state(self, msg: String) -> None:
        with self._lock:
            self._telemetry.local_planner_state = (msg.data or "").strip()
            self._telemetry.t_local_planner_state = time.monotonic()

    def _on_local_planner_debug(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return

        debug = dict(data)
        committed = data.get("committed")
        if isinstance(committed, dict):
            event_id = str(committed.get("event_id") or "")
            debug["committed"] = {
                "event_id": event_id,
                "event_short": self._get_short_id(event_id) if event_id else "—",
                "mode": str(committed.get("mode") or ""),
                "gate_radius_m": (
                    float(committed["gate_radius_m"])
                    if committed.get("gate_radius_m") is not None
                    else None
                ),
                "selected_arm_angle_rad": (
                    float(committed["selected_arm_angle_rad"])
                    if committed.get("selected_arm_angle_rad") is not None
                    else None
                ),
                "event_center_xy": committed.get("event_center_xy"),
            }

        with self._lock:
            old_safety = str(self._telemetry.planner_debug.get("safety_mode", ""))
            self._telemetry.planner_debug = debug
            self._telemetry.t_local_planner_debug = time.monotonic()

        new_safety = str(debug.get("safety_mode", "")).upper()
        if new_safety != old_safety:
            if new_safety == "BLOCKED":
                self._add_activity("Planner blocked by front obstacle", "warn")
            elif new_safety == "TURN_TO_OPENING":
                self._add_activity("Planner turning toward detected opening", "warn")
            elif new_safety == "DEAD_END":
                self._add_activity("Planner declared DEAD_END", "error")

    def _on_graph_target(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return

        kind = str(data.get("kind") or "").strip().lower()
        normalized: Dict[str, Any] = {"kind": kind}
        event_id = str(data.get("event_id") or "")
        if kind == "commit":
            normalized.update(
                {
                    "event_id": event_id,
                    "event_short": self._get_short_id(event_id) if event_id else "—",
                    "mode": str(data.get("mode") or ""),
                    "event_center_xy": data.get("event_center_xy"),
                    "selected_arm_angle_rad": (
                        float(data["selected_arm_angle_rad"])
                        if data.get("selected_arm_angle_rad") is not None
                        else None
                    ),
                    "gate_radius_m": (
                        float(data["gate_radius_m"])
                        if data.get("gate_radius_m") is not None
                        else None
                    ),
                }
            )
        elif kind == "reverse":
            normalized["mode"] = str(data.get("mode") or "")

        with self._lock:
            old_target = dict(self._telemetry.graph_target)
            self._telemetry.graph_target = normalized
            self._telemetry.t_graph_target = time.monotonic()

        old_sig = (
            str(old_target.get("kind", "")),
            str(old_target.get("event_id", "")),
            str(old_target.get("mode", "")),
        )
        new_sig = (
            str(normalized.get("kind", "")),
            str(normalized.get("event_id", "")),
            str(normalized.get("mode", "")),
        )
        if new_sig != old_sig:
            if kind == "commit" and event_id:
                self._add_activity(
                    f"Graph target committed: {self._get_short_id(event_id)} ({normalized.get('mode', '')})",
                    "info",
                )
            elif kind == "reverse":
                mode = str(normalized.get("mode", "")).strip() or "unspecified"
                self._add_activity(f"Graph target reverse requested ({mode})", "warn")
            elif kind == "clear":
                self._add_activity("Graph target cleared", "info")

    def _on_event_stored(self, msg: String) -> None:
        """Track newly stored events."""
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
            if data.get("status") == "stored":
                event_id = data.get("event_id", "")
                if not isinstance(event_id, str) or not event_id:
                    return
                short_id = self._get_short_id(event_id)
                px = float(data.get("x", 0) or 0)
                py = float(data.get("y", 0) or 0)
                jt = str(data.get("junction_type", "") or "").upper()
                et = str(data.get("event_type", "") or "").upper()
                num_paths = int(data.get("num_paths", 0) or 0)
                if et == "START" or jt == "START":
                    event_type = "start"
                elif jt == "DEAD_END":
                    event_type = "dead_end"
                else:
                    event_type = "junction"
                entry_angle = data.get("entry_angle")
                selected_idx = data.get("selected_path_index")
                path_angles = data.get("path_angles")
                dead_end_paths = data.get("dead_end_paths")
                
                is_new = False
                is_dead_end = (event_type == "dead_end")
                with self._lock:
                    if event_id not in self._known_stored_junction_ids:
                        self._known_stored_junction_ids.add(event_id)
                        self._telemetry.total_events_discovered += 1
                        if is_dead_end:
                            self._telemetry.dead_ends_encountered += 1
                        else:
                            self._telemetry.junction_events_discovered += 1
                        is_new = True
                    previous = self._telemetry.event_map.get(event_id)
                    known_dead_end_paths = (
                        list(previous.dead_end_paths)
                        if previous is not None
                        else []
                    )
                    if isinstance(dead_end_paths, list):
                        known_dead_end_paths = [
                            int(index)
                            for index in dead_end_paths
                            if isinstance(index, (int, float))
                        ]
                    self._telemetry.event_map[event_id] = _EventMapEntry(
                        x=px, y=py, event_id=event_id, short_id=short_id,
                        event_type=event_type, arms_count=num_paths,
                        entry_angle=float(entry_angle) if entry_angle is not None else None,
                        selected_path_index=int(selected_idx) if selected_idx is not None else None,
                        path_angles=[float(a) for a in path_angles] if path_angles else None,
                        dead_end_paths=known_dead_end_paths,
                        matched_during_return=(
                            previous.matched_during_return
                            if previous is not None
                            else (
                                event_id
                                in self._telemetry.matched_event_ids
                            )
                        ),
                    )
                
                if is_new:
                    label = "dead end" if is_dead_end else "junction"
                    self._add_activity(f"New {label} {short_id} stored", "success" if not is_dead_end else "warning")
                else:
                    self._add_activity(f"Event {short_id} updated", "info")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    def _on_event_detected(self, msg: String) -> None:
        """Track detected events (including dead ends)."""
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
            event_type = data.get("event_type", "")
            jc = data.get("junction_config", {})
            jt = jc.get("junction_type", "") if isinstance(jc, dict) else ""
            
            if jt == "DEAD_END":
                with self._lock:
                    self._telemetry.dead_ends_encountered += 1
                self._add_activity("Dead end detected", "error")
        except json.JSONDecodeError:
            pass

    def _on_event_deleted(self, msg: String) -> None:
        """Track deleted events."""
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
            event_id = data.get("event_id", "")
            reason = data.get("reason", "unknown")
            short_id = self._get_short_id(event_id)
            with self._lock:
                if isinstance(event_id, str) and event_id in self._known_stored_junction_ids:
                    self._known_stored_junction_ids.discard(event_id)
                    if self._telemetry.junction_events_discovered > 0:
                        self._telemetry.junction_events_discovered -= 1
                if isinstance(event_id, str) and event_id in self._telemetry.event_map:
                    del self._telemetry.event_map[event_id]
            self._add_activity(f"Event {short_id} removed ({reason})", "warn")
        except json.JSONDecodeError:
            pass

    def _on_current_descriptor(self, msg: String) -> None:
        """Cache the live radial profile, capped at 72 points for the UI."""
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                angles = data.get("angles", [])
                dists = data.get("filtered_distances", data.get("raw_distances", []))
                if isinstance(angles, list) and isinstance(dists, list) and len(angles) > 2:
                    max_pts = 72
                    angles = angles[:max_pts] if len(angles) > max_pts else angles
                    dists = dists[:max_pts] if len(dists) > max_pts else dists
                    profile = _RadialProfileData(
                        angles=angles,
                        distances=dists,
                        mean_radius=float(data.get("mean_radius", 0) or 0),
                    )
                    with self._lock:
                        self._telemetry.current_profile = profile
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    def _snapshot(self) -> Dict[str, Any]:
        now_mono = time.monotonic()
        with self._lock:
            t = self._telemetry
            # Without a remaining-count topic, RETURN/BACKTRACK uses stored-junction count as "to exit".
            events_remaining = None
            mode_upper = (t.mode or "").upper()
            if "RETURN" in mode_upper or "BACKTRACK" in mode_upper:
                events_remaining = int(t.junction_events_discovered)
            activity_log = []
            for entry in reversed(list(t.activity_log)):
                mins = int(entry.timestamp // 60)
                secs = int(entry.timestamp % 60)
                activity_log.append({
                    "time": f"{mins}:{secs:02d}",
                    "message": entry.message,
                    "event_type": entry.event_type,
                })
            events_list = []
            for ev in t.event_map.values():
                entry = {
                    "x": ev.x,
                    "y": ev.y,
                    "id": ev.event_id,
                    "short_id": ev.short_id,
                    "type": ev.event_type,
                    "arms": ev.arms_count,
                    "matched": ev.matched_during_return,
                }
                if ev.entry_angle is not None:
                    entry["entry_angle"] = ev.entry_angle
                if ev.selected_path_index is not None:
                    entry["selected_idx"] = ev.selected_path_index
                if ev.path_angles:
                    entry["path_angles"] = ev.path_angles
                if ev.dead_end_paths:
                    entry["dead_end_paths"] = ev.dead_end_paths
                events_list.append(entry)
            
            return_progress = dict(t.return_progress)
            anchor_event_id = return_progress.get("distance_anchor_event_id")
            if isinstance(anchor_event_id, str) and anchor_event_id:
                return_progress["distance_anchor_short"] = self._get_short_id(
                    anchor_event_id
                )

            return {
                "server_age_s": float(now_mono - self._start_time_mono_s),
                "mode": t.mode,
                "expected_event_id": t.expected_event_id,
                "expected_short": self._get_short_id(t.expected_event_id) if t.expected_event_id else "—",
                "return_target_event_id": t.return_target_event_id,
                "return_target_short": self._get_short_id(t.return_target_event_id) if t.return_target_event_id else "—",
                "return_status": t.return_status,
                "return_progress": return_progress,
                "pose_correction_policy": dict(t.pose_correction_policy),
                "mission_complete_elapsed_s": (now_mono - t.mission_complete_time) if t.mission_complete_time else None,
                "match_confidence": t.match_confidence,
                "last_match_event_id": t.last_match_event_id,
                "last_match_short": self._get_short_id(t.last_match_event_id) if t.last_match_event_id else "—",
                "speed_m_s": t.speed_m_s,
                "pitch_deg": t.pitch_deg,
                "yaw_deg": t.yaw_heading_deg if t.yaw_heading_deg is not None else t.yaw_deg,
                "battery_percent": t.battery_percent,
                "local_planner_state": t.local_planner_state,
                "graph_target": t.graph_target,
                "planner_debug": t.planner_debug,
                "total_events": t.total_events_discovered,
                "junction_events": t.junction_events_discovered,
                "dead_ends": t.dead_ends_encountered,
                "events_remaining": events_remaining,
                "activity_log": activity_log,
                "current_profile": {
                    "angles": t.current_profile.angles,
                    "distances": t.current_profile.distances,
                } if t.current_profile and t.current_profile.angles else None,
                "stored_profile": {
                    "angles": t.stored_profile.angles,
                    "distances": t.stored_profile.distances,
                } if t.stored_profile and t.stored_profile.angles else None,
                "drone_x": t.drone_x,
                "drone_y": t.drone_y,
                "drone_yaw_rad": t.drone_yaw_rad,
                "start_x": t.start_x,
                "start_y": t.start_y,
                "trajectory": list(t.trajectory)[-300:] if len(t.trajectory) > 300 else list(t.trajectory),
                "events": events_list,
            }

    def _publish_mission_start(self) -> bool:
        if not self._enable_mission_controls:
            return False
        msg = Bool()
        msg.data = True
        self._start_pub.publish(msg)
        self._add_activity("Mission started", "success")
        return True

    def _publish_mission_stop(self) -> bool:
        if not self._enable_mission_controls:
            return False
        msg = Bool()
        msg.data = True
        self._stop_pub.publish(msg)
        self._add_activity("Mission stopped", "error")
        return True

    def _publish_mission_reset(self) -> bool:
        if not self._enable_mission_controls:
            return False
        msg = Bool()
        msg.data = True
        self._reset_pub.publish(msg)
        return True

    def _reset_dashboard_state(self) -> None:
        """Reset dashboard-local state as if it was freshly started."""
        with self._lock:
            self._telemetry = _Telemetry()
            self._known_stored_junction_ids.clear()
            self._start_time_wall_s = time.time()
            self._start_time_mono_s = time.monotonic()
        self._add_activity("Dashboard reset - ready for new mission", "info")

    def _handle_manual_cmd(self, data: Dict[str, Any]) -> bool:
        """Publish a Twist from joystick input. Returns True on success."""
        clamp = lambda v: max(-1.0, min(1.0, float(v)))  # noqa: E731
        vx = clamp(data.get("vx", 0.0)) * self._manual_max_vx
        vy = clamp(data.get("vy", 0.0)) * self._manual_max_vy
        vz = clamp(data.get("vz", 0.0)) * self._manual_max_vz
        yaw_rate = clamp(data.get("yaw_rate", 0.0)) * self._manual_max_yaw_rate

        tw = Twist()
        tw.linear.x = vx
        tw.linear.y = vy
        tw.linear.z = vz
        tw.angular.z = yaw_rate
        self._manual_cmd_vel_pub.publish(tw)
        self._manual_active = True
        self._manual_last_cmd_time = time.monotonic()
        return True

    def _manual_watchdog(self) -> None:
        """Zero cmd_vel if no joystick input for _manual_timeout_s."""
        if not self._manual_active:
            return
        if (time.monotonic() - self._manual_last_cmd_time) > self._manual_timeout_s:
            self._manual_cmd_vel_pub.publish(Twist())
            self._manual_active = False

    def _make_handler(self, *, poll_hz: float):
        node = self
        page = _html_page(poll_hz=poll_hz)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                return

            def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
                # JSON cannot encode NaN/Inf; map them to 0 so the poller never fails.
                def sanitize(obj):
                    if isinstance(obj, float):
                        if obj != obj or obj == float('inf') or obj == float('-inf'):
                            return 0.0
                        return obj
                    if isinstance(obj, dict):
                        return {k: sanitize(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [sanitize(v) for v in obj]
                    return obj
                try:
                    body = json.dumps(sanitize(payload)).encode("utf-8")
                except (TypeError, ValueError):
                    body = json.dumps({"error": "serialization_error"}).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                try:
                    path = urlparse(self.path).path
                    if path == "/" or path == "/index.html":
                        self._send_bytes(int(HTTPStatus.OK), page, "text/html; charset=utf-8")
                        return
                    if path == "/api/state":
                        self._send_json(int(HTTPStatus.OK), node._snapshot())
                        return
                    self._send_json(int(HTTPStatus.NOT_FOUND), {"ok": False, "error": "not_found"})
                except Exception:
                    try:
                        self._send_json(int(HTTPStatus.INTERNAL_SERVER_ERROR), {"ok": False, "error": "internal"})
                    except Exception:
                        pass

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path == "/api/mission/start":
                    ok = node._publish_mission_start()
                    self._send_json(int(HTTPStatus.OK), {"ok": ok})
                    return
                if path == "/api/mission/stop":
                    ok_stop = node._publish_mission_stop()
                    ok_reset = node._publish_mission_reset()
                    node._reset_dashboard_state()
                    self._send_json(int(HTTPStatus.OK), {"ok": bool(ok_stop and ok_reset)})
                    return
                if path == "/api/return_home":
                    ok = node._trigger_return_home()
                    self._send_json(int(HTTPStatus.OK), {"ok": ok})
                    return
                if path == "/api/manual_cmd":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(length) if length > 0 else b""
                        data = json.loads(body) if body else {}
                        ok = node._handle_manual_cmd(data)
                        self._send_json(int(HTTPStatus.OK), {"ok": ok})
                    except (json.JSONDecodeError, ValueError, TypeError):
                        self._send_json(int(HTTPStatus.BAD_REQUEST), {"ok": False, "error": "bad_json"})
                    return
                self._send_json(int(HTTPStatus.NOT_FOUND), {"ok": False, "error": "not_found"})

        return Handler

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "_httpd") and self._httpd is not None:
                self._httpd.shutdown_server()
        finally:
            return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = WebDashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
