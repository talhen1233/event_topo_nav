### LocalPathPlanner
`local_path_planner_node.py` is a graph-following local planner for the tiny drone.
It combines:

- a fast reactive safety layer from the point cloud
- a sticky low-compute graph intent layer for yaw
- local `vy` / `vz` control from point-cloud statistics, with an optional gentle `vy` pull toward the graph projection

The planner does **not** choose topology on its own after an event is committed.
The authoritative committed arm arrives on `/navigation/graph_target`, while the planner only keeps a speculative continuation before an event exists.

### Main Inputs
- `/map_pointcloud` (`sensor_msgs/PointCloud2`): local occupied voxels
- `/drone/state_estimate` (`nav_msgs/Odometry`): pose and velocity
- `/skeleton_graph/json` (`std_msgs/String`): graph extracted by the event system
- `/navigation/graph_target` (`std_msgs/String`): committed intent from navigation
- `/mission_control/start` / `/mission_control/stop` (`std_msgs/Bool`): mission gating

### Main Outputs
- `/crazyflie/cmd_vel` (`geometry_msgs/Twist`): body-frame velocity command
- `/local_planner/state` (`std_msgs/String`): high-level planner mode
- `/local_planner/debug_state` (`std_msgs/String`): JSON telemetry for the dashboard
- `/cmd_vel_markers` (`visualization_msgs/Marker`): commanded linear/yaw visualization
- `/local_planner/debug_markers` (`visualization_msgs/Marker`): graph, yaw-intent, speculative-arm, and planner-status markers

### Planner Modes
- `IDLE`: no mission or no valid graph, hover in place
- `FOLLOW_INTENT`: speculative graph-following before a committed event arm exists
- `COMMITTED_ARM`: following the arm selected by navigation
- `REVERSE_FOLLOW`: navigation-commanded return/backtrack along the rear-facing graph tangent
- `BLOCKED`: front obstacle near a graph node, evaluating whether motion is still possible
- `TURN_TO_OPENING`: rotating toward a locally detected opening
- `DEAD_END`: turning back along the graph after safety confirmed no viable exit

### Tuning Order
1. Tune front safety distances first: `front_block_dist_m`, `front_clear_dist_m`, `front_stop_dist_m`, `open_dist_m`, `dead_end_time_s`.
2. Tune graph intent stability: `graph_lookahead_m`, `min_arm_length_m`, `yaw_intent_*`, `commit_support_angle_deg`.
3. Tune lateral centering: `d0_side_m`, `k_y`, `center_deadband_m`, `side_dist_percentile`, `vy_filter_tau_s`, then `graph_center_*` if you want additional pull to the graph centerline.
4. Tune vertical behavior: `vertical_dist_percentile`, `d0_floor_m`, `d0_ceiling_m`, `k_floor`, `k_ceiling`, `vz_filter_tau_s`, `vz_deadband_mps`.
5. Tune forward motion and yaw gating last: `v_nom`, `vx_max`, `yaw_kp`, `yaw_rate_max_rps`, `yaw_align_gate_rad`.

### Parameter Source
Planner parameters are declared in `local_path_planner_node.py`.
Launch defaults live in:

- `src/local_path_planner/config/params.yaml`
- `src/local_path_planner/launch/local_path_planner.launch.py`

If you run the node directly with `python3`, the code defaults are used unless you explicitly pass ROS parameters.