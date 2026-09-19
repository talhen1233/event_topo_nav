# Gazebo simulation

Requires an NVIDIA GPU, the NVIDIA Container Toolkit, and an X11 display. Run all commands from the repository root.

```bash
xhost +local:docker
docker compose -f docker-compose.sim.yml up
```

Default world: `crazyflie_ideal_maze`.

Edgar Mine stress case:

```bash
docker compose -f docker-compose.sim.yml run --rm gazebo \
  ros2 launch drone_simulation crazyflie_final_prelim_03.launch.py
```

Keyboard teleop (with the simulator already up):

```bash
docker compose -f docker-compose.sim.yml exec gazebo \
  ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/crazyflie/cmd_vel_user
```
