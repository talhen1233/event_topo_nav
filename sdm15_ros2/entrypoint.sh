#!/bin/bash
. /root/ros2_ws/install/setup.bash
. /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
exec "$@"