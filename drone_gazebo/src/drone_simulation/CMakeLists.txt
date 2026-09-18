cmake_minimum_required(VERSION 3.8)
project(drone_simulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  worlds
  config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  drone_simulation/crazyflie_odom_to_tf.py
  drone_simulation/crazyflie_firmware_controller.py
  drone_simulation/crazyflie_camera_info_publisher.py
  drone_simulation/kinematic_hover_shim.py
  drone_simulation/odometry_to_imu.py
  drone_simulation/sim_time_enforcer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  drone_simulation/kinematic_hover_shim.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME kinematic_hover_shim
)

install(PROGRAMS
  drone_simulation/crazyflie_firmware_controller.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME crazyflie_firmware_controller
)

install(PROGRAMS
  drone_simulation/crazyflie_odom_to_tf.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME crazyflie_odom_to_tf
)

install(PROGRAMS
  drone_simulation/odometry_to_imu.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME odometry_to_imu
)

install(PROGRAMS
  drone_simulation/crazyflie_camera_info_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME crazyflie_camera_info_publisher
)

install(PROGRAMS
  drone_simulation/sim_time_enforcer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sim_time_enforcer
)

install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
