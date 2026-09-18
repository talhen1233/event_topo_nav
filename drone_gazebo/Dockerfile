FROM ros:humble as base

SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/ros2*.list && apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
          signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
          http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
          > /etc/apt/sources.list.d/ros2.list && \
    apt-get update

RUN apt-get update && apt-get install -y \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3-pip \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-map-msgs \
        ros-humble-tf-transformations \
        ros-humble-nav-msgs \
        ros-humble-geometry-msgs \
        ros-humble-sensor-msgs \
        ros-humble-ackermann-msgs \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-image-transport \
        ros-humble-cv-bridge \
        ros-humble-vision-opencv \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-xacro \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-gazebo-msgs \
        ros-humble-camera-info-manager \
        ros-humble-compressed-image-transport \
        ros-humble-theora-image-transport \
        graphviz \
        graphviz-dev \
        git \
        wget \
        sudo \
        vim \
        htop

FROM base as gazebo
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install -y \
    gz-garden \
    git \
    cmake \
    build-essential \
    python3-vcstool

RUN mkdir -p /root/ros_gz_ws/src
WORKDIR /root/ros_gz_ws

RUN cd src && \
    git clone https://github.com/gazebosim/ros_gz.git -b humble

RUN apt-get update && \
    rosdep init || true && \
    rosdep update && \
    rosdep install -r --from-paths src --ignore-src -y

RUN source /opt/ros/humble/setup.bash && \
    colcon build --merge-install --cmake-args -DBUILD_TESTING=OFF

RUN echo 'source /root/ros_gz_ws/install/setup.bash' >> /root/.bashrc

FROM gazebo as dds
RUN apt-get install -y ros-humble-rmw-cyclonedds-cpp && \
    echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> /root/.bashrc

RUN apt-get update && apt-get install -y \
    ros-humble-robot-localization \
    ros-humble-imu-tools \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-depth-image-proc \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-laser-geometry \
    ros-humble-teleop-twist-keyboard \
    ros-humble-joy \
    ros-humble-teleop-twist-joy \
    ros-humble-rqt-image-view \
    ros-humble-rviz2 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

FROM dds as release

WORKDIR /workspaces/drone_gazebo
COPY src /workspaces/drone_gazebo/src
COPY crazyflie-simulation /workspaces/drone_gazebo/crazyflie-simulation

RUN rosdep init || true
RUN rosdep update

RUN source /opt/ros/humble/setup.bash && \
    source /root/ros_gz_ws/install/setup.bash && \
    colcon build --packages-select drone_simulation \
        --symlink-install --cmake-args -DBUILD_TESTING=OFF

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc
RUN echo 'source /root/ros_gz_ws/install/setup.bash' >> /root/.bashrc
RUN echo 'source /workspaces/drone_gazebo/install/setup.bash' >> /root/.bashrc

RUN echo 'export GAZEBO_MODEL_PATH=/root/.gazebo/models:$GAZEBO_MODEL_PATH' >> /root/.bashrc
RUN echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> /root/.bashrc
RUN echo 'export GZ_SIM_RESOURCE_PATH=/workspaces/drone_gazebo/src/drone_simulation/worlds:$GZ_SIM_RESOURCE_PATH' >> /root/.bashrc
RUN echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=/root/ros_gz_ws/install/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/root/ros_gz_ws/install/lib:$LD_LIBRARY_PATH' >> /root/.bashrc

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_INDIRECT=0

ENV GZ_SIM_RESOURCE_PATH="/workspaces/drone_gazebo/src/drone_simulation/worlds"
ENV GZ_SIM_SYSTEM_PLUGIN_PATH="/root/ros_gz_ws/install/lib"
ENV GAZEBO_MODEL_PATH="/root/.gazebo/models"

RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'if [ -f /root/ros_gz_ws/install/setup.bash ]; then' >> /entrypoint.sh && \
    echo '    source /root/ros_gz_ws/install/setup.bash' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '    echo "WARNING: ros_gz workspace not found!"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'if [ -f /workspaces/drone_gazebo/install/setup.bash ]; then' >> /entrypoint.sh && \
    echo '    source /workspaces/drone_gazebo/install/setup.bash' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> /entrypoint.sh && \
    echo 'export GAZEBO_MODEL_PATH=/root/.gazebo/models:$GAZEBO_MODEL_PATH' >> /entrypoint.sh && \
    echo 'export GZ_SIM_RESOURCE_PATH=/workspaces/drone_gazebo/src/drone_simulation/worlds:$GZ_SIM_RESOURCE_PATH' >> /entrypoint.sh && \
    echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=/root/ros_gz_ws/install/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH' >> /entrypoint.sh && \
    echo 'export LD_LIBRARY_PATH=/root/ros_gz_ws/install/lib:$LD_LIBRARY_PATH' >> /entrypoint.sh && \
    echo 'export NVIDIA_VISIBLE_DEVICES=all' >> /entrypoint.sh && \
    echo 'export NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'ros2 daemon stop' >> /entrypoint.sh && \
    echo 'ros2 daemon start' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/bin/bash" ]
