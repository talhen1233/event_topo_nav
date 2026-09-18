FROM ros:humble as base
##################################

SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

# Refresh the ROS repository key before the first apt update. Older Humble
# base-image layers may contain an expired key and stale package indexes.
RUN python3 -c "import urllib.request; \
p='https://raw.githubusercontent.com/ros/rosdistro/master/ros.key'; \
urllib.request.urlretrieve(p, '/usr/share/keyrings/ros2-latest-archive-keyring.gpg'); \
urllib.request.urlretrieve(p, '/usr/share/keyrings/ros-archive-keyring.gpg')"

RUN     apt-get update && \
        apt-get upgrade -y && \
        apt-get install python3-pip -y

RUN pip3 install --upgrade numpy==1.26.4
RUN pip3 install transforms3d 

RUN pip3 install opencv-python && \
        apt-get update && \
        apt-get install libgl1-mesa-glx -y

RUN     apt-get update && \
        apt-get install ros-humble-map-msgs -y && \
        apt-get install ros-humble-tf-transformations -y && \
        apt-get install ros-humble-nav-msgs -y && \
        apt-get install ros-humble-geometry-msgs -y && \
        apt-get install ros-humble-sensor-msgs -y

RUN     apt-get update && \
        apt-get upgrade -y

RUN pip3 install --upgrade --user pip

RUN apt install python3-libgpiod python3-smbus2 i2c-tools -y
#################       CYCLONE DDS         ###################################
RUN     apt-get install ros-humble-rmw-cyclonedds-cpp -y && \
        echo export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp >> /root/.bashrc && \
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp 
###############################################################################


FROM base as release
####################

ADD src/ /root/ros2_ws/src

WORKDIR /root/ros2_ws

RUN . /opt/ros/humble/setup.bash && \
        colcon build  --symlink-install

COPY    entrypoint.sh /entrypoint.sh
RUN     chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
