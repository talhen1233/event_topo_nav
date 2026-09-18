FROM ros:humble as base
##################################

SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

# The ROS apt key baked into older ros:humble images expires and then
# apt falls back to stale indexes (404 on package debs).
RUN python3 -c "import urllib.request; \
p='https://raw.githubusercontent.com/ros/rosdistro/master/ros.key'; \
urllib.request.urlretrieve(p, '/usr/share/keyrings/ros2-latest-archive-keyring.gpg'); \
urllib.request.urlretrieve(p, '/usr/share/keyrings/ros-archive-keyring.gpg')"

RUN     apt-get update && \
        apt-get install python3-pip -y

RUN pip3 install scikit-image \
                 scikit-learn

RUN pip3 install --upgrade numpy==1.26.4
RUN pip3 install transforms3d scipy

RUN pip3 install opencv-python opencv-python-headless opencv-contrib-python && \
        apt-get update && \
        apt-get install libgl1-mesa-glx -y

RUN pip3 install networkx numba

# Install the geometry2 stack in one shot. apt will otherwise upgrade
# tf2-py/tf2-ros to 0.25.22 while leaving libtf2 at the image's 0.25.7,
# and tf2_py then fails with undefined symbol lookupVelocity.
RUN     apt-get update && \
        apt-get install -y \
        ros-humble-map-msgs \
        ros-humble-tf-transformations \
        ros-humble-tf2 \
        ros-humble-tf2-py \
        ros-humble-tf2-ros \
        ros-humble-tf2-ros-py \
        ros-humble-tf2-msgs \
        ros-humble-nav-msgs \
        ros-humble-geometry-msgs \
        ros-humble-sensor-msgs \
        ros-humble-sensor-msgs-py \
        ros-humble-visualization-msgs \
        graphviz graphviz-dev

RUN pip3 install --upgrade --user pip
 
#################       CYCLONE DDS         ###################################
RUN     apt-get install ros-humble-rmw-cyclonedds-cpp -y && \
        echo export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp >> /root/.bashrc && \
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp 
###############################################################################


FROM base as release
####################

ADD src/ /root/ros2_ws/src
COPY config/ /root/ros2_ws/src/event_navigation_system/config/

WORKDIR /root/ros2_ws

RUN . /opt/ros/humble/setup.bash && \
        colcon build  --symlink-install

COPY    entrypoint.sh /entrypoint.sh
RUN     chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
