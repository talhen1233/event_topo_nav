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

RUN pip3 install --upgrade numpy==1.26.4
RUN pip3 install scipy

RUN     apt-get update && \
        apt-get install ros-humble-map-msgs -y && \
        apt-get install ros-humble-tf-transformations -y && \
        apt-get install ros-humble-nav-msgs -y && \
        apt-get install ros-humble-geometry-msgs -y && \
        apt-get install ros-humble-sensor-msgs -y && \
        apt-get install graphviz graphviz-dev -y && \
        apt-get install ros-humble-sensor-msgs-py -y && \
        apt-get install ros-humble-visualization-msgs -y
        
RUN pip3 install --upgrade --user pip
 
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
