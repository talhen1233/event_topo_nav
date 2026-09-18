# Event-Based Topological Navigation for Micro-Drones

Low-compute stack for a sub-250 g micro-drone in confined tunnels. The vehicle explores an unknown branching network with sparse ToF, optical flow, and IMU, stores junctions as a topological graph, and returns to start without GPS or a global metric map.

[Supplementary video](https://youtu.be/x1kHfgQJU5w)

ROS 2 Humble. Build and run with Docker Compose.

**Requirements:** Linux, Docker Engine with Compose v2, NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), X11.

`ROS_DOMAIN_ID` defaults to `0`.

## First-time build

```bash
xhost +local:docker
mkdir -p ~/.gz
docker compose -f docker-compose.sim.yml build
docker compose -f docker-compose.pil.yml build
```

The simulation image is large. The first Gazebo launch also downloads Fuel world models (internet required); later launches reuse `~/.gz`.

## Simulation

See [`drone_gazebo/README.md`](drone_gazebo/README.md).

```bash
docker compose -f docker-compose.sim.yml up
```

## PIL

Start the simulation first, then:

```bash
docker compose -f docker-compose.pil.yml up
```

Mission control is at [http://localhost:8000/](http://localhost:8000/). **Start** / **Stop** run explore-and-return; **Return Home** forces return. The map is the event graph in the odometry frame (junctions, selected/rejected arms, dead ends, clearance). Profile Matching overlays the live radial profile on a stored landmark during return.

For the matching 3D view, load `config/rviz/pil.rviz`:

```bash
docker exec drone_gazebo rviz2 -d /config/rviz/pil.rviz
```

Runtime parameters are in `config/`. `docker-compose.yml` is the onboard vehicle stack and is not required here.
