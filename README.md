# Event-Based Topological Navigation

ROS 2 Humble stack for confined-space micro-drone navigation. Build and run everything from Docker Compose.

**Requirements:** Linux, Docker Engine with Compose v2, NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), X11.

`ROS_DOMAIN_ID` defaults to `0`.

## First-time build

```bash
xhost +local:docker
mkdir -p ~/.gz
docker compose -f docker-compose.sim.yml build
docker compose -f docker-compose.hil.yml build
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
docker compose -f docker-compose.hil.yml up
```

Runtime parameters are in `config/`. `docker-compose.yml` is the onboard vehicle stack and is not required here.
