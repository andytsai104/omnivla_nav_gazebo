# OmniVLA Docker Starter

## Build the image
```bash
docker build -t omnivla:jazzy-harmonic .
```

## Run the container
```bash
./run_omnivla_container.sh ~/ros2_projects/omnivla_ws
```

If your workspace was previously built on the host, rebuild inside the container:
```bash
cd /omnivla_ws
rm -rf build install log
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Quick verification inside the container
```bash
printenv ROS_DISTRO
gz sim --versions
dpkg -l | grep -E 'ros-jazzy-ros-gz|gz-sim'
which ros2
which colcon
nvidia-smi
```

Expected:
- `ROS_DISTRO=jazzy`
- `gz-sim8` for Harmonic
- `ros2`, `colcon`, and `gz` commands available
- `nvidia-smi` works when the host NVIDIA runtime is configured

## Notes
- GPU support depends on the host having NVIDIA drivers and NVIDIA Container Toolkit configured.
- GUI support depends on X11 forwarding from the host.
- Docker Desktop on Linux can run GPU-enabled containers, but the host still needs NVIDIA Container Toolkit / driver support.
