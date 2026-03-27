FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=jazzy \
    ROS_ROOT=/opt/ros/jazzy \
    ROS_WS=/omnivla_ws

SHELL ["/bin/bash", "-c"]

# Base OS packages first
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    tzdata \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    ca-certificates \
    build-essential \
    cmake \
    git \
    vim \
    nano \
    less \
    python3 \
    python3-pip \
    python3-venv \
    python3-argcomplete \
    python3-dev \
    iputils-ping \
    net-tools \
    mesa-utils \
    x11-apps \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Enable Ubuntu universe and ROS 2 apt repository
RUN add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list

# ROS 2 Jazzy + Gazebo Harmonic + ROS dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    ros-jazzy-ros-gz \
    ros-jazzy-gz-ros2-control \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# Python packages commonly useful for robotics / ML prototyping.
# Add or remove to match your actual VLA stack.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    numpy \
    scipy \
    matplotlib \
    pyyaml \
    jupyterlab

# Initialize rosdep (safe if it already exists during derived builds)
RUN rosdep init 2>/dev/null || true

# Create workspace root and convenient shell setup
RUN mkdir -p ${ROS_WS}/src && \
    echo "source ${ROS_ROOT}/setup.bash" >> /etc/bash.bashrc && \
    echo "export ROS_WS=${ROS_WS}" >> /etc/bash.bashrc && \
    echo "if [ -f ${ROS_WS}/install/setup.bash ]; then source ${ROS_WS}/install/setup.bash; fi" >> /etc/bash.bashrc && \
    echo "export LIBGL_ALWAYS_INDIRECT=0" >> /etc/bash.bashrc

WORKDIR ${ROS_WS}

CMD ["bash"]
