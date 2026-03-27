#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="omnivla:jazzy-harmonic"
CONTAINER_NAME="omnivla_jazzy_harmonic"
HOST_WS_DEFAULT="$HOME/ros2_projects/omnivla_ws"
HOST_WS="${1:-$HOST_WS_DEFAULT}"

if [ ! -d "$HOST_WS" ]; then
  echo "Workspace directory not found: $HOST_WS"
  exit 1
fi

xhost +local:docker >/dev/null 2>&1 || true

exec docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --network host \
  --ipc host \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOST_WS":/omnivla_ws \
  "$IMAGE_NAME" \
  bash
