#!/usr/bin/env bash
# Copy a reviewed capture profile into the running image. Run after starting
# the container and before `run_data_capture.launch`.
set -euo pipefail

script_dir=$(dirname "$(readlink -f "$0")")
mode=${1:-multi}
container=${2:-pika-ros}
container=${PIKA_CONTAINER:-$container}
ros_root=${PIKA_ROS_ROOT:-/root/pika_ros}

case "$mode" in
  single|multi)
    profile="${mode}_pika_data_params.yaml"
    ;;
  *)
    echo "Usage: $0 [single|multi] [container-name]" >&2
    exit 2
    ;;
esac

docker cp "$script_dir/../config/$profile" \
  "$container:$ros_root/install/share/data_tools/config/$profile"
echo "Installed $mode-Pika IMU capture profile in $container."
