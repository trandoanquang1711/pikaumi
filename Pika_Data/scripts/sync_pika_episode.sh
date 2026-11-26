#!/usr/bin/env bash
# Synchronize a completed Pika episode. Run on the host.
set -euo pipefail

usage() {
  echo "Usage: $0 <single|multi> <episode-index>" >&2
  exit 2
}

mode=${1:-}
episode_index=${2:-}
[[ "$mode" == "single" || "$mode" == "multi" ]] || usage
[[ "$episode_index" =~ ^[0-9]+$ ]] || usage

container=${PIKA_CONTAINER:-pika-ros}
container_data_dir=${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}
ros_root=${PIKA_ROS_ROOT:-/root/pika_ros}
capture_type="${mode}_pika"

if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container '$container' is not running." >&2
  exit 1
fi

docker exec -it "$container" bash -lc \
  'source "$1/install/setup.bash" && roslaunch data_tools run_data_sync.launch "type:=$2" "datasetDir:=$3" "episodeIndex:=$4"' \
  _ "$ros_root" "$capture_type" "$container_data_dir" "$episode_index"
