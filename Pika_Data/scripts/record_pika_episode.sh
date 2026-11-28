#!/usr/bin/env bash
# Start a Pika capture and attach a human-readable task instruction on exit.
# Run on the host after the sensor stack is already running.
set -euo pipefail

usage() {
  echo "Usage: $0 <single|multi> <episode-index> [instruction]" >&2
  exit 2
}

mode=${1:-}
episode_index=${2:-}
instruction=${3:-}
container=${PIKA_CONTAINER:-pika-ros}
container_data_dir=${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}
ros_root=${PIKA_ROS_ROOT:-/root/pika_ros}

[[ "$mode" == "single" || "$mode" == "multi" ]] || usage
[[ "$episode_index" =~ ^[0-9]+$ ]] || usage

if [[ -z "$instruction" ]]; then
  read -r -p "Task instruction: " instruction
fi
[[ -n "$instruction" ]] || { echo "Instruction cannot be empty." >&2; exit 2; }

capture_type="${mode}_pika"
episode_dir="$container_data_dir/episode${episode_index}"

if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container '$container' is not running. Start the sensor stack first." >&2
  exit 1
fi

if docker exec "$container" test -e "$episode_dir"; then
  echo "Episode already exists: $episode_dir. Choose an unused episode number." >&2
  exit 1
fi

echo "Recording ${capture_type} episode${episode_index}."
echo "Instruction: $instruction"
if [[ "$mode" == "multi" ]]; then
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  launch_code="$(cat "$script_dir/pika_capture_service.py")"
  docker exec -it -e "PIKA_USE_HEAD=${PIKA_USE_HEAD:-false}" \
    -e "PIKA_EXPORT_FPS=${PIKA_EXPORT_FPS:-}" \
    -e "PIKA_WRIST_CAMERA=${PIKA_WRIST_CAMERA:-fisheye}" "$container" bash -lc \
    'source "$1/install/setup.bash" && export ROS_LOG_DIR="$3/.ros_logs" && exec python3 -u -c "$2" "$3" --terminal "$4"' \
    _ "$ros_root" "$launch_code" "$container_data_dir" "$episode_index"
else
  docker exec -it "$container" bash -lc \
    'source "$1/install/setup.bash" && export ROS_LOG_DIR="$3/.ros_logs" && roslaunch data_tools run_data_capture.launch "type:=$2" "datasetDir:=$3" "episodeIndex:=$4"' \
    _ "$ros_root" "$capture_type" "$container_data_dir" "$episode_index"
fi

# The capture node creates the episode directory. Write only after it exits so
# its default null instruction cannot overwrite the requested task text.
docker exec -i "$container" python3 - "$episode_dir" "$instruction" <<'PY'
import json
import sys
from pathlib import Path

episode_dir = Path(sys.argv[1])
instruction = sys.argv[2]
if not episode_dir.is_dir():
    raise SystemExit(f"Capture did not create {episode_dir}; instruction was not written.")
path = episode_dir / "instructions.json"
path.write_text(json.dumps({"instructions": [instruction]}, indent=2) + "\n")
print(f"Wrote {path}")
PY

if [[ "$mode" == "multi" ]]; then
  workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  echo "Export ${PIKA_EXPORT_FPS:-20} Hz source data:"
  printf '  bash %q export %s' "$workspace/pika_data.sh" "$episode_index"
  if [[ -n "${PIKA_DATA_DIR:-}" ]]; then
    printf ' --data-root %q' "$PIKA_DATA_DIR"
  fi
  printf '\n'
else
  echo "Now synchronize the episode:"
  echo "  bash scripts/sync_pika_episode.sh $mode $episode_index"
fi
