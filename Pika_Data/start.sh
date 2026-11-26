#!/bin/bash
# Start the PIKA ROS Docker container.
# Usage: bash start.sh [single|multi]
# Default mode: single

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MODE=${1:-single}
# Use the locally built, project-facing image name by default.  Override only
# when deliberately testing another image: PIKA_IMAGE=... bash start.sh multi
IMAGE=${PIKA_IMAGE:-pika-acquisition:latest}
CONTAINER_NAME=${PIKA_CONTAINER:-pika-ros}
HOST_DATA_DIR=${PIKA_DATA_DIR:-$SCRIPT_DIR/data}
CONTAINER_DATA_DIR=${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}
ROS_ROOT=${PIKA_ROS_ROOT:-/root/pika_ros}
SENSORS_YAML="$SCRIPT_DIR/config/sensors.yaml"

if [[ "$MODE" != "single" && "$MODE" != "multi" ]]; then
    echo "Usage: bash start.sh [single|multi]"
    exit 1
fi

# Read serial numbers from config/sensors.yaml
if [ ! -f "$SENSORS_YAML" ]; then
    echo "Error: $SENSORS_YAML not found"
    exit 1
fi

mkdir -p "$HOST_DATA_DIR"

if [ "$MODE" = "single" ]; then
    DEPTH_SN=$(python3 -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print(d["realsense"]["single"]["depth_camera_serial"])' "$SENSORS_YAML")
    if [ "$DEPTH_SN" = "CHANGE_ME" ]; then
        echo "Warning: realsense.single.depth_camera_serial is not set in config/sensors.yaml"
    fi
else
    L_SN=$(python3 -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print(d["realsense"]["multi"]["left_depth_camera_serial"])' "$SENSORS_YAML")
    R_SN=$(python3 -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print(d["realsense"]["multi"]["right_depth_camera_serial"])' "$SENSORS_YAML")
    L_VIVE=$(python3 -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print((d.get("vive_trackers") or {}).get("left", ""))' "$SENSORS_YAML")
    R_VIVE=$(python3 -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print((d.get("vive_trackers") or {}).get("right", ""))' "$SENSORS_YAML")
    if [ "$L_SN" = "CHANGE_ME" ] || [ "$R_SN" = "CHANGE_ME" ]; then
        echo "Warning: one or more realsense serial numbers are not set in config/sensors.yaml"
    fi
fi

# Allow Docker to connect to the host X11 display (required for RViz)
xhost +local:docker

# Re-attach or start existing container if it exists
if docker inspect "$CONTAINER_NAME" &>/dev/null; then
    STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
    if [ "$STATUS" = "running" ]; then
        echo "Attaching to running container: $CONTAINER_NAME"
        docker attach "$CONTAINER_NAME"
    else
        echo "Restarting existing container: $CONTAINER_NAME"
        docker start -ai "$CONTAINER_NAME"
    fi
    exit 0
fi

# Build docker run arguments for a new container
DOCKER_ARGS=(
    -it
    --name "$CONTAINER_NAME"
    --privileged
    --network host
    -v /dev:/dev
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -e DISPLAY="$DISPLAY"
    -v "$HOST_DATA_DIR:$CONTAINER_DATA_DIR"
)

if [ "$MODE" = "single" ]; then
    SCRIPT_PATH="$ROS_ROOT/install/share/sensor_tools/scripts/start_single_sensor.bash"
    DOCKER_ARGS+=(
        -e DEPTH_CAMERA_SERIAL="$DEPTH_SN"
        -e SCRIPT_PATH="$SCRIPT_PATH"
    )
else
    SCRIPT_PATH="$ROS_ROOT/install/share/sensor_tools/scripts/start_multi_sensor.bash"
    DOCKER_ARGS+=(
        -e L_DEPTH_CAMERA_SERIAL="$L_SN"
        -e R_DEPTH_CAMERA_SERIAL="$R_SN"
        -e SCRIPT_PATH="$SCRIPT_PATH"
    )
    if [ -n "${L_VIVE:-}" ]; then
        DOCKER_ARGS+=(-e pika_L_code="$L_VIVE")
    fi
    if [ -n "${R_VIVE:-}" ]; then
        DOCKER_ARGS+=(-e pika_R_code="$R_VIVE")
    fi
fi

echo "Starting new container in $MODE mode..."

if [ "$MODE" = "multi" ]; then
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c '
        sed -i "s|^l_depth_camera_no=.*|l_depth_camera_no=${L_DEPTH_CAMERA_SERIAL}|" "$SCRIPT_PATH" &&
        sed -i "s|^r_depth_camera_no=.*|r_depth_camera_no=${R_DEPTH_CAMERA_SERIAL}|" "$SCRIPT_PATH" &&
        if [ -n "${pika_L_code:-}" ]; then sed -i -E "s|^([[:space:]]*(export[[:space:]]+)?)pika_L_code=.*|\1pika_L_code=${pika_L_code}|" "$SCRIPT_PATH"; fi &&
        if [ -n "${pika_R_code:-}" ]; then sed -i -E "s|^([[:space:]]*(export[[:space:]]+)?)pika_R_code=.*|\1pika_R_code=${pika_R_code}|" "$SCRIPT_PATH"; fi &&
        exec bash
    '
else
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c '
        sed -i "s|^depth_camera_no=.*|depth_camera_no=${DEPTH_CAMERA_SERIAL}|" "$SCRIPT_PATH" &&
        exec bash
    '
fi
