# Pika Hardware Setup

Use this guide for installation and hardware troubleshooting. Recalibrate Vive when the tracking setup changes. See the [English setup and recording guide](../../../GETTING_STARTED.md).

A local ROS Noetic workspace for collecting synchronized AgileX Pika Sense
single- or dual-Pika episodes. Raw episodes are written to a host data folder;
validated episodes can be exported to one portable HDF5 file per episode.

## What is recorded

Dual Pika capture records the following streams for each synchronized frame:

| Stream | Typical rate | Contents |
|---|---:|---|
| D405 RGB | 30 Hz | JPEG color images for left and right Pikas |
| Fisheye RGB | 30 Hz | JPEG wrist/egocentric images for left and right Pikas |
| D405 aligned depth | 30 Hz | Raw 16-bit grayscale PNG depth images |
| Vive pose | 120 Hz raw, synchronized to cameras | 6-DoF position and orientation |
| Gripper | about 120 Hz raw | Angle and opening distance |
| IMU | 250 Hz raw | Angular velocity and linear acceleration |

The IMU orientation quaternion is all zero. Use the Vive 6-DoF pose stream for
end-effector orientation. The passive gripper does not provide force/effort.

## Requirements

- Ubuntu 20.04 host and Docker 20.10+
- Pika base stations powered and fixed in place
- Pika trackers paired through SteamVR
- USB 3 connections for every Pika camera/sensor bundle
- At least 1 TB of storage for raw collection

## Paths and names

No host-specific path is required. The scripts use these defaults:

| Variable | Default | Purpose |
|---|---|---|
| `PIKA_CONTAINER` | `pika-ros` | Docker container name |
| `PIKA_DATA_DIR` | `<this-repository>/data` | Host directory that stores episodes |
| `PIKA_CONTAINER_DATA_DIR` | `/home/agilex/data` | Same directory as seen inside the container |
| `PIKA_ROS_ROOT` | `/root/pika_ros` | ROS workspace inside the image |
| `PIKA_IMAGE` | `pika-acquisition:latest` | Docker image name |

For normal use, set nothing: episodes are saved automatically in this
repository's `../../data` directory.

Only set these variables when you deliberately want a different location or
container name. Replace `/your/chosen/path` with a directory you own:

```bash
export PIKA_DATA_DIR=/your/chosen/path
export PIKA_CONTAINER_DATA_DIR=/data
export PIKA_CONTAINER=my-pika-ros
```

Use the same exported variables in every terminal. Changing either data-path
variable requires recreating the container; the host data directory is not
removed by `docker rm`.

## First-time dual-Pika setup

### 1. Configure physical device mapping

Open `config/sensors.yaml` and set:

- left/right D405 serial numbers, found with `rs-enumerate-devices -s`
- left/right USB physical paths, found with the Pikas connected:

```bash
cd /path/to/Pika_Data
bash scripts/find_usb_kernel.bash
```

The `multi_sensor` USB paths must match the current host topology. If the host
USB parent number changes (for example `3-10` becomes `3-7`), update
`config/sensors.yaml`, reinstall rules, and reconnect the Pikas.

### 2. Install dual udev aliases

```bash
sudo bash setup_host.sh multi_sensor
```

Unplug and reconnect both complete Pika USB bundles. Verify all aliases exist:

```bash
ls -l /dev/ttyUSB50 /dev/ttyUSB51 /dev/video50 /dev/video51
rs-enumerate-devices -s
```

The configured mapping is:

```text
ttyUSB50 / video50 : left Pika
ttyUSB51 / video51 : right Pika
```

The target names such as `ttyUSB0` or `video16` may change; the aliases must
exist and point to the physical Pikas selected in `config/sensors.yaml`.

### 3. Build and create the image/container

```bash
docker build -t "${PIKA_IMAGE:-pika-acquisition:latest}" .
```

Create a dual container:

```bash
bash start.sh multi
```

If an existing container was created in single mode, or its data mount/config
changed, recreate only the container:

```bash
docker rm -f "${PIKA_CONTAINER:-pika-ros}"
bash start.sh multi
```

This does not remove episodes from `PIKA_DATA_DIR`.

## Vive calibration and tracker assignment

The current step-by-step procedure, including stopping sensors, backing up the
configuration, `survive-cli --force-calibrate`, USB cleanup and the final
tracking check, is in [the end-to-end guide](../../../PIKA_VISTA_PIPELINE.md#calib-vive).
Repeat that procedure after moving or changing the angle of either base station.
Do not run calibration and sensor launches concurrently, and retain the previous
configuration rather than deleting it before taking a backup.

Identify the tracker serial codes:

```bash
source "${PIKA_ROS_ROOT:-/root/pika_ros}/install/setup.bash"
roslaunch pika_locator get_code.launch
```

Assign the codes to the physical left/right Pikas, replacing the placeholders:

```bash
echo 'export pika_L_code=LHR-0AE19791' >> ~/.bashrc
echo 'export pika_R_code=LHR-8A6C081E' >> ~/.bashrc
source ~/.bashrc
```

Do not use LibSurvive labels such as `T20` or `T21` to assign left/right; those
labels depend on connection order.

## Record a dual episode

### 1. Start sensors and RViz

Inside the main container terminal:

```bash
cd "${PIKA_ROS_ROOT:-/root/pika_ros}/install/share/sensor_tools/scripts"
bash start_multi_sensor.bash
```

RViz should open. Set the fixed frame to `base_link`, add `TF`, and add Pose
displays for `/pika_pose_l` and `/pika_pose_r`. Both must move smoothly.

### 2. Verify tracking and enable IMU capture

On the host, in a second terminal:

```bash
cd /path/to/Pika_Data
bash scripts/enable_imu_capture.sh multi

docker exec -it "${PIKA_CONTAINER:-pika-ros}" bash
```

Inside that second container shell:

```bash
source "${PIKA_ROS_ROOT:-/root/pika_ros}/install/setup.bash"
rostopic echo -n 1 /pika_localization_status_l
rostopic echo -n 1 /pika_localization_status_r
```

Both must report `accurate: True`. Exit this second shell to return to the host.

### 3. Record with a task instruction

Run on the host. Choose an unused episode number:

```bash
cd /path/to/Pika_Data
bash scripts/record_pika_episode.sh multi 6
```

The script asks for a task instruction before capture, for example:

```text
Pick up the red block and place it in the tray.
```

Perform one complete manipulation for 30–60 seconds. Press Enter once to stop.
The script writes `instructions.json` automatically.

### 4. Synchronize and validate

```bash
bash scripts/sync_pika_episode.sh multi 6

python3 scripts/pika_episode_tools.py validate \
  "${PIKA_DATA_DIR:-$(pwd)/data}/episode6"
```

A valid dual episode has matching `sync.txt` counts for 12 streams, a non-null
instruction, both fisheye intrinsic files, and 16-bit raw depth.

## Fisheye intrinsic calibration

Calibrate each physical fisheye once while its lens, mount, and capture
resolution remain unchanged. The supported board is ChArUco 4×5 with 45 mm
squares, 35 mm markers, and `DICT_4X4_50`.

With the relevant fisheye stream running:

```bash
docker cp scripts/calibrate_charuco_fisheye.py "${PIKA_CONTAINER:-pika-ros}:/tmp/"
docker exec -it "${PIKA_CONTAINER:-pika-ros}" bash

python3 /tmp/calibrate_charuco_fisheye.py capture \
  --images-dir "${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}/calibration/right_fisheye/images"

python3 /tmp/calibrate_charuco_fisheye.py calibrate \
  --images-dir "${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}/calibration/right_fisheye/images" \
  --output "${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}/calibration/right_fisheye/fisheye_intrinsics.json"
```

Capture at least 25 sharp, diverse full-board views. Copy the correct left and
right intrinsic JSON files to each episode's `calibration/` directory before
validation/export.

Camera intrinsics are not camera-to-tool extrinsics. Camera-to-tool transforms
require a separate, validated hand-eye calibration with a fixed board and
stable Vive tracking. Do not deliver an unvalidated transform.

## HDF5 export

`pika_episode_tools.py` creates a portable Pika HDF5 file. It retains all
synchronized streams, JPEG/PNG bytes, timestamps, task instruction, and every
JSON calibration file attached to the episode.

The exporter needs `h5py`, which is available in the Docker image. Copy the
tool into the running container and export, for example episode 4. This command
expands the configurable data path on the host before calling Docker:

```bash
cd /path/to/Pika_Data

docker cp scripts/pika_episode_tools.py "${PIKA_CONTAINER:-pika-ros}:/tmp/pika_episode_tools.py"

docker exec -it "${PIKA_CONTAINER:-pika-ros}" python3 /tmp/pika_episode_tools.py \
  export-hdf5 \
  "${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}/episode4" \
  "${PIKA_CONTAINER_DATA_DIR:-/home/agilex/data}/exports/episode4_pika.hdf5"
```

The result is on the host at:

```text
$PIKA_DATA_DIR/exports/episode4_pika.hdf5
```

This command exports a Pika HDF5 package. To build the LeRobot v3.0 VLA dataset
from raw episodes, follow the workspace pipeline for `pika_data.sh export`,
`prepare`, and `lerobot`. [Workspace instructions](../../README.md).

## Troubleshooting

| Problem | Resolution |
|---|---|
| `/dev/ttyUSB50` or `/dev/video50` missing | Recheck `config/sensors.yaml`, run `sudo bash setup_host.sh multi_sensor`, then reconnect both Pikas. |
| Base tracking inaccurate or jittery | Keep base stations fixed, remove direct sunlight/IR interference, then rerun Vive calibration. |
| Fisheye `KeyError: 16` | In the container run `pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python`; use the image's system OpenCV. |
| RealSense frames stop arriving | Check `rs-enumerate-devices -s`, USB 3 connections, and avoid overloaded hubs/docks. |
| HDF5 export reports `h5py` missing | Run the export inside the running `pika-ros` container. |

## Delivery notes

- Raw capture creates many small JSON/image files by design; use HDF5 for a
  compact transfer package.
- Depth is raw 16-bit PNG, not a colorized preview.
- Synchronization uses nearest timestamps with a 30 ms limit; it is not
  hardware-triggered.
- A customer package should include a real labeled manipulation episode,
  fisheye intrinsics, and validated camera-to-tool extrinsics if available.
