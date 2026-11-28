"""Apply the workspace's camera serials and optional FPS to Pika startup."""

import fcntl
import os
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import json


def acquire_sensor_lock(lock_path=Path("/tmp/pika-sensors.lock"), proc_root=Path("/proc")):
    """Keep one owner in this wrapper process without leaking the lock FD to detached children."""
    message = ("Pika sensors are running or starting, or Vive calibration is using the devices. Keep only one sensor session. "
               "Run ./pika_data.sh status --check-streams, then open a recording session. "
               "To change settings, stop the existing sensor process with Ctrl+C and wait for it to exit.")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(message) from error
        active = []
        for directory in proc_root.iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid():
                continue
            try:
                args = (directory / "cmdline").read_bytes().split(b"\0")
            except (OSError, ProcessLookupError):
                continue
            names = {Path(arg.decode(errors="replace")).name for arg in args if arg}
            # Match argv tokens, never source code passed through python -c.
            if names.intersection({"open_multi_sensor.launch", "start_multi_sensor.bash",
                                   "serial_gripper_imu", "usb_camera.py", "pika_double_locator_node", "survive-cli"}):
                active.append(directory.name)
        if active:
            raise ValueError(message + " Sensor process IDs: " + ", ".join(sorted(active, key=int)))
        os.set_inheritable(fd, False)
        return fd
    except BaseException:
        os.close(fd)
        raise


def validate_vive_trackers(vive_trackers):
    """Validate optional left/right Vive tracker serial mapping (`LHR-XXXXXXXX`, both sides present & distinct)."""
    if not vive_trackers:
        return {}
    if not isinstance(vive_trackers, dict):
        raise ValueError("vive_trackers in config/sensors.yaml must be a mapping with left and right keys.")
    unknown = set(vive_trackers.keys()) - {"left", "right"}
    if unknown:
        raise ValueError(f"Unsupported vive_trackers keys: {sorted(unknown)}")
    if "left" not in vive_trackers or "right" not in vive_trackers:
        raise ValueError("vive_trackers in config/sensors.yaml must define both 'left' and 'right' codes.")
    normalized = {}
    for side in ("left", "right"):
        code = str(vive_trackers[side]).strip()
        if not re.fullmatch(r"LHR-[A-Za-z0-9]+", code):
            raise ValueError("Left and right vive_trackers codes in config/sensors.yaml must match LHR-XXXXXXXX.")
        normalized[side] = code
    if normalized["left"] == normalized["right"]:
        raise ValueError(f"vive_trackers left and right codes must be distinct (both are {normalized['left']}).")
    return normalized


def build_sensor_child_env(base_env, vive_trackers=None):
    """Build subprocess environment with lowercase `pika_L_code` / `pika_R_code` expected by `sensor_tools`."""
    child_env = dict(base_env)
    validated = validate_vive_trackers(vive_trackers)
    if validated:
        child_env["pika_L_code"] = validated["left"]
        child_env["pika_R_code"] = validated["right"]
    return child_env


def configured_script(path, camera_fps=None, depth_serials=None, vive_trackers=None):
    if camera_fps is not None and camera_fps not in (15, 30, 60, 90):
        raise ValueError("Camera FPS must be 15, 30, 60, or 90.")
    script = path.read_text()
    if camera_fps is not None:
        script, count = re.subn(r"(?m)^camera_fps=[0-9]+[ \t]*$", f"camera_fps={camera_fps}", script)
        if count != 1:
            raise ValueError("Expected exactly one camera_fps setting in the installed startup script.")
    for side, serial in (depth_serials or {}).items():
        if side not in ("left", "right") or not re.fullmatch(r"[0-9]+", str(serial)):
            raise ValueError("Left and right D405 serials in config/sensors.yaml must be strings of digits.")
        key = f"{side[0]}_depth_camera_no"
        script, count = re.subn(rf"(?m)^{key}=[^\n]*$", f"{key}={serial}", script)
        if count != 1:
            raise ValueError(f"Expected exactly one {key} setting in the installed startup script.")
    validated_vive = validate_vive_trackers(vive_trackers)
    exports_to_inject = []
    for side, code in validated_vive.items():
        key = f"pika_{side[0].upper()}_code"
        script, count = re.subn(
            rf"(?m)^([ \t]*(?:export[ \t]+)?){key}=[^\n]*$",
            rf"\g<1>{key}={code}",
            script,
        )
        if count == 0:
            exports_to_inject.append(f"export {key}={code}")
    if exports_to_inject:
        header = "\n".join(exports_to_inject) + "\n"
        if script.startswith("#!"):
            first_nl = script.find("\n")
            if first_nl != -1:
                script = script[: first_nl + 1] + header + script[first_nl + 1 :]
            else:
                script = script + "\n" + header
        else:
            script = header + script
    return script


def head_launch(installed_launch, profile, vive_trackers=None, wrist_camera="fisheye"):
    """Build one roslaunch configuration for the selected sensors."""
    enabled = profile.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("head_camera.enabled must be boolean")
    original = ET.parse(installed_launch).getroot()
    validated_vive = validate_vive_trackers(vive_trackers)
    vive_arg_overrides = {
        f"pika_{side[0].upper()}_code": code for side, code in validated_vive.items()
    }
    for arg in original.findall("arg"):
        if arg.attrib["name"] in vive_arg_overrides:
            arg.set("default", vive_arg_overrides[arg.attrib["name"]])
    if wrist_camera == "d405":
        for suffix in ("l", "r"):
            cameras = [node for node in original.findall("node")
                       if node.get("pkg") == "sensor_tools" and node.get("type") == "usb_camera.py"
                       and node.get("name", "").endswith(f"camera_fisheye_{suffix}")]
            if len(cameras) != 1:
                raise ValueError(f"Expected one fisheye camera node for side {suffix} in {installed_launch}")
            original.remove(cameras[0])
        root = original
    elif wrist_camera == "fisheye":
        root = ET.Element("launch")
        include = ET.Element("include", file=str(installed_launch))
        for arg in original.findall("arg"):
            root.append(arg)
            arg_name = arg.attrib["name"]
            ET.SubElement(include, "arg", name=arg_name, value=f"$(arg {arg_name})")
        root.append(include)
    else:
        raise ValueError(f"Unsupported wrist camera: {wrist_camera}")
    settings = {"enabled": enabled}
    if enabled:
        serial = profile.get("serial", "")
        if not isinstance(serial, str) or not re.fullmatch(r"[0-9]+", serial):
            raise ValueError("Set head_camera.serial to the quoted D435i driver serial")
        depth = profile.get("enable_depth", False)
        if not isinstance(depth, bool):
            raise ValueError("head_camera.enable_depth must be boolean")
        settings.update(serial=serial, enable_depth=depth, width=640, height=480, fps=30,
                        rgb_topic="/camera_head/color/image_raw",
                        depth_topic="/camera_head/aligned_depth_to_color/image_raw",
                        synchronization="ROS header timestamps; no hardware trigger")
        head = ET.SubElement(root, "include", file="$(find realsense2_camera)/launch/rs_camera.launch")
        args = dict(camera="camera_head", serial_no=serial, device_type="d435", color_width="640",
                    color_height="480", color_fps="30", depth_width="640", depth_height="480",
                    depth_fps="30", enable_color="true", enable_depth=str(depth).lower(),
                    align_depth=str(depth).lower(), enable_sync="true", enable_pointcloud="false",
                    enable_confidence="false", enable_infra1="false", enable_infra2="false",
                    enable_gyro="false", enable_accel="false", wait_for_device_timeout="15.0")
        for name, value in args.items():
            ET.SubElement(head, "arg", name=name, value=value)
    ET.SubElement(root, "rosparam", param="/pika_head_camera").text = json.dumps(settings)
    return ET.tostring(root, encoding="unicode")


def main():
    import yaml  # Provided by the ROS image; host CLI needs no YAML dependency.

    path = Path(sys.argv[1]) / "install/share/sensor_tools/scripts/start_multi_sensor.bash"
    config = yaml.safe_load(sys.argv[3])
    wrist_camera = os.environ.get("PIKA_WRIST_CAMERA", "fisheye")
    profile = config["realsense"]["multi"]
    serials = {side: str(profile[f"{side}_depth_camera_serial"]) for side in ("left", "right")}
    vive_trackers = validate_vive_trackers(config.get("vive_trackers") or None)
    script = configured_script(
        path,
        int(sys.argv[2]) if sys.argv[2] else None,
        serials,
        vive_trackers=vive_trackers,
    )
    head = dict(config.get("head_camera", {}))
    if "PIKA_USE_HEAD" in os.environ:
        head["enabled"] = os.environ["PIKA_USE_HEAD"] == "true"
    if head.get("enabled") and head.get("serial") in serials.values():
        raise ValueError("Head serial must differ from wrist D405 serials")
    wrapper = head_launch(
        Path(sys.argv[1]) / "install/share/sensor_tools/launch/open_multi_sensor.launch",
        head,
        vive_trackers=vive_trackers,
        wrist_camera=wrist_camera,
    )
    fd = acquire_sensor_lock()
    print(f"D405 serials from config/sensors.yaml: left={serials['left']}, right={serials['right']}", flush=True)
    # Preserve $0 so the installed script finds its USB helpers beside itself.
    try:
        with tempfile.TemporaryDirectory(prefix="pika-head-") as temporary:
            launch = Path(temporary) / "sensors.launch"
            launch.write_text(wrapper)
            command = "roslaunch sensor_tools open_multi_sensor.launch"
            if command not in script:
                raise ValueError("Installed sensor startup command changed; cannot attach head camera")
            script = script.replace(command, "roslaunch " + str(launch))
            child_env = build_sensor_child_env(os.environ, vive_trackers)
            import subprocess
            raise SystemExit(subprocess.call(["bash", "-c", script, str(path)], env=child_env))
    finally:
        os.close(fd)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Sensor startup stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
