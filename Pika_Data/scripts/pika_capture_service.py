"""JSON-line bridge to the installed Pika recorder, executed inside pika-ros."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from threading import Lock
import time

import rosgraph
import rospy
from data_msgs.msg import LocalizationStatus
from data_msgs.srv import CaptureService


LAUNCH = """<launch>
  <arg name="datasetDir"/>
  <arg name="useService" default="true"/>
  <arg name="episodeIndex" default="0"/>
  <remap from="/data_tools_dataCapture/capture_service" to="capture_service"/>
  <remap from="/data_tools_dataCapture/status" to="capture_status"/>
  <include file="$(find data_tools)/launch/run_data_capture.launch">
    <arg name="type" value="multi_pika"/>
    <arg name="useService" value="$(arg useService)"/>
    <arg name="episodeIndex" value="$(arg episodeIndex)"/>
    <arg name="datasetDir" value="$(arg datasetDir)"/>
  </include>
  <!-- Only the VISTA cameras are required. Missing D405 must not stop capture. -->
  <rosparam>
dataInfo:
  camera:
    color:
      names: ['pikaFisheyeCamera_l', 'pikaFisheyeCamera_r']
      parentFrames: ['camera_fisheye_l_link', 'camera_fisheye_r_link']
      topics: ['/camera_fisheye_l/color/image_raw', '/camera_fisheye_r/color/image_raw']
      configTopics: ['/camera_fisheye_l/color/camera_info', '/camera_fisheye_r/color/camera_info']
    depth: {names: [], parentFrames: [], topics: [], configTopics: []}
  </rosparam>
</launch>
"""


def selected_head_profile():
    import os
    profile = dict(rospy.get_param("/pika_head_camera", {"enabled": False}))
    requested = os.environ.get("PIKA_USE_HEAD")
    if requested == "true" and not profile.get("enabled"):
        raise ValueError("Head requested but not started. Run ./pika_data.sh sensors --use-head first.")
    if requested == "false":
        profile.update(enabled=False, enable_depth=False)
    return profile


def capture_launch():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(LAUNCH)
    profile = selected_head_profile()
    if profile.get("enabled"):
        color = {"names": ["pikaFisheyeCamera_l", "pikaFisheyeCamera_r", "headCamera"],
                 "parentFrames": ["camera_fisheye_l_link", "camera_fisheye_r_link", "camera_head_color_optical_frame"],
                 "topics": ["/camera_fisheye_l/color/image_raw", "/camera_fisheye_r/color/image_raw", "/camera_head/color/image_raw"],
                 "configTopics": ["/camera_fisheye_l/color/camera_info", "/camera_fisheye_r/color/camera_info", "/camera_head/color/camera_info"]}
        depth = {key: [] for key in color}
        if profile.get("enable_depth"):
            depth = dict(names=["headCamera"], parentFrames=["camera_head_color_optical_frame"],
                         topics=["/camera_head/aligned_depth_to_color/image_raw"],
                         configTopics=["/camera_head/aligned_depth_to_color/camera_info"])
        root.find("rosparam").text = json.dumps({"dataInfo": {"camera": {"color": color, "depth": depth}}})
    return ET.tostring(root, encoding="unicode")


def write_capture_metadata(path):
    profile = selected_head_profile()
    temporary = path / ".capture_metadata.tmp"
    temporary.write_text(json.dumps({"version": 1, "head_camera": profile}, indent=2) + "\n")
    temporary.replace(path / "capture_metadata.json")
    if profile.get("enabled"):
        streams = ["camera/color/headCamera"]
        if profile.get("enable_depth"):
            streams.append("camera/depth/headCamera")
        for relative in streams:
            directory = path / relative
            if not (directory / "config.json").is_file() or not any(
                    p.suffix.lower() in {".jpg", ".png"} for p in directory.glob("*")):
                raise ValueError(f"Missing configured head camera stream: {relative}")


class TrackingLog:
    def __init__(self):
        self.lock = Lock()
        self.samples = None
        self.subscribers = [
            rospy.Subscriber(f"/pika_localization_status_{suffix}", LocalizationStatus,
                             self.observe, callback_args=side, queue_size=128)
            for side, suffix in (("left", "l"), ("right", "r"))
        ]

    def observe(self, message, side):
        with self.lock:
            if self.samples is not None:
                # LocalizationStatus in pika-ros has no header.
                self.samples[side].append([rospy.Time.now().to_nsec(), bool(message.accurate)])

    def start(self):
        with self.lock:
            self.samples = {"left": [], "right": []}

    def stop(self):
        with self.lock:
            samples, self.samples = self.samples, None
            return samples


def write_tracking_log(path, samples):
    destination = path / "tracking_status.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=path, prefix=".tracking_status_", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump({"version": 1, "clock": "ros_receive_time_ns", "sides": samples}, file)
            file.write("\n")
            file.flush()
            temporary.chmod(0o644)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    summary = []
    for side, label in (("left", "left"), ("right", "right")):
        rows = samples[side]
        if not rows:
            summary.append(f"{label}: tracking log missing")
        elif any(not accurate for _, accurate in rows):
            summary.append(f"{label}: {sum(not accurate for _, accurate in rows)}/{len(rows)} samples reported tracking errors")
    return "; ".join(summary)


def wait_statistics(path):
    statistic = path / "statistic.txt"
    deadline = time.monotonic() + 30
    while not statistic.is_file() or not statistic.stat().st_size:
        if time.monotonic() >= deadline:
            raise ValueError(f"Missing statistic.txt for {path.name}. Check the recorder log.")
        time.sleep(0.1)


def check_saved_episode(path, instruction):
    wait_statistics(path)
    for side in ("l", "r"):
        streams = [
            (f"camera/color/pikaFisheyeCamera_{side}", {".jpg", ".png"}),
            (f"localization/pose/pika_{side}", {".json"}),
            (f"gripper/encoder/pika_{side}", {".json"}),
        ]
        for stream, suffixes in streams:
            directory = path / stream
            if not directory.is_dir() or not any(
                p.suffix in suffixes and p.name != "config.json" for p in directory.iterdir()
            ):
                raise ValueError(f"{path.name} is missing samples for {stream}. Check the sensors before recording again.")
    write_capture_metadata(path)
    temporary = path / ".pika_session_instruction.tmp"
    temporary.write_text(json.dumps({"instructions": [instruction]}, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path / "instructions.json")


def main(data_root):
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def reply(ok, active=False, message=""):
        print(json.dumps({"ok": ok, "active": active, "message": message}), file=protocol, flush=True)

    active = None
    if not rosgraph.is_master_online():
        reply(False, message="ROS master is unavailable. Run ./pika_data.sh sensors.")
        return
    with tempfile.TemporaryDirectory(prefix="pika-capture-") as temp:
        launch = Path(temp) / "capture.launch"
        launch.write_text(capture_launch())
        recorder = subprocess.Popen(
            ["roslaunch", str(launch), f"datasetDir:={data_root}"],
            stdin=subprocess.DEVNULL, stdout=sys.stderr,
        )
        try:
            rospy.init_node("session_client", disable_signals=True)
            tracking = TrackingLog()
            rospy.wait_for_service("capture_service", timeout=12)
            capture = rospy.ServiceProxy("capture_service", CaptureService)
            reply(True)
            for line in sys.stdin:
                request = json.loads(line)
                tracking_summary = ""
                try:
                    if request["operation"] == "start":
                        if active is not None:
                            raise ValueError("A recording is active. Save the current episode first.")
                        episode, instruction = request["episode"], request["instruction"]
                        if (data_root / f"episode{episode}").exists():
                            raise ValueError(f"episode{episode} already exists. Choose the next episode number.")
                        tracking.start()
                        try:
                            result = capture(True, False, episode, str(data_root), json.dumps([instruction]))
                            if not result.success:
                                raise ValueError(result.message or "The recorder refused to start.")
                        except (ValueError, rospy.ServiceException):
                            tracking.stop()
                            raise
                        active = (episode, instruction)
                    elif request["operation"] in ("save", "discard"):
                        if active is None:
                            raise ValueError("The recorder is waiting for a start command.")
                        episode, instruction = active
                        result = capture(False, True, episode, str(data_root), json.dumps([instruction]))
                        if not result.success:
                            raise ValueError(result.message or "The recorder did not confirm that it stopped.")
                        active = None
                        samples = tracking.stop()
                        path = data_root / f"episode{episode}"
                        if request["operation"] == "discard":
                            wait_statistics(path)
                        else:
                            check_saved_episode(path, instruction)
                            tracking_summary = write_tracking_log(path, samples)
                    else:
                        raise ValueError("Invalid recorder command. Use start, save, or discard.")
                    reply(True, active is not None, tracking_summary)
                except (OSError, ValueError, rospy.ServiceException) as error:
                    reply(False, active is not None, str(error))
        except rospy.ROSException as error:
            reply(False, active is not None, str(error))
        finally:
            if recorder.poll() is None:
                recorder.send_signal(signal.SIGINT)
                recorder.wait()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--terminal":
        if not rosgraph.is_master_online():
            raise SystemExit("ROS master is unavailable. Run ./pika_data.sh sensors.")
        with tempfile.TemporaryDirectory(prefix="pika-capture-") as temporary:
            launch = Path(temporary) / "capture.launch"
            launch.write_text(capture_launch())
            rospy.init_node("terminal_tracking_log", anonymous=True, disable_signals=True)
            tracking = TrackingLog()
            tracking.start()
            result = subprocess.call(["roslaunch", str(launch), f"datasetDir:={sys.argv[1]}",
                                      "useService:=false", f"episodeIndex:={sys.argv[3]}"])
            samples = tracking.stop()
            if result == 0:
                write_capture_metadata(Path(sys.argv[1]) / f"episode{sys.argv[3]}")
                summary = write_tracking_log(Path(sys.argv[1]) / f"episode{sys.argv[3]}", samples)
                if summary:
                    print(f"Tracking: {summary}. See tracking_status.json for this episode.")
            raise SystemExit(result)
    else:
        main(Path(sys.argv[1]))
