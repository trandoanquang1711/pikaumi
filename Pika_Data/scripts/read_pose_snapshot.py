"""Collect fresh pose feedback in an existing ROS 1 or ROS 2 environment."""

import json
from collections import Counter
import sys
import time


def collect(ros_version, topic, status_topic=""):
    poses, stamps = [], set()
    status = [False, 0.0]
    counts = Counter()
    frame = [None]
    device_serial = None

    def callback(message):
        counts["feedback"] += 1
        if status_topic and (not status[0] or time.monotonic() - status[1] > 0.5):
            counts["tracking_rejected"] += 1
            return
        header = getattr(message, "header", None)
        if header is not None:
            stamp = header.stamp
            seconds = getattr(stamp, "secs", getattr(stamp, "sec", 0))
            nanoseconds = getattr(stamp, "nsecs", getattr(stamp, "nanosec", 0))
            key = seconds * 10**9 + nanoseconds
            if not key or key in stamps:
                counts["stamp_rejected"] += 1
                return
            stamps.add(key)
            frame[0] = header.frame_id
        pose = getattr(message, "pose", message)
        p, q = pose.position, pose.orientation
        poses.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    deadline = time.monotonic() + 8
    if ros_version == "1":
        import rospy
        import rostopic

        rospy.init_node("tcp_axes_snapshot", anonymous=True, disable_signals=True)
        if status_topic:
            side = "left" if status_topic.endswith("_l") else "right"
            device_serial = rospy.get_param(f"/pika_double_locator/{side}_hand_code", None)
        pose_type, _, _ = rostopic.get_topic_class(topic, blocking=False)
        if pose_type is None:
            raise ValueError(f"No pose publisher for {topic}. Start the sensors.")
        subscribers = []
        if status_topic:
            status_type, _, _ = rostopic.get_topic_class(status_topic, blocking=False)
            if status_type is None:
                raise ValueError(f"Missing tracking status topic {status_topic}.")

            def tracking(message):
                accurate = bool(message.accurate)
                counts["accurate" if accurate else "inaccurate"] += 1
                status[:] = [accurate, time.monotonic()]

            subscribers.append(rospy.Subscriber(status_topic, status_type, tracking, queue_size=1))
        subscribers.append(rospy.Subscriber(topic, pose_type, callback, queue_size=1))
        while len(poses) < 30 and time.monotonic() < deadline and not rospy.is_shutdown():
            time.sleep(0.01)
        rospy.signal_shutdown("snapshot complete")
    else:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rosidl_runtime_py.utilities import get_message

        rclpy.init()
        node = rclpy.create_node("tcp_axes_snapshot")
        try:
            types = {}
            while topic not in types and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                types = dict(node.get_topic_names_and_types())
            if topic not in types:
                raise ValueError(f"No feedback publisher for {topic}.")
            node.create_subscription(get_message(types[topic][0]), topic, callback, qos_profile_sensor_data)
            while len(poses) < 30 and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    if len(poses) < 30:
        detail = f"Received {counts['feedback']} feedback messages and {counts['stamp_rejected']} empty or repeated timestamps."
        if status_topic:
            detail += (f" {status_topic}: accurate=True {counts['accurate']}, "
                       f"accurate=False {counts['inaccurate']}; "
                       f"rejected {counts['tracking_rejected']} feedback messages because tracking was invalid or stale.")
        raise ValueError(f"Received only {len(poses)}/30 valid samples from {topic}. {detail}")
    return {"poses": poses[:30], "frame": frame[0], "topic": topic,
            "received_at_ns": time.time_ns(), "tracking_required": bool(status_topic),
            "device_serial": device_serial}


if __name__ == "__main__":
    try:
        print(json.dumps(collect(*sys.argv[1:]), allow_nan=False))
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
