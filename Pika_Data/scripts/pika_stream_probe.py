"""Observe VISTA input streams without starting an episode."""

import functools
import json
import time


def collect():
    import rospy
    from data_msgs.msg import Gripper, LocalizationStatus
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image

    rows, tracking, invalid_images = {}, {}, {}
    subscribers = []
    rospy.init_node("pika_sync_preflight", anonymous=True, disable_signals=True)

    def observe(key, message):
        rows[key].append([message.header.stamp.to_nsec(), rospy.Time.now().to_nsec()])
        if isinstance(message, Image) and (
                message.width <= 0 or message.height <= 0 or message.step <= 0
                or len(message.data) < message.step * message.height):
            invalid_images[key] += 1

    def accuracy(side, message):
        tracking[side].append(bool(message.accurate))

    for side, suffix in (("left", "l"), ("right", "r")):
        streams = [(f"{side}_image", f"/camera_fisheye_{suffix}/color/image_raw", Image),
                   (f"{side}_pose", f"/pika_pose_{suffix}", PoseStamped),
                   (f"{side}_gripper", f"/gripper_{suffix}/data", Gripper)]
        for key, topic, message_type in streams:
            rows[key] = []
            invalid_images[key] = 0
            subscribers.append(rospy.Subscriber(topic, message_type, functools.partial(observe, key),
                                                 queue_size=1, buff_size=8 * 1024**2))
        tracking[side] = []
        subscribers.append(rospy.Subscriber(f"/pika_localization_status_{suffix}", LocalizationStatus,
                                             functools.partial(accuracy, side), queue_size=1))
    import os
    profile = dict(rospy.get_param("/pika_head_camera", {"enabled": False}))
    requested = os.environ.get("PIKA_USE_HEAD")
    if requested == "true" and not profile.get("enabled"):
        raise ValueError("Head requested but not started. Run ./pika_data.sh sensors --use-head first.")
    if requested == "false":
        profile["enabled"] = False
    if profile.get("enabled"):
        head_topics = {"head_image": "/camera_head/color/image_raw"}
        if profile.get("enable_depth"):
            head_topics["head_depth_image"] = "/camera_head/aligned_depth_to_color/image_raw"
        for key, topic in head_topics.items():
            rows[key], invalid_images[key] = [], 0
            subscribers.append(rospy.Subscriber(topic, Image, functools.partial(observe, key),
                                                queue_size=1, buff_size=8 * 1024**2))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not rospy.is_shutdown():
        if all(rows.values()) and all(tracking.values()):
            break
        time.sleep(0.02)
    if all(rows.values()) and all(tracking.values()):
        for values in [*rows.values(), *tracking.values()]:
            values.clear()
        for key in invalid_images:
            invalid_images[key] = 0
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not rospy.is_shutdown():
            time.sleep(0.02)
    ended_at_ns = rospy.Time.now().to_nsec()
    for subscriber in subscribers:
        subscriber.unregister()
    rospy.signal_shutdown("preflight complete")
    return {"streams": rows, "tracking": tracking, "invalid_images": invalid_images,
            "ended_at_ns": ended_at_ns}


if __name__ == "__main__":
    print(json.dumps(collect()))
