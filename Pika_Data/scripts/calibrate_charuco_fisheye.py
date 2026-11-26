#!/usr/bin/env python3
"""Capture and calibrate a Pika fisheye using the project's ChArUco board.

Defaults match the verified 4x5 target: 45 mm squares, 35 mm markers, and
OpenCV DICT_4X4_50. Calibration writes measured values; it never substitutes
missing values.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def cv2_with_aruco():
    import cv2
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support is unavailable. Install an OpenCV contrib build in a separate calibration environment.")
    return cv2


def dictionary_id(cv2, name):
    value = getattr(cv2.aruco, name, None)
    if value is None:
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {name}")
    return value


def make_board(cv2, args):
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id(cv2, args.dictionary))
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return dictionary, cv2.aruco.CharucoBoard_create(
            args.squares_x, args.squares_y, args.square_m, args.marker_m, dictionary)
    return dictionary, cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_m, args.marker_m, dictionary)


def board_corners(board):
    return board.chessboardCorners if hasattr(board, "chessboardCorners") else board.getChessboardCorners()


def capture(args):
    import rospy
    import cv2
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    output = Path(args.images_dir)
    output.mkdir(parents=True, exist_ok=True)
    bridge = CvBridge()
    last_save = [0.0]
    saved = [0]

    def callback(message):
        now = time.monotonic()
        if now - last_save[0] < args.interval:
            return
        try:
            frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            filename = output / f"frame_{message.header.stamp.to_nsec():019d}.jpg"
            if cv2.imwrite(str(filename), frame):
                last_save[0] = now
                saved[0] += 1
                rospy.loginfo("Saved %s (%d images)", filename.name, saved[0])
        except Exception as exc:
            rospy.logwarn("Could not save fisheye frame: %s", exc)

    rospy.init_node("pika_charuco_capture", anonymous=True)
    rospy.Subscriber(args.topic, Image, callback, queue_size=1)
    print(f"Capturing {args.topic} every {args.interval:.2f}s into {output}. Press Ctrl+C when 30-40 good views are saved.")
    rospy.spin()


def calibrate(args):
    cv2 = cv2_with_aruco()
    dictionary, board = make_board(cv2, args)
    detector_params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters()
    object_points, image_points, accepted = [], [], []
    image_size = None

    for path in sorted(Path(args.images_dir).glob("*")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
        if ids is None:
            continue
        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_ids is None or count < args.min_corners:
            continue
        ids_flat = charuco_ids.reshape(-1)
        object_points.append(board_corners(board)[ids_flat].reshape(-1, 1, 3).astype("float64"))
        image_points.append(charuco_corners.reshape(-1, 1, 2).astype("float64"))
        accepted.append(path.name)
        image_size = (image.shape[1], image.shape[0])

    if len(accepted) < args.min_views:
        raise RuntimeError(f"Only {len(accepted)} usable views; need at least {args.min_views}. Capture more sharp, diverse full-board views.")

    import numpy as np
    k = np.eye(3, dtype=np.float64)
    d = np.zeros((4, 1), dtype=np.float64)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(object_points, image_points, image_size, k, d, None, None, flags=flags)
    errors = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)
        errors.extend(np.linalg.norm(projected.reshape(-1, 2) - observed.reshape(-1, 2), axis=1).tolist())

    result = {
        "camera": "Pika fisheye",
        "model": "fisheye",
        "image_width": image_size[0], "image_height": image_size[1],
        "K": k.tolist(), "D": d.reshape(-1).tolist(),
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "accepted_view_count": len(accepted), "accepted_images": accepted,
        "board": {"type": "ChArUco", "dictionary": args.dictionary,
                  "squares_x": args.squares_x, "squares_y": args.squares_y,
                  "square_length_m": args.square_m, "marker_length_m": args.marker_m},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--squares-x", type=int, default=4)
    common.add_argument("--squares-y", type=int, default=5)
    common.add_argument("--square-m", type=float, default=0.045)
    common.add_argument("--marker-m", type=float, default=0.035)
    common.add_argument("--dictionary", default="DICT_4X4_50")
    main = argparse.ArgumentParser(description=__doc__)
    commands = main.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture", parents=[common])
    cap.add_argument("--topic", default="/camera_fisheye/color/image_raw")
    cap.add_argument("--images-dir", required=True)
    cap.add_argument("--interval", type=float, default=0.5)
    cal = commands.add_parser("calibrate", parents=[common])
    cal.add_argument("--images-dir", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--min-views", type=int, default=25)
    cal.add_argument("--min-corners", type=int, default=6)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        capture(args) if args.command == "capture" else calibrate(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
