"""Shared defaults and CLI validation for the capture-to-dataset workflow."""

import argparse
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
CALIBRATION = WORKSPACE / "data/calibration"

FPS_CHOICES = range(5, 91)
CAMERA_FPS_CHOICES = (15, 30, 60, 90)
WRIST_CAMERAS = ("fisheye", "d405")
EXPORT_DEFAULTS = {
    "fps": 20, "image_size": None, "max_image_offset_ms": 40.0,
    "max_image_skew_ms": 10.0, "max_sensor_gap_ms": 30.0,
}
PREPARE_DEFAULTS = {
    "min_segment_seconds": 1.0, "max_speed_m_s": 2.0,
    "max_angular_speed_rad_s": 15.0, "pose_hold_ms": 250.0,
}
SESSION_DEFAULTS = {
    **EXPORT_DEFAULTS, **PREPARE_DEFAULTS,
    "calibration_dir": str(CALIBRATION),
    "repo_id": "local/dual_pika", "use_head": False, "wrist_camera": "fisheye",
}
TIMING_KEYS = ("fps", "max_image_offset_ms", "max_image_skew_ms", "max_sensor_gap_ms")


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Enter a finite number greater than zero.")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Enter an integer greater than zero.")
    return number


def add_head_option(parser, default=None):
    parser.add_argument("--use-head", action=argparse.BooleanOptionalAction, default=default,
                        help="Include D435i head capture/checks; --no-use-head disables it. "
                             "New session/record default: off; sensors/status default: sensor profile.")


def head_arguments(enabled):
    return [] if enabled is None else ["--use-head" if enabled else "--no-use-head"]


def add_wrist_option(parser, default="fisheye"):
    parser.add_argument("--wrist-camera", type=str.lower, choices=WRIST_CAMERAS, default=default,
                        help="Wrist RGB camera pair. Default: fisheye, or the saved mode when reopening a session.")


def capture_metadata(episode):
    metadata = episode / "capture_metadata.json"
    return json.loads(metadata.read_text()) if metadata.is_file() else {}


def recorded_wrist_camera(episode):
    mode = capture_metadata(episode).get("wrist_camera", "fisheye")
    if mode not in WRIST_CAMERAS:
        raise ValueError(f"Invalid wrist_camera in {episode / 'capture_metadata.json'}: {mode!r}")
    return mode


def recorded_export_fps(episode):
    fps = capture_metadata(episode).get("export_fps", EXPORT_DEFAULTS["fps"])
    if not isinstance(fps, int) or fps not in FPS_CHOICES:
        raise ValueError(f"Invalid export_fps in {episode / 'capture_metadata.json'}: {fps!r}")
    return fps


def wrist_skew_default(mode, wrist_only=True):
    return EXPORT_DEFAULTS["max_image_skew_ms"] if mode == "fisheye" and wrist_only else 40.0


def session_settings(overrides):
    settings = {**SESSION_DEFAULTS, **(overrides or {})}
    if not overrides or "max_image_skew_ms" not in overrides:
        settings["max_image_skew_ms"] = wrist_skew_default(settings["wrist_camera"])
    return settings


def fps_value(value):
    fps = int(value)
    if fps not in FPS_CHOICES:
        raise argparse.ArgumentTypeError(f"Enter an integer between {FPS_CHOICES.start} and {FPS_CHOICES[-1]}.")
    return fps


def add_fps_option(parser, *, description=None):
    parser.add_argument("--fps", type=fps_value, metavar="HZ",
                        help=description or "Action, dataset, or validation grid rate (5–90 Hz). Default: 20 Hz")


def add_timing_options(parser, *, fps_help=None):
    add_fps_option(parser, description=fps_help)
    parser.add_argument("--max-image-offset-ms", type=positive_float,
                        help="Maximum image age relative to the observation timestamp. Default: 40 ms")
    parser.add_argument("--max-image-skew-ms", type=positive_float,
                        help="Maximum wrist image skew. Defaults to 10 ms for fisheye RGB and 40 ms for D405 or all streams")
    parser.add_argument("--max-sensor-gap-ms", type=positive_float,
                        help="Maximum pose or gripper gap for interpolation. Default: 30 ms")


def add_export_options(parser, *, fps_help=None):
    add_timing_options(parser, fps_help=fps_help)
    parser.add_argument("--image-size", type=positive_int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Resize exported images. Defaults to the original dimensions")


def add_prepare_options(parser):
    parser.add_argument("--min-segment-seconds", type=positive_float,
                        help="Minimum duration of a continuous action segment. Default: 1 second")
    parser.add_argument("--max-speed-m-s", type=positive_float,
                        help="Raw pose translation speed limit. Default: 2 m/s")
    parser.add_argument("--max-angular-speed-rad-s", type=positive_float,
                        help="Raw pose angular speed limit. Default: 15 rad/s")
    parser.add_argument("--pose-hold-ms", type=positive_float,
                        help="Reject identical poses repeated for at least this long. Default: 250 ms")


def option_arguments(settings, keys):
    arguments = []
    for key in keys:
        value = settings.get(key)
        if value is None:
            continue
        arguments.append("--" + key.replace("_", "-"))
        values = value if isinstance(value, (list, tuple)) else [value]
        arguments.extend(map(str, values))
    return arguments


def source_filename(fps):
    return f"pika_source_{fps}hz.hdf5"
