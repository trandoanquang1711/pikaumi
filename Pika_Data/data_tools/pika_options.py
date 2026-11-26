"""Shared defaults and CLI validation for the capture-to-dataset workflow."""

import argparse
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
CALIBRATION = WORKSPACE / "data/calibration"

FPS_CHOICES = (5, 10, 15, 20, 25, 30)
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
    "repo_id": "local/dual_pika", "use_head": False,
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


def add_timing_options(parser):
    parser.add_argument("--fps", type=int, choices=FPS_CHOICES,
                        help="Action, dataset, or validation grid rate. Default: 20 Hz")
    parser.add_argument("--max-image-offset-ms", type=positive_float,
                        help="Maximum image age relative to the observation timestamp. Default: 40 ms")
    parser.add_argument("--max-image-skew-ms", type=positive_float,
                        help="Maximum timestamp difference between wrist images. Defaults to 10 ms for VISTA and 40 ms for all streams")
    parser.add_argument("--max-sensor-gap-ms", type=positive_float,
                        help="Maximum pose or gripper gap for interpolation. Default: 30 ms")


def add_export_options(parser):
    add_timing_options(parser)
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
