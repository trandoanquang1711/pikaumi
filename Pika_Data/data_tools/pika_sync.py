"""Check the live timestamp streams with the same pairing rules used by export."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from pika_options import (EXPORT_DEFAULTS, TIMING_KEYS, add_timing_options, add_head_option,
                          add_wrist_option, wrist_skew_default)
from prepare_pika_dataset import brackets, sample_head, sample_wrist_pair, timestamp_grid, valid_runs


def analyze(data, fps=20, max_image_offset_ms=40, max_image_skew_ms=10, max_sensor_gap_ms=30):
    errors, statistics, times = [], {}, {}
    for key, rows in data["streams"].items():
        if len(rows) < 2:
            errors.append(f"Missing samples for {key}")
            continue
        values = np.asarray(rows, dtype=np.int64)
        timestamps = values[:, 0]
        if np.any(timestamps <= 0) or np.any(np.diff(timestamps) <= 0):
            errors.append(f"Non-increasing timestamps for {key}")
            continue
        age = (values[:, 1] - timestamps) / 1e6
        age_p95 = float(np.percentile(age, 95))
        if age.min() < -5 or age_p95 > 200 or (data["ended_at_ns"] - timestamps[-1]) > 250_000_000:
            errors.append(f"Stale timestamps or clock mismatch for {key}")
        if data["invalid_images"].get(key, 0):
            errors.append(f"Empty image or unexpected dimensions for {key}")
        statistics[key] = {"samples": len(rows), "hz": float((len(rows) - 1) * 1e9 / np.ptp(timestamps)),
                           "receive_age_ms_p95": age_p95, "max_gap_ms": float(np.diff(timestamps).max() / 1e6)}
        times[key] = timestamps
    for side in ("left", "right"):
        flags = data["tracking"].get(side, [])
        if not flags or not all(flags):
            errors.append(f"Unstable {side} tracking")
    report = {"streams": statistics, "errors": errors, "ready": False,
              "scope": "ROS header timestamp consistency; exposure-to-pose latency not calibrated"}
    required = {f"{side}_{kind}" for side in ("left", "right") for kind in ("image", "pose", "gripper")}
    required.update(key for key in data["streams"] if key.startswith("head_"))
    if not required.issubset(times):
        return report
    grid = timestamp_grid(max(series[0] for series in times.values()),
                          min(series[-1] for series in times.values()), fps)
    if len(grid) < 2:
        errors.append("The streams have no overlapping time range")
        return report
    pairs = sample_wrist_pair(times["left_image"], times["right_image"], grid, "causal",
                              round(max_image_offset_ms * 1e6))
    skew = np.abs(times["left_image"][pairs[0][0]] - times["right_image"][pairs[1][0]]) / 1e6
    valid = pairs[0][2] & pairs[1][2] & (skew <= max_image_skew_ms)
    for key in ("left_pose", "right_pose", "left_gripper", "right_gripper"):
        valid &= brackets(times[key], grid, round(max_sensor_gap_ms * 1e6))[3]
    wrist_times = np.column_stack([times["left_image"][pairs[0][0]], times["right_image"][pairs[1][0]]])
    for key in times:
        if key.startswith("head_"):
            indices, _, ok = sample_head(times[key], grid, wrist_times, round(max_image_offset_ms * 1e6))
            head_skew = np.ptp(np.column_stack([wrist_times, times[key][indices]]), axis=1) / 1e6
            valid &= ok & (head_skew <= 40)
            statistics[key]["pairwise_skew_ms_p95"] = float(np.percentile(head_skew, 95))
    fraction = float(valid.mean())
    longest = max((b - a - 1 for a, b in valid_runs(valid)), default=0) / fps
    report.update(fps=fps, valid_fraction=fraction, longest_valid_seconds=longest,
                  wrist_skew_ms_p95=float(np.percentile(skew, 95)),
                  wrist_skew_ms_max=float(skew.max()))
    if fraction < 0.9 or longest < 1:
        errors.append(f"Only {fraction:.1%} of samples passed alignment. Longest valid interval: {longest:.2f} seconds")
    report["ready"] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--ros-root", required=True)
    add_timing_options(parser)
    add_head_option(parser)
    add_wrist_option(parser)
    parser.set_defaults(**{key: EXPORT_DEFAULTS[key] for key in TIMING_KEYS if key != "max_image_skew_ms"})
    args = parser.parse_args()
    if args.max_image_skew_ms is None:
        args.max_image_skew_ms = wrist_skew_default(args.wrist_camera)
    code = (Path(__file__).resolve().parents[1] / "scripts/pika_stream_probe.py").read_text()
    head_env = [] if args.use_head is None else ["-e", f"PIKA_USE_HEAD={str(args.use_head).lower()}"]
    result = subprocess.run(["docker", "exec", *head_env, "-e", f"PIKA_WRIST_CAMERA={args.wrist_camera}",
                             args.container, "bash", "-lc",
                             'source "$1/install/setup.bash" && exec python3 -c "$2"',
                             "_", args.ros_root, code], check=True, capture_output=True, text=True, timeout=18)
    report = analyze(json.loads(result.stdout), **{key: getattr(args, key) for key in TIMING_KEYS})
    for key, values in report["streams"].items():
        print(f"{key}: {values['hz']:.1f} Hz, timestamp age p95 {values['receive_age_ms_p95']:.1f} ms")
    if "valid_fraction" in report:
        print(f"Timestamp alignment: {report['valid_fraction']:.1%} valid, wrist skew p95 "
              f"{report['wrist_skew_ms_p95']:.2f} ms, longest valid interval {report['longest_valid_seconds']:.2f} s")
    if not report["ready"]:
        parser.exit(1, "Recording preflight failed: " + "; ".join(report["errors"]) + "\n")
    print("Live timestamp check passed for both images, poses, and grippers.")


if __name__ == "__main__":
    main()
