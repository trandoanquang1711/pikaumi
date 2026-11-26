"""Measure tool axes from signed translations while holding tool orientation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

PIKA = Path(__file__).resolve().parents[1]
WORKSPACE = PIKA.parent
AXES = "xyz"
DEFINITION = {
    "x": "From wrist toward fingertips, the approach direction",
    "y": "Toward the device left finger along the jaw opening axis",
    "z": "From the device underside toward its top, +Z = +X cross +Y",
}


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stationary_pose(snapshot):
    poses = np.asarray(snapshot["poses"], dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 20 or not np.isfinite(poses).all():
        raise ValueError("Provide at least 20 finite poses [x,y,z,qx,qy,qz,qw].")
    if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=0.01):
        raise ValueError("Invalid pose quaternion.")
    rotation = Rotation.from_quat(poses[:, 3:])
    mean = rotation.mean()
    position = np.median(poses[:, :3], axis=0)
    jitter_mm = float(np.linalg.norm(poses[:, :3] - position, axis=1).max() * 1000)
    jitter_deg = float(np.degrees((mean.inv() * rotation).magnitude()).max())
    if jitter_mm > 2.5 or jitter_deg > 1:
        raise ValueError(f"The gripper moved or tracking was noisy: {jitter_mm:.2f} mm / {jitter_deg:.2f}°.")
    return np.r_[position, mean.as_quat()], {"max_position_jitter_mm": jitter_mm,
                                           "max_rotation_jitter_deg": jitter_deg}


def translation_delta(origin, endpoint, offset, distance_mm, label):
    if not np.isfinite(distance_mm) or distance_mm < 40:
        raise ValueError("Use a measured translation of at least 40 mm.")
    r0, r1 = Rotation.from_quat(origin[3:]), Rotation.from_quat(endpoint[3:])
    delta = r0.inv().apply(endpoint[:3] + r1.apply(offset) - origin[:3] - r0.apply(offset))
    drift = float(np.degrees((r0.inv() * r1).magnitude()))
    length_mm = float(np.linalg.norm(delta) * 1000)
    if drift > 2 or abs(length_mm - distance_mm) > 5:
        raise ValueError(f"Sample {label} rotated by {drift:.2f}° and traveled {length_mm:.2f} mm. "
                         f"Keep the tool orientation fixed and translate {distance_mm:g} ±5 mm.")
    return delta, {"length_mm": length_mm, "orientation_drift_deg": drift}


def capture_feedback(device, side, topic, ros2_setup):
    suffix = {"left": "l", "right": "r"}[side]
    code = (PIKA / "scripts/read_pose_snapshot.py").read_text()
    if device == "pika":
        setup = os.environ.get("PIKA_ROS_ROOT", "/root/pika_ros") + "/install/setup.bash"
        command = ["docker", "exec", os.environ.get("PIKA_CONTAINER", "pika-ros"), "bash", "-lc",
                   'source "$1" && exec python3 -c "$2" 1 "$3" "$4"', "_", setup, code, topic,
                   f"/pika_localization_status_{suffix}"]
    else:
        command = ["bash", "-c", 'source "$1" && exec /usr/bin/python3 -c "$2" 2 "$3"',
                   "_", str(ros2_setup), code, topic]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "Could not read pose feedback.")
    return json.loads(result.stdout)


def _default_tcp_file(side, device):
    name = f"{side}_{'pika' if device == 'pika' else 'piper'}_tcp.json"
    preferred = WORKSPACE / "data/calibration" / name
    legacy = WORKSPACE / "data/pika_calibration/tcp" / name
    return preferred if preferred.is_file() or not legacy.is_file() else legacy


def sample(args):
    path = args.directory / f"{args.label}.json"
    if path.exists():
        raise ValueError(f"Sample already exists at {path}. Choose a new measurement directory or label.")
    origin = None
    if args.label[-1] in AXES:
        origin = json.loads((args.directory / f"{args.label[0]}0.json").read_text())
    suffix = {"left": "l", "right": "r"}[args.side]
    topic = args.topic or (f"/pika_pose_{suffix}" if args.device == "pika" else f"/end_pose_{args.side}")
    snapshot = capture_feedback(args.device, args.side, topic, args.ros2_setup)
    pose, quality = stationary_pose(snapshot)
    snapshot.update(device=args.device, side=args.side, label=args.label,
                    pose=pose.tolist(), stationary_quality=quality, tcp_axes_definition=DEFINITION)
    if origin is not None:
        for key in ("device", "side", "topic", "frame", "device_serial"):
            if origin.get(key) != snapshot.get(key):
                raise ValueError(f"The origin and sample {args.label} differ in {key}.")
        tcp_file = args.tcp_file or _default_tcp_file(args.side, args.device)
        calibration = json.loads(tcp_file.read_text())
        offset = np.asarray(calibration["transform"], dtype=float)[:3, 3]
        delta, metrics = translation_delta(np.asarray(origin["pose"]), pose, offset,
                                           args.distance_mm, args.label)
        snapshot["translation_from_origin"] = {**metrics, "delta_reported_frame_m": delta.tolist()}
    save_new(path, snapshot)
    print(f"Saved {path.resolve()}\nAverage pose {np.round(pose, 5).tolist()}")


def solve_axes(samples, calibration, distance_mm=80):
    if not np.isfinite(distance_mm) or distance_mm < 40:
        raise ValueError("Use a measured translation of at least 40 mm.")
    matrix = np.asarray(calibration["transform"], dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("TCP calibration must contain a valid 4×4 transform.")
    identities = {(s["device"], s["side"], s.get("topic"), s.get("frame"), s.get("device_serial"))
                  for s in samples.values()}
    if len(identities) != 1:
        raise ValueError("All samples must share the same device, side, topic, and frame.")
    poses = {name: stationary_pose(s)[0] for name, s in samples.items()}
    directions, errors = {}, []
    for group in "ab":
        origin = poses[f"{group}0"]
        vectors, metrics = [], {}
        for axis in AXES:
            pose = poses[f"{group}{axis}"]
            delta, measurement = translation_delta(origin, pose, matrix[:3, 3], distance_mm, group + axis)
            vectors.append(delta / np.linalg.norm(delta))
            metrics[axis] = measurement
        directions[group] = np.column_stack(vectors)
        if np.linalg.det(directions[group]) <= 0:
            raise ValueError(f"Group {group.upper()} has a reversed axis. Check the right-hand rule.")
        errors.append(metrics)
    r_a = Rotation.from_quat(poses["a0"][3:])
    r_b = Rotation.from_quat(poses["b0"][3:])
    change_deg = float(np.degrees((r_a.inv() * r_b).magnitude()))
    if change_deg < 20:
        raise ValueError("Tilt the gripper at least 20° from group A when measuring independent validation group B.")
    # Fit only A. B is a separate validation with the tool held at a new orientation.
    u, _, vt = np.linalg.svd(directions["a"])
    rotation = u @ vt
    angles = {}
    for group in "ab":
        cosines = np.sum(rotation * directions[group], axis=0)
        angles[group] = np.degrees(np.arccos(np.clip(cosines, -1, 1))).tolist()
    worst = max(max(values) for values in angles.values())
    report = {"numerical_checks_passed": worst <= 5, "max_axis_error_deg": worst,
              "axis_errors_deg": angles, "measured_motion": dict(zip("ab", errors)),
              "validation_orientation_change_deg": change_deg,
              "thresholds": {"max_axis_error_deg": 5, "max_orientation_drift_deg": 2,
                             "max_distance_error_mm": 5, "min_validation_rotation_deg": 20},
              "tcp_axes_definition": DEFINITION,
              "physical_axis_directions_require_operator_confirmation": True}
    candidate = dict(calibration)
    matrix[:3, :3] = rotation
    candidate.update(transform=matrix.tolist(), tcp_axes=DEFINITION, rotation_estimated=True,
                     quality_status="axes_candidate", axes_validation=report)
    candidate["device_serial"] = samples["a0"].get("device_serial")
    return report, candidate


def solve(args):
    names = [group + axis for group in "ab" for axis in "0xyz"]
    samples = {name: json.loads((args.directory / f"{name}.json").read_text()) for name in names}
    first = samples["a0"]
    tcp_file = args.tcp_file or _default_tcp_file(first["side"], first["device"])
    calibration = json.loads(tcp_file.read_text())
    report_path = args.directory / "axes_report.json"
    candidate_path = args.directory / "tcp_candidate.json"
    if report_path.exists() or candidate_path.exists():
        raise ValueError("Axis fit results already exist in this directory. Choose a new measurement directory.")
    report, candidate = solve_axes(samples, calibration, args.distance_mm)
    report["input_tcp_file"] = str(tcp_file.resolve())
    report["measured_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_new(report_path, report)
    if not report["numerical_checks_passed"]:
        raise ValueError(f"Maximum axis error {report['max_axis_error_deg']:.2f}° exceeds 5°. See {report_path}.")
    candidate["axes_measurements_dir"] = str(args.directory.resolve())
    save_new(candidate_path, candidate)
    print(f"Maximum axis error: {report['max_axis_error_deg']:.2f}°.\n"
          f"Calibration candidate: {candidate_path.resolve()}\n"
          "Verify axis directions and gripper opening before using this calibration to generate labels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    capture = sub.add_parser("sample", help="Save pose feedback while the gripper is stationary")
    capture.add_argument("device", choices=("pika", "piper"))
    capture.add_argument("side", choices=("left", "right"))
    capture.add_argument("label", choices=[group + axis for group in "ab" for axis in "0xyz"])
    capture.add_argument("--directory", required=True, type=Path)
    capture.add_argument("--topic", help="Override the default pose feedback topic")
    capture.add_argument("--ros2-setup", type=Path, default=Path("/opt/ros/humble/setup.bash"))
    capture.add_argument("--tcp-file", type=Path)
    capture.add_argument("--distance-mm", type=float, default=80)
    fit = sub.add_parser("solve", help="Fit tool axes and validate them using group B")
    fit.add_argument("--directory", required=True, type=Path)
    fit.add_argument("--tcp-file", type=Path)
    fit.add_argument("--distance-mm", type=float, default=80)
    args = parser.parse_args()
    try:
        (sample if args.operation == "sample" else solve)(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"TCP measurement stopped: {error}\n")


if __name__ == "__main__":
    main()
