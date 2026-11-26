"""Prepare bimanual TCP labels and export them with VISTA's LeRobot writer."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import tempfile

import numpy as np


SIDES = {"left": "pika_l", "right": "pika_r"}
CAMERAS = {f"observation.images.{side}_wrist": f"images/{side}_fisheye_rgb/frames" for side in SIDES}
ACTION_NAMES = [
    f"{side}_{field}" for side in SIDES
    for field in ("dx_m", "dy_m", "dz_m", "rx_rad", "ry_rad", "rz_rad", "gripper_width_m")
]
STATE_NAMES = [
    "right_in_left_x_m", "right_in_left_y_m", "right_in_left_z_m",
    "right_in_left_rot_col0_x", "right_in_left_rot_col0_y", "right_in_left_rot_col0_z",
    "right_in_left_rot_col1_x", "right_in_left_rot_col1_y", "right_in_left_rot_col1_z",
    "left_gripper_width_m", "right_gripper_width_m",
]
PIKA_TCP_NAMES = [
    f"{side}_{field}" for side in SIDES
    for field in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
] + [f"{side}_gripper_width_m" for side in SIDES]


TRACKING_TIMEOUT_NS = 250_000_000
TRACKING_RECOVERY_NS = 250_000_000
TRACKING_LOSS_GUARD_NS = 50_000_000


def motion_screen(group, raw_episode, times, max_speed=2.0, max_angular_speed=15.0, pose_hold_ms=250.0):
    from scipy.spatial.transform import Rotation
    from prepare_pika_dataset import read_stream

    min_pose_hold_ns = round(pose_hold_ms * 1e6)
    rejected = np.zeros(len(times) - 1, dtype=bool)
    checks = {}
    for side, label in SIDES.items():
        stream = read_stream(raw_episode / "localization/pose" / label, {".json"},
                             ["x", "y", "z", "roll", "pitch", "yaw"])
        rotations = Rotation.from_euler("xyz", stream.values[:, 3:])
        dt = np.diff(stream.times) * 1e-9
        speed = np.linalg.norm(np.diff(stream.values[:, :3], axis=0), axis=1) / dt
        angular_speed = (rotations[:-1].inv() * rotations[1:]).magnitude() / dt
        flags = np.flatnonzero((speed > max_speed) | (angular_speed > max_angular_speed))
        left = group[f"sync/pose_{side}/left_timestamp_ns"][:]
        right = group[f"sync/pose_{side}/right_timestamp_ns"][:]
        if not np.isin(left, stream.times).all() or not np.isin(right, stream.times).all():
            raise ValueError(f"Raw pose timestamps do not match source export for {side}")
        for i in flags:
            # Reject transitions whose endpoints or interpolation span touch the jump.
            rejected |= (left[:-1] < stream.times[i + 1]) & (right[1:] > stream.times[i])
        repeated = np.all(stream.values[1:] == stream.values[:-1], axis=1)
        edges = np.flatnonzero(np.diff(np.r_[False, repeated, False]))
        holds = []
        for start, last in edges.reshape(-1, 2):
            if stream.times[last] - stream.times[start] < min_pose_hold_ns:
                continue
            # Include the first change after the stale pose in the excluded interval.
            end = min(last + 1, len(stream.times) - 1)
            rejected |= (left[:-1] < stream.times[end]) & (right[1:] > stream.times[start])
            holds.append({
                "start_s": float((stream.times[start] - times[0]) * 1e-9),
                "held_until_s": float((stream.times[last] - times[0]) * 1e-9),
                "end_s": float((stream.times[end] - times[0]) * 1e-9),
            })
        checks[side] = {
            "peak_speed_m_s": float(speed.max()),
            "peak_angular_speed_rad_s": float(angular_speed.max()),
            "flagged_raw_intervals": [
                {"start_s": float((stream.times[i] - times[0]) * 1e-9),
                 "end_s": float((stream.times[i + 1] - times[0]) * 1e-9),
                 "speed_m_s": float(speed[i]), "angular_speed_rad_s": float(angular_speed[i])}
                for i in flags
            ],
            "suspected_stale_pose_intervals": holds,
        }
    return rejected, {"max_speed_m_s": max_speed, "max_angular_speed_rad_s": max_angular_speed,
                      "min_exact_pose_hold_s": min_pose_hold_ns * 1e-9,
                      "tracking_status_checked": False, "sides": checks}


def tracking_screen(group, raw_episode, times):
    """Accept only action spans contained in a continuous, stable tracking interval."""
    path = raw_episode / "tracking_status.json"
    rejected = np.zeros(len(times) - 1, dtype=bool)
    if not path.exists():
        return rejected, {"checked": False, "reason": "legacy_episode_without_tracking_status"}

    log = json.loads(path.read_text())
    if (not isinstance(log, dict) or log.get("version") != 1
            or log.get("clock") != "ros_receive_time_ns" or not isinstance(log.get("sides"), dict)):
        raise ValueError(f"Invalid tracking log format: {path}")
    report = {"checked": True, "clock": log["clock"], "timeout_ms": TRACKING_TIMEOUT_NS / 1e6,
              "recovery_ms": TRACKING_RECOVERY_NS / 1e6, "loss_guard_ms": TRACKING_LOSS_GUARD_NS / 1e6,
              "sides": {}}
    for side in SIDES:
        samples = log["sides"].get(side, [])
        if not isinstance(samples, list) or any(
            not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int
            or not 0 < row[0] < np.iinfo(np.int64).max - TRACKING_TIMEOUT_NS or type(row[1]) is not bool
            for row in samples
        ):
            raise ValueError(f"Invalid {side} tracking sample: {path}")
        stamps = np.array([stamp for stamp, _ in samples], dtype=np.int64)
        accurate = np.array([ok for _, ok in samples], dtype=bool)
        if np.any(np.diff(stamps) < 0):
            raise ValueError(f"ROS time moved backwards in the {side} tracking log: {path}")
        # Multiple flags can arrive at the same ROS time. A False wins.
        stamps, unique_indices = np.unique(stamps, return_index=True)
        accurate = np.logical_and.reduceat(accurate, unique_indices)
        gaps = np.diff(stamps)
        connected = accurate[:-1] & accurate[1:] & (gaps <= TRACKING_TIMEOUT_NS)
        # Pad an empty stream with no boundary flags.
        boundary = np.zeros(min(len(stamps), 1), dtype=bool)
        starts = stamps[accurate & ~np.r_[boundary, connected]] + TRACKING_RECOVERY_NS
        last = np.flatnonzero(accurate & ~np.r_[connected, boundary])
        ends = stamps[last] + TRACKING_TIMEOUT_NS
        has_next = last + 1 < len(stamps)
        ends[has_next] = np.minimum(ends[has_next], stamps[last[has_next] + 1]) - TRACKING_LOSS_GUARD_NS
        long_enough = (stamps[last] >= starts) & (ends > starts)
        starts, ends = starts[long_enough], ends[long_enough]

        # Include the raw pose samples on either side of interpolated grid endpoints.
        left = group[f"sync/pose_{side}/left_timestamp_ns"][:-1]
        right = group[f"sync/pose_{side}/right_timestamp_ns"][1:]
        accepted = np.zeros(len(rejected), dtype=bool)
        if len(starts):
            interval = np.searchsorted(starts, left, side="right") - 1
            accepted = (interval >= 0) & (right < ends[np.maximum(interval, 0)])
        rejected |= ~accepted
        report["sides"][side] = {"status_samples": len(samples), "inaccurate_samples": sum(not ok for _, ok in samples),
                                 "max_status_gap_ms": float(gaps.max() / 1e6) if len(gaps) else None,
                                 "transitions_rejected": int((~accepted).sum())}
    return rejected, report


def calibrated_tcp_poses(pose, measurements):
    from scipy.spatial.transform import Rotation

    positions, rotations = [], []
    for i, side in enumerate(SIDES):
        matrix = np.asarray(measurements[side]["transform"], dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
            raise ValueError(f"Invalid pose-to-TCP transform for {side}")
        pika_rotation = Rotation.from_quat(pose[:, i, 3:])
        positions.append(pose[:, i, :3] + pika_rotation.apply(matrix[:3, 3]))
        rotations.append(pika_rotation * Rotation.from_matrix(matrix[:3, :3]))
    return positions, rotations


def tcp_labels(pose, width, calibration):
    measurements = {side: json.loads((calibration / f"{side}_pika_tcp.json").read_text())
                    for side in SIDES}
    positions, rotations = calibrated_tcp_poses(pose, measurements)
    actions = []
    for i, (p, r) in enumerate(zip(positions, rotations, strict=True)):
        dr = (r[:-1].inv() * r[1:]).as_rotvec()
        dp = r[:-1].inv().apply(np.diff(p, axis=0))
        actions.append(np.c_[dp, dr, width[1:, i]])
    left_inverse = rotations[0].inv()
    right_in_left = left_inverse.apply(positions[1] - positions[0])
    rotation = (left_inverse * rotations[1]).as_matrix()
    state = np.c_[right_in_left, rotation[:, :, 0], rotation[:, :, 1], width]
    return state[:-1].astype(np.float32), np.column_stack(actions).astype(np.float32), measurements


def dataset_cameras(group):
    cameras = dict(CAMERAS)
    if "images/head_rgb/frames" in group:
        cameras["observation.images.head"] = "images/head_rgb/frames"
    return cameras


def prepare(source, raw_episode, calibration, output, min_segment_seconds=1.0,
            max_speed_m_s=2.0, max_angular_speed_rad_s=15.0, pose_hold_ms=250.0):
    import h5py
    from pika_options import FPS_CHOICES
    from prepare_pika_dataset import timestamp_grid, valid_runs

    with h5py.File(source, "r") as file:
        fps = file.attrs["fps"]
        if file.attrs["format"] != "pika-robomimic-source-v1" or fps not in FPS_CHOICES:
            raise ValueError(f"Expected a Dual Pika source HDF5 with supported fps: {source}")
        fps = int(fps)
        group = file[f"source_episodes/{raw_episode.name}"]
        times = group["timestamp_ns"][:]
        if len(times) < 2 or not np.array_equal(times, timestamp_grid(int(times[0]), int(times[-1]), fps)):
            raise ValueError(f"The source timestamps are not on the {fps} Hz grid")
        # VISTA observes only the wrist pair. D405 RGB/depth gaps do not invalidate its labels.
        motion_valid = group.get("motion_valid", group["valid"])[:].astype(bool)
        wrist_times = np.column_stack([
            group[f"images/{side}_fisheye_rgb/timestamp_ns"][:] for side in SIDES
        ])
        wrist_age_ns = times[:, None] - wrist_times
        wrist_skew_ns = np.ptp(wrist_times, axis=1)
        max_age_ms = float(file.attrs.get("max_image_offset_ms", 40))
        max_skew_ms = float(file.attrs.get("max_image_skew_ms", 40))
        wrist_age_valid = ((wrist_age_ns >= 0) & (wrist_age_ns <= round(max_age_ms * 1e6))).all(axis=1)
        wrist_skew_valid = wrist_skew_ns <= round(max_skew_ms * 1e6)
        valid = motion_valid & wrist_age_valid & wrist_skew_valid
        head_sync = None
        if "images/head_rgb" in group:
            head_times = group["images/head_rgb/timestamp_ns"][:]
            head_age = times - head_times
            head_skew = np.ptp(np.column_stack([wrist_times, head_times]), axis=1)
            head_ok = ((head_age >= 0) & (head_age <= round(max_age_ms * 1e6))
                       & (head_skew <= round(float(group.attrs.get("head_max_skew_ms", 40)) * 1e6)))
            valid &= head_ok
            head_sync = {"max_skew_ms": float(group.attrs.get("head_max_skew_ms", 40)),
                         "rows_rejected": int((~head_ok).sum()), "hardware_triggered": False}

        jumps, screen = motion_screen(group, raw_episode, times, max_speed_m_s,
                                      max_angular_speed_rad_s, pose_hold_ms)
        unreliable_tracking, tracking = tracking_screen(group, raw_episode, times)
        screen["tracking_status_checked"] = tracking["checked"]
        state, action, measurements = tcp_labels(group["pose"][:], group["width"][:], calibration)
        eligible = valid[:-1] & valid[1:]
        clean = eligible & ~jumps & ~unreliable_tracking
        min_transitions = math.ceil(min_segment_seconds * fps)
        runs = [(a, b) for a, b in valid_runs(clean)
                if b - a >= min_transitions]
        if not runs:
            if tracking["checked"] and (eligible & unreliable_tracking).any():
                count = int((eligible & unreliable_tracking).sum())
                raise ValueError(f"{raw_episode.name}: no segment with valid tracking lasts at least "
                                 f"{min_segment_seconds:g} s. Tracking rejected {count}/{int(eligible.sum())} "
                                 "otherwise aligned actions. See tracking_status.json and check Vive.")
            raise ValueError(f"No contiguous segment with at least {min_transitions} valid transitions")
        rows = np.concatenate([np.arange(a, b) for a, b in runs])
        if not (np.isfinite(state[rows]).all() and np.isfinite(action[rows]).all()):
            raise ValueError("Non-finite TCP labels")
        report = {
            "episode": raw_episode.name, "source_hdf5": str(source.resolve()), "fps": fps,
            "task": json.loads(group.attrs["instruction"])[0],
            "action_fields": ACTION_NAMES, "state_fields": STATE_NAMES,
            "action_semantics": "Per-step TCP-local pose increment (t to t+1) and next absolute width",
            "tcp_measurements": measurements, "motion_screen": screen, "tracking_status": tracking,
            "cameras": dataset_cameras(group), "head_sync": head_sync,
            "wrist_sync": {
                "pairing": str(file.attrs.get("wrist_pairing", "legacy_independent")),
                "max_age_ms": max_age_ms, "max_skew_ms": max_skew_ms,
                "rows_rejected_by_age": int((~wrist_age_valid).sum()),
                "rows_rejected_by_skew": int((~wrist_skew_valid).sum()),
                "kept_age_ms_p95": dict(zip(SIDES, np.percentile(wrist_age_ns[rows], 95, axis=0) / 1e6)),
                "kept_skew_ms_p95": float(np.percentile(wrist_skew_ns[rows], 95) / 1e6),
                "kept_skew_ms_max": float(wrist_skew_ns[rows].max() / 1e6),
            },
            "rows_rejected_by_pose_or_gripper": int((~motion_valid).sum()),
            "source_transitions": int(eligible.sum()),
            "transitions_rejected_by_motion_screen": int((eligible & jumps).sum()),
            "transitions_rejected_by_tracking_status": int((eligible & unreliable_tracking).sum()),
            "transitions_rejected_by_motion_or_tracking": int((eligible & (jumps | unreliable_tracking)).sum()),
            "transitions_rejected_by_short_segment": int(clean.sum()) - len(rows),
            "min_segment_transitions": min_transitions,
            "min_segment_seconds": min_segment_seconds,
            "kept_transitions": len(rows),
            "segments": [
                {"first_source_row": a, "last_source_row": b - 1, "transitions": b - a,
                 "start_s": float((times[a] - times[0]) * 1e-9),
                 "end_s": float((times[b] - times[0]) * 1e-9)}
                for a, b in runs
            ],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pika_vla_", dir=output.parent) as temp:
        staging = Path(temp) / "labels"
        staging.mkdir()
        np.savez_compressed(staging / "labels.npz", state=state[rows], action=action[rows],
                            source_row=rows, timestamp_ns=times[rows],
                            segment_index=np.repeat(np.arange(len(runs)), [b - a for a, b in runs]))
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    print(f"{raw_episode.name}: retained {len(rows)}/{int(eligible.sum())} actions in {len(runs)} segments → {output}")


def write_dataset_provenance(staging, group, rows, report):
    """Portable row-to-sensor timestamps and calibration alongside encoded videos."""
    directory = staging / "provenance"
    directory.mkdir(exist_ok=True)
    name = report["episode"]
    timestamps = {"source_row": rows, "timestamp_ns": group["timestamp_ns"][rows]}
    calibration = {}
    for key, path in dataset_cameras(group).items():
        camera = group[path.rsplit("/", 1)[0]]
        label = key.rsplit(".", 1)[-1]
        timestamps[label + "_timestamp_ns"] = camera["timestamp_ns"][rows]
        calibration[key] = {field: np.asarray(camera.attrs[field]).tolist()
                            for field in ("K", "D", "camera_model")}
        calibration[key]["image_size"] = list(reversed(camera["frames"].shape[1:3]))
    np.savez_compressed(directory / (name + "_timestamps.npz"), **timestamps)
    if "calibration_json" in group:
        calibration["source_calibration"] = json.loads(group["calibration_json"].asstr()[()])
    (directory / (name + "_calibration.json")).write_text(json.dumps(calibration, indent=2) + "\n")
    return {"timestamps": f"provenance/{name}_timestamps.npz",
            "calibration": f"provenance/{name}_calibration.json"}


def _compute_stats_dict(arr: np.ndarray) -> dict:
    arr64 = np.asarray(arr, dtype=np.float64)
    return {
        "min": np.min(arr64, axis=0).tolist(),
        "max": np.max(arr64, axis=0).tolist(),
        "mean": np.mean(arr64, axis=0).tolist(),
        "std": np.std(arr64, axis=0).tolist(),
        "q01": np.quantile(arr64, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr64, 0.99, axis=0).tolist(),
        "count": [int(arr64.shape[0])],
    }


def _export_lerobot_v3_native(prepared_dirs, output: Path, repo_id: str, representation: str):
    """Zero-`umi-vista-main` LeRobot v3.0 writer using pyarrow + OpenCV."""
    import cv2
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pika_options import FPS_CHOICES

    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    absolute_tcp = representation == "pika-tcp"
    state_names = PIKA_TCP_NAMES if absolute_tcp else STATE_NAMES
    action_names = PIKA_TCP_NAMES if absolute_tcp else ACTION_NAMES

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pika_lerobot_", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        (staging / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        (staging / "meta").mkdir(parents=True, exist_ok=True)

        reports = []
        all_states, all_actions = [], []
        episodes_meta = []
        tasks_map: dict[str, int] = {}
        ep_idx = 0
        global_frame_idx = 0
        expected_cameras = None
        features = None
        fps_val = None

        for prepared in prepared_dirs:
            report = json.loads((prepared / "report.json").read_text())
            if (report["fps"] not in FPS_CHOICES or report["action_fields"] != ACTION_NAMES
                    or report["state_fields"] != STATE_NAMES):
                raise ValueError(f"Label schema mismatch: {prepared}")
            if reports and report["tcp_measurements"] != reports[0]["tcp_measurements"]:
                raise ValueError(f"TCP measurements differ across prepared episodes: {prepared}")
            if reports and report["fps"] != reports[0]["fps"]:
                raise ValueError(f"Episode frame rates differ: {prepared}")
            fps_val = int(report["fps"])

            with ExitStack() as stack:
                labels = stack.enter_context(np.load(prepared / "labels.npz"))
                source = stack.enter_context(h5py.File(report["source_hdf5"], "r"))
                group = source[f"source_episodes/{report['episode']}"]
                cameras = dataset_cameras(group)
                if "observation.images.head" in cameras and report.get("head_sync") is None:
                    raise ValueError("Head camera labels must be regenerated with head synchronization checks")
                images = {key: group[path] for key, path in cameras.items()}
                state, action = labels["state"].astype(np.float32), labels["action"].astype(np.float32)
                rows, segments = labels["source_row"], labels["segment_index"]
                times = group["timestamp_ns"][:]
                if not np.array_equal(labels["timestamp_ns"], times[rows]):
                    raise ValueError(f"Source timestamps changed: {prepared}")
                if absolute_tcp:
                    positions, rotations = calibrated_tcp_poses(group["pose"][:], report["tcp_measurements"])
                    widths = group["width"][:]
                    tcp = np.column_stack([
                        positions[0], rotations[0].as_quat(), positions[1], rotations[1].as_quat(), widths,
                    ]).astype(np.float32)
                    state, action = tcp[rows], tcp[rows + 1]
                    if not (np.isfinite(state).all() and np.isfinite(action).all()):
                        raise ValueError(f"Non-finite absolute TCP poses: {prepared}")
                    report.update(
                        state_fields=state_names,
                        action_fields=action_names,
                        action_semantics="Absolute Pika TCP poses and widths at t+1 in the source Vive tracking frame",
                    )

                cur_features = {
                    "observation.state": {"dtype": "float32", "shape": [len(state_names)], "names": state_names},
                    "action": {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
                    **{
                        key: {"dtype": "video", "shape": [3, int(arr.shape[1]), int(arr.shape[2])],
                              "names": ["channels", "height", "width"]}
                        for key, arr in images.items()
                    },
                }
                if features is None:
                    features = cur_features
                    expected_cameras = set(images)
                elif set(images) != expected_cameras or any(
                    cur_features[k]["shape"] != features[k]["shape"] for k in images
                ):
                    raise ValueError(f"Episode image dimensions differ: {prepared}")

                report["provenance"] = write_dataset_provenance(staging, group, rows, report)
                all_states.append(state)
                all_actions.append(action)

                task_str = str(report["task"])
                if task_str not in tasks_map:
                    tasks_map[task_str] = len(tasks_map)
                task_idx = tasks_map[task_str]

                seg_splits = np.flatnonzero(segments[1:] != segments[:-1]) + 1 if len(segments) > 1 else []
                idx_runs = np.split(np.arange(len(rows)), seg_splits)
                ep_lerobot_indices: list[int] = []
                for run_idx in idx_runs:
                    if len(run_idx) == 0:
                        continue
                    seg_rows = rows[run_idx]
                    seg_state = state[run_idx]
                    seg_action = action[run_idx]
                    n_frames = len(run_idx)
                    ts_rel = (np.arange(n_frames, dtype=np.float32) / float(fps_val))
                    done_flags = [False] * (n_frames - 1) + [True]

                    table = pa.table({
                        "observation.state": [row.tolist() for row in seg_state],
                        "action": [row.tolist() for row in seg_action],
                        "timestamp": ts_rel.tolist(),
                        "frame_index": list(range(n_frames)),
                        "episode_index": [ep_idx] * n_frames,
                        "index": list(range(global_frame_idx, global_frame_idx + n_frames)),
                        "task_index": [task_idx] * n_frames,
                        "next.done": done_flags,
                    })
                    chunk_dir = staging / f"data/chunk-{ep_idx // 1000:03d}"
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, chunk_dir / f"episode_{ep_idx:06d}.parquet")

                    for cam_key, arr in images.items():
                        vid_dir = staging / "videos" / cam_key
                        vid_dir.mkdir(parents=True, exist_ok=True)
                        vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
                        h, w = int(arr.shape[1]), int(arr.shape[2])
                        writer = cv2.VideoWriter(
                            str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps_val), (w, h)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Failed to open VideoWriter: {vid_path}")
                        try:
                            for r in seg_rows:
                                frame_rgb = arr[int(r)]
                                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        finally:
                            writer.release()

                    episodes_meta.append({"episode_index": ep_idx, "tasks": [task_str], "length": n_frames})
                    ep_lerobot_indices.append(ep_idx)
                    global_frame_idx += n_frames
                    ep_idx += 1

            report["lerobot_episode_indices"] = ep_lerobot_indices
            reports.append(report)

        if ep_idx == 0 or not all_states:
            raise ValueError("No prepared episodes")

        stats = {
            "observation.state": _compute_stats_dict(np.concatenate(all_states, axis=0)),
            "action": _compute_stats_dict(np.concatenate(all_actions, axis=0)),
        }
        stats_text = json.dumps(stats, indent=2) + "\n"
        (staging / "meta/stats.json").write_text(stats_text)
        (staging / "meta/abs_action_norm_stats.json").write_text(stats_text)
        (staging / "meta/tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": idx, "task": t}) + "\n" for t, idx in tasks_map.items())
        )
        (staging / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(ep) + "\n" for ep in episodes_meta)
        )
        info = {
            "codebase_version": "v3.0",
            "repo_id": repo_id,
            "robot_type": "dual_pika",
            "fps": fps_val,
            "total_episodes": ep_idx,
            "total_frames": global_frame_idx,
            "total_tasks": len(tasks_map),
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        if absolute_tcp:
            info["pika"] = {
                "state_layout": "dual_pika_tcp_pose7_plus_gripper_widths",
                "reference_frame": "source_vive_tracking_world",
                "action_semantics": "next_absolute_tcp_pose_and_gripper_width",
            }
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        (staging / "pika_source_reports.json").write_text(json.dumps(reports, indent=2) + "\n")
        staging.rename(output)
    print(f"LeRobot v3.0 ({representation}, native pyarrow+mp4) → {output}")


def export_lerobot(prepared_dirs, output, repo_id, format_version="v3.0", representation="umi",
                   ik_calibration=None):
    if representation not in {"umi", "pika-tcp", "piper-joint"}:
        raise ValueError(f"Unsupported representation: {representation}")
    absolute_tcp = representation == "pika-tcp"
    joint_space = representation == "piper-joint"
    if joint_space != (ik_calibration is not None):
        raise ValueError("--ik-calibration is required for piper-joint and applies only to that representation")
    if absolute_tcp and format_version != "v3.0":
        raise ValueError(f"{representation} export requires LeRobot v3.0")
    if format_version == "v2.1" or joint_space:
        raise ValueError(
            "Legacy v2.1 / piper-joint exporter was replaced by standardized C-STRO retargeting. "
            "Run: ./pika import-raw ... && ./pika convert --format robot-joints --robot piper_pika --solver cstro ... "
            "&& ./pika dataset export --layout bi_piper_pika_follower ..."
        )
    if format_version != "v3.0":
        raise ValueError(f"Unsupported LeRobot format: {format_version}")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.normalize import RunningStats, serialize_json
    except ImportError:
        return _export_lerobot_v3_native(prepared_dirs, output, repo_id, representation)
    import h5py
    from pika_options import FPS_CHOICES

    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    state_names = PIKA_TCP_NAMES if absolute_tcp else STATE_NAMES
    action_names = PIKA_TCP_NAMES if absolute_tcp else ACTION_NAMES
    state_stats, action_stats = RunningStats(), RunningStats()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pika_lerobot_", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        dataset = None
        reports = []
        ep_counter = 0
        try:
            for prepared in prepared_dirs:
                report = json.loads((prepared / "report.json").read_text())
                if (report["fps"] not in FPS_CHOICES or report["action_fields"] != ACTION_NAMES
                        or report["state_fields"] != STATE_NAMES):
                    raise ValueError(f"Label schema mismatch: {prepared}")
                if reports and report["tcp_measurements"] != reports[0]["tcp_measurements"]:
                    raise ValueError(f"TCP measurements differ across prepared episodes: {prepared}")
                if reports and report["fps"] != reports[0]["fps"]:
                    raise ValueError(f"Episode frame rates differ: {prepared}")
                with ExitStack() as stack:
                    labels = stack.enter_context(np.load(prepared / "labels.npz"))
                    source = stack.enter_context(h5py.File(report["source_hdf5"], "r"))
                    group = source[f"source_episodes/{report['episode']}"]
                    cameras = dataset_cameras(group)
                    if "observation.images.head" in cameras and report.get("head_sync") is None:
                        raise ValueError("Head camera labels must be regenerated with head synchronization checks")
                    images = {key: group[path] for key, path in cameras.items()}
                    state, action = labels["state"], labels["action"]
                    rows, segments = labels["source_row"], labels["segment_index"]
                    times = group["timestamp_ns"][:]
                    if not np.array_equal(labels["timestamp_ns"], times[rows]):
                        raise ValueError(f"Source timestamps changed: {prepared}")
                    if absolute_tcp:
                        positions, rotations = calibrated_tcp_poses(group["pose"][:], report["tcp_measurements"])
                        widths = group["width"][:]
                        tcp = np.column_stack([
                            positions[0], rotations[0].as_quat(), positions[1], rotations[1].as_quat(), widths,
                        ]).astype(np.float32)
                        state, action = tcp[rows], tcp[rows + 1]
                        if not (np.isfinite(state).all() and np.isfinite(action).all()):
                            raise ValueError(f"Non-finite absolute TCP poses: {prepared}")
                        report.update(state_fields=state_names, action_fields=action_names,
                                      action_semantics="Absolute Pika TCP poses and widths at t+1 in the source Vive tracking frame")
                    features = {
                        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
                        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
                        **{key: {"dtype": "video", "shape": (3, *array.shape[1:3]),
                                 "names": ["channels", "height", "width"]} for key, array in images.items()},
                    }
                    if dataset is None:
                        dataset = LeRobotDataset.create(repo_id, fps=int(report["fps"]), root=staging, features=features,
                                                        robot_type="dual_pika", video_backend="pyav")
                    elif (set(images) != set(dataset.meta.camera_keys)
                          or any(features[key]["shape"] != dataset.features[key]["shape"] for key in images)):
                        raise ValueError(f"Episode image dimensions differ: {prepared}")
                    report["provenance"] = write_dataset_provenance(staging, group, rows, report)
                    state_stats.update(state)
                    action_stats.update(action)
                    ep_lerobot_indices = []
                    for i, row in enumerate(rows):
                        dataset.add_frame({"task": report["task"], "observation.state": state[i], "action": action[i],
                                           **{key: array[row] for key, array in images.items()}})
                        if i + 1 == len(rows) or segments[i + 1] != segments[i]:
                            dataset.save_episode()
                            ep_lerobot_indices.append(ep_counter)
                            ep_counter += 1
                    report["lerobot_episode_indices"] = ep_lerobot_indices
                reports.append(report)
        finally:
            if dataset is not None:
                dataset.finalize()
        if dataset is None:
            raise ValueError("No prepared episodes")
        stats = {"observation.state": state_stats.get_statistics(), "action": action_stats.get_statistics()}
        (staging / "meta/abs_action_norm_stats.json").write_text(serialize_json(stats))
        (staging / "pika_source_reports.json").write_text(json.dumps(reports, indent=2) + "\n")
        if absolute_tcp:
            info_path = staging / "meta/info.json"
            info = json.loads(info_path.read_text())
            info["pika"] = {
                "state_layout": "dual_pika_tcp_pose7_plus_gripper_widths",
                "reference_frame": "source_vive_tracking_world",
                "action_semantics": "next_absolute_tcp_pose_and_gripper_width",
            }
            info_path.write_text(json.dumps(info, indent=2) + "\n")
        loaded = LeRobotDataset(repo_id, root=staging, revision="v3.0", video_backend="pyav",
                               use_delta_action=False)
        loaded.load_abs_action_norm_stats()
        sample = loaded[0]
        if (sample["action"].shape != (len(action_names),)
                or sample["observation.state"].shape != (len(state_names),)
                or sample["task"] != reports[0]["task"]
                or any(tuple(sample[key].shape) != tuple(loaded.features[key]["shape"]) for key in loaded.meta.camera_keys)):
            raise ValueError("VISTA's LeRobot reader did not load the expected sample")
        del loaded
        staging.rename(output)
    print(f"LeRobot v3.0 ({representation}) → {output}")


def main():
    from pika_options import PREPARE_DEFAULTS, add_prepare_options

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("raw_episode", type=Path)
    prep.add_argument("calibration", type=Path)
    prep.add_argument("output", type=Path)
    add_prepare_options(prep)
    prep.set_defaults(**PREPARE_DEFAULTS)
    lerobot = sub.add_parser("lerobot")
    lerobot.add_argument("prepared", type=Path, nargs="+")
    lerobot.add_argument("--output-dir", type=Path, required=True)
    lerobot.add_argument("--repo-id", default="local/dual_pika")
    lerobot.add_argument("--format-version", choices=("v2.1", "v3.0"), default="v3.0")
    lerobot.add_argument("--representation", choices=("umi", "pika-tcp", "piper-joint"), default="umi")
    lerobot.add_argument("--ik-calibration", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            if args.output.exists():
                raise ValueError(f"Output already exists: {args.output}")
            prepare(args.source, args.raw_episode, args.calibration, args.output,
                    **{key: getattr(args, key) for key in PREPARE_DEFAULTS})
        else:
            export_lerobot(args.prepared, args.output_dir, args.repo_id, args.format_version,
                           args.representation, args.ik_calibration)
    except ImportError as error:
        parser.exit(1, f"Missing Python dependency: {error}. Set up the environment using the README.\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Processing stopped: {error}\n")


if __name__ == "__main__":
    main()
