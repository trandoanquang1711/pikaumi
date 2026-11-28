#!/usr/bin/env python3
"""Resample raw dual-Pika episodes and export portable source demonstrations.

These are Pika observations and inferred Cartesian actions, not robot feedback.
No ROS installation, tactile sensor, pressure channel, or CAN connection is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pika_options import add_fps_option, recorded_export_fps, recorded_wrist_camera, wrist_skew_default


SIDES = {"left": "pika_l", "right": "pika_r"}
IMAGE_STREAMS = {
    "left_d405_rgb": "camera/color/pikaDepthCamera_l",
    "right_d405_rgb": "camera/color/pikaDepthCamera_r",
    "left_fisheye_rgb": "camera/color/pikaFisheyeCamera_l",
    "right_fisheye_rgb": "camera/color/pikaFisheyeCamera_r",
    "left_depth": "camera/depth/pikaDepthCamera_l",
    "right_depth": "camera/depth/pikaDepthCamera_r",
}


def timestamp_ns(path):
    """Parse microsecond filenames without rounding Unix time through float."""
    return int(Decimal(Path(path).stem) * 1_000_000_000)


@dataclass
class Stream:
    files: list
    times: np.ndarray
    values: np.ndarray | None = None


def read_stream(directory, extensions, fields=None):
    files = []
    for p in Path(directory).iterdir():
        if p.suffix.lower() in extensions:
            try:
                timestamp_ns(p)
            except Exception:
                continue
            files.append(p)
    files.sort(key=timestamp_ns)
    if len(files) < 2:
        raise ValueError(f"Need at least two samples: {directory}")
    times = np.asarray([timestamp_ns(p) for p in files], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Duplicate/non-increasing timestamps: {directory}")
    values = None
    if fields:
        rows = [json.loads(p.read_text()) for p in files]
        values = np.asarray([[row[k] for k in fields] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite sensor data: {directory}")
    return Stream(files, times, values)


def brackets(times, grid, max_gap_ns):
    right = np.searchsorted(times, grid, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    exact = times[right] == grid
    left = np.where(exact, right, left)
    gap = times[right] - times[left]
    valid = ((grid >= times[0]) & (grid <= times[-1])
             & ((gap <= max_gap_ns) | exact))
    alpha = np.divide(grid - times[left], gap, out=np.zeros(len(grid)), where=gap > 0)
    return left, right, alpha, valid


def sample_images(times, grid, policy, tolerance_ns):
    right = np.searchsorted(times, grid, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    if policy == "causal":
        index = np.searchsorted(times, grid, side="right") - 1
        available = index >= 0
        index = index.clip(0)
    else:
        index = np.where(np.abs(times[right] - grid) < np.abs(times[left] - grid), right, left)
        available = np.ones(len(grid), dtype=bool)
    offsets = times[index] - grid
    valid = available & (np.abs(offsets) <= tolerance_ns)
    return index, offsets, valid


def sample_wrist_pair(left_times, right_times, grid, policy, tolerance_ns):
    streams = (left_times, right_times)
    samples = [sample_images(times, grid, policy, tolerance_ns) for times in streams]
    if policy == "causal":
        for i, times in enumerate(streams):
            index, _, valid = samples[i]
            other_index, _, other_valid = samples[1 - i]
            peer = streams[1 - i][other_index]
            right = np.searchsorted(times, peer)
            # Try the frames bracketing the peer's time, within the original causal age limit.
            for candidate in (right.clip(0, len(times) - 1), (right - 1).clip(0)):
                age = grid - times[candidate]
                better = np.abs(times[candidate] - peer) < np.abs(times[index] - peer)
                use = valid & other_valid & (age >= 0) & (age <= tolerance_ns) & better
                index[use] = candidate[use]
            samples[i] = index, times[index] - grid, valid
    return samples


def sample_head(times, grid, wrist_timestamps, tolerance_ns):
    """Choose the closest head frame to the wrist pair, without future images."""
    low = wrist_timestamps.min(axis=1)
    target = low + (wrist_timestamps.max(axis=1) - low) // 2
    index, _, _ = sample_images(times, target, "nearest", tolerance_ns)
    latest = np.searchsorted(times, grid, side="right") - 1
    index = np.minimum(index, latest).clip(0)
    offset = times[index] - grid
    valid = (latest >= 0) & (offset <= 0) & (offset >= -tolerance_ns)
    return index, offset, valid


def head_streams(episode):
    metadata = episode / "capture_metadata.json"
    profile = json.loads(metadata.read_text()).get("head_camera", {}) if metadata.is_file() else {}
    present = (episode / "camera/color/headCamera").is_dir()
    if profile.get("enabled") or present:
        streams = {"head_rgb": "camera/color/headCamera"}
        if profile.get("enable_depth") or (episode / "camera/depth/headCamera").is_dir():
            streams["head_depth"] = "camera/depth/headCamera"
        return streams
    return {}


def valid_runs(valid):
    """Return observation runs; never create transitions across a missing row."""
    padded = np.r_[False, valid, False].astype(np.int8)
    return [(int(a), int(b)) for a, b in zip(
        np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    ) if b - a >= 2]


def resolve_intrinsic(data_root, episode, side, explicit=None):
    name = f"{side}_fisheye_intrinsics.json"
    local = (Path(explicit) if explicit else episode / "calibration") / name
    if local.is_file():
        candidates = [local]
    elif explicit:
        raise ValueError(f"Missing calibration: {local}")
    else:
        candidates = sorted(data_root.glob(f"*/calibration/{name}"))
        central = data_root / "calibration" / name
        if central.is_file():
            candidates.append(central)
    if not candidates:
        raise ValueError(f"No measured {name}; supply --calibration-dir")
    signatures = {}
    for path in candidates:
        d = json.loads(path.read_text())
        k, distortion = np.asarray(d["K"], float), np.asarray(d["D"], float)
        if (k.shape != (3, 3) or distortion.size != 4 or not np.isfinite(k).all()
                or not np.isfinite(distortion).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or d.get("model") != "fisheye"
                or d.get("image_width", 0) <= 0 or d.get("image_height", 0) <= 0):
            raise ValueError(f"Invalid measured fisheye calibration: {path}")
        signature = json.dumps([k.tolist(), distortion.ravel().tolist(),
                                d["image_width"], d["image_height"]])
        signatures[signature] = (path, d)
    if len(signatures) != 1:
        raise ValueError(f"Conflicting {side} calibrations; select --calibration-dir")
    path, d = next(iter(signatures.values()))
    try:
        source = str(path.relative_to(data_root))
    except ValueError:
        source = str(path)
    return {"source": source, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **d}


def cartesian_actions(pose, width):
    """[world dp(m), world rotvec(rad), next absolute width(m)] for each arm."""
    arms = []
    for i in range(2):
        rotations = Rotation.from_quat(pose[:, i, 3:])
        delta_rotation = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
        arms.append(np.c_[np.diff(pose[:, i, :3], axis=0), delta_rotation, width[1:, i]])
    return np.concatenate(arms, axis=1).astype(np.float32)


def timestamp_grid(start, stop, fps):
    if fps <= 0:
        raise ValueError("fps must be positive")
    if stop < start:
        return np.empty(0, dtype=np.int64)
    # Round relative offsets, preserving nanosecond precision at Unix epoch timestamps.
    count = (int(stop) - int(start)) * fps // 1_000_000_000 + 2
    ticks = np.arange(count, dtype=np.int64)
    offsets = (ticks * 1_000_000_000 + fps // 2) // fps
    grid = np.int64(start) + offsets
    return grid[grid <= stop]


def prepare_episode(data_root, name, args):
    episode = data_root / name
    instruction = json.loads((episode / "instructions.json").read_text())["instructions"]
    if not instruction or any(not isinstance(x, str) or not x.strip() or x == "null"
                              for x in instruction):
        raise ValueError(f"Missing task instruction: {episode}")
    streams = {}
    for side, label in SIDES.items():
        streams[f"pose_{side}"] = read_stream(
            episode / "localization/pose" / label, {".json"},
            ["x", "y", "z", "roll", "pitch", "yaw"])
        streams[f"gripper_{side}"] = read_stream(
            episode / "gripper/encoder" / label, {".json"}, ["angle", "distance"])
    wrist_camera = recorded_wrist_camera(episode)
    wrist_keys = tuple(f"{side}_{wrist_camera}_rgb" for side in SIDES)
    image_streams = {key: IMAGE_STREAMS[key] for key in wrist_keys} if args.vista_only else dict(IMAGE_STREAMS)
    if not getattr(args, "exclude_head", False):
        image_streams.update(head_streams(episode))
    for key, relative in image_streams.items():
        streams[key] = read_stream(episode / relative, {".jpg", ".png"})

    start = max(int(s.times[0]) for s in streams.values())
    stop = min(int(s.times[-1]) for s in streams.values())
    grid = timestamp_grid(start, stop, args.fps)
    if len(grid) < 2:
        raise ValueError(f"No overlapping sensor interval: {episode}")
    valid = np.ones(len(grid), dtype=bool)
    poses, widths, angles = [], [], []
    numeric_provenance, image_provenance, report = {}, {}, {}
    for key, stream in streams.items():
        gaps = np.diff(stream.times) / 1e6
        report[key] = {"raw_samples": len(stream.times), "median_rate_hz": float(1000 / np.median(gaps)),
                       "max_raw_gap_ms": float(gaps.max())}
    for side in SIDES:
        for kind in ["pose", "gripper"]:
            key = f"{kind}_{side}"
            s = streams[key]
            left, right, alpha, ok = brackets(s.times, grid, int(args.max_sensor_gap_ms * 1e6))
            values = s.values[left] * (1 - alpha[:, None]) + s.values[right] * alpha[:, None]
            if kind == "pose":
                # Relative seconds keep precision at Unix timestamps around 1e18 ns.
                rotations = Rotation.from_euler("xyz", s.values[:, 3:])
                quaternions = Slerp((s.times - s.times[0]) / 1e9, rotations)(
                    (grid - s.times[0]) / 1e9).as_quat()
                # Give the stored quaternion sequence a consistent sign.
                for i in range(1, len(quaternions)):
                    if quaternions[i] @ quaternions[i - 1] < 0:
                        quaternions[i] *= -1
                poses.append(np.c_[values[:, :3], quaternions])
            else:
                angles.append(values[:, 0])
                widths.append(values[:, 1])
                ok &= (values[:, 1] >= 0) & (values[:, 1] <= args.max_pika_width_m)
            numeric_provenance[key] = {"left_timestamp_ns": s.times[left],
                                       "right_timestamp_ns": s.times[right], "alpha": alpha}
            report[key]["invalid_grid_rows"] = int((~ok).sum())
            valid &= ok
    motion_valid = valid.copy()
    max_age_ns = int(args.max_image_offset_ms * 1e6)
    wrist_samples = {}
    if args.vista_only:
        left_wrist, right_wrist = (streams[key].times for key in wrist_keys)
        wrist_samples = dict(zip(wrist_keys, sample_wrist_pair(
            left_wrist, right_wrist, grid, args.image_policy, max_age_ns)))
    for key in image_streams:
        s = streams[key]
        if key in wrist_samples:
            index, offset, ok = wrist_samples[key]
        elif key.startswith("head_") and wrist_samples:
            wrist_times = np.column_stack([streams[k].times[wrist_samples[k][0]] for k in wrist_keys])
            index, offset, ok = sample_head(s.times, grid, wrist_times, max_age_ns)
        else:
            index, offset, ok = sample_images(s.times, grid, args.image_policy, max_age_ns)
        image_provenance[key] = {"index": index, "timestamp_ns": s.times[index], "offset_ns": offset}
        report[key].update(invalid_grid_rows=int((~ok).sum()),
                          max_offset_ms=float(np.abs(offset).max() / 1e6),
                          reused_frames=int(np.sum(np.diff(index) == 0)))
        valid &= ok
    image_times = np.stack([x["timestamp_ns"] for x in image_provenance.values()], axis=1)
    skew_ms = np.ptp(image_times, axis=1) / 1e6
    # Independent head exposures have a separate, explicit 40 ms skew budget.
    wrist_times = np.column_stack([image_provenance[k]["timestamp_ns"] for k in wrist_keys])
    if "head_rgb" in image_provenance:
        nonhead_times = np.column_stack([v["timestamp_ns"] for k, v in image_provenance.items()
                                         if not k.startswith("head_")])
        valid &= np.ptp(nonhead_times, axis=1) / 1e6 <= args.max_image_skew_ms
        for key in image_provenance:
            if key.startswith("head_"):
                all_times = np.column_stack([wrist_times, image_provenance[key]["timestamp_ns"]])
                valid &= np.ptp(all_times, axis=1) <= 40_000_000
    else:
        valid &= skew_ms <= args.max_image_skew_ms
    runs = valid_runs(valid)
    if not runs:
        raise ValueError(f"No valid transitions: {episode}")
    intrinsics = {side: resolve_intrinsic(data_root, episode, side, args.calibration_dir)
                  for side in SIDES if f"{side}_fisheye_rgb" in image_streams}
    # Check actual image dimensions and type, rather than zero fisheye CameraInfo.
    camera_configs = {}
    for key, relative in image_streams.items():
        first = cv2.imread(str(streams[key].files[0]), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise ValueError(f"Cannot decode {streams[key].files[0]}")
        if key.endswith("depth"):
            if first.dtype != np.uint16 or first.ndim != 2:
                raise ValueError(f"Expected raw uint16 depth: {key}")
        else:
            if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
                raise ValueError(f"Expected uint8 three-channel RGB source: {key}")
        if "fisheye" in key:
            c = intrinsics[key.split("_")[0]]
            if (first.shape[1], first.shape[0]) != (c["image_width"], c["image_height"]):
                raise ValueError(f"Fisheye calibration resolution mismatch: {key}")
        else:
            camera_configs[key] = json.loads((episode / relative / "config.json").read_text())
            if key.startswith("head_") or (wrist_camera == "d405" and key in wrist_keys):
                c = camera_configs[key]
                k = np.asarray(c["K"], dtype=float).reshape(3, 3)
                if ((c["width"], c["height"]) != (first.shape[1], first.shape[0])
                        or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0):
                    raise ValueError(f"Invalid CameraInfo or resolution mismatch: {key}")
        report[key]["shape"] = list(first.shape)
    return dict(name=name, episode=episode, grid=grid, valid=valid, motion_valid=motion_valid, runs=runs,
                pose=np.stack(poses, axis=1), width=np.stack(widths, axis=1),
                angle=np.stack(angles, axis=1), streams=streams,
                numeric_provenance=numeric_provenance, image_provenance=image_provenance,
                instruction=instruction, intrinsics=intrinsics, camera_configs=camera_configs,
                wrist_camera=wrist_camera,
                report={"streams": report, "grid_rows": len(grid), "valid_rows": int(valid.sum()),
                        "rejected_rows": int((~valid).sum()), "segments": len(runs),
                        "isolated_valid_rows": int(valid.sum())-sum(b-a for a,b in runs),
                        "transitions": sum(b-a-1 for a,b in runs),
                        "image_pairwise_skew_ms_p95": float(np.percentile(skew_ms, 95)),
                        "image_pairwise_skew_ms_max": float(skew_ms.max())})


def _letterbox_rgb(image, target):
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = int(target[0]), int(target[1])
    if (src_w, src_h) == (dst_w, dst_h):
        return image, (1.0, 1.0, 0, 0)
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    if pad_x == 0 and pad_y == 0 and (new_w, new_h) == (dst_w, dst_h):
        return resized, (new_w / float(src_w), new_h / float(src_h), 0, 0)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=image.dtype)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, (new_w / float(src_w), new_h / float(src_h), pad_x, pad_y)


def _letterbox_intrinsics(k_mat, src_w, src_h, target_w, target_h):
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    transform = np.array(
        [[new_w / float(src_w), 0.0, float(pad_x)],
         [0.0, new_h / float(src_h), float(pad_y)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return transform @ np.asarray(k_mat, dtype=np.float64).reshape(3, 3)


def write_images(group, paths, depth, image_size):
    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"Corrupt image: {paths[0]}")
    target = (first.shape[1], first.shape[0]) if depth or image_size is None else tuple(image_size)
    shape = (len(paths), target[1], target[0]) + (() if depth else (3,))
    ds = group.create_dataset("frames", shape=shape, dtype="uint16" if depth else "uint8",
                              chunks=(1,) + shape[1:], compression="gzip", compression_opts=1)
    pad_xy = (0, 0)
    for i, path in enumerate(paths):
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None or a.shape != first.shape or a.dtype != first.dtype:
            raise ValueError(f"Corrupt or inconsistent image: {path}")
        if not depth:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            if (a.shape[1], a.shape[0]) != target:
                a, (_, _, pad_x, pad_y) = _letterbox_rgb(a, target)
                pad_xy = (pad_x, pad_y)
        ds[i] = a
    ds.attrs["color_order"] = "raw_depth" if depth else "RGB"
    ds.attrs["rectified"] = False
    ds.attrs["aspect_preserved"] = True
    ds.attrs["letterbox_pad_xy"] = list(pad_xy)
    return target


def export_episode(root, output, args):
    import h5py

    string = h5py.string_dtype("utf-8")
    name = output["name"]
    g = root.require_group("source_episodes").create_group(name)
    g.attrs["source_episode"] = name  # relative to --data-root; movable with the dataset
    g.attrs["wrist_camera"] = output["wrist_camera"]
    g.attrs["instruction"] = json.dumps(output["instruction"], ensure_ascii=False)
    g.create_dataset("timestamp_ns", data=output["grid"])
    g.create_dataset("valid", data=output["valid"])
    g.create_dataset("motion_valid", data=output["motion_valid"])
    for key in ["pose", "width", "angle"]:
        g.create_dataset(key, data=output[key])
    for key, fields in output["numeric_provenance"].items():
        pg = g.require_group("sync").create_group(key)
        for field, values in fields.items():
            pg.create_dataset(field, data=values)
    calibration = {"fisheye": output["intrinsics"], "camera_info": output["camera_configs"],
                   "handeye": None, "tracker_to_tcp": None, "depth_scale_m_per_unit": None}
    metadata = output["episode"] / "capture_metadata.json"
    if metadata.is_file():
        calibration["capture_metadata"] = json.loads(metadata.read_text())
    g.attrs["head_max_skew_ms"] = 40.0
    for key, fields in output["image_provenance"].items():
        stream = output["streams"][key]
        paths = [stream.files[int(i)] for i in fields["index"]]
        ig = g.require_group("images").create_group(key)
        ig.create_dataset("source_files", data=np.asarray(
            [str(p.relative_to(output["episode"])) for p in paths], dtype=string))
        for field in ["timestamp_ns", "offset_ns"]:
            ig.create_dataset(field, data=fields[field])
        target = write_images(ig, paths, key.endswith("depth"), args.image_size)
        if key == "head_depth":
            ig.attrs["aligned_to"] = "head_rgb"
            ig.attrs["depth_scale_m_per_unit"] = 0.001  # ROS realsense 16UC1 is millimetres.
        if "fisheye" in key:
            c = output["intrinsics"][key.split("_")[0]]
            ig.attrs["K"] = _letterbox_intrinsics(c["K"], c["image_width"], c["image_height"], target[0], target[1])
            ig.attrs["D"] = np.asarray(c["D"])
            ig.attrs["camera_model"] = "fisheye"
        else:
            c = output["camera_configs"][key]
            ig.attrs["K"] = _letterbox_intrinsics(c["K"], c["width"], c["height"], target[0], target[1])
            ig.attrs["D"] = np.asarray(c["D"])
            ig.attrs["camera_model"] = c["distortion_model"]
    g.create_dataset("calibration_json", data=json.dumps(calibration), dtype=string)
    demos = root["data"]
    for a, b in output["runs"]:
        count = b-a-1
        demo = demos.create_group(f"demo_{len(demos)}")
        demo.attrs.update(num_samples=count, source_episode=name,
                          source_embodiment="pika", source_tcp_calibrated=False,
                          robot_replay_validated=False, success_known=False,
                          task_instruction=output["instruction"][0])
        demo.create_dataset("timestamp_ns", data=output["grid"][a:b-1])
        demo.create_dataset("actions", data=cartesian_actions(output["pose"][a:b], output["width"][a:b]))
        demo.create_dataset("rewards", data=np.zeros(count, dtype=np.float32))
        demo.create_dataset("dones", data=np.r_[np.zeros(count-1, dtype=np.uint8), np.uint8(1)])
        # These source transitions have no measured robot joint positions.
        for group_name, lo, hi in [("obs", a, b-1), ("next_obs", a+1, b)]:
            obs = demo.create_group(group_name)
            obs.create_dataset("eef_pose", data=output["pose"][lo:hi].astype(np.float32))
            obs.create_dataset("gripper_width", data=output["width"][lo:hi].astype(np.float32))
            state = np.concatenate([np.c_[output["pose"][lo:hi,i], output["width"][lo:hi,i]]
                                    for i in range(2)], axis=1)
            obs.create_dataset("state", data=state.astype(np.float32))
            for key in output["image_provenance"]:
                source = g[f"images/{key}/frames"]
                # A same-file VDS avoids duplicating every image for obs/next_obs.
                layout = h5py.VirtualLayout(shape=(count,)+source.shape[1:], dtype=source.dtype)
                layout[:] = h5py.VirtualSource(".", source.name, source.shape)[lo:hi]
                obs.create_virtual_dataset(key, layout)
    return output["report"]


def main(argv=None):
    import h5py
    from pika_options import source_filename

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path)
    add_fps_option(parser, description="Dataset rate (5–90 Hz). Defaults to the recorded export rate or 20 Hz.")
    parser.add_argument("--image-policy", choices=["nearest", "causal"], default="causal")
    parser.add_argument("--max-image-offset-ms", type=float, default=40)
    parser.add_argument("--max-image-skew-ms", type=float, help="Default: 10 ms for fisheye RGB, 40 ms for D405 or all streams")
    parser.add_argument("--max-sensor-gap-ms", type=float, default=30)
    parser.add_argument("--max-pika-width-m", type=float, default=.11)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--vista-only", action="store_true", help="Keep only the two wrist RGB image streams")
    parser.add_argument("--exclude-head", action="store_true", help="Exclude head from synchronization and output without altering raw data")
    args = parser.parse_args(argv)
    if args.image_size and any(x <= 0 for x in args.image_size):
        parser.error("Image dimensions must be positive")
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("Duplicate episodes")
    if any(Path(x).name != x for x in args.episodes):
        parser.error("Episodes must be directory names within data-root")
    data_root, destination = args.data_root.resolve(), args.output_dir.resolve()
    modes = {recorded_wrist_camera(data_root / name) for name in args.episodes}
    if len(modes) != 1:
        parser.error("All episodes in a source export must use the same wrist camera")
    if args.fps is None:
        recorded_rates = {recorded_export_fps(data_root / name) for name in args.episodes}
        if len(recorded_rates) != 1:
            parser.error("Episodes have different export FPS settings. Pass --fps to use one rate.")
        args.fps = recorded_rates.pop()
    if args.max_image_skew_ms is None:
        args.max_image_skew_ms = wrist_skew_default(next(iter(modes)), args.vista_only)
    if any(x <= 0 for x in [args.max_image_offset_ms, args.max_image_skew_ms,
                           args.max_sensor_gap_ms, args.max_pika_width_m]):
        parser.error("All physical/time limits must be positive")
    if destination.exists():
        parser.error(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pika_export_", dir=destination.parent))
    try:
        report = {"fps": args.fps, "image_policy": args.image_policy,
                  "max_image_offset_ms": args.max_image_offset_ms,
                  "max_image_skew_ms": args.max_image_skew_ms,
                  "max_sensor_gap_ms": args.max_sensor_gap_ms,
                  "source_embodiment": "pika", "robot_replay_validated": False,
                  "training_ready": False,
                  "quality_checks": {"timestamp_alignment_checked": True,
                                     "tracking_status_checked": False,
                                     "motion_jump_checked": False,
                                     "tcp_transform_applied": False},
                  "episodes": {}}
        with h5py.File(staging / source_filename(args.fps), "w", libver="latest") as f:
            f.attrs["format"] = "pika-robomimic-source-v1"
            f.attrs["fps"] = args.fps
            f.attrs["max_image_offset_ms"] = args.max_image_offset_ms
            f.attrs["max_image_skew_ms"] = args.max_image_skew_ms
            f.attrs["wrist_pairing"] = (
                "causal_pair" if args.vista_only and args.image_policy == "causal"
                else f"independent_{args.image_policy}"
            )
            f.attrs["training_ready"] = False
            f.attrs["quality_checks"] = json.dumps(report["quality_checks"])
            f.attrs["action_spec"] = json.dumps({
                "per_arm": ["world_dx_m","world_dy_m","world_dz_m",
                            "world_rotvec_x_rad","world_rotvec_y_rad","world_rotvec_z_rad",
                            "next_absolute_gripper_width_m"], "order": ["left", "right"],
                "semantics": "inferred source motion from obs[k] to next_obs[k]; not robot commands",
                "source_pose_frame": "reported Pika pose; tracker/TCP identity not yet verified"})
            d = f.create_group("data")
            d.attrs["env_args"] = json.dumps({"env_name":"DualPikaSource", "env_type":2,
                "env_kwargs":{"fps":args.fps,"offline_only":True}})
            for name in args.episodes:
                result = prepare_episode(data_root, name, args)
                report["episodes"][name] = export_episode(f, result, args)
                timeline_dir = staging / "timelines"
                timeline_dir.mkdir(exist_ok=True)
                np.savez_compressed(timeline_dir / f"{name}.npz", timestamp_ns=result["grid"],
                                    pose=result["pose"], width=result["width"],
                                    valid=result["valid"], motion_valid=result["motion_valid"], source_episode=name,
                                    pose_frame="reported_pika_pose_unverified")
                print(f"{name}: {result['report']['valid_rows']}/{result['report']['grid_rows']} valid rows; "
                      f"{result['report']['transitions']} transitions", flush=True)
            d.attrs["total"] = sum(int(x.attrs["num_samples"]) for x in d.values())
            report["total_transitions"] = int(d.attrs["total"])
        (staging / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
