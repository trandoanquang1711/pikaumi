"""`umi-delta` LeRobot v3.0 VLA exporter (`GEMINI.md` 11D relative TCP state + 14D local delta action)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import (
    HAS_PYARROW,
    contiguous_valid_runs,
    extract_episode_tcp_targets,
    load_dataset_episodes,
    load_dataset_info,
    load_episode_raw,
    natural_sort_key,
)
from pikaumi.robots.utils import pose7_to_mat

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq


def _compute_quantile_stats(arr: np.ndarray) -> dict[str, Any]:
    x = np.asarray(arr, dtype=np.float64)
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
        "count": [int(x.shape[0])],
    }


def _slice_video_segment(
    src_mp4: Path,
    dst_mp4: Path,
    start_idx: int,
    end_idx_exclusive: int,
    fps: float,
) -> list[int] | None:
    """Copy exact frame range `[start_idx, end_idx_exclusive)` from `src_mp4` to `dst_mp4`."""
    import cv2

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {src_mp4}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(dst_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {dst_mp4}")
        written = 0
        idx = 0
        try:
            while idx < end_idx_exclusive:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if idx >= start_idx:
                    writer.write(frame)
                    written += 1
                idx += 1
        finally:
            writer.release()
        expected = end_idx_exclusive - start_idx
        if written != expected:
            raise ValueError(f"Video {src_mp4} yielded {written} frames for [{start_idx}, {end_idx_exclusive})")
        return [3, h, w]
    finally:
        cap.release()


def compute_umi_delta_tensors(
    left_tcp_poses7: np.ndarray,
    right_tcp_poses7: np.ndarray,
    left_widths_m: np.ndarray,
    right_widths_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `GEMINI.md` bimanual VLA tensors over `T - 1` real transitions (Left-before-Right):

    - `observation.state` `(T - 1, 11)`: Right TCP in Left TCP frame
      `[x, y, z (m), rot_col0 (3), rot_col1 (3), left_width_m, right_width_m]`.
    - `action` `(T - 1, 14)`: Left `[0:7]` then Right `[7:14]`, each
      `[dx_m, dy_m, dz_m, rx_rad, ry_rad, rz_rad, next_gripper_width_m]`
      where `dx..rz` are **TCP-local deltas in the frame at step `t`** (`T_t^{-1} T_{t+1}`)
      and `next_gripper_width_m` is target absolute width at `t+1` in meters.
    """
    T = len(left_tcp_poses7)
    if T < 2:
        raise ValueError(f"Need at least 2 frames to compute real transitions, got {T}")
    N = T - 1
    T_l = [pose7_to_mat(p) for p in left_tcp_poses7]
    T_r = [pose7_to_mat(p) for p in right_tcp_poses7]
    wl = np.clip(np.asarray(left_widths_m, dtype=np.float64), 0.0, 0.10)
    wr = np.clip(np.asarray(right_widths_m, dtype=np.float64), 0.0, 0.10)

    state_11d = np.zeros((N, 11), dtype=np.float32)
    action_14d = np.zeros((N, 14), dtype=np.float32)

    for t in range(N):
        T_rel = np.linalg.inv(T_l[t]) @ T_r[t]
        state_11d[t, 0:3] = T_rel[:3, 3]
        state_11d[t, 3:6] = T_rel[:3, 0]
        state_11d[t, 6:9] = T_rel[:3, 1]
        state_11d[t, 9] = wl[t]
        state_11d[t, 10] = wr[t]

        t_next = t + 1
        for offset, mats, widths in ((0, T_l, wl), (7, T_r, wr)):
            T_delta = np.linalg.inv(mats[t]) @ mats[t_next]
            dx_local = T_delta[:3, 3]
            rvec_local = Rotation.from_matrix(T_delta[:3, :3]).as_rotvec()
            action_14d[t, offset : offset + 3] = dx_local
            action_14d[t, offset + 3 : offset + 6] = rvec_local
            action_14d[t, offset + 6] = widths[t_next]

    return state_11d, action_14d


def export_umi_delta(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    chunk_size: int = 50,
    allow_flagged_episodes: bool = False,
    min_segment_seconds: float = 0.0,
    max_sync_skew_s: float = 0.05,
) -> dict[str, Any]:
    """Export `pika_raw` dataset into LeRobot v3.0 `umi-delta` format with dropout splitting & strict raw-episode splits."""
    from pikaumi.dataset.curation import _link_or_copy

    src = Path(dataset_dir)
    dst = Path(output_dir)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(src)
    episodes = load_dataset_episodes(src)
    fps = int(info.get("fps", 30))
    default_task = str(info.get("task", "bimanual manipulation"))

    # Filter episodes flagged by QA / direction analysis unless allow_flagged_episodes=True
    rejected_ids: set[int] = set()
    if not allow_flagged_episodes:
        q_path = src / "meta" / "quality_report.json"
        if q_path.is_file():
            q_rep = json.loads(q_path.read_text(encoding="utf-8"))
            for item in q_rep.get("episodes", []):
                if not item.get("passed", True):
                    rejected_ids.add(int(item["episode_index"]))
        d_path = src / "meta" / "direction_report.json"
        if d_path.is_file():
            d_rep = json.loads(d_path.read_text(encoding="utf-8"))
            for idx in d_rep.get("flagged_reversed_episodes", []):
                rejected_ids.add(int(idx))

    min_seg_frames = max(2, int(round(float(min_segment_seconds) * fps)) + 1) if min_segment_seconds > 0.0 else 2

    raw_episode_groups: dict[str, list[int]] = {}
    tasks_map: dict[str, int] = {}
    out_episodes: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    video_shapes: dict[str, list[int]] = {}
    out_idx = 0
    global_frame_idx = 0

    for ep_meta in episodes:
        old_idx = int(ep_meta["episode_index"])
        if old_idx in rejected_ids:
            continue
        src_ep_name = str(ep_meta.get("source_episode", f"episode{old_idx}"))
        ep_task = str(ep_meta.get("task", default_task))
        if ep_task not in tasks_map:
            tasks_map[ep_task] = len(tasks_map)
        task_idx = tasks_map[ep_task]

        raw_ep = load_episode_raw(src, old_idx)
        valid_mask = raw_ep["left_accurate"] & raw_ep["right_accurate"]
        if max_sync_skew_s is not None and "sync_skew_s" in raw_ep:
            skew = np.asarray(raw_ep["sync_skew_s"], dtype=np.float64)
            valid_mask = valid_mask & np.isfinite(skew) & (skew <= float(max_sync_skew_s))
        runs = contiguous_valid_runs(valid_mask, min_len=min_seg_frames)
        if not runs:
            continue

        tcp_by_side, widths_by_side, ts_full = extract_episode_tcp_targets(src, old_idx)

        for seg_idx, (a, b) in enumerate(runs):
            state_11d, action_14d = compute_umi_delta_tensors(
                tcp_by_side["left"][a:b],
                tcp_by_side["right"][a:b],
                widths_by_side["left"][a:b],
                widths_by_side["right"][a:b],
            )
            ts = ts_full[a : b - 1] - ts_full[a]
            N = len(ts)
            all_states.append(state_11d)
            all_actions.append(action_14d)
            raw_episode_groups.setdefault(src_ep_name, []).append(out_idx)

            stem = f"episode_{out_idx:06d}"
            np.savez_compressed(
                dst / "data" / f"{stem}.npz",
                timestamps_s=ts,
                observation_state=state_11d,
                action=action_14d,
            )
            if HAS_PYARROW:
                done_flags = np.zeros(N, dtype=bool)
                if N > 0:
                    done_flags[-1] = True
                tbl = pa.table(
                    {
                        "index": np.arange(global_frame_idx, global_frame_idx + N, dtype=np.int64),
                        "frame_index": np.arange(N, dtype=np.int64),
                        "episode_index": np.full(N, out_idx, dtype=np.int64),
                        "task_index": np.full(N, task_idx, dtype=np.int64),
                        "timestamp": ts,
                        "next.done": done_flags,
                        "observation.state": [r.tolist() for r in state_11d],
                        "action": [r.tolist() for r in action_14d],
                    }
                )
                chunk_dir = dst / "data" / f"chunk-{out_idx // 1000:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                chunk_pq = chunk_dir / f"{stem}.parquet"
                pq.write_table(tbl, chunk_pq)
                _link_or_copy(chunk_pq, dst / "data" / f"{stem}.parquet")
            global_frame_idx += N

            for cam in ("left_wrist", "right_wrist"):
                vid_src = src / "videos" / cam / f"episode_{old_idx:06d}.mp4"
                if vid_src.is_file():
                    dst_cam = dst / "videos" / f"observation.images.{cam}" / f"{stem}.mp4"
                    shape_chw = _slice_video_segment(vid_src, dst_cam, a, b - 1, float(fps))
                    if shape_chw is not None:
                        video_shapes[f"observation.images.{cam}"] = shape_chw
                    _link_or_copy(dst_cam, dst / "videos" / cam / f"{stem}.mp4")

            out_episodes.append(
                {
                    "episode_index": out_idx,
                    "source_episode_index": old_idx,
                    "source_episode": src_ep_name,
                    "segment_index": seg_idx,
                    "source_frame_range": [int(a), int(b - 1)],
                    "task": ep_task,
                    "task_index": task_idx,
                    "num_frames": int(N),
                    "fps": fps,
                }
            )
            out_idx += 1

    # Write LeRobot v3.0 tasks.jsonl and q01/q99 normalization stats
    tasks_lines = [
        json.dumps({"task_index": idx, "task": t_str}, ensure_ascii=False)
        for t_str, idx in sorted(tasks_map.items(), key=lambda kv: kv[1])
    ]
    (dst / "meta" / "tasks.jsonl").write_text("\n".join(tasks_lines) + ("\n" if tasks_lines else ""), encoding="utf-8")

    if all_states:
        stats = {
            "observation.state": _compute_quantile_stats(np.concatenate(all_states, axis=0)),
            "action": _compute_quantile_stats(np.concatenate(all_actions, axis=0)),
        }
        save_json_atomic(dst / "meta" / "stats.json", stats)
        save_json_atomic(dst / "meta" / "abs_action_norm_stats.json", stats)

    # Group strictly by raw episodeN (naturally sorted) so split segments never leak across train/val (`GEMINI.md`)
    raw_names = sorted(raw_episode_groups.keys(), key=natural_sort_key)
    n_val = max(1, int(round(len(raw_names) * val_fraction))) if len(raw_names) >= 2 else 0
    val_raw = set(raw_names[-n_val:]) if n_val > 0 else set()
    train_ep_ids = [idx for rn in raw_names if rn not in val_raw for idx in raw_episode_groups[rn]]
    val_ep_ids = [idx for rn in raw_names if rn in val_raw for idx in raw_episode_groups[rn]]

    source_reports = {
        "schema": "pika_source_reports_v1",
        "split_unit": "raw_episode",
        "train_raw_episodes": [rn for rn in raw_names if rn not in val_raw],
        "val_raw_episodes": [rn for rn in raw_names if rn in val_raw],
        "train_episode_indices": train_ep_ids,
        "val_episode_indices": val_ep_ids,
        "mandatory_trainer_flags": {
            "policy.use_delta_action": False,
            "use_corr_state": False,
            "drop_n_last_frames": max(chunk_size - 1, 0),
            "enable_fisheye_warp": False,
        },
    }
    save_json_atomic(dst / "meta" / "pika_source_reports.json", source_reports)

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [11],
            "names": [
                "right_in_left_x_m",
                "right_in_left_y_m",
                "right_in_left_z_m",
                "rot6d_c0x",
                "rot6d_c0y",
                "rot6d_c0z",
                "rot6d_c1x",
                "rot6d_c1y",
                "rot6d_c1z",
                "left_width_m",
                "right_width_m",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": [
                "left_dx_m", "left_dy_m", "left_dz_m", "left_rx_rad", "left_ry_rad", "left_rz_rad", "left_next_width_m",
                "right_dx_m", "right_dy_m", "right_dz_m", "right_rx_rad", "right_ry_rad", "right_rz_rad", "right_next_width_m",
            ],
        },
    }
    for vkey, vshape in video_shapes.items():
        features[vkey] = {
            "dtype": "video",
            "shape": [int(vshape[0]), int(vshape[1]), int(vshape[2])],
            "names": ["channels", "height", "width"],
        }

    out_info = {
        "codebase_version": "v3.0",
        "format": "umi-delta",
        "fps": fps,
        "num_episodes": len(out_episodes),
        "total_frames": sum(e["num_frames"] for e in out_episodes),
        "total_tasks": len(tasks_map),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "mandatory_trainer_flags": source_reports["mandatory_trainer_flags"],
    }
    save_json_atomic(dst / "meta" / "info.json", out_info)
    save_json_atomic(dst / "meta" / "episodes.json", out_episodes)
    return out_info
