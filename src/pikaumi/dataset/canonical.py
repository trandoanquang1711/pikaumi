"""Canonical joint-space LeRobot v3.0 dataset converter (`pika convert --format robot-joints`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import (
    HAS_PYARROW,
    contiguous_valid_runs,
    load_dataset_episodes,
    load_dataset_info,
    load_episode_raw,
)
from pikaumi.dataset.screening import (
    compute_solve_cache_signature,
    solve_episode_with_method,
)
from pikaumi.dataset.umi_delta import _compute_quantile_stats, _slice_video_segment
from pikaumi.retargeting.pika_to_robot import prepare_episode_robot_targets
from pikaumi.robots.registry import RobotProfile, load_robot_profile

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq


def canonical_joint_feature_names(profile: RobotProfile) -> list[str]:
    """Ordered canonical joint vector: per active side `[<arm_joint>.pos ..., <side>_gripper.width_m]`."""
    names: list[str] = []
    for side in profile.active_sides:
        for jn in profile.arms[side].arm_joint_names:
            names.append(f"{jn}.pos")
        names.append(f"{side}_gripper.width_m")
    return names


def pack_canonical_joint_vector(
    profile: RobotProfile,
    q_traj: np.ndarray,
    robot_widths_by_side: dict[str, np.ndarray],
) -> np.ndarray:
    cols: list[np.ndarray] = []
    for side in profile.active_sides:
        for act_idx in profile.arms[side].arm_actuated_indices:
            cols.append(q_traj[:, act_idx])
        cols.append(robot_widths_by_side[side])
    return np.column_stack(cols).astype(np.float32)


def convert_dataset_to_robot_joints(
    dataset_dir: Path,
    output_dir: Path,
    *,
    robot: str = "piper_pika",
    solver: str = "cstro",
    retarget_mode: str = "absolute-table",
    allow_flagged_episodes: bool = False,
) -> dict[str, Any]:
    """Convert `pika_raw` dataset into `<robot>-joints` LeRobot v3.0 dataset (`action[t] = state[t+1]`)."""
    from pikaumi.dataset.curation import _link_or_copy

    src = Path(dataset_dir)
    dst = Path(output_dir)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(src)
    episodes = load_dataset_episodes(src)
    fps = int(info.get("fps", 30))
    default_task = str(info.get("task", "bimanual manipulation"))

    prof = load_robot_profile(robot)
    tcal = load_table_calibration(robot, require_physical=False)
    mode = "relative-anchor" if len(prof.active_sides) == 1 and retarget_mode == "absolute-table" else retarget_mode
    expected_sig = compute_solve_cache_signature(info, prof, tcal.sha256, solver, mode, episodes=episodes)

    # Check screening verdicts unless allow_flagged_episodes=True
    rejected_ids: set[int] = set()
    screen_file = src / "meta" / f"screen_{robot}.json"
    if screen_file.is_file() and not allow_flagged_episodes:
        s_rep = json.loads(screen_file.read_text(encoding="utf-8"))
        for item in s_rep.get("episodes", []):
            if item.get("verdict") == "REJECT":
                rejected_ids.add(int(item["episode_index"]))

    cache_path = src / "meta" / f"_solves_{robot}_{solver}.npz"
    cache_z = np.load(cache_path) if cache_path.is_file() else None
    cache_valid = (
        cache_z is not None
        and "signature" in cache_z
        and str(cache_z["signature"].item()) == expected_sig
    )

    feat_names = canonical_joint_feature_names(prof)
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

        raw_ep = load_episode_raw(src, old_idx)
        ts_full = raw_ep["timestamps_s"]
        valid_mask = raw_ep["left_accurate"] & raw_ep["right_accurate"]
        runs = contiguous_valid_runs(valid_mask, min_len=2)
        if not runs:
            continue

        ep_task = str(ep_meta.get("task", default_task))
        if ep_task not in tasks_map:
            tasks_map[ep_task] = len(tasks_map)
        task_idx = tasks_map[ep_task]

        is_full_clean = len(runs) == 1 and runs[0] == (0, len(ts_full))
        if is_full_clean and cache_valid and f"ep_{old_idx:06d}_q" in cache_z:
            seg_solutions = [
                (
                    0,
                    len(ts_full),
                    cache_z[f"ep_{old_idx:06d}_q"].astype(np.float64),
                    {
                        s: cache_z[f"ep_{old_idx:06d}_w_{s}"].astype(np.float64)
                        for s in prof.active_sides
                    },
                )
            ]
        else:
            ee_tgts_full, widths_full, _, table_z = prepare_episode_robot_targets(
                str(src), old_idx, prof, table_calib=tcal, retarget_mode=mode
            )
            seg_solutions = []
            for a, b in runs:
                ee_seg = {s: ee_tgts_full[s][a:b] for s in prof.active_sides}
                w_seg = {s: widths_full[s][a:b] for s in prof.active_sides}
                q_seg, w_rob, _ = solve_episode_with_method(
                    prof, ee_seg, w_seg, solver_name=solver, fps=float(fps), table_z_robot=table_z
                )
                seg_solutions.append((a, b, q_seg, w_rob))

        for seg_idx, (a, b, q_traj, w_robot) in enumerate(seg_solutions):
            state_full = pack_canonical_joint_vector(prof, q_traj, w_robot)
            if len(state_full) < 2:
                continue
            state_j = state_full[:-1]
            action_j = state_full[1:]
            ts = ts_full[a : b - 1] - ts_full[a]
            q_out = q_traj[:-1].astype(np.float32)
            N = len(ts)
            all_states.append(state_j)
            all_actions.append(action_j)

            stem = f"episode_{out_idx:06d}"
            np.savez_compressed(
                dst / "data" / f"{stem}.npz",
                timestamps_s=ts,
                observation_state=state_j,
                action=action_j,
                urdf_q_full=q_out,
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
                        "observation.state": [r.tolist() for r in state_j],
                        "action": [r.tolist() for r in action_j],
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
                    "source_episode": ep_meta.get("source_episode", f"episode{old_idx}"),
                    "segment_index": seg_idx,
                    "source_frame_range": [int(a), int(b - 1)],
                    "task": ep_task,
                    "task_index": task_idx,
                    "num_frames": int(N),
                    "fps": fps,
                }
            )
            out_idx += 1

    if cache_z is not None:
        cache_z.close()

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

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [len(feat_names)], "names": feat_names},
        "action": {"dtype": "float32", "shape": [len(feat_names)], "names": feat_names},
    }
    for vkey, vshape in video_shapes.items():
        features[vkey] = {
            "dtype": "video",
            "shape": [int(vshape[0]), int(vshape[1]), int(vshape[2])],
            "names": ["channels", "height", "width"],
        }

    out_info = {
        "codebase_version": "v3.0",
        "format": f"{robot}-joints",
        "robot": robot,
        "gripper_max_width_m": prof.gripper_max_width_m,
        "solver": solver,
        "retarget_mode": mode,
        "table_calib_sha256": tcal.sha256,
        "cache_signature": expected_sig,
        "fps": fps,
        "num_episodes": len(out_episodes),
        "total_frames": sum(e["num_frames"] for e in out_episodes),
        "total_tasks": len(tasks_map),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
    }
    save_json_atomic(dst / "meta" / "info.json", out_info)
    save_json_atomic(dst / "meta" / "episodes.json", out_episodes)
    return out_info
