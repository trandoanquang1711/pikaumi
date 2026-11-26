"""Parallel LeRobot v2.1 joint-action exporter from raw Dual Pika Sense episodes (`pika convert-v21`).

Uses `pikaumi`'s 5-layer calibration (`session.yaml`, `pika_tcp.yaml`, table calibration),
causal L-R wrist camera pair alignment, contiguous valid tracking segmentation, and C-STRO
bimanual IK to produce a strict LeRobot v2.1 dataset:
  - `data/chunk-000/episode_XXXXXX.parquet` (HuggingFace `FixedSizeList<float32>[D]`)
  - `videos/chunk-000/observation.images.{left_wrist,right_wrist}/episode_XXXXXX.mp4` (H.264 `yuv420p`)
  - `meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl,stats.json,abs_action_norm_stats.json}`
  - `pika_source_reports.json`
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.config import save_json_atomic
from pikaumi.dataset.canonical import pack_canonical_joint_vector
from pikaumi.dataset.raw import (
    _align_causal_wrist_pair,
    _infer_distance_to_meters,
    _load_tracking_status_json,
    _numeric_timestamp_files,
    apply_tracking_status_with_padding,
    contiguous_valid_runs,
    default_calibration_metadata,
    natural_sort_key,
)
from pikaumi.dataset.screening import solve_episode_with_method
from pikaumi.retargeting.pika_to_robot import map_table_tcp_to_robot_ee
from pikaumi.robots.registry import RobotProfile, load_robot_profile
from pikaumi.robots.utils import mat_to_pose7, pose7_to_mat


def v21_joint_feature_names(profile: RobotProfile) -> list[str]:
    """Ordered LeRobot v2.1 joint feature names: `<joint>_rad ..., <side>_gripper_width_m`."""
    names: list[str] = []
    for side in profile.active_sides:
        for jn in profile.arms[side].arm_joint_names:
            names.append(f"{jn}_rad")
        names.append(f"{side}_gripper_width_m")
    return names


def _hf_schema_metadata(dim: int) -> dict[bytes, bytes]:
    return {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "observation.state": {
                            "feature": {"dtype": "float32", "_type": "Value"},
                            "length": int(dim),
                            "_type": "List",
                        },
                        "action": {
                            "feature": {"dtype": "float32", "_type": "Value"},
                            "length": int(dim),
                            "_type": "List",
                        },
                        "timestamp": {"dtype": "float32", "_type": "Value"},
                        "frame_index": {"dtype": "int64", "_type": "Value"},
                        "episode_index": {"dtype": "int64", "_type": "Value"},
                        "index": {"dtype": "int64", "_type": "Value"},
                        "task_index": {"dtype": "int64", "_type": "Value"},
                    }
                }
            }
        ).encode("utf-8")
    }


def _vector_stats(arr: np.ndarray) -> dict[str, Any]:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "count": [int(x.shape[0])],
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q10": np.quantile(x, 0.10, axis=0).tolist(),
        "q50": np.quantile(x, 0.50, axis=0).tolist(),
        "q90": np.quantile(x, 0.90, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def _image_stats_from_samples(samples_rgb: np.ndarray, total_frames: int) -> dict[str, Any]:
    flat = samples_rgb.reshape(-1, 3).astype(np.float64)

    def _c311(vec: np.ndarray) -> list[list[list[float]]]:
        return [[[float(vec[c])]] for c in range(3)]

    return {
        "min": _c311(np.min(flat, axis=0)),
        "max": _c311(np.max(flat, axis=0)),
        "mean": _c311(np.mean(flat, axis=0)),
        "std": _c311(np.std(flat, axis=0)),
        "count": [int(total_frames)],
        "q01": _c311(np.quantile(flat, 0.01, axis=0)),
        "q10": _c311(np.quantile(flat, 0.10, axis=0)),
        "q50": _c311(np.quantile(flat, 0.50, axis=0)),
        "q90": _c311(np.quantile(flat, 0.90, axis=0)),
        "q99": _c311(np.quantile(flat, 0.99, axis=0)),
    }


def _encode_h264_and_sample_stats(
    frame_paths: list[Path],
    dst_mp4: Path,
    *,
    fps: int = 30,
    max_stat_frames: int = 100,
    spatial_stride: int = 16,
) -> tuple[dict[str, Any], np.ndarray, tuple[int, int]]:
    """Encode selected raw frames to H.264 `yuv420p` MP4 (exact PTS) and sample RGB stats."""
    import av

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    n = len(frame_paths)
    sample_indices = set(
        np.unique(np.linspace(0, n - 1, min(n, max_stat_frames), dtype=int)).tolist()
    )
    sampled_rgb: list[np.ndarray] = []

    first_bgr = cv2.imread(str(frame_paths[0]))
    if first_bgr is None:
        raise RuntimeError(f"Failed to read image: {frame_paths[0]}")
    h, w = first_bgr.shape[:2]

    container = av.open(str(dst_mp4), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        stream.options = {
            "crf": "18",
            "preset": "veryfast",
            "g": str(fps),
            "threads": "1",
            "movflags": "+faststart",
        }
        for i, fp in enumerate(frame_paths):
            bgr = first_bgr if i == 0 else cv2.imread(str(fp))
            if bgr is None:
                raise RuntimeError(f"Failed to read image: {fp}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if i in sample_indices:
                sampled_rgb.append(rgb[::spatial_stride, ::spatial_stride, :].astype(np.float32) / 255.0)
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            vf.pts = i
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    finally:
        container.close()

    samples_arr = np.stack(sampled_rgb, axis=0)
    return _image_stats_from_samples(samples_arr, n), samples_arr, (h, w)


def _read_raw_episode_resampled(
    ep_dir: Path,
    cal: dict[str, Any],
    fps: int,
    default_task: str = "bimanual manipulation",
) -> dict[str, Any]:
    """Read raw Pika episode with causal L-R wrist alignment & padded tracking validity."""
    pose_l_dir = ep_dir / "localization" / "pose" / "pika_l"
    pose_r_dir = ep_dir / "localization" / "pose" / "pika_r"
    grip_l_dir = ep_dir / "gripper" / "encoder" / "pika_l"
    grip_r_dir = ep_dir / "gripper" / "encoder" / "pika_r"
    cam_l_dir = ep_dir / "camera" / "color" / "pikaFisheyeCamera_l"
    cam_r_dir = ep_dir / "camera" / "color" / "pikaFisheyeCamera_r"

    ep_task = default_task
    inst_file = ep_dir / "instructions.json"
    if inst_file.is_file():
        inst_list = json.loads(inst_file.read_text(encoding="utf-8")).get("instructions") or []
        if inst_list and isinstance(inst_list[0], str) and inst_list[0].strip():
            ep_task = inst_list[0].strip()

    pose_l_items = _numeric_timestamp_files(pose_l_dir, {".json"})
    pose_r_items = _numeric_timestamp_files(pose_r_dir, {".json"})
    grip_l_items = _numeric_timestamp_files(grip_l_dir, {".json"})
    grip_r_items = _numeric_timestamp_files(grip_r_dir, {".json"})
    cam_l_items = _numeric_timestamp_files(cam_l_dir, {".jpg", ".jpeg", ".png"})
    cam_r_items = _numeric_timestamp_files(cam_r_dir, {".jpg", ".jpeg", ".png"})

    t0_dec = pose_l_items[0][0]

    def _load_pose(items: list[tuple[Any, Path]], side: str):
        ts_arr = np.array([float(ts - t0_dec) for ts, _ in items], dtype=np.float64)
        p7_list = []
        acc_list = []
        for _, p in items:
            r = json.loads(p.read_text(encoding="utf-8"))
            xyz = r.get("position") or [r["x"], r["y"], r["z"]]
            if "quaternion" in r:
                q = r["quaternion"]
            elif all(k in r for k in ("qx", "qy", "qz", "qw")):
                q = [r["qx"], r["qy"], r["qz"], r["qw"]]
            else:
                q = Rotation.from_euler("xyz", [r["roll"], r["pitch"], r["yaw"]]).as_quat().tolist()
            p7_list.append([float(v) for v in xyz] + [float(v) for v in q])
            acc_list.append(bool(r.get("accurate", True)))
        acc_arr = np.asarray(acc_list, dtype=bool)
        tr = _load_tracking_status_json(ep_dir, side, t0_dec)
        if tr is not None:
            st_ts, st_acc = tr
            acc_arr = apply_tracking_status_with_padding(
                ts_arr, st_ts, st_acc, acc_arr, pre_pad_s=0.05, post_pad_s=0.25
            )
        return ts_arr, np.asarray(p7_list, dtype=np.float64), acc_arr

    def _load_grip(items: list[tuple[Any, Path]]):
        ts_arr = np.array([float(ts - t0_dec) for ts, _ in items], dtype=np.float64)
        vals = [float(json.loads(p.read_text(encoding="utf-8")).get("distance", 0.0)) for _, p in items]
        return ts_arr, _infer_distance_to_meters(vals)

    ts_pl, p7_l, acc_l = _load_pose(pose_l_items, "left")
    ts_pr, p7_r, acc_r = _load_pose(pose_r_items, "right")
    ts_gl, w_l = _load_grip(grip_l_items)
    ts_gr, w_r = _load_grip(grip_r_items)
    ts_cl = np.array([float(ts - t0_dec) for ts, _ in cam_l_items], dtype=np.float64)
    ts_cr = np.array([float(ts - t0_dec) for ts, _ in cam_r_items], dtype=np.float64)

    t_start = max(ts_pl[0], ts_pr[0], ts_gl[0], ts_gr[0], ts_cl[0], ts_cr[0])
    t_end = min(ts_pl[-1], ts_pr[-1], ts_gl[-1], ts_gr[-1], ts_cl[-1], ts_cr[-1])
    grid = np.arange(t_start, t_end + 1e-9, 1.0 / float(fps), dtype=np.float64)

    def _asof(src_ts: np.ndarray, arr: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(src_ts, grid, side="right") - 1, 0, len(src_ts) - 1)
        return arr[idx]

    idx_cl, idx_cr = _align_causal_wrist_pair(ts_cl, ts_cr, grid, max_age_s=0.040)
    skew_ms = np.abs(ts_cl[idx_cl] - ts_cr[idx_cr]) * 1000.0

    p7_l_g = _asof(ts_pl, p7_l)
    p7_r_g = _asof(ts_pr, p7_r)
    acc_l_g = _asof(ts_pl, acc_l)
    acc_r_g = _asof(ts_pr, acc_r)
    w_l_g = _asof(ts_gl, w_l)
    w_r_g = _asof(ts_gr, w_r)

    T_tv = pose7_to_mat(np.asarray(cal["T_table_vive"], dtype=np.float64))
    T_tr_tcp_l = pose7_to_mat(np.asarray(cal["T_tracker_tcp"]["left"], dtype=np.float64))
    T_tr_tcp_r = pose7_to_mat(np.asarray(cal["T_tracker_tcp"]["right"], dtype=np.float64))

    tcp_l = np.zeros_like(p7_l_g)
    tcp_r = np.zeros_like(p7_r_g)
    for i in range(len(grid)):
        tcp_l[i] = mat_to_pose7(T_tv @ pose7_to_mat(p7_l_g[i]) @ T_tr_tcp_l)
        tcp_r[i] = mat_to_pose7(T_tv @ pose7_to_mat(p7_r_g[i]) @ T_tr_tcp_r)

    return {
        "task": ep_task,
        "grid_s": grid - grid[0],
        "tcp_table": {"left": tcp_l, "right": tcp_r},
        "widths": {"left": w_l_g, "right": w_r_g},
        "valid_mask": acc_l_g & acc_r_g,
        "cam_l_paths": [cam_l_items[int(i)][1] for i in idx_cl],
        "cam_r_paths": [cam_r_items[int(i)][1] for i in idx_cr],
        "max_cam_skew_ms": float(np.max(skew_ms)),
    }


def _process_one_episode(task_args: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        raw_order_idx,
        ep_dir_str,
        cal_dir_str,
        out_root_str,
        robot,
        solver,
        retarget_mode,
        require_physical,
        table_calib_path,
        fps,
        min_seg_transitions,
        default_task,
    ) = task_args
    t_wall0 = time.perf_counter()
    cv2.setNumThreads(1)

    ep_dir = Path(ep_dir_str)
    out_root = Path(out_root_str)
    cal = default_calibration_metadata(Path(cal_dir_str) if cal_dir_str else None)
    prof = load_robot_profile(robot)
    tcal = load_table_calibration(
        robot,
        path=table_calib_path,
        require_physical=bool(require_physical),
    )

    raw = _read_raw_episode_resampled(ep_dir, cal, fps=fps, default_task=default_task)
    min_pose_samples = max(2, int(min_seg_transitions) + 1)
    all_runs = contiguous_valid_runs(raw["valid_mask"], min_len=2)
    runs = [r for r in all_runs if (r[1] - r[0]) >= min_pose_samples]
    if not runs:
        raise RuntimeError(
            f"{ep_dir.name} produced 0 valid runs with >= {min_seg_transitions} transitions (all_runs={all_runs})"
        )

    staging_dir = out_root / "videos" / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    segments_out: list[dict[str, Any]] = []
    total_valid_samples = int(sum(b - a for a, b in runs))
    dropped_frames = int(len(raw["valid_mask"]) - total_valid_samples)

    for seg_idx, (a, b) in enumerate(runs):
        tcp_seg = {s: raw["tcp_table"][s][a:b] for s in ("left", "right")}
        w_seg = {s: raw["widths"][s][a:b] for s in ("left", "right")}

        ee_tgts, table_z = map_table_tcp_to_robot_ee(
            prof, tcp_seg, table_calib=tcal, retarget_mode=retarget_mode
        )
        q_traj, w_rob, _ = solve_episode_with_method(
            prof, ee_tgts, w_seg, solver_name=solver, fps=float(fps), table_z_robot=table_z
        )

        pos_errs_mm: dict[str, list[float]] = {s: [] for s in prof.active_sides}
        rot_errs_deg: dict[str, list[float]] = {s: [] for s in prof.active_sides}
        for k in range(len(q_traj)):
            fk, _ = prof.urdf_model.forward_kinematics_all(q_traj[k])
            for side in prof.active_sides:
                T_sol = fk[prof.arms[side].ee_link]
                T_tgt = pose7_to_mat(ee_tgts[side][k])
                pos_errs_mm[side].append(float(np.linalg.norm(T_sol[:3, 3] - T_tgt[:3, 3]) * 1000.0))
                R_diff = T_sol[:3, :3].T @ T_tgt[:3, :3]
                ang = math.degrees(math.acos(float(np.clip((np.trace(R_diff) - 1.0) * 0.5, -1.0, 1.0))))
                rot_errs_deg[side].append(ang)

        state_full = pack_canonical_joint_vector(prof, q_traj, w_rob)
        state_j = state_full[:-1].astype(np.float32)
        action_j = state_full[1:].astype(np.float32)
        n_frames = int(len(state_j))
        assert n_frames == (b - a - 1) and n_frames >= min_seg_transitions
        assert np.all(np.isfinite(state_j)) and np.all(np.isfinite(action_j))

        stage_l = staging_dir / f"{ep_dir.name}_seg{seg_idx:02d}_left.mp4"
        stage_r = staging_dir / f"{ep_dir.name}_seg{seg_idx:02d}_right.mp4"

        img_stats_l, img_samp_l, hw_l = _encode_h264_and_sample_stats(
            raw["cam_l_paths"][a : b - 1], stage_l, fps=fps
        )
        img_stats_r, img_samp_r, hw_r = _encode_h264_and_sample_stats(
            raw["cam_r_paths"][a : b - 1], stage_r, fps=fps
        )

        g_idx_l = np.unique(np.linspace(0, len(img_samp_l) - 1, min(len(img_samp_l), 15), dtype=int))
        g_idx_r = np.unique(np.linspace(0, len(img_samp_r) - 1, min(len(img_samp_r), 15), dtype=int))

        ik_metrics: dict[str, float] = {}
        for side in prof.active_sides:
            ik_metrics[f"{side}_pos_mean_mm"] = round(float(np.mean(pos_errs_mm[side])), 2)
            ik_metrics[f"{side}_pos_max_mm"] = round(float(np.max(pos_errs_mm[side])), 2)
            ik_metrics[f"{side}_rot_max_deg"] = round(float(np.max(rot_errs_deg[side])), 2)

        segments_out.append(
            {
                "raw_order_idx": raw_order_idx,
                "segment_index": seg_idx,
                "num_segments_from_source": len(runs),
                "source_episode": ep_dir.name,
                "task": raw["task"],
                "length": n_frames,
                "source_frame_range": [int(a), int(b - 1)],
                "dropped_frames": dropped_frames,
                "max_cam_skew_ms": round(raw["max_cam_skew_ms"], 3),
                "state": state_j,
                "action": action_j,
                "stage_mp4_l": str(stage_l),
                "stage_mp4_r": str(stage_r),
                "img_hw_l": hw_l,
                "img_hw_r": hw_r,
                "img_stats_l": img_stats_l,
                "img_stats_r": img_stats_r,
                "img_global_l": img_samp_l[g_idx_l],
                "img_global_r": img_samp_r[g_idx_r],
                "ik_metrics": ik_metrics,
                "elapsed_s": round(time.perf_counter() - t_wall0, 2),
            }
        )

    return segments_out


def export_raw_to_lerobot_v21(
    raw_dir: Path,
    output_dir: Path,
    *,
    calibration_dir: Path | None = None,
    robot: str = "piper_pika",
    solver: str = "cstro",
    retarget_mode: str = "absolute-table",
    exclude_ids: set[int] | None = None,
    fps: int = 30,
    min_segment_seconds: float = 1.0,
    workers: int = 1,
    require_physical_table: bool = False,
    table_calib_path: Path | None = None,
    default_task: str = "bimanual manipulation",
) -> dict[str, Any]:
    """Export raw Dual Pika episodes (`episodeN/`) to a strict LeRobot v2.1 joint-action dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t_start = time.perf_counter()
    raw_dir = Path(raw_dir)
    out_dir = Path(output_dir)
    excl = set(exclude_ids or set())
    min_seg_transitions = max(2, int(math.ceil(float(min_segment_seconds) * float(fps))))

    all_ep_dirs = sorted(
        (p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("episode")),
        key=lambda p: natural_sort_key(p.name),
    )
    kept_ep_dirs = [
        p for p in all_ep_dirs if int(p.name.replace("episode", "")) not in excl
    ]
    if not kept_ep_dirs:
        raise FileNotFoundError(f"No matching episode* directories found in {raw_dir} (excluded={sorted(excl)})")

    prof = load_robot_profile(robot)
    feat_names = v21_joint_feature_names(prof)
    dim = len(feat_names)
    tcal = load_table_calibration(
        robot,
        path=table_calib_path,
        require_physical=require_physical_table,
    )
    cal_meta = default_calibration_metadata(calibration_dir)

    print(
        f"[START] Exporting {len(kept_ep_dirs)}/{len(all_ep_dirs)} raw episodes -> {out_dir} "
        f"(robot={robot}, solver={solver}, excluded={sorted(excl)}, min_seg={min_segment_seconds:g}s, workers={workers})",
        flush=True,
    )

    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENCV_NUM_THREADS"] = "1"

    jobs = [
        (
            raw_idx,
            str(ep_dir),
            str(calibration_dir) if calibration_dir else "",
            str(out_dir),
            robot,
            solver,
            retarget_mode,
            bool(require_physical_table),
            str(table_calib_path) if table_calib_path else None,
            int(fps),
            min_seg_transitions,
            default_task,
        )
        for raw_idx, ep_dir in enumerate(kept_ep_dirs)
    ]

    results_by_raw_idx: dict[int, list[dict[str, Any]]] = {}
    eff_workers = max(1, min(int(workers), len(jobs)))
    if eff_workers == 1:
        for done, j in enumerate(jobs, start=1):
            segs = _process_one_episode(j)
            results_by_raw_idx[segs[0]["raw_order_idx"]] = segs
            if done % 10 == 0 or done == len(jobs) or len(segs) > 1:
                r0 = segs[0]
                lens = [s["length"] for s in segs]
                print(
                    f"  [{done:3d}/{len(jobs)}] {r0['source_episode']} -> {len(segs)} seg(s) {lens}f ({r0['elapsed_s']}s)",
                    flush=True,
                )
    else:
        pool = ProcessPoolExecutor(max_workers=eff_workers, mp_context=mp.get_context("spawn"))
        try:
            futs = {pool.submit(_process_one_episode, j): j[0] for j in jobs}
            done = 0
            for fut in as_completed(futs):
                segs = fut.result()
                results_by_raw_idx[segs[0]["raw_order_idx"]] = segs
                done += 1
                if done % 10 == 0 or done == len(jobs) or len(segs) > 1:
                    r0 = segs[0]
                    lens = [s["length"] for s in segs]
                    print(
                        f"  [{done:3d}/{len(jobs)}] {r0['source_episode']} -> {len(segs)} seg(s) {lens}f ({r0['elapsed_s']}s)",
                        flush=True,
                    )
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    ordered: list[dict[str, Any]] = []
    for raw_idx in range(len(jobs)):
        for seg in results_by_raw_idx[raw_idx]:
            ep_idx = len(ordered)
            seg["episode_index"] = ep_idx
            chunk_idx = ep_idx // 1000
            stem = f"episode_{ep_idx:06d}.mp4"
            dst_l = out_dir / "videos" / f"chunk-{chunk_idx:03d}" / "observation.images.left_wrist" / stem
            dst_r = out_dir / "videos" / f"chunk-{chunk_idx:03d}" / "observation.images.right_wrist" / stem
            dst_l.parent.mkdir(parents=True, exist_ok=True)
            dst_r.parent.mkdir(parents=True, exist_ok=True)
            os.replace(seg["stage_mp4_l"], dst_l)
            os.replace(seg["stage_mp4_r"], dst_r)
            ordered.append(seg)

    staging_dir = out_dir / "videos" / "_staging"
    if staging_dir.exists():
        staging_dir.rmdir()

    tasks_map: dict[str, int] = {}
    episodes_jsonl_lines: list[str] = []
    episodes_stats_lines: list[str] = []
    source_reports: list[dict[str, Any]] = []

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_ts: list[np.ndarray] = []
    all_fidx: list[np.ndarray] = []
    all_eidx: list[np.ndarray] = []
    all_gidx: list[np.ndarray] = []
    all_tidx: list[np.ndarray] = []
    all_img_l: list[np.ndarray] = []
    all_img_r: list[np.ndarray] = []

    hf_meta = _hf_schema_metadata(dim)
    schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), dim)),
            pa.field("action", pa.list_(pa.float32(), dim)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ],
        metadata=hf_meta,
    )

    global_index = 0
    for res in ordered:
        ep_idx = res["episode_index"]
        n = res["length"]
        ep_task = res["task"]
        if ep_task not in tasks_map:
            tasks_map[ep_task] = len(tasks_map)
        task_idx = tasks_map[ep_task]

        state_j = res["state"]
        action_j = res["action"]
        ts = (np.arange(n, dtype=np.float64) / float(fps)).astype(np.float32)
        frame_idx = np.arange(n, dtype=np.int64)
        ep_idx_arr = np.full(n, ep_idx, dtype=np.int64)
        idx_arr = np.arange(global_index, global_index + n, dtype=np.int64)
        task_idx_arr = np.full(n, task_idx, dtype=np.int64)
        global_index += n

        state_col = pa.FixedSizeListArray.from_arrays(pa.array(state_j.reshape(-1), type=pa.float32()), dim)
        action_col = pa.FixedSizeListArray.from_arrays(pa.array(action_j.reshape(-1), type=pa.float32()), dim)
        tbl = pa.table(
            {
                "observation.state": state_col,
                "action": action_col,
                "timestamp": pa.array(ts, type=pa.float32()),
                "frame_index": pa.array(frame_idx, type=pa.int64()),
                "episode_index": pa.array(ep_idx_arr, type=pa.int64()),
                "index": pa.array(idx_arr, type=pa.int64()),
                "task_index": pa.array(task_idx_arr, type=pa.int64()),
            },
            schema=schema,
        )
        chunk_dir = out_dir / "data" / f"chunk-{ep_idx // 1000:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, chunk_dir / f"episode_{ep_idx:06d}.parquet")

        episodes_jsonl_lines.append(
            json.dumps({"episode_index": ep_idx, "tasks": [ep_task], "length": n}, ensure_ascii=False)
        )
        ep_stats = {
            "observation.state": _vector_stats(state_j),
            "action": _vector_stats(action_j),
            "timestamp": _vector_stats(ts),
            "frame_index": _vector_stats(frame_idx),
            "episode_index": _vector_stats(ep_idx_arr),
            "index": _vector_stats(idx_arr),
            "task_index": _vector_stats(task_idx_arr),
            "observation.images.left_wrist": res["img_stats_l"],
            "observation.images.right_wrist": res["img_stats_r"],
        }
        episodes_stats_lines.append(
            json.dumps({"episode_index": ep_idx, "stats": ep_stats}, ensure_ascii=False)
        )

        source_reports.append(
            {
                "episode_index": ep_idx,
                "source_episode": res["source_episode"],
                "segment_index": res["segment_index"],
                "num_segments_from_source": res["num_segments_from_source"],
                "fps": int(fps),
                "task": ep_task,
                "num_frames": n,
                "source_frame_range": res["source_frame_range"],
                "dropped_frames": res["dropped_frames"],
                "max_cam_skew_ms": res["max_cam_skew_ms"],
                "ik_metrics": res["ik_metrics"],
            }
        )

        all_states.append(state_j)
        all_actions.append(action_j)
        all_ts.append(ts)
        all_fidx.append(frame_idx)
        all_eidx.append(ep_idx_arr)
        all_gidx.append(idx_arr)
        all_tidx.append(task_idx_arr)
        all_img_l.append(res["img_global_l"])
        all_img_r.append(res["img_global_r"])

    tasks_lines = [
        json.dumps({"task_index": idx, "task": t_str}, ensure_ascii=False)
        for t_str, idx in sorted(tasks_map.items(), key=lambda kv: kv[1])
    ]
    (out_dir / "meta" / "tasks.jsonl").write_text("\n".join(tasks_lines) + "\n", encoding="utf-8")
    (out_dir / "meta" / "episodes.jsonl").write_text("\n".join(episodes_jsonl_lines) + "\n", encoding="utf-8")
    (out_dir / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(episodes_stats_lines) + "\n", encoding="utf-8"
    )

    cat_states = np.concatenate(all_states, axis=0)
    cat_actions = np.concatenate(all_actions, axis=0)
    global_stats = {
        "observation.state": _vector_stats(cat_states),
        "action": _vector_stats(cat_actions),
        "timestamp": _vector_stats(np.concatenate(all_ts, axis=0)),
        "frame_index": _vector_stats(np.concatenate(all_fidx, axis=0)),
        "episode_index": _vector_stats(np.concatenate(all_eidx, axis=0)),
        "index": _vector_stats(np.concatenate(all_gidx, axis=0)),
        "task_index": _vector_stats(np.concatenate(all_tidx, axis=0)),
        "observation.images.left_wrist": _image_stats_from_samples(
            np.concatenate(all_img_l, axis=0), global_index
        ),
        "observation.images.right_wrist": _image_stats_from_samples(
            np.concatenate(all_img_r, axis=0), global_index
        ),
    }
    save_json_atomic(out_dir / "meta" / "stats.json", global_stats)
    save_json_atomic(
        out_dir / "meta" / "abs_action_norm_stats.json",
        {
            "observation.state": global_stats["observation.state"],
            "action": global_stats["action"],
        },
    )

    h_l, w_l = ordered[0]["img_hw_l"]
    h_r, w_r = ordered[0]["img_hw_r"]
    n_eps = len(ordered)
    n_chunks = max(1, (n_eps + 999) // 1000)
    robot_type = f"dual_{robot}" if len(prof.active_sides) == 2 else robot
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": n_eps,
        "total_frames": int(global_index),
        "total_tasks": len(tasks_map),
        "total_videos": n_eps * 2,
        "total_chunks": n_chunks,
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{n_eps}"},
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [dim], "names": feat_names},
            "action": {"dtype": "float32", "shape": [dim], "names": feat_names},
            "observation.images.left_wrist": {
                "dtype": "video",
                "shape": [3, int(h_l), int(w_l)],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.fps": int(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.right_wrist": {
                "dtype": "video",
                "shape": [3, int(h_r), int(w_r)],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.fps": int(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "pika": {
            "state_layout": "canonical_joint_positions_plus_gripper_width_per_side",
            "action_semantics": "next_absolute_joint_position_and_gripper_width",
            "label_source": f"pikaumi_{solver}_ik",
            "robot_profile": robot,
            "retarget_mode": retarget_mode,
            "gripper_max_width_m": float(prof.gripper_max_width_m),
            "excluded_raw_episodes": sorted(excl),
            "table_calib_sha256": tcal.sha256,
            "table_calib_verified": bool(tcal.verified),
            "spatial_sha256": cal_meta.get("spatial_sha256"),
            "pika_tcp_sha256": cal_meta.get("pika_tcp_sha256"),
            "raw_source_dir": str(raw_dir),
        },
    }
    save_json_atomic(out_dir / "meta" / "info.json", info)
    save_json_atomic(
        out_dir / "pika_source_reports.json",
        {
            "raw_source_dir": str(raw_dir),
            "excluded_raw_episodes": sorted(excl),
            "total_episodes": n_eps,
            "total_frames": int(global_index),
            "table_calib_sha256": tcal.sha256,
            "episodes": source_reports,
        },
    )

    elapsed = time.perf_counter() - t_start
    print(
        f"[DONE] Exported {n_eps} LeRobot v2.1 episode(s) ({global_index} frames) to {out_dir} in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "output_dir": str(out_dir),
        "total_episodes": n_eps,
        "total_frames": int(global_index),
        "excluded_raw_episodes": sorted(excl),
        "elapsed_s": round(elapsed, 2),
    }
