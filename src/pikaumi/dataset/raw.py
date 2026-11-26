"""Robot-agnostic `pika_raw` dataset format with deferred TCP/table re-application & Pika_Data importer."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.calibration.pika_tcp import compute_pika_tcp_sha256
from pikaumi.config import (
    PIKA_TCP_CALIB_DIR,
    load_yaml,
    save_json_atomic,
    sha256_object,
)
from pikaumi.robots.utils import IDENTITY_POSE7, mat_to_pose7, pose7_to_mat, rpy_xyz_to_pose7
from pikaumi.synchronization import (
    build_causal_sync_grid,
    interpolate_jaw_widths,
    interpolate_se3_trajectory,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except Exception:  # pragma: no cover
    HAS_PYARROW = False


def natural_sort_key(text: str) -> list[int | str]:
    """Sort strings with embedded integers numerically (`episode2` before `episode10`)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def contiguous_valid_runs(valid_mask: np.ndarray, *, min_len: int = 2) -> list[tuple[int, int]]:
    """Return half-open `[start, end)` intervals of contiguous `True` frames with `end - start >= min_len`."""
    mask = np.asarray(valid_mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def _load_legacy_pika_tcp_jsons(cal_dir: Path) -> dict[str, list[float]] | None:
    """Load measured legacy `{left,right}_pika_tcp.json` transforms when `pika_tcp.yaml` is absent."""
    candidates = [cal_dir, cal_dir / "tcp"]
    for base in candidates:
        left_p = base / "left_pika_tcp.json"
        right_p = base / "right_pika_tcp.json"
        if left_p.is_file() and right_p.is_file():
            out: dict[str, list[float]] = {}
            for side, path in (("left", left_p), ("right", right_p)):
                payload = json.loads(path.read_text(encoding="utf-8"))
                mat = np.asarray(payload["transform"], dtype=np.float64)
                if mat.shape != (4, 4) or not np.isfinite(mat).all():
                    raise ValueError(f"Invalid 4x4 transform in legacy TCP file: {path}")
                out[side] = mat_to_pose7(mat).tolist()
            return out
    return None


def default_calibration_metadata(calibration_dir: str | Path | None = None) -> dict[str, Any]:
    """Load `T_table_vive` and `T_tracker_tcp` (left/right) with SHA-256 fingerprints."""
    cal_dir = Path(calibration_dir) if calibration_dir is not None else None
    tcp_cfg: dict[str, Any] = {}
    T_trk_tcp: dict[str, list[float]] = {}

    if cal_dir is not None and (cal_dir / "pika_tcp.yaml").is_file():
        tcp_cfg = load_yaml(cal_dir / "pika_tcp.yaml")
    else:
        legacy_tcp = _load_legacy_pika_tcp_jsons(cal_dir) if cal_dir is not None else None
        if legacy_tcp is not None:
            T_trk_tcp = legacy_tcp
        else:
            tcp_yaml_path = PIKA_TCP_CALIB_DIR / "pika_sense_v1.yaml"
            tcp_cfg = load_yaml(tcp_yaml_path) if tcp_yaml_path.is_file() else {}

    if not T_trk_tcp:
        sides = tcp_cfg.get("sides") or {}
        for side in ("left", "right"):
            if side in sides:
                T_trk_tcp[side] = [*sides[side]["position"], *sides[side]["quaternion"]]
            else:
                T_trk_tcp[side] = IDENTITY_POSE7.tolist()

    T_table_vive = IDENTITY_POSE7.tolist()
    spatial_sha = "none"
    if cal_dir is not None and (cal_dir / "session.yaml").is_file():
        ses = load_yaml(cal_dir / "session.yaml")
        tv = ses.get("T_table_vive") or {}
        if "position" in tv and "quaternion" in tv:
            T_table_vive = [*tv["position"], *tv["quaternion"]]
        spatial_sha = str(ses.get("spatial_sha256", "none"))
    elif cal_dir is not None and (cal_dir / "spatial.yaml").is_file():
        spatial_sha = sha256_object(load_yaml(cal_dir / "spatial.yaml"))

    pika_tcp_sha = compute_pika_tcp_sha256(tcp_cfg) if tcp_cfg else sha256_object(T_trk_tcp)
    return {
        "T_table_vive": T_table_vive,
        "T_tracker_tcp": T_trk_tcp,
        "spatial_sha256": spatial_sha,
        "pika_tcp_sha256": pika_tcp_sha,
        "gripper_max_width_m": float(tcp_cfg.get("gripper_max_width_m", 0.10)),
    }


def write_episode_table(
    episode_idx: int,
    data_dir: Path,
    *,
    timestamps_s: np.ndarray,
    state_16d: np.ndarray,
    action_16d: np.ndarray,
    left_accurate: np.ndarray,
    right_accurate: np.ndarray,
    sync_skew_s: np.ndarray,
    lr_image_skew_s: np.ndarray | None = None,
) -> Path:
    """Write episode tabular trajectory as `.parquet` (when `pyarrow` is installed) plus `.npz` sidecar."""
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_idx:06d}"
    npz_path = data_dir / f"{stem}.npz"
    npz_payload: dict[str, Any] = {
        "timestamps_s": np.asarray(timestamps_s, dtype=np.float64),
        "observation_state": np.asarray(state_16d, dtype=np.float32),
        "action": np.asarray(action_16d, dtype=np.float32),
        "left_accurate": np.asarray(left_accurate, dtype=bool),
        "right_accurate": np.asarray(right_accurate, dtype=bool),
        "sync_skew_s": np.asarray(sync_skew_s, dtype=np.float32),
    }
    if lr_image_skew_s is not None:
        npz_payload["lr_image_skew_s"] = np.asarray(lr_image_skew_s, dtype=np.float32)
    np.savez_compressed(npz_path, **npz_payload)
    if HAS_PYARROW:
        pq_path = data_dir / f"{stem}.parquet"
        T = len(timestamps_s)
        cols: dict[str, Any] = {
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, episode_idx, dtype=np.int64),
            "timestamp": np.asarray(timestamps_s, dtype=np.float64),
            "observation.state": [row.tolist() for row in np.asarray(state_16d, dtype=np.float32)],
            "action": [row.tolist() for row in np.asarray(action_16d, dtype=np.float32)],
            "left_accurate": np.asarray(left_accurate, dtype=bool),
            "right_accurate": np.asarray(right_accurate, dtype=bool),
            "sync_skew_s": np.asarray(sync_skew_s, dtype=np.float32),
        }
        if lr_image_skew_s is not None:
            cols["lr_image_skew_s"] = np.asarray(lr_image_skew_s, dtype=np.float32)
        pq.write_table(pa.table(cols), pq_path)
        return pq_path
    return npz_path


def load_episode_raw(dataset_dir: str | Path, episode_idx: int) -> dict[str, np.ndarray]:
    """Load a `pika_raw` episode seamlessly from `.npz` or `.parquet`."""
    root = Path(dataset_dir)
    stem = f"episode_{episode_idx:06d}"
    npz_path = root / "data" / f"{stem}.npz"
    pq_path = root / "data" / f"{stem}.parquet"
    if npz_path.is_file():
        with np.load(npz_path) as z:
            out = {
                "timestamps_s": z["timestamps_s"].astype(np.float64),
                "observation_state": z["observation_state"].astype(np.float64),
                "action": z["action"].astype(np.float64),
                "left_accurate": z["left_accurate"].astype(bool),
                "right_accurate": z["right_accurate"].astype(bool),
                "sync_skew_s": z["sync_skew_s"].astype(np.float64),
            }
            if "lr_image_skew_s" in z:
                out["lr_image_skew_s"] = z["lr_image_skew_s"].astype(np.float64)
            return out
    if pq_path.is_file() and HAS_PYARROW:
        tbl = pq.read_table(pq_path)
        out = {
            "timestamps_s": np.asarray(tbl["timestamp"].to_pylist(), dtype=np.float64),
            "observation_state": np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float64),
            "action": np.asarray(tbl["action"].to_pylist(), dtype=np.float64),
            "left_accurate": np.asarray(tbl["left_accurate"].to_pylist(), dtype=bool),
            "right_accurate": np.asarray(tbl["right_accurate"].to_pylist(), dtype=bool),
            "sync_skew_s": np.asarray(tbl["sync_skew_s"].to_pylist(), dtype=np.float64),
        }
        if "lr_image_skew_s" in tbl.column_names:
            out["lr_image_skew_s"] = np.asarray(tbl["lr_image_skew_s"].to_pylist(), dtype=np.float64)
        return out
    raise FileNotFoundError(f"Episode {episode_idx} not found under {root / 'data'}")


def load_dataset_info(dataset_dir: str | Path) -> dict[str, Any]:
    info_path = Path(dataset_dir) / "meta" / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_dataset_episodes(dataset_dir: str | Path) -> list[dict[str, Any]]:
    ep_path = Path(dataset_dir) / "meta" / "episodes.json"
    return list(json.loads(ep_path.read_text(encoding="utf-8")))


def _episode_calibration(dataset_dir: str | Path, episode_idx: int, info: dict[str, Any]) -> dict[str, Any]:
    """Return per-episode calibration if stored in `episodes.json` (e.g. after merge), else dataset default."""
    cal = dict(info.get("calibration") or {})
    ep_path = Path(dataset_dir) / "meta" / "episodes.json"
    if ep_path.is_file():
        for meta in json.loads(ep_path.read_text(encoding="utf-8")):
            if int(meta.get("episode_index", -1)) == int(episode_idx):
                ep_cal = meta.get("calibration")
                if isinstance(ep_cal, dict):
                    cal.update(ep_cal)
                break
    return cal


def extract_episode_tcp_targets(
    dataset_dir: str | Path,
    episode_idx: int,
    *,
    override_T_table_vive: np.ndarray | None = None,
    override_T_tracker_tcp: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Deferred post-hoc TCP & table frame reconstruction:

    `T_table_tcp[side](t) = T_table_vive @ T_vive_tracker[side](t) @ T_tracker_tcp[side]`.

    Allows updating `pika_tcp.yaml` or `session.yaml` without re-recording raw Vive data!
    Returns `(tcp_poses_table_by_side, widths_m_by_side, timestamps_s)`.
    """
    info = load_dataset_info(dataset_dir)
    cal = _episode_calibration(dataset_dir, episode_idx, info)
    ep = load_episode_raw(dataset_dir, episode_idx)
    state = ep["observation_state"]  # (T, 16)

    T_tv = pose7_to_mat(
        override_T_table_vive
        if override_T_table_vive is not None
        else np.asarray(cal.get("T_table_vive", IDENTITY_POSE7), dtype=np.float64)
    )
    trk_tcp_cfg = cal.get("T_tracker_tcp") or {}

    slices = {"left": (0, 7, 7), "right": (8, 15, 15)}
    tcp_out: dict[str, np.ndarray] = {}
    widths_out: dict[str, np.ndarray] = {}

    for side, (p0, p1, w_idx) in slices.items():
        if override_T_tracker_tcp is not None and side in override_T_tracker_tcp:
            T_tt = pose7_to_mat(override_T_tracker_tcp[side])
        else:
            T_tt = pose7_to_mat(np.asarray(trk_tcp_cfg.get(side, IDENTITY_POSE7), dtype=np.float64))

        raw_poses = state[:, p0:p1]
        T = len(raw_poses)
        poses_table = np.zeros((T, 7), dtype=np.float64)
        for t in range(T):
            T_vt = pose7_to_mat(raw_poses[t])
            poses_table[t] = mat_to_pose7(T_tv @ T_vt @ T_tt)
        tcp_out[side] = poses_table
        widths_out[side] = np.clip(state[:, w_idx].copy(), 0.0, 0.10)

    return tcp_out, widths_out, ep["timestamps_s"]


def generate_synthetic_bimanual_dataset(
    output_dir: str | Path,
    *,
    num_episodes: int = 3,
    frames_per_episode: int = 60,
    fps: int = 30,
    include_reversed_episode: bool = False,
    include_bad_tracking_episode: bool = False,
    include_branch_challenge: bool = True,
) -> Path:
    """Create a realistic, deterministic bimanual towel-fold `pika_raw` dataset for QA & retargeting benchmarks."""
    out = Path(output_dir)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    cal = default_calibration_metadata(None)
    T_tt_l_inv = np.linalg.inv(pose7_to_mat(np.asarray(cal["T_tracker_tcp"]["left"])))
    T_tt_r_inv = np.linalg.inv(pose7_to_mat(np.asarray(cal["T_tracker_tcp"]["right"])))

    episodes_meta: list[dict[str, Any]] = []
    ep_specs: list[str] = ["normal_fold" for _ in range(num_episodes)]
    if include_branch_challenge and len(ep_specs) >= 2:
        ep_specs[1] = "cross_reach_co_grasp"
    if include_reversed_episode:
        ep_specs.append("reversed_reset")
    if include_bad_tracking_episode:
        ep_specs.append("dropout_fault")

    for ep_idx, kind in enumerate(ep_specs):
        T = frames_per_episode
        ts = np.arange(T, dtype=np.float64) / float(fps)
        phase = np.linspace(0.0, 1.0, T)
        if kind == "reversed_reset":
            phase = phase[::-1]

        state = np.zeros((T, 16), dtype=np.float64)
        left_acc = np.ones(T, dtype=bool)
        right_acc = np.ones(T, dtype=bool)
        skew = np.full(T, 0.004, dtype=np.float64)

        for t_i, u in enumerate(phase):
            # In table coordinates (+X left-to-right=0? Wait: table frame matches robot_from_table convention:
            # In sim table YAMLs: robot_from_table has pos=[0.34, 0, -0.03], quat=[0,0,-0.7071,0.7071] (Rz(-90 deg)).
            # With Rz(-90 deg): table +X -> robot -Y (right), table +Y -> robot +X (forward), table +Z -> robot +Z (up)!
            # So in table coordinates: x in [-0.20, +0.20] is lateral, y in [-0.08, +0.12] is depth, z in [0.04, 0.20] is height!
            co_grasp_window = 0.28 <= u <= 0.78
            lift = 0.10 * np.sin(np.pi * np.clip((u - 0.25) / 0.55, 0.0, 1.0))
            fold_fwd = 0.10 * (1.0 - np.cos(np.pi * u)) * 0.5

            if kind == "cross_reach_co_grasp":
                yaw_sweep = 0.55 * np.sin(2.0 * np.pi * u)
                lat = 0.13 - 0.03 * np.sin(np.pi * u)
            else:
                yaw_sweep = 0.20 * np.sin(np.pi * u) + 0.04 * ep_idx
                lat = 0.15

            # Pika TCP orientation in table frame: pointing forward (+Y_table) and pitched down ~35 deg
            rpy_l = [0.0, 0.55, np.pi / 2.0 + yaw_sweep]
            rpy_r = [0.0, 0.55, np.pi / 2.0 - yaw_sweep]
            pos_l = np.array([-lat, -0.04 + fold_fwd, 0.07 + lift])
            pos_r = np.array([+lat, -0.04 + fold_fwd, 0.07 + lift])

            T_tcp_l = pose7_to_mat(rpy_xyz_to_pose7(pos_l, rpy_l))
            T_tcp_r = pose7_to_mat(rpy_xyz_to_pose7(pos_r, rpy_r))
            p_trk_l = mat_to_pose7(T_tcp_l @ T_tt_l_inv)
            p_trk_r = mat_to_pose7(T_tcp_r @ T_tt_r_inv)

            w_val = 0.012 if co_grasp_window else 0.082
            state[t_i, 0:7] = p_trk_l
            state[t_i, 7] = w_val
            state[t_i, 8:15] = p_trk_r
            state[t_i, 15] = w_val

        if kind == "dropout_fault":
            left_acc[T // 3 : T // 3 + 4] = False
            skew[T // 2] = 0.090

        action = np.vstack([state[1:], state[-1:]])
        write_episode_table(
            ep_idx,
            out / "data",
            timestamps_s=ts,
            state_16d=state,
            action_16d=action,
            left_accurate=left_acc,
            right_accurate=right_acc,
            sync_skew_s=skew,
        )

        # Save lightweight 32x32 start/end visual thumbnails for direction detection (`direction.py`)
        img_start = np.full((32, 32, 3), 40, dtype=np.uint8)
        img_end = np.full((32, 32, 3), 40, dtype=np.uint8)
        img_start[18:28, 6:26, 0] = 210  # unfolded cloth at bottom
        img_end[8:18, 10:22, 1] = 220    # folded cloth moved up-center
        if kind == "reversed_reset":
            img_start, img_end = img_end, img_start
        np.savez_compressed(
            out / "frames" / f"episode_{ep_idx:06d}_thumbs.npz",
            left_start=img_start,
            left_end=img_end,
            right_start=img_start,
            right_end=img_end,
        )

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "source_episode": f"episode{ep_idx}",
                "kind": kind,
                "num_frames": T,
                "fps": fps,
                "duration_s": round(T / float(fps), 4),
            }
        )

    info_payload = {
        "format": "pika_raw",
        "version": "3.0",
        "fps": fps,
        "task": "fold towel bimanual",
        "num_episodes": len(episodes_meta),
        "total_frames": sum(e["num_frames"] for e in episodes_meta),
        "state_dim": 16,
        "action_dim": 16,
        "feature_names": [
            "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw", "left_width_m",
            "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw", "right_width_m",
        ],
        "calibration": cal,
    }
    save_json_atomic(out / "meta" / "info.json", info_payload)
    save_json_atomic(out / "meta" / "episodes.json", episodes_meta)
    return out


def _numeric_timestamp_files(directory: Path, suffixes: set[str]) -> list[tuple[Any, Path]]:
    from decimal import Decimal, InvalidOperation

    if not directory.is_dir():
        return []
    out: list[tuple[Any, Path]] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            try:
                ts_dec = Decimal(p.stem)
            except InvalidOperation:
                continue
            out.append((ts_dec, p))
    out.sort(key=lambda item: item[0])
    return out


def _infer_distance_to_meters(values: list[float]) -> np.ndarray:
    """Infer whether raw jaw distance stream is in millimeters (> 0.25 max) or meters."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size > 0 and float(np.max(np.abs(arr))) > 0.25:
        return arr / 1000.0
    return arr


def _load_tracking_status_json(ep_dir: Path, side: str, t0_dec: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Load optional `<ep_dir>/tracking_status.json` (`{"sides": {"left": [[ns, bool], ...]}}`)."""
    from decimal import Decimal

    status_file = ep_dir / "tracking_status.json"
    if not status_file.is_file():
        return None
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    entries = (payload.get("sides") or {}).get(side)
    if not entries:
        return None
    ts_list: list[float] = []
    ok_list: list[bool] = []
    for item in entries:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        raw_ts = Decimal(str(item[0]))
        ts_sec = raw_ts / Decimal("1000000000") if raw_ts > Decimal("100000000000") else raw_ts
        ts_list.append(float(ts_sec - t0_dec))
        ok_list.append(bool(item[1]))
    if not ts_list:
        return None
    order = np.argsort(ts_list)
    return np.asarray(ts_list, dtype=np.float64)[order], np.asarray(ok_list, dtype=bool)[order]


def apply_tracking_status_with_padding(
    ts_arr: np.ndarray,
    st_ts: np.ndarray,
    st_acc: np.ndarray,
    acc_arr: np.ndarray | None = None,
    *,
    pre_pad_s: float = 0.05,
    post_pad_s: float = 0.25,
) -> np.ndarray:
    """Apply zero-order-hold tracking status and pad each dropout run by pre/post safety margins."""
    ts = np.asarray(ts_arr, dtype=np.float64)
    base = np.ones(len(ts), dtype=bool) if acc_arr is None else np.asarray(acc_arr, dtype=bool).copy()
    if len(ts) == 0 or len(st_ts) == 0:
        return base
    st_ts_a = np.asarray(st_ts, dtype=np.float64)
    st_acc_a = np.asarray(st_acc, dtype=bool)
    idx_near = np.clip(np.searchsorted(st_ts_a, ts, side="right") - 1, 0, len(st_acc_a) - 1)
    raw_valid = base & st_acc_a[idx_near]
    if np.all(raw_valid):
        return raw_valid

    padded = raw_valid.copy()
    n = len(ts)
    i = 0
    while i < n:
        if not raw_valid[i]:
            start = i
            while i < n and not raw_valid[i]:
                i += 1
            end_excl = i
            t_start = float(ts[start])
            t_recover = float(ts[end_excl]) if end_excl < n else float(ts[end_excl - 1])
            padded &= ~((ts >= t_start - pre_pad_s) & (ts <= t_recover + post_pad_s))
        else:
            i += 1
    return padded


def _align_causal_wrist_pair(
    ts_cl: np.ndarray,
    ts_cr: np.ndarray,
    grid: np.ndarray,
    *,
    max_age_s: float = 0.040,
) -> tuple[np.ndarray, np.ndarray]:
    """Select strictly causal (`ts <= grid`) wrist camera indices while minimizing left-right skew.

    Mirrors `Pika_Data/data_tools/prepare_pika_dataset.py:sample_wrist_pair`: when a synthetic
    grid timestamp falls inside a sub-millisecond hardware offset between left and right cameras,
    snaps the newer stream back by one frame (within `max_age_s`) so the pair stays locked without
    ever peeking into future frames.
    """
    streams = (np.asarray(ts_cl, dtype=np.float64), np.asarray(ts_cr, dtype=np.float64))
    indices = [
        np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)
        for times in streams
    ]
    for i, times in enumerate(streams):
        idx = indices[i].copy()
        other_idx = indices[1 - i]
        peer = streams[1 - i][other_idx]
        right = np.searchsorted(times, peer)
        for cand in (np.clip(right, 0, len(times) - 1), np.clip(right - 1, 0, len(times) - 1)):
            age = grid - times[cand]
            better = np.abs(times[cand] - peer) < np.abs(times[idx] - peer)
            use = (age >= 0.0) & (age <= float(max_age_s)) & better
            idx[use] = cand[use]
        indices[i] = idx
    return indices[0], indices[1]


def import_pika_episodes(
    input_dir: Path,
    output_dir: Path,
    *,
    calibration_dir: Path | None = None,
    fps: int = 30,
    task: str = "bimanual manipulation",
) -> dict[str, Any]:
    """Import raw `Pika_Data` episode directories (`localization/pose/...` or `pose/*.jsonl`) with wrist videos."""
    from decimal import Decimal

    src = Path(input_dir)
    dst = Path(output_dir)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    cal = default_calibration_metadata(calibration_dir)

    cand_dirs = (
        sorted(
            (p for p in src.iterdir() if p.is_dir() and p.name.startswith("episode")),
            key=lambda p: natural_sort_key(p.name),
        )
        if src.is_dir()
        else []
    )
    episodes_meta: list[dict[str, Any]] = []
    video_shapes: dict[str, list[int]] = {}

    for ep_idx, ep_dir in enumerate(cand_dirs):
        pose_l_dir = ep_dir / "localization" / "pose" / "pika_l"
        pose_r_dir = ep_dir / "localization" / "pose" / "pika_r"
        stat_l_dir = ep_dir / "localization" / "status" / "pika_l"
        stat_r_dir = ep_dir / "localization" / "status" / "pika_r"
        grip_l_dir = ep_dir / "gripper" / "encoder" / "pika_l"
        grip_r_dir = ep_dir / "gripper" / "encoder" / "pika_r"
        cam_l_dir = ep_dir / "camera" / "color" / "pikaFisheyeCamera_l"
        cam_r_dir = ep_dir / "camera" / "color" / "pikaFisheyeCamera_r"

        pose_l_file = ep_dir / "pose" / "pika_pose_l.jsonl"
        pose_r_file = ep_dir / "pose" / "pika_pose_r.jsonl"
        grip_l_file = ep_dir / "gripper" / "pika_gripper_l.jsonl"
        grip_r_file = ep_dir / "gripper" / "pika_gripper_r.jsonl"

        ep_task = task
        inst_file = ep_dir / "instructions.json"
        if inst_file.is_file():
            try:
                inst_list = json.loads(inst_file.read_text(encoding="utf-8")).get("instructions") or []
                if inst_list and isinstance(inst_list[0], str) and inst_list[0].strip():
                    ep_task = inst_list[0].strip()
            except Exception:
                pass

        t0_dec: Decimal | None = None

        def _apply_tracking_status(
            side: str,
            ts_arr: np.ndarray,
            acc_arr: np.ndarray,
            *,
            pre_pad_s: float = 0.05,
            post_pad_s: float = 0.25,
        ) -> np.ndarray:
            assert t0_dec is not None
            tr = _load_tracking_status_json(ep_dir, side, t0_dec)
            if tr is None:
                return acc_arr
            st_ts, st_acc = tr
            return apply_tracking_status_with_padding(
                ts_arr,
                st_ts,
                st_acc,
                acc_arr,
                pre_pad_s=pre_pad_s,
                post_pad_s=post_pad_s,
            )

        def _read_pose_json_dir(d: Path, sd: Path, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            nonlocal t0_dec
            items = _numeric_timestamp_files(d, {".json"})
            if len(items) < 2:
                raise ValueError(f"Need >= 2 pose JSON files in {d}")
            if t0_dec is None:
                t0_dec = items[0][0]
            ts_arr = np.array([float(ts - t0_dec) for ts, _ in items], dtype=np.float64)
            p7_list = []
            acc_list = []
            for _, p in items:
                r = json.loads(p.read_text(encoding="utf-8"))
                xyz = r.get("position") or [r["x"], r["y"], r["z"]]
                quat = (
                    r.get("quaternion")
                    or Rotation.from_euler("xyz", [r["roll"], r["pitch"], r["yaw"]]).as_quat().tolist()
                )
                p7_list.append([*xyz, *quat])
                acc_list.append(bool(r.get("accurate", True)))
            acc_arr = np.array(acc_list, dtype=bool)
            stat_items = _numeric_timestamp_files(sd, {".json"})
            if stat_items:
                st_ts = np.array([float(ts - t0_dec) for ts, _ in stat_items], dtype=np.float64)
                st_acc = np.array(
                    [bool(json.loads(p.read_text(encoding="utf-8")).get("accurate", True)) for _, p in stat_items],
                    dtype=bool,
                )
                idx_near = np.clip(np.searchsorted(st_ts, ts_arr, side="right") - 1, 0, len(st_acc) - 1)
                acc_arr = acc_arr & st_acc[idx_near]
            acc_arr = _apply_tracking_status(side, ts_arr, acc_arr)
            return ts_arr, np.array(p7_list, dtype=np.float64), acc_arr

        def _read_grip_json_dir(d: Path) -> tuple[np.ndarray, np.ndarray]:
            assert t0_dec is not None
            items = _numeric_timestamp_files(d, {".json"})
            if not items:
                raise FileNotFoundError(f"Missing gripper encoder JSON stream in {d}")
            ts_arr = np.array([float(ts - t0_dec) for ts, _ in items], dtype=np.float64)
            records = [json.loads(p.read_text(encoding="utf-8")) for _, p in items]
            if all("width_m" in r for r in records):
                vals = np.array([float(r["width_m"]) for r in records], dtype=np.float64)
            else:
                raw_dists = [float(r["width_m"] if "width_m" in r else r["distance"]) for r in records]
                vals = _infer_distance_to_meters(raw_dists)
            return ts_arr, vals

        def _read_pose_jsonl(f: Path, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            nonlocal t0_dec
            rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) < 2:
                raise ValueError(f"Need >= 2 pose rows in {f}")
            if t0_dec is None:
                t0_dec = Decimal(str(rows[0]["timestamp"]))
            ts_arr = np.array([float(Decimal(str(r["timestamp"])) - t0_dec) for r in rows], dtype=np.float64)
            if "pose7" in rows[0]:
                p7 = np.array([r["pose7"] for r in rows], dtype=np.float64)
            else:
                p7_list = []
                for r in rows:
                    xyz = r.get("position") or [r["x"], r["y"], r["z"]]
                    quat = (
                        r.get("quaternion")
                        or Rotation.from_euler("xyz", [r["roll"], r["pitch"], r["yaw"]]).as_quat().tolist()
                    )
                    p7_list.append([*xyz, *quat])
                p7 = np.array(p7_list, dtype=np.float64)
            acc_arr = np.array([bool(r.get("accurate", True)) for r in rows], dtype=bool)
            acc_arr = _apply_tracking_status(side, ts_arr, acc_arr)
            return ts_arr, p7, acc_arr

        def _read_grip_jsonl(f: Path) -> tuple[np.ndarray, np.ndarray]:
            assert t0_dec is not None
            if not f.is_file():
                raise FileNotFoundError(f"Missing gripper encoder JSONL stream: {f}")
            rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                raise FileNotFoundError(f"Empty gripper encoder JSONL stream: {f}")
            ts_arr = np.array([float(Decimal(str(r["timestamp"])) - t0_dec) for r in rows], dtype=np.float64)
            if all("width_m" in r for r in rows):
                vals = np.array([float(r["width_m"]) for r in rows], dtype=np.float64)
            else:
                raw_dists = [float(r["width_m"] if "width_m" in r else r["distance"]) for r in rows]
                vals = _infer_distance_to_meters(raw_dists)
            return ts_arr, vals

        cam_l_items = _numeric_timestamp_files(cam_l_dir, {".jpg", ".jpeg", ".png"})
        cam_r_items = _numeric_timestamp_files(cam_r_dir, {".jpg", ".jpeg", ".png"})

        if pose_l_dir.is_dir() and pose_r_dir.is_dir():
            ts_pl, p7_l, acc_l = _read_pose_json_dir(pose_l_dir, stat_l_dir, "left")
            ts_pr, p7_r, acc_r = _read_pose_json_dir(pose_r_dir, stat_r_dir, "right")
            ts_gl, w_l = _read_grip_json_dir(grip_l_dir)
            ts_gr, w_r = _read_grip_json_dir(grip_r_dir)
        elif pose_l_file.is_file() and pose_r_file.is_file():
            ts_pl, p7_l, acc_l = _read_pose_jsonl(pose_l_file, "left")
            ts_pr, p7_r, acc_r = _read_pose_jsonl(pose_r_file, "right")
            ts_gl, w_l = _read_grip_jsonl(grip_l_file)
            ts_gr, w_r = _read_grip_jsonl(grip_r_file)
        else:
            continue

        assert t0_dec is not None
        starts = [float(ts_pl[0]), float(ts_pr[0]), float(ts_gl[0]), float(ts_gr[0])]
        ends = [float(ts_pl[-1]), float(ts_pr[-1]), float(ts_gl[-1]), float(ts_gr[-1])]
        if cam_l_items and cam_r_items:
            ts_cl = np.array([float(ts - t0_dec) for ts, _ in cam_l_items], dtype=np.float64)
            ts_cr = np.array([float(ts - t0_dec) for ts, _ in cam_r_items], dtype=np.float64)
            starts.extend([float(ts_cl[0]), float(ts_cr[0])])
            ends.extend([float(ts_cl[-1]), float(ts_cr[-1])])
        else:
            ts_cl = ts_cr = None

        t_start = max(starts)
        t_end = min(ends)
        grid = build_causal_sync_grid(t_start, t_end, fps=fps)
        if len(grid) < 2:
            continue

        # Record real pre-resampling sensor timestamp gap excess (ms) so quality.py can detect dropped packets
        raw_streams = [ts_pl, ts_pr, ts_gl, ts_gr]
        if ts_cl is not None and ts_cr is not None:
            raw_streams.extend([ts_cl, ts_cr])
        nominal_dt_ms = 1000.0 / float(fps)
        raw_max_gap_excess_ms = 0.0
        for st_arr in raw_streams:
            in_win = st_arr[(st_arr >= t_start) & (st_arr <= t_end)]
            if len(in_win) >= 2:
                raw_max_gap_excess_ms = max(
                    raw_max_gap_excess_ms,
                    max(0.0, float(np.max(np.diff(in_win)) * 1000.0) - nominal_dt_ms),
                )

        ip_l, val_pl, sk_pl = interpolate_se3_trajectory(ts_pl, p7_l, grid, accurate_flags=acc_l)
        ip_r, val_pr, sk_pr = interpolate_se3_trajectory(ts_pr, p7_r, grid, accurate_flags=acc_r)
        iw_l, val_gl, sk_gl = interpolate_jaw_widths(ts_gl, w_l, grid)
        iw_r, val_gr, sk_gr = interpolate_jaw_widths(ts_gr, w_r, grid)

        state = np.hstack([ip_l, iw_l[:, None], ip_r, iw_r[:, None]])
        action = np.vstack([state[1:], state[-1:]])
        skew = np.maximum.reduce([sk_pl, sk_pr, sk_gl, sk_gr])
        lr_image_skew_s: np.ndarray | None = None

        if ts_cl is not None and ts_cr is not None:
            import cv2

            idx_cl, idx_cr = _align_causal_wrist_pair(ts_cl, ts_cr, grid, max_age_s=0.040)
            cam_skew = np.maximum(np.abs(grid - ts_cl[idx_cl]), np.abs(grid - ts_cr[idx_cr]))
            skew = np.maximum(skew, cam_skew)
            lr_image_skew_s = np.abs(ts_cl[idx_cl] - ts_cr[idx_cr])

            (dst / "frames").mkdir(parents=True, exist_ok=True)
            thumbs: dict[str, np.ndarray] = {}
            for side_lbl, items_c, idx_c, cam_key in (
                ("left", cam_l_items, idx_cl, "left_wrist"),
                ("right", cam_r_items, idx_cr, "right_wrist"),
            ):
                first_path = items_c[int(idx_c[0])][1]
                last_path = items_c[int(idx_c[-1])][1]
                first_bgr = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
                last_bgr = cv2.imread(str(last_path), cv2.IMREAD_COLOR)
                if first_bgr is None or last_bgr is None:
                    raise ValueError(f"Unreadable wrist camera frame in {ep_dir.name} ({first_path} / {last_path})")
                h_im, w_im = first_bgr.shape[:2]
                video_shapes[f"observation.images.{cam_key}"] = [3, int(h_im), int(w_im)]
                thumbs[f"{side_lbl}_start"] = cv2.cvtColor(cv2.resize(first_bgr, (32, 32)), cv2.COLOR_BGR2RGB)
                thumbs[f"{side_lbl}_end"] = cv2.cvtColor(cv2.resize(last_bgr, (32, 32)), cv2.COLOR_BGR2RGB)

                vid_dir = dst / "videos" / cam_key
                vid_dir.mkdir(parents=True, exist_ok=True)
                vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
                writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w_im, h_im))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open cv2.VideoWriter for {vid_path}")
                try:
                    for fi in idx_c:
                        frame_path = items_c[int(fi)][1]
                        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                        if frame_bgr is None:
                            raise ValueError(f"Corrupt or unreadable image frame: {frame_path}")
                        if frame_bgr.shape[:2] != (h_im, w_im):
                            frame_bgr = cv2.resize(frame_bgr, (w_im, h_im))
                        writer.write(frame_bgr)
                finally:
                    writer.release()

            if len(thumbs) == 4:
                np.savez_compressed(dst / "frames" / f"episode_{ep_idx:06d}_thumbs.npz", **thumbs)

        write_episode_table(
            ep_idx,
            dst / "data",
            timestamps_s=grid - grid[0],
            state_16d=state,
            action_16d=action,
            left_accurate=val_pl & val_gl,
            right_accurate=val_pr & val_gr,
            sync_skew_s=skew,
            lr_image_skew_s=lr_image_skew_s,
        )
        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "source_episode": ep_dir.name,
                "task": ep_task,
                "kind": "imported_pika_data",
                "num_frames": int(len(grid)),
                "fps": fps,
                "duration_s": round(len(grid) / float(fps), 4),
                "max_sensor_gap_ms": round(raw_max_gap_excess_ms, 2),
            }
        )

    info_payload: dict[str, Any] = {
        "format": "pika_raw",
        "version": "3.0",
        "fps": fps,
        "task": episodes_meta[0]["task"] if episodes_meta else task,
        "num_episodes": len(episodes_meta),
        "total_frames": sum(e["num_frames"] for e in episodes_meta),
        "state_dim": 16,
        "action_dim": 16,
        "calibration": cal,
    }
    if video_shapes:
        info_payload["video_features"] = video_shapes
    save_json_atomic(dst / "meta" / "info.json", info_payload)
    save_json_atomic(dst / "meta" / "episodes.json", episodes_meta)
    return info_payload
