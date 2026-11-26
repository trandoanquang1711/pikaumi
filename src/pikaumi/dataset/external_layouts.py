"""External LeRobot follower plugin layout exporters (`bi_piper_pika_follower`, `bi_piper_follower`, `bi_openarm_follower`)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import HAS_PYARROW

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq

# Exact firmware limits (in tenths of a degree) and joint sign conventions matching
# `xhuman/robots/piper/piper_sdk_interface.py` and `handumi-sw/src/handumi/dataset/external_layouts.py`.
_PIPER_LIMITS_TENTH_DEG: tuple[tuple[float, float], ...] = (
    (-1500.0, 1500.0),
    (0.0, 1800.0),
    (-1700.0, 0.0),
    (-1000.0, 1000.0),
    (-700.0, 700.0),
    (-1200.0, 1200.0),
)
_PIPER_SIGNS: tuple[float, ...] = (-1.0, 1.0, 1.0, -1.0, 1.0, -1.0)
_PIPER_LIMITS_RAD: tuple[tuple[float, float], ...] = tuple(
    (float(np.deg2rad(lo / 10.0)), float(np.deg2rad(hi / 10.0))) for lo, hi in _PIPER_LIMITS_TENTH_DEG
)

_OPENARM_GRIPPER_CLOSED_DEG = 0.0
_OPENARM_GRIPPER_OPEN_DEG = -60.0  # -1.0471975511965976 rad

LAYOUT_SPECS: dict[str, dict[str, Any]] = {
    "bi_piper_pika_follower": {
        "expected_robot": "piper_pika",
        "arm_encoding": "radians",
        "gripper_encoding": "meters",
        "default_max_width_m": 0.10,
        "names": [
            "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
            "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos", "left_gripper.width_m",
            "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
            "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos", "right_gripper.width_m",
        ],
    },
    "bi_piper_follower": {
        "expected_robot": "piper",
        "arm_encoding": "piper_m100_100",
        "gripper_encoding": "range_0_100",
        "default_max_width_m": 0.070,
        "names": [
            "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
            "left_forearm_roll.pos", "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
            "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
            "right_forearm_roll.pos", "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
        ],
    },
    "bi_openarm_follower": {
        "expected_robot": "openarmv1",
        "arm_encoding": "degrees",
        "gripper_encoding": "openarm_degrees",
        "default_max_width_m": 0.088,
        "names": [
            "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos", "left_joint_4.pos",
            "left_joint_5.pos", "left_joint_6.pos", "left_joint_7.pos", "left_gripper.pos",
            "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos", "right_joint_4.pos",
            "right_joint_5.pos", "right_joint_6.pos", "right_joint_7.pos", "right_gripper.pos",
        ],
    },
}


def encode_layout_vectors(
    arr: np.ndarray,
    layout: str,
    *,
    max_width_m: float | None = None,
) -> np.ndarray:
    """Convert canonical `(T, D)` `[left_arm_rad, left_w_m, right_arm_rad, right_w_m]` into plugin units."""
    spec = LAYOUT_SPECS[layout]
    out = np.asarray(arr, dtype=np.float64).copy()
    half = out.shape[1] // 2
    n_arm = half - 1
    w_max = float(max_width_m if max_width_m is not None and max_width_m > 0 else spec["default_max_width_m"])

    arm_enc = spec["arm_encoding"]
    grip_enc = spec["gripper_encoding"]

    for side_idx in range(2):
        col0 = side_idx * half
        arm_slice = out[:, col0 : col0 + n_arm]
        g_col = col0 + n_arm

        if arm_enc == "piper_m100_100":
            if n_arm != len(_PIPER_SIGNS):
                raise ValueError(f"bi_piper_follower expects 6 arm joints per side, got {n_arm}")
            for j in range(n_arm):
                lo, hi = _PIPER_LIMITS_RAD[j]
                signed = _PIPER_SIGNS[j] * arm_slice[:, j]
                arm_slice[:, j] = (signed - lo) * 200.0 / (hi - lo) - 100.0
        elif arm_enc == "degrees":
            arm_slice[:] = np.degrees(arm_slice)
        elif arm_enc == "radians":
            pass
        else:
            raise ValueError(f"Unsupported arm_encoding: {arm_enc!r}")

        frac = np.clip(out[:, g_col] / w_max, 0.0, 1.0)
        if grip_enc == "meters":
            pass
        elif grip_enc == "range_0_100":
            out[:, g_col] = 100.0 * frac
        elif grip_enc == "openarm_degrees":
            out[:, g_col] = _OPENARM_GRIPPER_CLOSED_DEG + frac * (
                _OPENARM_GRIPPER_OPEN_DEG - _OPENARM_GRIPPER_CLOSED_DEG
            )
        else:
            raise ValueError(f"Unsupported gripper_encoding: {grip_enc!r}")

    return out.astype(np.float32)


def export_external_follower_layout(
    joint_dataset_dir: str | Path,
    *,
    layout: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    if layout not in LAYOUT_SPECS:
        raise ValueError(f"Unknown layout={layout!r}; expected one of {tuple(LAYOUT_SPECS)}")
    spec = LAYOUT_SPECS[layout]
    src = Path(joint_dataset_dir)
    dst = Path(output_dir)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("robot") != spec["expected_robot"]:
        raise ValueError(
            f"Layout {layout} requires converted dataset for robot={spec['expected_robot']!r}, "
            f"got {info.get('robot')!r}"
        )
    episodes = json.loads((src / "meta" / "episodes.json").read_text(encoding="utf-8"))
    names = spec["names"]
    max_width_m = info.get("gripper_max_width_m") or spec["default_max_width_m"]

    from pikaumi.dataset.curation import _link_or_copy

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    has_videos: dict[str, bool] = {"left_wrist": False, "right_wrist": False}
    video_shapes_chw: dict[str, list[int]] = {}
    src_features = info.get("features") or {}
    global_frame_idx = 0

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        task_idx = int(ep.get("task_index", 0))
        stem = f"episode_{ep_idx:06d}"
        with np.load(src / "data" / f"{stem}.npz") as z:
            ts = z["timestamps_s"].copy()
            st_raw = z["observation_state"].copy()
            ac_raw = z["action"].copy()
            extra = {
                k: z[k].copy()
                for k in z.files
                if k not in ("timestamps_s", "observation_state", "action")
            }

        st = encode_layout_vectors(st_raw, layout, max_width_m=float(max_width_m))
        ac = encode_layout_vectors(ac_raw, layout, max_width_m=float(max_width_m))
        all_states.append(st)
        all_actions.append(ac)

        np.savez_compressed(
            dst / "data" / f"{stem}.npz",
            timestamps_s=ts,
            observation_state=st,
            action=ac,
            **extra,
        )
        T = len(ts)
        if HAS_PYARROW:
            done_flags = np.zeros(T, dtype=bool)
            if T > 0:
                done_flags[-1] = True
            tbl = pa.table(
                {
                    "index": np.arange(global_frame_idx, global_frame_idx + T, dtype=np.int64),
                    "frame_index": np.arange(T, dtype=np.int64),
                    "episode_index": np.full(T, ep_idx, dtype=np.int64),
                    "task_index": np.full(T, task_idx, dtype=np.int64),
                    "timestamp": ts,
                    "next.done": done_flags,
                    "observation.state": [r.tolist() for r in st],
                    "action": [r.tolist() for r in ac],
                }
            )
            chunk_dir = dst / "data" / f"chunk-{ep_idx // 1000:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_pq = chunk_dir / f"{stem}.parquet"
            pq.write_table(tbl, chunk_pq)
            _link_or_copy(chunk_pq, dst / "data" / f"{stem}.parquet")
        global_frame_idx += T

        for cam in ("left_wrist", "right_wrist"):
            vid_src = src / "videos" / cam / f"{stem}.mp4"
            if vid_src.is_file():
                has_videos[cam] = True
                _link_or_copy(vid_src, dst / "videos" / f"observation.images.{cam}" / f"{stem}.mp4")
                _link_or_copy(vid_src, dst / "videos" / cam / f"{stem}.mp4")
                vkey = f"observation.images.{cam}"
                if vkey not in video_shapes_chw:
                    detected = _detect_video_shape_chw(vid_src, src_features.get(vkey))
                    video_shapes_chw[vkey] = detected

    # Copy tasks.jsonl if present
    src_tasks = src / "meta" / "tasks.jsonl"
    if src_tasks.is_file():
        dst_tasks = dst / "meta" / "tasks.jsonl"
        if src_tasks.resolve() != dst_tasks.resolve():
            shutil.copy2(src_tasks, dst_tasks)

    # Recompute quantile stats for the exported units (save FULL dict matching LeRobot v3.0)
    if all_states:
        st_cat = np.concatenate(all_states, axis=0)
        ac_cat = np.concatenate(all_actions, axis=0)
        stats = {
            "observation.state": _feature_stats(st_cat),
            "action": _feature_stats(ac_cat),
        }
        save_json_atomic(dst / "meta" / "stats.json", stats)
        save_json_atomic(dst / "meta" / "abs_action_norm_stats.json", stats)

    out_info = dict(info)
    out_info["external_layout"] = layout
    out_info["arm_encoding"] = spec["arm_encoding"]
    out_info["gripper_encoding"] = spec["gripper_encoding"]
    out_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        "action": {"dtype": "float32", "shape": [len(names)], "names": names},
    }
    for cam, present in has_videos.items():
        if present:
            vkey = f"observation.images.{cam}"
            features[vkey] = {
                "dtype": "video",
                "shape": video_shapes_chw.get(vkey, [3, 480, 640]),
                "names": ["channels", "height", "width"],
            }
    out_info["features"] = features
    save_json_atomic(dst / "meta" / "info.json", out_info)
    save_json_atomic(dst / "meta" / "episodes.json", episodes)
    return out_info


def _detect_video_shape_chw(vid_path: Path, src_feat: dict[str, Any] | None) -> list[int]:
    """Return `[3, height, width]` (`CHW`) from actual video frame or source metadata."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(vid_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if h > 0 and w > 0:
                return [3, h, w]
        else:
            cap.release()
    except Exception:
        pass
    if isinstance(src_feat, dict) and isinstance(src_feat.get("shape"), list) and len(src_feat["shape"]) == 3:
        shp = [int(v) for v in src_feat["shape"]]
        if shp[0] == 3:
            return shp
        if shp[2] == 3:
            return [3, shp[0], shp[1]]
    return [3, 480, 640]


def _feature_stats(arr: np.ndarray) -> dict[str, Any]:
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

