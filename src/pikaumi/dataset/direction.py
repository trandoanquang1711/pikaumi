"""Reversed / scene-reset episode detector (`pika dataset direction`) via MAD-weighted start/end change."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import load_dataset_episodes, load_episode_raw


def _has_complete_thumbs(root: Path, ep_idx: int) -> bool:
    thumb_path = root / "frames" / f"episode_{ep_idx:06d}_thumbs.npz"
    if not thumb_path.is_file():
        return False
    try:
        with np.load(thumb_path) as z:
            return all(k in z for k in ("left_start", "left_end", "right_start", "right_end"))
    except Exception:
        return False


def _episode_change_descriptor(root: Path, ep_idx: int, *, use_thumbs: bool) -> np.ndarray:
    """Compute signed start-to-end descriptor from visual thumbnails (when uniformly present) + pose displacement."""
    parts: list[np.ndarray] = []
    if use_thumbs:
        thumb_path = root / "frames" / f"episode_{ep_idx:06d}_thumbs.npz"
        with np.load(thumb_path) as z:
            for side in ("left", "right"):
                diff = z[f"{side}_end"].astype(np.float64) - z[f"{side}_start"].astype(np.float64)
                parts.append(diff.ravel() / 255.0)
    ep = load_episode_raw(root, ep_idx)
    state = ep["observation_state"]
    w = max(1, min(5, len(state) // 6))
    pose_delta = np.mean(state[-w:, [0, 1, 2, 8, 9, 10]], axis=0) - np.mean(state[:w, [0, 1, 2, 8, 9, 10]], axis=0)
    parts.append(pose_delta * 10.0)
    return np.concatenate(parts)


def analyze_dataset_direction(dataset_dir: str | Path) -> dict[str, Any]:
    """Flag reversed / reset episodes whose start->end change opposes the MAD-weighted dataset consensus."""
    root = Path(dataset_dir)
    episodes = load_dataset_episodes(root)
    if len(episodes) < 3:
        report = {
            "dataset_dir": str(root),
            "num_episodes": len(episodes),
            "consensus_defined": False,
            "flagged_reversed_episodes": [],
            "episodes": [
                {"episode_index": int(e["episode_index"]), "cosine_to_consensus": 1.0, "reversed": False}
                for e in episodes
            ],
        }
        save_json_atomic(root / "meta" / "direction_report.json", report)
        return report

    ep_ids = [int(e["episode_index"]) for e in episodes]
    use_thumbs = all(_has_complete_thumbs(root, idx) for idx in ep_ids)
    descs = np.stack([_episode_change_descriptor(root, idx, use_thumbs=use_thumbs) for idx in ep_ids], axis=0)
    med = np.median(descs, axis=0)
    mad = np.median(np.abs(descs - med), axis=0) + 1e-3
    weighted_consensus = med / mad
    c_norm = float(np.linalg.norm(weighted_consensus))
    if c_norm < 1e-6:
        # Closed-loop task (start == end) or symmetric null consensus: do not fabricate a fake direction.
        report = {
            "dataset_dir": str(root),
            "num_episodes": len(episodes),
            "consensus_defined": False,
            "flagged_reversed_episodes": [],
            "episodes": [
                {"episode_index": idx, "cosine_to_consensus": 0.0, "reversed": False}
                for idx in ep_ids
            ],
        }
        save_json_atomic(root / "meta" / "direction_report.json", report)
        return report

    per_ep: list[dict[str, Any]] = []
    flagged: list[int] = []
    for i, ep_idx in enumerate(ep_ids):
        v = descs[i] / mad
        v_norm = float(np.linalg.norm(v))
        cos_sim = float(np.dot(v, weighted_consensus) / max(v_norm * c_norm, 1e-9))
        is_rev = cos_sim < 0.0
        if is_rev:
            flagged.append(ep_idx)
        per_ep.append(
            {
                "episode_index": ep_idx,
                "cosine_to_consensus": round(cos_sim, 5),
                "reversed": is_rev,
            }
        )

    report = {
        "dataset_dir": str(root),
        "num_episodes": len(per_ep),
        "consensus_defined": True,
        "flagged_reversed_episodes": flagged,
        "episodes": per_ep,
    }
    save_json_atomic(root / "meta" / "direction_report.json", report)
    return report
