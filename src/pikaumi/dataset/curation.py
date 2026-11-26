"""Lossless dataset curation (`pika dataset curate`) preserving video streams without re-encoding."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import (
    load_dataset_episodes,
    load_dataset_info,
    load_episode_raw,
    write_episode_table,
)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            if src.resolve() == dst.resolve():
                return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def curate_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    robot: str | None = None,
    exclude_episodes: list[int] | None = None,
) -> dict[str, Any]:
    """Create a clean curated `pika_raw` dataset dropping QA-failed, reversed, or robot-unscreened episodes."""
    src = Path(dataset_dir)
    dst = Path(output_dir)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(src)
    episodes = load_dataset_episodes(src)
    rejected: dict[int, list[str]] = {int(e["episode_index"]): [] for e in episodes}

    for manual_idx in exclude_episodes or []:
        rejected.setdefault(int(manual_idx), []).append("manual_exclude")

    q_path = src / "meta" / "quality_report.json"
    if q_path.is_file():
        q_rep = json.loads(q_path.read_text(encoding="utf-8"))
        for item in q_rep.get("episodes", []):
            if not item.get("passed", True):
                rejected.setdefault(int(item["episode_index"]), []).extend(item.get("reasons", ["quality_failed"]))

    d_path = src / "meta" / "direction_report.json"
    if d_path.is_file():
        d_rep = json.loads(d_path.read_text(encoding="utf-8"))
        for idx in d_rep.get("flagged_reversed_episodes", []):
            rejected.setdefault(int(idx), []).append("reversed_direction")

    if robot is not None:
        s_path = src / "meta" / f"screen_{robot}.json"
        if s_path.is_file():
            s_rep = json.loads(s_path.read_text(encoding="utf-8"))
            for item in s_rep.get("episodes", []):
                if item.get("verdict") == "REJECT":
                    rejected.setdefault(int(item["episode_index"]), []).extend(
                        item.get("reasons", [f"screen_reject:{robot}"])
                    )

    kept_episodes: list[dict[str, Any]] = []
    dropped_summary: list[dict[str, Any]] = []
    new_idx = 0

    for ep_meta in episodes:
        old_idx = int(ep_meta["episode_index"])
        reasons = rejected.get(old_idx, [])
        if reasons:
            dropped_summary.append({"old_episode_index": old_idx, "reasons": reasons})
            continue

        raw_ep = load_episode_raw(src, old_idx)
        write_episode_table(
            new_idx,
            dst / "data",
            timestamps_s=raw_ep["timestamps_s"],
            state_16d=raw_ep["observation_state"],
            action_16d=raw_ep["action"],
            left_accurate=raw_ep["left_accurate"],
            right_accurate=raw_ep["right_accurate"],
            sync_skew_s=raw_ep["sync_skew_s"],
            lr_image_skew_s=raw_ep.get("lr_image_skew_s"),
        )

        thumb_src = src / "frames" / f"episode_{old_idx:06d}_thumbs.npz"
        if thumb_src.is_file():
            _link_or_copy(thumb_src, dst / "frames" / f"episode_{new_idx:06d}_thumbs.npz")

        for cam in ("left_wrist", "right_wrist"):
            vid_src = src / "videos" / cam / f"episode_{old_idx:06d}.mp4"
            if vid_src.is_file():
                _link_or_copy(vid_src, dst / "videos" / cam / f"episode_{new_idx:06d}.mp4")

        new_meta = dict(ep_meta)
        new_meta["episode_index"] = new_idx
        new_meta["curated_from_episode"] = old_idx
        kept_episodes.append(new_meta)
        new_idx += 1

    new_info = dict(info)
    new_info["num_episodes"] = len(kept_episodes)
    new_info["total_frames"] = sum(int(e["num_frames"]) for e in kept_episodes)
    save_json_atomic(dst / "meta" / "info.json", new_info)
    save_json_atomic(dst / "meta" / "episodes.json", kept_episodes)

    curation_manifest = {
        "source_dataset": str(src),
        "output_dataset": str(dst),
        "robot_filter": robot,
        "kept_episodes": len(kept_episodes),
        "dropped_episodes": dropped_summary,
    }
    save_json_atomic(dst / "meta" / "curation_manifest.json", curation_manifest)
    return curation_manifest
