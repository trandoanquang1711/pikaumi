"""Multi-session `pika_raw` dataset merger (`pika dataset merge`) with provenance tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pikaumi.config import save_json_atomic
from pikaumi.dataset.curation import _link_or_copy
from pikaumi.dataset.raw import (
    load_dataset_episodes,
    load_dataset_info,
    load_episode_raw,
    write_episode_table,
)


def merge_datasets(
    source_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    if not source_dirs:
        raise ValueError("merge_datasets requires at least one source dataset")
    dst = Path(output_dir)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    base_info = load_dataset_info(source_dirs[0])
    fps = int(base_info["fps"])
    merged_episodes: list[dict[str, Any]] = []
    provenance_sources: list[dict[str, Any]] = []
    global_idx = 0

    src_paths = [Path(s) for s in source_dirs]
    name_counts: dict[str, int] = {}
    for p in src_paths:
        name_counts[p.name] = name_counts.get(p.name, 0) + 1
    seen_source_episodes: set[str] = set()

    for src_idx, src in enumerate(src_paths):
        s_info = load_dataset_info(src)
        if int(s_info["fps"]) != fps:
            raise ValueError(f"FPS mismatch merging {src}: {s_info['fps']} != {fps}")
        eps = load_dataset_episodes(src)
        provenance_sources.append(
            {
                "source_index": src_idx,
                "source_dir": str(src),
                "num_episodes": len(eps),
                "calibration": s_info.get("calibration"),
            }
        )
        s_cal = s_info.get("calibration")
        has_name_collision = name_counts[src.name] > 1
        src_prefix = f"s{src_idx:02d}_{src.name}" if has_name_collision else src.name
        for ep_meta in eps:
            old_idx = int(ep_meta["episode_index"])
            raw_ep = load_episode_raw(src, old_idx)
            write_episode_table(
                global_idx,
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
                _link_or_copy(thumb_src, dst / "frames" / f"episode_{global_idx:06d}_thumbs.npz")
            for cam in ("left_wrist", "right_wrist"):
                vid_src = src / "videos" / cam / f"episode_{old_idx:06d}.mp4"
                if vid_src.is_file():
                    _link_or_copy(vid_src, dst / "videos" / cam / f"episode_{global_idx:06d}.mp4")
            new_meta = dict(ep_meta)
            new_meta["episode_index"] = global_idx
            orig_src_ep = str(ep_meta.get("source_episode", f"episode{old_idx}"))
            if has_name_collision:
                cand_src_ep = f"{src_prefix}/{orig_src_ep}"
            elif "/" not in orig_src_ep:
                cand_src_ep = f"{src_prefix}/{orig_src_ep}"
            else:
                cand_src_ep = orig_src_ep
            if cand_src_ep in seen_source_episodes:
                cand_src_ep = f"s{src_idx:02d}_{src.name}/{orig_src_ep}"
            seen_source_episodes.add(cand_src_ep)
            new_meta["source_episode"] = cand_src_ep
            if "calibration" not in new_meta and s_cal is not None:
                new_meta["calibration"] = s_cal
            if task is not None:
                new_meta["task"] = task
            new_meta["merged_from"] = {
                "source_index": src_idx,
                "dataset": src.name,
                "source_dir": str(src),
                "episode_index": old_idx,
            }
            merged_episodes.append(new_meta)
            global_idx += 1

    out_info = dict(base_info)
    if task is not None:
        out_info["task"] = task
    out_info["num_episodes"] = len(merged_episodes)
    out_info["total_frames"] = sum(int(e["num_frames"]) for e in merged_episodes)
    out_info["per_episode_calibration"] = True
    save_json_atomic(dst / "meta" / "info.json", out_info)
    save_json_atomic(dst / "meta" / "episodes.json", merged_episodes)
    prov = {"output_dir": str(dst), "num_episodes": len(merged_episodes), "sources": provenance_sources}
    save_json_atomic(dst / "meta" / "merge_provenance.json", prov)
    return prov
