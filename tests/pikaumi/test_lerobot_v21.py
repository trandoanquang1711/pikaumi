"""Offline unit tests for `pika convert-v21` (LeRobot v2.1 joint-action exporter)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pikaumi.cli import main


def _write_synthetic_raw_episode(ep_dir: Path, n_steps: int = 12, fps: int = 30) -> None:
    for rel in (
        "localization/pose/pika_l",
        "localization/pose/pika_r",
        "gripper/encoder/pika_l",
        "gripper/encoder/pika_r",
        "camera/color/pikaFisheyeCamera_l",
        "camera/color/pikaFisheyeCamera_r",
    ):
        (ep_dir / rel).mkdir(parents=True, exist_ok=True)

    (ep_dir / "instructions.json").write_text(
        json.dumps({"instructions": ["pick up synthetic cube"]}), encoding="utf-8"
    )
    img = np.full((64, 80, 3), 120, dtype=np.uint8)
    for k in range(n_steps):
        ts = f"{1000.0 + k / float(fps):.6f}"
        # Keep poses comfortably inside Piper workspace around tabletop origin
        pl = {"x": -0.22, "y": 0.05 + 0.002 * k, "z": 0.16, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        pr = {"x": 0.22, "y": 0.05 + 0.002 * k, "z": 0.16, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        (ep_dir / f"localization/pose/pika_l/{ts}.json").write_text(json.dumps(pl), encoding="utf-8")
        (ep_dir / f"localization/pose/pika_r/{ts}.json").write_text(json.dumps(pr), encoding="utf-8")
        (ep_dir / f"gripper/encoder/pika_l/{ts}.json").write_text(json.dumps({"distance": 0.05}), encoding="utf-8")
        (ep_dir / f"gripper/encoder/pika_r/{ts}.json").write_text(json.dumps({"distance": 0.06}), encoding="utf-8")
        cv2.imwrite(str(ep_dir / f"camera/color/pikaFisheyeCamera_l/{ts}.jpg"), img)
        cv2.imwrite(str(ep_dir / f"camera/color/pikaFisheyeCamera_r/{ts}.jpg"), img)


def test_convert_v21_cli_and_schema(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_root = tmp_path / "raw"
    _write_synthetic_raw_episode(raw_root / "episode0", n_steps=10)
    _write_synthetic_raw_episode(raw_root / "episode1", n_steps=10)

    out_dir = tmp_path / "v21_out"
    rc = main(
        [
            "convert-v21",
            str(raw_root),
            "--output-dir",
            str(out_dir),
            "--robot",
            "piper_pika",
            "--exclude",
            "1",
            "--min-segment-seconds",
            "0.2",
            "--workers",
            "1",
        ]
    )
    assert rc == 0

    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 1
    assert info["total_videos"] == 2
    assert info["pika"]["excluded_raw_episodes"] == [1]

    pq_path = out_dir / "data" / "chunk-000" / "episode_000000.parquet"
    assert pq_path.is_file()
    tbl = pq.read_table(pq_path)
    assert b"huggingface" in (tbl.schema.metadata or {})
    assert pa.types.is_fixed_size_list(tbl.schema.field("observation.state").type)
    assert tbl.schema.field("observation.state").type.list_size == 14
    assert (out_dir / "videos" / "chunk-000" / "observation.images.left_wrist" / "episode_000000.mp4").is_file()
    assert (out_dir / "videos" / "chunk-000" / "observation.images.right_wrist" / "episode_000000.mp4").is_file()
    assert (out_dir / "meta" / "episodes_stats.jsonl").is_file()
    assert (out_dir / "meta" / "abs_action_norm_stats.json").is_file()
    assert (out_dir / "pika_source_reports.json").is_file()
