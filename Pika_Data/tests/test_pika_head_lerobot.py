"""Real CPU video/Parquet round-trip; requires the dataset dependency environment."""
import json
from pathlib import Path
import shutil
import sys

import pytest


def test_three_camera_lerobot_roundtrip_and_mixed_schema_rejection(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from datasets import config as datasets_config
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "hf_cache")
    import numpy as np
    from test_pika_pipeline import synthetic_episode, pipeline, vla
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = tmp_path / "raw"
    raw.mkdir()
    episode = synthetic_episode(raw)
    shutil.copytree(episode / "camera/color/pikaDepthCamera_l", episode / "camera/color/headCamera")
    (episode / "capture_metadata.json").write_text(json.dumps({"head_camera": dict(enabled=True, enable_depth=False)}))
    exported = tmp_path / "export"
    pipeline.main(["--data-root", str(raw), "--output-dir", str(exported), "--episodes", "episode3",
                   "--vista-only", "--image-size", "128", "96"])
    source = exported / "pika_source_20hz.hdf5"
    calibration = tmp_path / "tcp"
    calibration.mkdir()
    for side in ("left", "right"):
        (calibration / f"{side}_pika_tcp.json").write_text(json.dumps({"transform": np.eye(4).tolist()}))
    prepared = tmp_path / "prepared"
    vla.prepare(source, episode, calibration, prepared, min_segment_seconds=0.1)
    output = tmp_path / "lerobot"
    vla.export_lerobot([prepared], output, "local/head_test")
    # Load after relocating the completed dataset; original source files are absent.
    relocated = tmp_path / "relocated"
    shutil.move(output, relocated)
    hidden = source.with_suffix(".hidden")
    source.rename(hidden)
    try:
        dataset = LeRobotDataset("local/head_test", root=relocated, revision="v3.0", video_backend="pyav", use_delta_action=False)
        assert set(dataset.meta.camera_keys) == {"observation.images.left_wrist", "observation.images.right_wrist", "observation.images.head"}
        sample = dataset[0]
        assert tuple(sample["observation.images.head"].shape) == (3, 96, 128)
        assert tuple(sample["action"].shape) == (14,)
        assert tuple(sample["observation.state"].shape) == (11,)
        assert sample["task"] == "fold a towel"
        assert sample["observation.images.head"][0].mean() > .8  # red, not BGR blue
        report = json.loads((relocated / "pika_source_reports.json").read_text())[0]
        assert (relocated / report["provenance"]["timestamps"]).is_file()
        with np.load(relocated / report["provenance"]["timestamps"]) as times:
            assert len(times["head_timestamp_ns"]) == len(dataset)
        del dataset
    finally:
        hidden.rename(source)

    tcp_output = tmp_path / "pika-tcp"
    vla.export_lerobot([prepared], tcp_output, "local/pika_tcp", representation="pika-tcp")
    tcp_dataset = LeRobotDataset("local/pika_tcp", root=tcp_output, revision="v3.0",
                                  video_backend="pyav", use_delta_action=False)
    assert tcp_dataset.meta.info["pika"]["reference_frame"] == "source_vive_tracking_world"
    assert set(tcp_dataset.meta.camera_keys) == {"observation.images.left_wrist",
                                                "observation.images.right_wrist", "observation.images.head"}
    with np.load(prepared / "labels.npz") as labels, h5py.File(source, "r") as file:
        rows = labels["source_row"]
        group = file["source_episodes/episode3"]
        expected = np.column_stack([group["pose"][:, 0], group["pose"][:, 1], group["width"][:]])
        np.testing.assert_allclose(np.asarray(tcp_dataset.hf_dataset["observation.state"]), expected[rows], atol=1e-7)
        np.testing.assert_allclose(np.asarray(tcp_dataset.hf_dataset["action"]), expected[rows + 1], atol=1e-7)

    legacy_source = tmp_path / "legacy.hdf5"
    shutil.copyfile(source, legacy_source)
    with h5py.File(legacy_source, "r+") as file:
        del file["source_episodes/episode3/images/head_rgb"]
    legacy = tmp_path / "legacy_labels"
    shutil.copytree(prepared, legacy)
    report = json.loads((legacy / "report.json").read_text())
    report.update(source_hdf5=str(legacy_source), head_sync=None)
    report["cameras"].pop("observation.images.head")
    (legacy / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="image dimensions differ"):
        vla.export_lerobot([legacy, prepared], tmp_path / "mixed", "local/mixed_test")
    assert not (tmp_path / "mixed").exists()
