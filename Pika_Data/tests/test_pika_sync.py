"""Preflight rejects missing streams, clock errors and unusable camera pairing."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_tools"))
from pika_sync import analyze
from pika_session import CaptureClient
from prepare_pika_dataset import timestamp_grid


def live_sample():
    start = 1794022000000000000
    streams = {}
    for side, skew in (("left", 0), ("right", 2_000_000)):
        for kind, fps in (("image", 30), ("pose", 120), ("gripper", 125)):
            timestamps = timestamp_grid(start + (skew if kind == "image" else 0),
                                        start + 3 * 10**9 + (skew if kind == "image" else 0), fps)
            streams[f"{side}_{kind}"] = np.column_stack([timestamps, timestamps + 2_000_000]).tolist()
    return {"streams": streams, "tracking": {"left": [True] * 30, "right": [True] * 30},
            "invalid_images": {}, "ended_at_ns": start + 3_030_000_000}


def test_live_sync_accepts_fresh_well_paired_inputs():
    report = analyze(live_sample())
    assert report["ready"]
    assert report["valid_fraction"] == 1
    assert report["wrist_skew_ms_p95"] == 2
    assert report["longest_valid_seconds"] >= 2


@pytest.mark.parametrize("fault", ["missing_image", "duplicate_pose", "stale_gripper", "tracking_lost", "empty_image"])
def test_live_sync_rejects_faults_that_tracking_snapshots_alone_would_miss(fault):
    sample = live_sample()
    if fault == "missing_image":
        sample["streams"]["right_image"] = []
    elif fault == "duplicate_pose":
        sample["streams"]["left_pose"][3][0] = sample["streams"]["left_pose"][2][0]
    elif fault == "stale_gripper":
        for row in sample["streams"]["right_gripper"]:
            row[0] -= 10**9
    elif fault == "tracking_lost":
        sample["tracking"]["right"][4] = False
    else:
        sample["invalid_images"]["left_image"] = 1
    report = analyze(sample)
    assert not report["ready"]
    assert report["errors"]


def test_live_sync_rejects_camera_pairs_outside_skew_limit():
    sample = live_sample()
    for row in sample["streams"]["right_image"]:
        row[0] += 14_000_000
        row[1] += 14_000_000
    report = analyze(sample)
    assert not report["ready"]
    assert report["valid_fraction"] == 0


def test_capture_does_not_launch_recorder_when_preflight_fails(tmp_path, monkeypatch):
    checked = []
    def run(command, **kwargs):
        checked.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr("pika_session.subprocess.run", run)
    monkeypatch.setattr("pika_session.subprocess.Popen", lambda *args, **kwargs: pytest.fail("recorder started"))
    client = CaptureClient("sensors", "/raw", "/root/pika_ros", tmp_path / "capture.log",
                           tmp_path / "raw", {"fps": 10, "max_image_offset_ms": 70.0})
    with pytest.raises(ValueError, match="alignment"):
        client.start(10, "fold a towel")
    assert "--check-streams" in checked[0]
    assert checked[0][checked[0].index("--fps") + 1] == "10"
    assert checked[0][checked[0].index("--max-image-offset-ms") + 1] == "70.0"
    assert not client.active
