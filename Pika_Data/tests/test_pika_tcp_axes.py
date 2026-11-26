"""Detect axis inversions and validate a fitted frame at a held-out orientation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pika_tcp_axes", ROOT / "data_tools/pika_tcp_axes.py")
axes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(axes)


def measurements():
    correction = Rotation.from_euler("xyz", [35, -20, 75], degrees=True).as_matrix()
    samples = {}
    for group, rotation in zip("ab", Rotation.from_euler("xyz", [[5, 15, 25], [35, -25, 80]], degrees=True)):
        for index, label in enumerate("0xyz"):
            position = np.array([0.2, -0.3, 0.4])
            if index:
                position += rotation.apply(correction[:, index - 1] * 0.08)
            pose = np.r_[position, rotation.as_quat()]
            poses = np.repeat(pose[None], 30, axis=0)
            poses[::2, 3:] *= -1  # quaternion signs do not change orientation
            samples[group + label] = {"poses": poses.tolist(), "device": "pika", "side": "left",
                                      "topic": "/pika_pose_l", "frame": "base_link"}
    calibration = {"transform": np.eye(4).tolist(), "quality_status": "provisional"}
    return samples, calibration, correction


def test_axis_fit_and_independent_validation_recover_reported_to_tcp_rotation():
    samples, calibration, expected = measurements()
    report, candidate = axes.solve_axes(samples, calibration)
    assert report["numerical_checks_passed"]
    np.testing.assert_allclose(np.array(candidate["transform"])[:3, :3], expected, atol=1e-12)
    np.testing.assert_allclose(calibration["transform"], np.eye(4))
    assert report["validation_orientation_change_deg"] >= 20


def test_axis_fit_rejects_signed_axis_reflection():
    samples, calibration, _ = measurements()
    origin = np.array(samples["a0"]["poses"])[0, :3]
    reversed_poses = np.array(samples["ay"]["poses"])
    reversed_poses[:, :3] = 2 * origin - reversed_poses[:, :3]
    samples["ay"]["poses"] = reversed_poses.tolist()
    with pytest.raises(ValueError, match="reversed axis"):
        axes.solve_axes(samples, calibration)


def test_validation_at_another_orientation_catches_inconsistent_physical_axes():
    samples, calibration, _ = measurements()
    samples["bx"], samples["by"] = samples["by"], samples["bx"]
    samples["by"], samples["bz"] = samples["bz"], samples["by"]
    report, _ = axes.solve_axes(samples, calibration)
    assert not report["numerical_checks_passed"]
    assert report["max_axis_error_deg"] > 80


def test_snapshot_rejects_motion_and_invalid_quaternions():
    samples, _, _ = measurements()
    moved = np.array(samples["a0"]["poses"])
    moved[-1, 0] += 0.01
    with pytest.raises(ValueError, match="gripper moved"):
        axes.stationary_pose({"poses": moved.tolist()})
    moved[:, 3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        axes.stationary_pose({"poses": moved.tolist()})


@pytest.mark.parametrize("angle_deg,length_m", [(5.1, 0.08), (0, 0.16)])
def test_translation_rejects_rotation_or_wrong_measured_distance(angle_deg, length_m):
    origin = np.r_[np.zeros(3), Rotation.identity().as_quat()]
    endpoint = np.r_[[length_m, 0, 0], Rotation.from_euler("z", angle_deg, degrees=True).as_quat()]
    with pytest.raises(ValueError, match="Keep the tool orientation fixed"):
        axes.translation_delta(origin, endpoint, np.zeros(3), 80, "ax")
