"""Offline tests for strictly read-only Layer 5 3-point tabletop calibration (`capture-table-robot`)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.calibration.table_capture import (
    ReadOnlyPiperJointListener,
    capture_table_robot_three_points,
    compute_tcp_position_from_arm_joints,
    parse_manual_point_input,
)
from pikaumi.cli import main
from pikaumi.real.piper_driver import MDEG_TO_RAD
from pikaumi.robots.kinematics import BimanualIKSolver
from pikaumi.robots.registry import load_robot_profile
from pikaumi.robots.utils import pose7_to_mat


class _StrictReadOnlyPyAgxMock:
    """Mock pyAgxArm handle that fails hard if ANY motion/enable/disable/reset method is called."""

    def __init__(self, joint_seq_rad: list[np.ndarray]) -> None:
        self._seq = list(joint_seq_rad)
        self._idx = 0
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def get_joint_angles(self) -> SimpleNamespace:
        q = self._seq[min(self._idx, len(self._seq) - 1)]
        self._idx += 1
        return SimpleNamespace(msg=q.tolist(), timestamp=100.0 + self._idx * 0.01)

    def __getattr__(self, name: str):
        raise AssertionError(f"FORBIDDEN TX/CONTROL CALL on read-only Layer 5 listener: {name}")


class _StrictReadOnlyPiperSdkMock:
    """Mock piper_sdk handle that fails hard if ANY motion/enable/disable/reset method is called."""

    def __init__(self, q_rad: np.ndarray) -> None:
        self._q_rad = np.asarray(q_rad, dtype=np.float64)
        self.connected = False
        self.disconnected = False

    def ConnectPort(self) -> None:
        self.connected = True

    def DisconnectPort(self) -> None:
        self.disconnected = True

    def GetArmJointMsgs(self) -> SimpleNamespace:
        units = np.round(self._q_rad / MDEG_TO_RAD).astype(int)
        js = SimpleNamespace(
            joint_1=int(units[0]),
            joint_2=int(units[1]),
            joint_3=int(units[2]),
            joint_4=int(units[3]),
            joint_5=int(units[4]),
            joint_6=int(units[5]),
        )
        return SimpleNamespace(joint_state=js, time_stamp=123.45)

    def __getattr__(self, name: str):
        raise AssertionError(f"FORBIDDEN TX/CONTROL CALL on read-only piper_sdk listener: {name}")


def test_readonly_can_capture_pyagxarm_recovers_exact_table_pose_with_zero_tx(tmp_path: Path) -> None:
    prof = load_robot_profile("piper_pika")
    solver = BimanualIKSolver(prof)

    # Solve 3 reachable left-arm joint configurations touching 3 exact points on a known table plane
    p0_gt = np.array([0.34, 0.05, 0.02], dtype=np.float64)
    px_gt = np.array([0.34, -0.10, 0.02], dtype=np.float64)  # +X table -> -Y robot (150 mm)
    py_gt = np.array([0.49, 0.05, 0.02], dtype=np.float64)  # +Y table -> +X robot (150 mm)

    fk_home = solver.fk_poses7(prof.home_q)
    q_pts: list[np.ndarray] = []
    xyz_pts: list[np.ndarray] = []
    for target_p in (p0_gt, px_gt, py_gt):
        tgt = {
            "left": np.concatenate([target_p, fk_home["left"][3:7]]),
            "right": fk_home["right"].copy(),
        }
        q_sol = solver.solve_frame_lm(prof.home_q, tgt, apply_delta_cap=False)
        q_arm = q_sol[list(prof.arms["left"].arm_actuated_indices)]
        q_pts.append(q_arm)
        xyz_pts.append(compute_tcp_position_from_arm_joints(prof, "left", q_arm))

    # Build a mock handle yielding 1 probe sample + 3 samples per point
    seq = [q_pts[0]] + [q_pts[0]] * 3 + [q_pts[1]] * 3 + [q_pts[2]] * 3
    mock_handle = _StrictReadOnlyPyAgxMock(seq)

    def _factory(port: str, *, sdk_backend: str, firmware: str) -> ReadOnlyPiperJointListener:
        return ReadOnlyPiperJointListener(
            port,
            sdk_backend="pyagxarm",
            firmware=firmware,
            sdk_factory=lambda _p, _fw: mock_handle,
        )

    out_yaml = tmp_path / "piper_pika_local.yaml"
    out_contacts = tmp_path / "contacts.json"
    res = capture_table_robot_three_points(
        robot="piper_pika",
        side="left",
        source="can",
        output_yaml=out_yaml,
        contacts_json=out_contacts,
        samples_per_point=3,
        verified=True,
        input_fn=lambda _prompt: "",
        output_fn=lambda _msg: None,
        listener_factory=_factory,
    )

    assert mock_handle.connected and mock_handle.disconnected
    tcal = load_table_calibration("piper_pika", path=out_yaml, require_physical=True)
    assert tcal.scope == "physical" and tcal.verified is True
    T = pose7_to_mat(tcal.T_robot_table_pose7)
    assert np.allclose(T[:3, 3], xyz_pts[0], atol=1e-5)
    assert T[2, 2] > 0.99  # table normal pointing +Z up
    assert res.metrics["table_tilt_deg"] < 2.0
    assert abs(res.metrics["xy_touch_angle_deg"] - 90.0) < 3.0


def test_readonly_piper_sdk_listener_reads_si_radians_with_zero_tx() -> None:
    q_gt = np.array([0.1, 1.2, -1.1, 0.05, 0.4, -0.2], dtype=np.float64)
    mock_arm = _StrictReadOnlyPiperSdkMock(q_gt)
    listener = ReadOnlyPiperJointListener(
        "can0",
        sdk_backend="piper_sdk",
        sdk_factory=lambda _p, _fw: mock_arm,
    )
    assert listener.connect() == "piper_sdk"
    q_med, spread = listener.sample_stationary_q_rad(samples=3, interval_s=0.0)
    listener.disconnect()
    assert np.allclose(q_med, q_gt, atol=1e-4)
    assert spread < 0.01
    assert mock_arm.disconnected


def test_manual_and_mock_layer5_capture_cli(tmp_path: Path) -> None:
    prof = load_robot_profile("piper_pika")

    # Parse direct xyz
    xyz_direct, q_none = parse_manual_point_input("0.34, 0.00, -0.03", prof, "left")
    assert q_none is None
    assert np.allclose(xyz_direct, [0.34, 0.0, -0.03])

    # Parse deg: 6 arm joints -> exact URDF FK
    q_home_left = prof.home_q[list(prof.arms["left"].arm_actuated_indices)]
    deg_str = "deg: " + " ".join(f"{v:.4f}" for v in np.rad2deg(q_home_left))
    xyz_fk, q_back = parse_manual_point_input(deg_str, prof, "left")
    assert q_back is not None
    assert np.allclose(xyz_fk, compute_tcp_position_from_arm_joints(prof, "left", q_home_left), atol=1e-4)

    # CLI mock run (offline, zero CAN I/O)
    out_yaml = tmp_path / "mock_table.yaml"
    out_contacts = tmp_path / "mock_contacts.json"
    rc = main(
        [
            "calibrate",
            "capture-table-robot",
            "--robot",
            "piper_pika",
            "--source",
            "mock",
            "--contacts-json",
            str(out_contacts),
            "--output",
            str(out_yaml),
        ]
    )
    assert rc == 0
    assert out_yaml.is_file()
    assert json.loads(out_contacts.read_text(encoding="utf-8"))["robot"] == "piper_pika"
    # Mock source never marks physical calibration verified
    tcal = load_table_calibration("piper_pika", path=out_yaml, require_physical=False)
    assert tcal.verified is False
