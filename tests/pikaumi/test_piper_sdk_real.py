"""Comprehensive offline tests for direct `piper_sdk` SocketCAN driver and all 3 real execution flows."""

from __future__ import annotations

import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any

import numpy as np
import pytest

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.cli import main
from pikaumi.config import PACKAGE_DIR
from pikaumi.dataset.raw import generate_synthetic_bimanual_dataset
from pikaumi.dataset.umi_delta import compute_umi_delta_tensors
from pikaumi.real.base import ArmStateMode
from pikaumi.real.piper_driver import (
    MockPiperSdkArm,
    MockPyAgxArm,
    PiperCanBackend,
    PiperGripperRange,
    PiperSdkArm,
    PyAgxPiperArm,
    load_piper_can_settings,
    piper_mdeg_to_q,
    q_to_piper_mdeg,
    resolve_pyagxarm_firmware_profile,
)
from pikaumi.real.policy_bridge import (
    HttpJsonPolicyClient,
    MockPolicyClient,
    build_umi_observation_state_11d,
    run_policy_bridge_loop,
)
from pikaumi.real.replay_real import execute_real_trajectory
from pikaumi.real.streamer import AccelerationLimitedJointTrajectory
from pikaumi.real.teleop import (
    MockPikaPoseSource,
    PikaDualPoseSample,
    PikaLiveRetargeter,
    UdpJsonPikaPoseSource,
    run_teleop_loop,
)
from pikaumi.robots.registry import load_robot_profile
from pikaumi.robots.utils import mat_to_pose7, pose7_to_mat


def _make_mock_piper_backend(
    robot_name: str = "piper_pika",
    *,
    enable_arms: bool = True,
    ctrl_mode: int = 0x01,
    arm_status: int = 0,
    teach_status: int = 0,
    mode_feed: int = 0,
    rate_hz: float = 200.0,
) -> tuple[PiperCanBackend, dict[str, MockPiperSdkArm]]:
    prof = load_robot_profile(robot_name)
    base_settings = load_piper_can_settings(prof, sdk_backend="piper_sdk", enable_arms=enable_arms)
    settings = type(base_settings)(
        **{
            **base_settings.__dict__,
            "command_rate_hz": rate_hz,
            "home_max_joint_speed_deg_s": 180.0,
            "home_timeout_s": 5.0,
        }
    )
    fakes: dict[str, MockPiperSdkArm] = {}

    def _factory(side: str, port: str) -> PiperSdkArm:
        q_arm_init = prof.home_q[list(prof.arms[side].arm_actuated_indices)]
        fake = MockPiperSdkArm(
            port,
            initial_mdeg=q_to_piper_mdeg(q_arm_init),
            initial_gripper_microm=50_000,
            ctrl_mode=ctrl_mode,
            arm_status=arm_status,
            teach_status=teach_status,
            mode_feed=mode_feed,
        )
        fakes[side] = fake
        return PiperSdkArm(side, port, sdk_factory=lambda _p, f=fake: f)

    backend = PiperCanBackend(prof, settings, arm_factory=_factory)
    return backend, fakes


def _make_mock_pyagx_backend(
    robot_name: str = "piper_pika",
    *,
    enable_arms: bool = True,
    ctrl_mode: int = 0x01,
    arm_status: int = 0,
    teach_status: int = 0,
    mode_feedback: int = 0x01,
    rate_hz: float = 200.0,
) -> tuple[PiperCanBackend, dict[str, MockPyAgxArm]]:
    prof = load_robot_profile(robot_name)
    base_settings = load_piper_can_settings(prof, sdk_backend="pyagxarm", enable_arms=enable_arms)
    settings = type(base_settings)(
        **{
            **base_settings.__dict__,
            "command_rate_hz": rate_hz,
            "home_max_joint_speed_deg_s": 180.0,
            "home_timeout_s": 5.0,
        }
    )
    fakes: dict[str, MockPyAgxArm] = {}

    def _factory(side: str, port: str) -> PyAgxPiperArm:
        q_arm_init = tuple(float(v) for v in prof.home_q[list(prof.arms[side].arm_actuated_indices)])
        fake = MockPyAgxArm(
            port,
            firmware_profile="v189",
            initial_q_rad=q_arm_init,
            initial_gripper_m=0.05,
            ctrl_mode=ctrl_mode,
            arm_status=arm_status,
            teach_status=teach_status,
            mode_feedback=mode_feedback,
        )
        fakes[side] = fake
        return PyAgxPiperArm(side, port, firmware="S-V1.8-9", sdk_factory=lambda _p, _fw, f=fake: f)

    backend = PiperCanBackend(prof, settings, arm_factory=_factory)
    return backend, fakes


def test_ast_strict_zero_torque_release_or_reset_in_real_package() -> None:
    """Firmware S-V1.8-9: `src/pikaumi/real/*.py` must NEVER call any disable, reset, or E-stop API."""
    forbidden = {
        "DisableArm",
        "DisablePiper",
        "ResetArm",
        "Reset",
        "ResetPiper",
        "EmergencyStop",
        "electronic_emergency_stop",
        "disable_gripper",
    }
    real_dir = PACKAGE_DIR / "real"
    for py_file in sorted(real_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
                    raise AssertionError(f"Forbidden S-V1.8-9 call {node.func.attr} in {py_file}")
                if isinstance(node.func, ast.Name) and node.func.id in forbidden:
                    raise AssertionError(f"Forbidden S-V1.8-9 call {node.func.id} in {py_file}")


def test_piper_unit_conversions_and_gripper_geometry() -> None:
    q = np.array([0.0, 1.45, -1.35, 0.25, 0.45, -0.5], dtype=np.float64)
    mdeg = q_to_piper_mdeg(q)
    q_back = piper_mdeg_to_q(mdeg)
    assert np.allclose(q, q_back, atol=2e-5)

    # piper_pika: 100 mm = 100,000 um physical-width
    prof_pp = load_robot_profile("piper_pika")
    s_pp = load_piper_can_settings(prof_pp)
    assert s_pp.ports == {"left": "can0", "right": "can1"}
    assert s_pp.auto_enable is False
    assert s_pp.gripper_ranges["left"].max_microm == 100_000
    assert s_pp.gripper_ranges["left"].width_m_to_microm(0.08, retarget_mode="physical-width") == 80_000
    assert s_pp.gripper_ranges["left"].width_m_to_microm(0.12, retarget_mode="physical-width") == 100_000

    # stock piper: 70 mm = 70,000 um normalized
    prof_p = load_robot_profile("piper")
    s_p = load_piper_can_settings(prof_p)
    assert s_p.gripper_ranges["left"].max_microm == 70_000
    assert s_p.gripper_ranges["left"].width_m_to_microm(0.05, retarget_mode="normalized") == 35_000
    assert s_p.gripper_ranges["left"].width_m_to_microm(0.10, retarget_mode="normalized") == 70_000

    # Custom calibrated offset
    rng = PiperGripperRange(min_microm=2_000, max_microm=98_000)
    assert rng.width_m_to_microm(0.0, retarget_mode="physical-width") == 2_000
    assert rng.width_m_to_microm(0.10, retarget_mode="physical-width") == 98_000


def test_piper_sdk_arm_sv189_safety_startup_and_fault_rejection() -> None:
    # 1. auto_enable=False refuses to silently enable when arm is idle (ctrl_mode=0x00)
    fake_idle = MockPiperSdkArm("can0", ctrl_mode=0x00)
    arm_idle = PiperSdkArm("left", "can0", sdk_factory=lambda _p: fake_idle)
    with pytest.raises(RuntimeError, match="auto_enable=False"):
        arm_idle.connect(enable=False, startup_speed_percent=10, speed_percent=60)
    assert all(name != "EnablePiper" for name, _ in fake_idle.call_log)

    # 2. Teach mode / MIT mode / non-zero arm_status MUST raise and NEVER auto-reset
    for kwargs in (
        {"ctrl_mode": 0x02},
        {"teach_status": 1},
        {"mode_feed": 0xAD},
        {"arm_status": 4},
    ):
        fake_bad = MockPiperSdkArm("can0", **kwargs)
        arm_bad = PiperSdkArm("left", "can0", sdk_factory=lambda _p, f=fake_bad: f)
        with pytest.raises(RuntimeError, match="S-V1.8-9|MIT mode"):
            arm_bad.connect(enable=True, startup_speed_percent=10, speed_percent=60)
        assert all(name not in {"EnablePiper", "JointCtrl"} for name, _ in fake_bad.call_log)

    # 3. enable=True startup sequence: slow speed -> EnablePiper -> JointCtrl(measured) -> target speed
    init_mdeg = (1000, 80000, -70000, 0, 20000, 500)
    fake_ok = MockPiperSdkArm("can0", initial_mdeg=init_mdeg, ctrl_mode=0x00)
    arm_ok = PiperSdkArm("left", "can0", sdk_factory=lambda _p: fake_ok)
    arm_ok.connect(enable=True, startup_speed_percent=10, speed_percent=65)
    names = [name for name, _ in fake_ok.call_log]
    assert names == ["ConnectPort", "MotionCtrl_2", "EnablePiper", "JointCtrl", "MotionCtrl_2"]
    assert fake_ok.call_log[1][1] == (0x01, 0x01, 10, 0x00)
    assert fake_ok.call_log[3][1] == init_mdeg
    assert fake_ok.call_log[4][1] == (0x01, 0x01, 65, 0x00)
    arm_ok.disconnect()
    assert fake_ok.call_log[-1][0] == "DisconnectPort"


def test_acceleration_limited_trajectory_bounds_and_hold() -> None:
    q0 = np.zeros(6, dtype=np.float64)
    traj = AccelerationLimitedJointTrajectory(
        q0,
        max_velocity_rad_s=1.0,
        max_acceleration_rad_s2=4.0,
    )
    traj.set_target(np.full(6, 0.5))
    dt = 0.01
    prev_p = q0.copy()
    prev_v = np.zeros(6)
    for _ in range(100):
        p = traj.step(dt)
        v = traj.velocity
        assert float(np.max(np.abs(v))) <= 1.0 + 1e-9
        assert float(np.max(np.abs(v - prev_v))) / dt <= 4.0 + 1e-6
        assert float(np.max(np.abs(p - prev_p))) <= 1.0 * dt + 1e-6
        prev_p = p
        prev_v = v
    assert np.allclose(traj.position, 0.5, atol=1e-3)

    # Hold immediately freezes target and zeroes velocity
    traj.set_target(np.full(6, -0.5))
    traj.step(dt)
    held = traj.hold_current()
    assert np.allclose(traj.velocity, 0.0)
    assert np.allclose(traj.step(dt), held)


def test_flow1_replay_real_with_mocked_piper_sdk_and_midrun_fault(tmp_path: Path) -> None:
    prof = load_robot_profile("piper_pika")
    T = 12
    q_traj = np.tile(prof.home_q, (T, 1))
    for t in range(T):
        q_traj[t, prof.arms["left"].arm_actuated_indices[0]] += 0.008 * t
        q_traj[t, prof.arms["right"].arm_actuated_indices[0]] -= 0.008 * t
    widths = {
        "left": np.linspace(0.08, 0.02, T),
        "right": np.linspace(0.08, 0.02, T),
    }

    # Nominal execution via PiperCanBackend + MockPiperSdkArm
    backend, fakes = _make_mock_piper_backend("piper_pika", enable_arms=True)
    rep = execute_real_trajectory(
        backend,
        prof,
        q_traj,
        widths,
        fps=30.0,
        speed_scale=0.5,
        ramp_speed_deg_s=180.0,
        sleep_realtime=False,
        dry_run=False,
    )
    assert rep["passed"] is True
    assert rep["ramped_to_start"] is True
    assert rep["completed"] is True
    assert rep["executed_frames"] == T
    assert backend.is_connected is False
    # Verify both arms received JointCtrl and GripperCtrl frames in um (<= 100,000 um)
    for side in ("left", "right"):
        j_calls = [c for c in fakes[side].call_log if c[0] == "JointCtrl"]
        g_calls = [c for c in fakes[side].call_log if c[0] == "GripperCtrl"]
        assert len(j_calls) >= 2
        assert len(g_calls) >= 1
        assert all(0 <= args[0] <= 100_000 for _, args in g_calls)

    # Mid-trajectory arm fault MUST latch HOLD_LAST_SAFE and abort cleanly
    backend2, fakes2 = _make_mock_piper_backend("piper_pika", enable_arms=True)
    orig_check = backend2.check_health
    calls = {"n": 0}

    def _inject_fault() -> None:
        calls["n"] += 1
        if calls["n"] == 4:
            fakes2["left"].arm_status = 7
        orig_check()

    backend2.check_health = _inject_fault  # type: ignore[method-assign]
    rep2 = execute_real_trajectory(
        backend2,
        prof,
        q_traj,
        widths,
        fps=30.0,
        speed_scale=0.5,
        ramp_speed_deg_s=180.0,
        sleep_realtime=False,
        dry_run=False,
    )
    assert rep2["completed"] is False
    assert rep2["final_mode"] == ArmStateMode.HOLD_LAST_SAFE.value
    assert rep2["executed_frames"] == 3
    assert any("backend_health_fault" in r for r in rep2["fault_reasons"])

    # CLI dry-run + --mock-robot smoke check
    ds_dir = generate_synthetic_bimanual_dataset(tmp_path / "raw_ds", num_episodes=1, frames_per_episode=12)
    out_json = tmp_path / "replay_mock.json"
    assert (
        main(
            [
                "replay-real",
                str(ds_dir),
                "--episode",
                "0",
                "--robot",
                "piper_pika",
                "--mock-robot",
                "--speed-scale",
                "0.4",
                "--json-out",
                str(out_json),
            ]
        )
        == 0
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["mock_robot"] is True
    assert payload["completed"] is True


def test_flow2_live_teleop_retargeting_udp_and_safety_faults(tmp_path: Path) -> None:
    prof = load_robot_profile("piper_pika")
    tcal = load_table_calibration("piper_pika", require_physical=False)

    # 1. Verify exact SE(3) chain on PikaLiveRetargeter
    retargeter = PikaLiveRetargeter(prof, table_calib=tcal, retarget_mode="absolute-table")
    now = time.monotonic()
    raw_pose = np.array([0.12, 0.20, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    sample = PikaDualPoseSample(
        timestamp_s=now,
        received_monotonic_s=now,
        poses7={"left": raw_pose, "right": raw_pose * np.array([1, -1, 1, 1, 1, 1, 1])},
        widths_m={"left": 0.07, "right": 0.04},
        accurate={"left": True, "right": True},
        pose_frame="vive_tracker",
    )
    ee_tgts, widths = retargeter.compute_ee_targets(sample, prof.home_q)
    expected_left = mat_to_pose7(
        retargeter.T_robot_table
        @ retargeter.T_table_vive
        @ pose7_to_mat(raw_pose)
        @ retargeter.T_tracker_tcp["left"]
        @ retargeter.T_pika_to_ee
    )
    assert np.allclose(ee_tgts["left"], expected_left, atol=1e-6)
    assert widths["left"] == pytest.approx(0.07)

    # 2. Loopback UDP JSON source -> PiperCanBackend(MockPiperSdkArm)
    udp_src = UdpJsonPikaPoseSource(host="127.0.0.1", port=0)
    udp_src.start()
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stop_tx = threading.Event()

    def _udp_sender() -> None:
        k = 0
        while not stop_tx.is_set():
            pkt = {
                "timestamp_s": time.time(),
                "pose_frame": "table_tcp",
                "left": {"pose7": [-0.28 + 0.002 * math.sin(k * 0.2), 0.20, 0.28, -0.1622, 0.1622, 0.6883, 0.6883], "width_m": 0.06, "accurate": True},
                "right": {"pose7": [0.28 + 0.002 * math.sin(k * 0.2), 0.20, 0.28, -0.1622, 0.1622, 0.6883, 0.6883], "width_m": 0.05, "accurate": True},
            }
            tx_sock.sendto(json.dumps(pkt).encode("utf-8"), ("127.0.0.1", udp_src.bound_port))
            k += 1
            time.sleep(0.01)

    sender_thread = threading.Thread(target=_udp_sender, daemon=True)
    sender_thread.start()
    try:
        backend, fakes = _make_mock_piper_backend("piper_pika", enable_arms=True)
        summary = run_teleop_loop(
            backend,
            udp_src,
            retargeter,
            fps=30.0,
            steps=6,
            stale_timeout_s=0.5,
            ramp_to_first=True,
            ramp_speed_deg_s=180.0,
            sleep_realtime=False,
            dry_run=False,
        )
        assert summary["executed_steps"] == 6
        assert summary["ramped_to_first"] is True
        assert summary["final_mode"] == ArmStateMode.ACTIVE_TRACKING.value
        assert any(name == "JointCtrl" for name, _ in fakes["left"].call_log)
    finally:
        stop_tx.set()
        sender_thread.join(timeout=1.0)
        tx_sock.close()
        udp_src.close()

    # 3. Tracking inaccuracy MUST immediately latch HOLD_LAST_SAFE
    class _DroppingSource(MockPikaPoseSource):
        def latest_sample(self) -> PikaDualPoseSample:
            s = super().latest_sample()
            if self._step >= 3:
                return PikaDualPoseSample(
                    timestamp_s=s.timestamp_s,
                    received_monotonic_s=s.received_monotonic_s,
                    poses7=s.poses7,
                    widths_m=s.widths_m,
                    accurate={"left": False, "right": True},
                    pose_frame=s.pose_frame,
                )
            return s

    backend_drop, _ = _make_mock_piper_backend("piper_pika", enable_arms=True)
    drop_sum = run_teleop_loop(
        backend_drop,
        _DroppingSource(),
        retargeter,
        fps=30.0,
        steps=10,
        ramp_to_first=False,
        sleep_realtime=False,
        dry_run=False,
    )
    assert drop_sum["final_mode"] == ArmStateMode.HOLD_LAST_SAFE.value
    assert "pika_tracking_inaccurate" in drop_sum["fault_reasons"]
    assert drop_sum["executed_steps"] < 10

    # 4. CLI `pika teleop --dry-run --mock-robot`
    tel_out = tmp_path / "teleop_dry.json"
    assert (
        main(
            [
                "teleop",
                "--robot",
                "piper_pika",
                "--source",
                "mock",
                "--steps",
                "6",
                "--out",
                str(tel_out),
            ]
        )
        == 0
    )
    tel_payload = json.loads(tel_out.read_text(encoding="utf-8"))
    assert tel_payload["dry_run"] is True
    assert tel_payload["executed_steps"] == 6


def test_flow3_policy_bridge_11d_state_http_chunking_and_rpc_fault(tmp_path: Path) -> None:
    # 1. Verify build_umi_observation_state_11d matches compute_umi_delta_tensors bit-for-bit
    left_p = np.array([0.15, 0.22, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    right_p = np.array([0.18, -0.20, 0.28, 0.0, 0.0, math.sin(0.2), math.cos(0.2)], dtype=np.float64)
    st_single = build_umi_observation_state_11d(left_p, right_p, 0.075, 0.042)
    st_batch, _ = compute_umi_delta_tensors(
        np.stack([left_p, left_p], axis=0),
        np.stack([right_p, right_p], axis=0),
        np.array([0.075, 0.075]),
        np.array([0.042, 0.042]),
    )
    assert st_single.shape == (11,)
    assert np.allclose(st_single, st_batch[0], atol=1e-6)

    # 2. Spin up local loopback HTTP JSON policy server returning 4-step action chunks
    received_obs: list[dict[str, Any]] = []

    class _PolicyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            received_obs.append(body)
            chunk = []
            for k in range(4):
                act = [0.0] * 14
                act[0] = 0.0015
                act[6] = 0.06 - 0.005 * k
                act[7] = 0.0015
                act[13] = 0.06 - 0.005 * k
                chunk.append(act)
            resp = json.dumps({"action_chunk": chunk}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        url = f"http://127.0.0.1:{port}/act"
        prof = load_robot_profile("piper_pika")
        backend, fakes = _make_mock_piper_backend("piper_pika", enable_arms=True)
        client = HttpJsonPolicyClient(url, timeout_s=1.0)
        summary = run_policy_bridge_loop(
            backend,
            prof,
            client,
            fps=30.0,
            steps=8,
            n_action_steps=4,
            task="fold towel",
            enforce_clearance=True,
            sleep_realtime=False,
            dry_run=False,
        )
        assert summary["steps"] == 8
        assert summary["rpc_queries"] == 2
        assert summary["final_mode"] == ArmStateMode.ACTIVE_TRACKING.value
        assert len(received_obs) == 2
        assert len(received_obs[0]["observation.state"]) == 11
        assert received_obs[0]["task"] == "fold towel"
        assert any(name == "JointCtrl" for name, _ in fakes["right"].call_log)

        # CLI dry-run against real local HTTP policy server
        pb_out = tmp_path / "pb_http.json"
        assert (
            main(
                [
                    "policy-bridge",
                    "--robot",
                    "piper_pika",
                    "--policy-url",
                    url,
                    "--task",
                    "pick cup",
                    "--n-action-steps",
                    "3",
                    "--steps",
                    "6",
                    "--out",
                    str(pb_out),
                ]
            )
            == 0
        )
        pb_payload = json.loads(pb_out.read_text(encoding="utf-8"))
        assert pb_payload["steps"] == 6
        assert pb_payload["rpc_queries"] == 2
    finally:
        server.shutdown()
        server.server_close()

    # 3. Failing policy client MUST latch HOLD_LAST_SAFE immediately
    class _BrokenPolicy(MockPolicyClient):
        def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
            raise TimeoutError("simulated VLA server timeout")

    backend_err, _ = _make_mock_piper_backend("piper_pika", enable_arms=True)
    err_sum = run_policy_bridge_loop(
        backend_err,
        load_robot_profile("piper_pika"),
        _BrokenPolicy(),
        fps=30.0,
        steps=5,
        sleep_realtime=False,
        dry_run=False,
    )
    assert err_sum["steps"] == 0
    assert err_sum["final_mode"] == ArmStateMode.HOLD_LAST_SAFE.value
    assert any("policy_rpc_fault" in r for r in err_sum["fault_reasons"])


def test_pyagxarm_v189_lifecycle_safety_and_si_units() -> None:
    # 1. Firmware profile resolution
    assert resolve_pyagxarm_firmware_profile("S-V1.8-9") == "v189"
    assert resolve_pyagxarm_firmware_profile("S-V1.8-10") == "v189"
    assert resolve_pyagxarm_firmware_profile("S-V1.8-8") == "v188"
    assert resolve_pyagxarm_firmware_profile("S-V1.8-5") == "v183"
    assert resolve_pyagxarm_firmware_profile("S-V1.8-2") == "default"

    # 2. auto_enable=False refuses to energize when ctrl_mode != 0x01
    fake_idle = MockPyAgxArm("can0", ctrl_mode=0x00)
    arm_idle = PyAgxPiperArm("left", "can0", firmware="S-V1.8-9", sdk_factory=lambda _p, _fw: fake_idle)
    with pytest.raises(RuntimeError, match="auto_enable=False"):
        arm_idle.connect(enable=False, startup_speed_percent=10, speed_percent=60)
    assert all(name != "enable" for name, _ in fake_idle.call_log)

    # 3. Teach mode (0x02), V189 MIT mode (mode_feedback=0x06), and non-zero arm_status MUST raise
    for kwargs in (
        {"ctrl_mode": 0x02},
        {"teach_status": 1},
        {"mode_feedback": 0x06},
        {"arm_status": 7},
    ):
        fake_bad = MockPyAgxArm("can0", **kwargs)
        arm_bad = PyAgxPiperArm("left", "can0", firmware="S-V1.8-9", sdk_factory=lambda _p, _fw, f=fake_bad: f)
        with pytest.raises(RuntimeError, match="S-V1.8-9|MIT mode"):
            arm_bad.connect(enable=True, startup_speed_percent=10, speed_percent=60)
        assert all(name not in {"enable", "move_j"} for name, _ in fake_bad.call_log)

    # 4. Exact pyAgxArm startup order: init_effector BEFORE connect -> normal -> slow -> enable -> move_j(measured) -> target speed
    q_init = (0.05, 1.40, -1.30, 0.0, 0.45, -0.10)
    fake_ok = MockPyAgxArm("can0", initial_q_rad=q_init, initial_gripper_m=0.06, ctrl_mode=0x00)
    arm_ok = PyAgxPiperArm("left", "can0", firmware="S-V1.8-9", sdk_factory=lambda _p, _fw: fake_ok)
    arm_ok.connect(enable=True, startup_speed_percent=10, speed_percent=55)
    names = [name for name, _ in fake_ok.call_log]
    assert names == [
        "init_effector",
        "connect",
        "set_normal_mode",
        "set_speed_percent",
        "enable",
        "move_j",
        "set_speed_percent",
    ]
    assert fake_ok.call_log[0][1] == ("agx_gripper",)
    assert fake_ok.call_log[3][1] == (10,)
    assert np.allclose(fake_ok.call_log[5][1], q_init)
    assert fake_ok.call_log[6][1] == (55,)

    # Native SI gripper command
    arm_ok.send_gripper_width_m(0.085, 1.2)
    assert fake_ok.effector is not None
    assert fake_ok.effector.call_log[-1] == ("move_gripper_m", (0.085, 1.2))
    assert arm_ok.read_gripper_width_m() == pytest.approx(0.085)

    arm_ok.disconnect()
    assert fake_ok.call_log[-1][0] == "disconnect"
    assert all(name not in {"disable", "reset", "electronic_emergency_stop"} for name, _ in fake_ok.call_log)


def test_pyagxarm_all_three_flows_end_to_end() -> None:
    prof = load_robot_profile("piper_pika")

    # Flow 1: replay-real over PyAgxPiperArm
    T = 8
    q_traj = np.tile(prof.home_q, (T, 1))
    for t in range(T):
        q_traj[t, prof.arms["left"].arm_actuated_indices[0]] += 0.006 * t
        q_traj[t, prof.arms["right"].arm_actuated_indices[0]] -= 0.006 * t
    widths = {"left": np.linspace(0.09, 0.03, T), "right": np.linspace(0.09, 0.03, T)}
    b1, f1 = _make_mock_pyagx_backend("piper_pika", enable_arms=True)
    r1 = execute_real_trajectory(
        b1,
        prof,
        q_traj,
        widths,
        fps=30.0,
        speed_scale=0.5,
        ramp_speed_deg_s=180.0,
        sleep_realtime=False,
        dry_run=False,
    )
    assert r1["completed"] is True
    assert r1["executed_frames"] == T
    assert any(n == "move_j" for n, _ in f1["left"].call_log)
    assert f1["left"].effector is not None and len(f1["left"].effector.call_log) >= 1
    assert all(0.0 <= args[0] <= 0.10 for _, args in f1["left"].effector.call_log)

    # Stock piper (70mm) scales 0.05m Pika width -> 0.035m hardware command
    b_stock, _ = _make_mock_pyagx_backend("piper", enable_arms=True)
    assert b_stock.map_widths_to_hardware_m({"left": 0.05, "right": 0.10}) == {
        "left": pytest.approx(0.035),
        "right": pytest.approx(0.070),
    }

    # Flow 2: live teleop over PyAgxPiperArm
    tcal = load_table_calibration("piper_pika", require_physical=False)
    retargeter = PikaLiveRetargeter(prof, table_calib=tcal, retarget_mode="absolute-table")
    b2, f2 = _make_mock_pyagx_backend("piper_pika", enable_arms=True)
    r2 = run_teleop_loop(
        b2,
        MockPikaPoseSource(fps=30.0),
        retargeter,
        fps=30.0,
        steps=5,
        ramp_to_first=True,
        ramp_speed_deg_s=180.0,
        sleep_realtime=False,
        dry_run=False,
    )
    assert r2["executed_steps"] == 5
    assert r2["final_mode"] == ArmStateMode.ACTIVE_TRACKING.value
    assert any(n == "move_j" for n, _ in f2["right"].call_log)

    # Flow 3: policy-bridge over PyAgxPiperArm
    b3, f3 = _make_mock_pyagx_backend("piper_pika", enable_arms=True)
    r3 = run_policy_bridge_loop(
        b3,
        prof,
        MockPolicyClient(chunk_size=2),
        fps=30.0,
        steps=6,
        n_action_steps=2,
        enforce_clearance=True,
        sleep_realtime=False,
        dry_run=False,
    )
    assert r3["steps"] == 6
    assert r3["rpc_queries"] == 3
    assert r3["final_mode"] == ArmStateMode.ACTIVE_TRACKING.value
    assert any(n == "move_j" for n, _ in f3["left"].call_log)


def test_round2_critical_2_3_docker_ros1_topics_types_and_packet_accuracy() -> None:
    from pikaumi.real.teleop import DockerRos1PikaPoseSource, parse_pika_json_packet

    src = DockerRos1PikaPoseSource(container_name="pika-ros")
    cmd = src.build_exec_command()
    shell_str = " ".join(cmd)
    assert "/root/pika_ros/install/setup.bash" in shell_str
    assert "/gripper_l/data" in shell_str
    assert "/gripper_r/data" in shell_str
    assert "/gripper/data_l" not in shell_str
    assert "/gripper/data_r" not in shell_str
    assert "LocalizationStatus" in shell_str
    assert "Gripper" in shell_str
    assert "std_msgs.msg" not in shell_str

    # Omitting "accurate" MUST default to False (never True)
    now = time.monotonic()
    pkt = parse_pika_json_packet(
        {
            "timestamp_s": 100.0,
            "left": {"pose7": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "width_m": 0.05},
            "right": {"pose7": [0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "width_m": 0.05},
        },
        received_monotonic_s=now,
    )
    assert pkt.accurate["left"] is False
    assert pkt.accurate["right"] is False

    # Missing right side MUST NOT fabricate a pose or mark right accurate
    pkt_one = parse_pika_json_packet(
        {
            "timestamp_s": 100.0,
            "left": {"pose7": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "width_m": 0.05, "accurate": True},
        },
        received_monotonic_s=now,
    )
    assert pkt_one.accurate["left"] is True
    assert "right" not in pkt_one.poses7 and not pkt_one.accurate.get("right", False)


def test_round2_critical_2_4_no_gripper_slam_on_connect() -> None:
    # Both pyAgxArm and piper_sdk backends must emit ZERO gripper commands during connect()
    # until an explicit target is sent via write_target().
    b_agx, f_agx = _make_mock_pyagx_backend("piper_pika", enable_arms=True)
    b_agx.connect()
    try:
        time.sleep(0.03)
        for side in ("left", "right"):
            eff = f_agx[side].effector
            assert eff is not None
            assert len(eff.call_log) == 0, f"Unexpected gripper slam on connect ({side}): {eff.call_log}"
    finally:
        b_agx.disconnect()

    b_sdk, f_sdk = _make_mock_piper_backend("piper_pika", enable_arms=True)
    b_sdk.connect()
    try:
        time.sleep(0.03)
        for side in ("left", "right"):
            gripper_calls = [c for c in f_sdk[side].call_log if c[0] == "GripperCtrl"]
            assert len(gripper_calls) == 0, f"Unexpected GripperCtrl slam on connect ({side}): {gripper_calls}"
    finally:
        b_sdk.disconnect()


def test_round2_critical_2_5_none_status_uninit_wait_and_stale_can_feedback() -> None:
    from pikaumi.real.piper_driver import _inspect_arm_status_for_unsafe_mode

    # None status MUST be treated as unsafe
    fault = _inspect_arm_status_for_unsafe_mode(None, "can0")
    assert fault is not None and "status_msg is None" in fault

    # PiperSdkArm waits past uninitialized time_stamp == 0.0 frames so it never reads (0,0,0,0,0,0)
    nonzero_mdeg = (15000, 45000, -30000, 5000, 20000, -10000)
    fake_sdk = MockPiperSdkArm("can0", initial_mdeg=nonzero_mdeg, uninitialized_frames=3)
    arm_sdk = PiperSdkArm(
        "left",
        "can0",
        stale_feedback_timeout_s=0.05,
        sdk_factory=lambda port: fake_sdk,
    )
    arm_sdk.connect(enable=True, startup_speed_percent=10, speed_percent=50)
    q_init = arm_sdk.read_q_rad()
    assert np.allclose(q_init, piper_mdeg_to_q(nonzero_mdeg), atol=1e-4)

    # Freezing timestamps triggers stale CAN feedback fault on PiperSdkArm
    fake_sdk.freeze_timestamps = True
    time.sleep(0.07)
    with pytest.raises(RuntimeError, match="stale"):
        arm_sdk.read_q_rad()
    with pytest.raises(RuntimeError, match="stale"):
        arm_sdk.check_arm_health()
    arm_sdk.disconnect()

    # Freezing timestamps triggers stale CAN feedback fault on PyAgxPiperArm too
    fake_agx = MockPyAgxArm("can0", initial_q_rad=(0.1, 0.5, -0.4, 0.0, 0.3, 0.0))
    arm_agx = PyAgxPiperArm(
        "left",
        "can0",
        stale_feedback_timeout_s=0.05,
        sdk_factory=lambda port, iface: fake_agx,
    )
    arm_agx.connect(enable=True, startup_speed_percent=10, speed_percent=50)
    arm_agx.check_arm_health()
    fake_agx.freeze_timestamps = True
    time.sleep(0.07)
    with pytest.raises(RuntimeError, match="stale"):
        arm_agx.read_q_rad()
    with pytest.raises(RuntimeError, match="stale"):
        arm_agx.check_arm_health()
    arm_agx.disconnect()


def test_round2_high_3_1_stock_piper_single_scale_and_inverse_obs_width() -> None:
    prof = load_robot_profile("piper")
    assert prof.gripper_max_width_m == pytest.approx(0.07)
    assert prof.gripper_retarget_mode == "normalized"

    # Single-scale check: Pika 0.10m -> robot physical 0.07m -> hardware command 0.07m (NOT 0.049m!)
    q_full, robot_w = prof.map_gripper_width_to_q(prof.home_q, "left", 0.10)
    assert robot_w == pytest.approx(0.070)
    assert prof.map_robot_width_to_pika("left", robot_w) == pytest.approx(0.10)

    b_stock, f_stock = _make_mock_pyagx_backend("piper", enable_arms=True)
    b_stock.connect()
    try:
        b_stock.write_target(q_full, {"left": 0.070, "right": 0.035})
        time.sleep(0.03)
        assert f_stock["left"].effector is not None
        assert f_stock["right"].effector is not None
        assert f_stock["left"].effector.width_m == pytest.approx(0.070, abs=1e-4)
        assert f_stock["right"].effector.width_m == pytest.approx(0.035, abs=1e-4)

        # Policy bridge must inverse-map physical 0.05m back to Pika 0.05/0.07*0.10 m in 11D obs
        received: list[np.ndarray] = []

        class _CapturePolicy(MockPolicyClient):
            def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
                received.append(np.asarray(observation["observation.state"], dtype=np.float64).copy())
                return super().predict_action_chunk(observation)

        b_stock.disconnect()
        b_stock2, _ = _make_mock_pyagx_backend("piper", enable_arms=True)
        run_policy_bridge_loop(
            b_stock2,
            prof,
            _CapturePolicy(chunk_size=1),
            fps=30.0,
            steps=2,
            n_action_steps=1,
            sleep_realtime=False,
            dry_run=False,
        )
        assert len(received) >= 1
        assert received[0][9] == pytest.approx(0.05 / 0.07 * 0.10, abs=1e-3)
        assert received[0][10] == pytest.approx(0.05 / 0.07 * 0.10, abs=1e-3)
    finally:
        b_stock.disconnect()


def test_round2_high_3_5_safety_guards_nan_width_preflight_live_calib_and_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    import pikaumi.real.replay_real as replay_real_mod
    from pikaumi.calibration.deployment import save_physical_table_calibration
    from pikaumi.dataset.canonical import convert_dataset_to_robot_joints
    from pikaumi.real.base import GuardedRobotController
    from pikaumi.real.replay_real import preflight_real_trajectory, run_replay_real_cli

    prof = load_robot_profile("piper_pika")

    # 1. NaN width latches HOLD_LAST_SAFE and preserves finite last safe widths
    guard = GuardedRobotController(prof)
    guard.seed_state(prof.home_q, {"left": 0.05, "right": 0.05})
    _, safe_w, mode = guard.filter_command(
        prof.home_q,
        {"left": float("nan"), "right": 0.05},
    )
    assert mode == ArmStateMode.HOLD_LAST_SAFE
    assert math.isfinite(safe_w["left"]) and safe_w["left"] == pytest.approx(0.05)

    # 2. preflight_real_trajectory rejects NaN, URDF joint limit violations, and reads nested openarmv1 limits
    q_nan = np.tile(prof.home_q, (4, 1))
    q_nan[2, 0] = float("nan")
    rep_nan = preflight_real_trajectory(prof, q_nan, fps=30.0)
    assert rep_nan["passed"] is False
    assert any("non_finite" in r for r in rep_nan["reasons"])

    q_oob = np.tile(prof.home_q, (4, 1))
    q_oob[1, prof.arms["left"].arm_actuated_indices[0]] = 99.0
    rep_oob = preflight_real_trajectory(prof, q_oob, fps=30.0)
    assert rep_oob["passed"] is False
    assert any("joint_limit" in r or "limit" in r for r in rep_oob["reasons"])

    oa_prof = load_robot_profile("openarmv1")
    # Constant-velocity ramp of 0.06 rad/step * 30 Hz = 1.8 rad/s (~103 deg/s) > openarmv1 nested 1.0 rad/s (57.3 deg/s)
    q_fast_oa = np.tile(oa_prof.home_q, (5, 1))
    for k in range(5):
        q_fast_oa[k, oa_prof.arms["left"].arm_actuated_indices[0]] += 0.06 * k
    rep_oa = preflight_real_trajectory(oa_prof, q_fast_oa, fps=30.0, speed_scale=1.0)
    assert rep_oa["passed"] is False
    assert any("peak_vel_deg_s" in r for r in rep_oa["reasons"])

    # 3. Live absolute-table teleop rejects missing session.yaml
    tcal = load_table_calibration("piper_pika", require_physical=False)
    empty_cal = tmp_path / "empty_cal"
    empty_cal.mkdir()
    with pytest.raises(PermissionError, match="session.yaml"):
        PikaLiveRetargeter(
            prof,
            table_calib=tcal,
            calibration_dir=empty_cal,
            retarget_mode="absolute-table",
            require_session_calibration=True,
        )

    # 4. Live replay-real rejects pre-solved joints dataset whose table_calib_sha256 mismatches
    raw_ds = generate_synthetic_bimanual_dataset(tmp_path / "raw_for_sha", num_episodes=1, frames_per_episode=8)
    joints_ds = tmp_path / "joints_for_sha"
    convert_dataset_to_robot_joints(raw_ds, joints_ds, robot="piper_pika")

    phys_yaml = save_physical_table_calibration(
        tmp_path / "phys_table.yaml",
        robot="piper_pika",
        T_robot_table_pose7=tcal.T_robot_table_pose7 + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        verified=True,
    )
    phys_tcal = load_table_calibration("piper_pika", path=phys_yaml, require_physical=True)
    monkeypatch.setattr(replay_real_mod, "require_live_authorization", lambda **_: phys_tcal)

    cli_args = argparse.Namespace(
        robot="piper_pika",
        dataset_dir=str(joints_ds),
        episode=0,
        live=True,
        operator_supported=True,
        speed_scale=0.5,
        json_out=None,
        mock_robot=False,
    )
    with pytest.raises(PermissionError, match="table_calib_sha256"):
        run_replay_real_cli(cli_args)


def test_live_preflight_clearance_and_pyagxarm_v1_motion_mode_regression() -> None:
    """Regression: live preflight uses real_config/urdf_model clearance and pyAgxArm 1.0.0 uses set_motion_mode('j')."""
    from types import SimpleNamespace

    from pikaumi.real.replay_real import preflight_real_trajectory

    prof = load_robot_profile("piper_pika")
    q_home_traj = np.tile(prof.home_q, (5, 1))

    rep_ok = preflight_real_trajectory(
        prof,
        q_home_traj,
        fps=30.0,
        speed_scale=0.3,
        table_z_robot=0.005,
        enforce_clearance=True,
    )
    assert rep_ok["passed"] is True, rep_ok

    rep_table_hit = preflight_real_trajectory(
        prof,
        q_home_traj,
        fps=30.0,
        speed_scale=0.3,
        table_z_robot=1.5,
        enforce_clearance=True,
    )
    assert rep_table_hit["passed"] is False
    assert any("table_clearance_m" in r for r in rep_table_hit["reasons"])

    class _StubPyAgxV1:
        """Mimics pyAgxArm 1.0.0 PiperV189Driver: has set_motion_mode, NO set_normal_mode."""

        class OPTIONS:
            class EFFECTOR:
                AGX_GRIPPER = "agx_gripper"

        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []
            self._ts = 100.0
            self._status_polls = 0
            self._q = [0.0, 0.2, -0.3, 0.0, 0.1, 0.0]

        def init_effector(self, kind: str) -> Any:
            self.calls.append(("init_effector", kind))
            return SimpleNamespace(
                move_gripper_m=lambda value, force: None,
                get_gripper_status=lambda: SimpleNamespace(msg=SimpleNamespace(value=0.05)),
            )

        def connect(self) -> None:
            self.calls.append(("connect", None))

        def get_arm_status(self) -> Any:
            self._ts += 0.01
            self._status_polls += 1
            hz = 0.0 if self._status_polls <= 3 else 200.0
            return SimpleNamespace(
                timestamp=self._ts,
                hz=hz,
                msg=SimpleNamespace(ctrl_mode=0x00, arm_status=0, teach_status=0),
            )

        def get_joint_angles(self) -> Any:
            self._ts += 0.01
            return SimpleNamespace(timestamp=self._ts, hz=200.0, msg=list(self._q))

        def set_motion_mode(self, mode: str) -> None:
            self.calls.append(("set_motion_mode", mode))

        def set_speed_percent(self, percent: int) -> None:
            self.calls.append(("set_speed_percent", percent))

        def enable(self) -> bool:
            self.calls.append(("enable", True))
            return True

        def move_j(self, joints: list[float]) -> None:
            self.calls.append(("move_j", joints))

        def disconnect(self) -> None:
            self.calls.append(("disconnect", None))

    stub = _StubPyAgxV1()
    arm = PyAgxPiperArm("left", "can0", sdk_factory=lambda _port, _fw: stub)
    arm.connect(enable=True, startup_speed_percent=5, speed_percent=20, timeout_s=1.0)
    assert stub._status_polls >= 4
    assert ("set_motion_mode", "j") in stub.calls
    assert ("enable", True) in stub.calls
    assert any(name == "move_j" for name, _ in stub.calls)
    arm.disconnect()


def test_smooth_replay_filter_eliminates_start_stop_chatter() -> None:
    """Zero-phase Butterworth + clamped CubicSpline + 3rd-order tracking filter eliminates 9 Hz start-stop chatter."""
    from pikaumi.real.replay_real import resample_trajectory_cubic, smooth_joint_trajectory

    prof = load_robot_profile("piper_pika")
    rng = np.random.default_rng(42)
    T = 60
    t = np.linspace(0.0, 2.0, T)
    q_noisy = np.tile(prof.home_q, (T, 1))
    right_idx = prof.arms["right"].arm_actuated_indices[1]
    # Smooth 0.5 Hz motion (starts from rest) + 12 Hz chatter noise
    q_noisy[:, right_idx] += 0.35 * 0.5 * (1.0 - np.cos(math.pi * t)) + 0.015 * np.sin(2.0 * math.pi * 12.0 * t)
    widths = {
        "left": np.full(T, 0.04, dtype=np.float64),
        "right": np.clip(0.05 + 0.02 * np.sin(math.pi * t) + rng.normal(0, 0.002, T), 0.0, 0.10),
    }

    q_smooth, w_smooth = smooth_joint_trajectory(prof, q_noisy, widths, fps=30.0, cutoff_hz=3.0)
    raw_acc = float(np.max(np.abs(np.diff(q_noisy[:, right_idx], n=2))))
    smooth_acc = float(np.max(np.abs(np.diff(q_smooth[:, right_idx], n=2))))
    assert smooth_acc < raw_acc * 0.25
    assert np.all((w_smooth["right"] >= 0.0) & (w_smooth["right"] <= prof.gripper_max_width_m))

    q_dense, _, actual_hz = resample_trajectory_cubic(q_smooth, w_smooth, source_hz=9.0, target_hz=100.0)
    assert actual_hz == pytest.approx(100.0, rel=0.02)
    assert len(q_dense) > len(q_smooth) * 10

    # Compare 200 Hz streamer with 3rd-order tracking filter vs trapezoidal on 100 Hz spline
    right_idxs = list(prof.arms["right"].arm_actuated_indices)
    traj_smooth = AccelerationLimitedJointTrajectory(
        q_dense[0, right_idxs],
        max_velocity_rad_s=math.radians(180.0),
        max_acceleration_rad_s2=math.radians(720.0),
        tracking_cutoff_hz=5.0,
    )
    traj_trap = AccelerationLimitedJointTrajectory(
        q_dense[0, right_idxs],
        max_velocity_rad_s=math.radians(180.0),
        max_acceleration_rad_s2=math.radians(720.0),
        tracking_cutoff_hz=None,
    )
    dt = 1.0 / 200.0
    vels_s, vels_t = [], []
    for k in range(400):
        idx = min(int(k * dt * actual_hz), len(q_dense) - 1)
        tgt = q_dense[idx, right_idxs]
        traj_smooth.set_target(tgt)
        traj_smooth.step(dt)
        vels_s.append(traj_smooth.velocity)
        traj_trap.set_target(tgt)
        traj_trap.step(dt)
        vels_t.append(traj_trap.velocity)
    peak_acc_smooth = float(np.degrees(np.max(np.abs(np.diff(np.asarray(vels_s), axis=0) / dt))))
    peak_acc_trap = float(np.degrees(np.max(np.abs(np.diff(np.asarray(vels_t), axis=0) / dt))))
    assert peak_acc_smooth < 120.0
    assert peak_acc_smooth < peak_acc_trap * 0.25

    # HOLD_LAST_SAFE still immediately zeroes velocity and locks position
    held = traj_smooth.hold_current()
    assert np.allclose(traj_smooth.velocity, 0.0)
    assert np.allclose(traj_smooth.step(dt), held)

