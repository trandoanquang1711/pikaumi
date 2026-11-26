"""Live online Dual Pika Sense -> Dual Piper retargeting (`pika teleop`).

Streams live 6-DoF Pika Sense tracker poses + jaw widths through the calibrated transform chain:
    T_robot_ee(t) = T_robot_table @ T_table_vive @ T_vive_tracker(t) @ T_tracker_tcp @ T_pika_to_ee
solves real-time bounded IK (`BimanualIKSolver`), enforces inter-arm & tabletop clearance plus
`GuardedRobotController` (`HOLD_LAST_SAFE`), and drives `PiperCanBackend` (`piper_sdk`) at fixed rate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
import socket
import subprocess
import threading
import time
from typing import Any, Protocol

import numpy as np

from pikaumi.calibration.deployment import TableCalibration, load_table_calibration
from pikaumi.calibration.verify import verify_calibration_bundle
from pikaumi.config import load_rig_config, save_json_atomic
from pikaumi.dataset.raw import default_calibration_metadata
from pikaumi.real.base import (
    ArmStateMode,
    GuardedRobotController,
    MockBimanualController,
    RealRobotBackend,
    require_live_authorization,
)
from pikaumi.real.piper_driver import PiperCanBackend, load_piper_can_settings
from pikaumi.real.streamer import next_periodic_deadline
from pikaumi.robots.kinematics import BimanualIKSolver
from pikaumi.robots.registry import RobotProfile, load_robot_profile
from pikaumi.robots.utils import mat_to_pose7, pose7_to_mat, se3_world_residual


@dataclass(frozen=True)
class PikaDualPoseSample:
    """Single synchronized Dual Pika Sense measurement for online teleoperation."""

    timestamp_s: float
    received_monotonic_s: float
    poses7: dict[str, np.ndarray]  # {"left": (7,), "right": (7,)}
    widths_m: dict[str, float]  # {"left": m, "right": m}
    accurate: dict[str, bool]  # {"left": bool, "right": bool}
    pose_frame: str = "vive_tracker"  # "vive_tracker" | "table_tcp"


class PikaPoseSource(Protocol):
    """Abstract live Pika Sense measurement stream."""

    def start(self) -> None: ...

    def latest_sample(self) -> PikaDualPoseSample | None: ...

    def close(self) -> None: ...


class MockPikaPoseSource:
    """Deterministic smooth bimanual tabletop Pika Sense generator for offline dry-run & tests."""

    def __init__(
        self,
        *,
        fps: float = 30.0,
        pose_frame: str = "table_tcp",
        center_z_m: float = 0.22,
        amplitude_m: float = 0.025,
    ) -> None:
        self.fps = float(fps)
        self.pose_frame = pose_frame
        self.center_z_m = float(center_z_m)
        self.amplitude_m = float(amplitude_m)
        self._step = 0

    def start(self) -> None:
        self._step = 0

    def latest_sample(self) -> PikaDualPoseSample:
        t = self._step / max(self.fps, 1.0)
        self._step += 1
        dx = self.amplitude_m * math.sin(2.0 * math.pi * 0.4 * t)
        dz = 0.015 * (1.0 - math.cos(2.0 * math.pi * 0.4 * t))
        width = 0.055 + 0.020 * math.sin(2.0 * math.pi * 0.5 * t)
        now = time.monotonic()
        # Reachable tabletop TCP poses in table frame (+X right, +Y forward, +Z up)
        q_nom = np.array([-0.1622, 0.1622, 0.6883, 0.6883], dtype=np.float64)
        q_nom = q_nom / float(np.linalg.norm(q_nom))
        z_nom = max(self.center_z_m, 0.28)
        poses = {
            "left": np.array(
                [-0.28 + dx, 0.20, z_nom + dz, q_nom[0], q_nom[1], q_nom[2], q_nom[3]],
                dtype=np.float64,
            ),
            "right": np.array(
                [0.28 + dx, 0.20, z_nom + dz, q_nom[0], q_nom[1], q_nom[2], q_nom[3]],
                dtype=np.float64,
            ),
        }
        return PikaDualPoseSample(
            timestamp_s=now,
            received_monotonic_s=now,
            poses7=poses,
            widths_m={"left": float(np.clip(width, 0.0, 0.10)), "right": float(np.clip(width, 0.0, 0.10))},
            accurate={"left": True, "right": True},
            pose_frame=self.pose_frame,
        )

    def close(self) -> None:
        return None


def parse_pika_json_packet(payload: dict[str, Any], *, received_monotonic_s: float) -> PikaDualPoseSample:
    """Parse a JSON dict emitted by UDP bridge or Docker ROS 1 helper into `PikaDualPoseSample`.

    Safety invariant: omitted `accurate` flag defaults to `False` (fail-closed).
    """
    poses: dict[str, np.ndarray] = {}
    widths: dict[str, float] = {}
    accurate: dict[str, bool] = {}
    for side in ("left", "right"):
        if side not in payload or not isinstance(payload[side], dict):
            continue
        sp = payload[side]
        arr = np.asarray(sp["pose7"], dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite pose7 for side={side}")
        poses[side] = arr
        widths[side] = float(sp.get("width_m", 0.05))
        accurate[side] = bool(sp.get("accurate", False))
    if not poses:
        raise ValueError("Packet contains neither 'left' nor 'right' pose")
    return PikaDualPoseSample(
        timestamp_s=float(payload.get("timestamp_s", received_monotonic_s)),
        received_monotonic_s=float(received_monotonic_s),
        poses7=poses,
        widths_m=widths,
        accurate=accurate,
        pose_frame=str(payload.get("pose_frame", "vive_tracker")),
    )


class UdpJsonPikaPoseSource:
    """Low-latency non-blocking UDP JSON receiver for live Dual Pika Sense poses."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        self.host = host
        self.port = int(port)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: PikaDualPoseSample | None = None
        self.bound_port: int = self.port
        self.parse_errors: int = 0

    def start(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.05)
        self.bound_port = int(sock.getsockname()[1])
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._rx_loop, name="pika-udp-rx", daemon=True)
        self._thread.start()

    def _rx_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                obj = json.loads(data.decode("utf-8"))
                sample = parse_pika_json_packet(obj, received_monotonic_s=time.monotonic())
                with self._lock:
                    self._latest = sample
            except Exception:
                self.parse_errors += 1

    def latest_sample(self) -> PikaDualPoseSample | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


_DOCKER_ROS1_BRIDGE_SCRIPT = r"""
import json, sys, time
import rospy
from geometry_msgs.msg import PoseStamped
from data_msgs.msg import LocalizationStatus, Gripper

state = {
    "left": {"pose7": None, "width_m": 0.05, "accurate": False},
    "right": {"pose7": None, "width_m": 0.05, "accurate": False},
}

def _on_pose(side):
    def cb(msg):
        p = msg.pose.position
        q = msg.pose.orientation
        state[side]["pose7"] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    return cb

def _on_acc(side):
    def cb(msg):
        state[side]["accurate"] = bool(msg.accurate)
    return cb

def _on_gripper(side):
    def cb(msg):
        state[side]["width_m"] = float(msg.distance)
    return cb

rospy.init_node("pikaumi_live_teleop_bridge", anonymous=True, disable_signals=True)
rospy.Subscriber("/pika_pose_l", PoseStamped, _on_pose("left"), queue_size=1)
rospy.Subscriber("/pika_pose_r", PoseStamped, _on_pose("right"), queue_size=1)
rospy.Subscriber("/pika_localization_status_l", LocalizationStatus, _on_acc("left"), queue_size=1)
rospy.Subscriber("/pika_localization_status_r", LocalizationStatus, _on_acc("right"), queue_size=1)
rospy.Subscriber("/gripper_l/data", Gripper, _on_gripper("left"), queue_size=1)
rospy.Subscriber("/gripper_r/data", Gripper, _on_gripper("right"), queue_size=1)

rate = rospy.Rate(60)
while not rospy.is_shutdown():
    if state["left"]["pose7"] is not None and state["right"]["pose7"] is not None:
        out = {
            "timestamp_s": time.time(),
            "pose_frame": "vive_tracker",
            "left": dict(state["left"]),
            "right": dict(state["right"]),
        }
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    rate.sleep()
"""


class DockerRos1PikaPoseSource:
    """Streams live Dual Pika Sense ROS 1 topics out of the `pika-ros` Docker container via JSONL."""

    def __init__(self, container_name: str = "pika-ros") -> None:
        self.container_name = container_name
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: PikaDualPoseSample | None = None

    def build_exec_command(self) -> list[str]:
        return [
            "docker",
            "exec",
            "-i",
            self.container_name,
            "bash",
            "-lc",
            "source /opt/ros/noetic/setup.bash && source /root/pika_ros/install/setup.bash && python3 -u -c "
            + shlex.quote(_DOCKER_ROS1_BRIDGE_SCRIPT),
        ]

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = self.build_exec_command()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="pika-docker-ros1-rx", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sample = parse_pika_json_packet(obj, received_monotonic_s=time.monotonic())
                with self._lock:
                    self._latest = sample
            except Exception:
                continue

    def latest_sample(self) -> PikaDualPoseSample | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class PikaLiveRetargeter:
    """Real-time SE(3) coordinate transformer & bounded-IK solver from Dual Pika Sense to robot."""

    def __init__(
        self,
        profile: RobotProfile,
        *,
        table_calib: TableCalibration,
        calibration_dir: str | Path | None = None,
        retarget_mode: str = "absolute-table",
        max_ik_nfev: int = 20,
        require_session_calibration: bool = False,
    ) -> None:
        if retarget_mode not in {"absolute-table", "relative-anchor"}:
            raise ValueError(f"Unknown retarget_mode={retarget_mode!r}")
        if require_session_calibration and retarget_mode == "absolute-table":
            if calibration_dir is None:
                raise PermissionError(
                    "Live absolute-table teleop requires --calibration-dir with verified session.yaml "
                    "(refusing silent Identity T_table_vive fallback)."
                )
            cal_path = Path(calibration_dir)
            if not (cal_path / "session.yaml").is_file():
                raise PermissionError(
                    f"Live absolute-table teleop requires {cal_path / 'session.yaml'} "
                    "(run `pika calibrate session` first)."
                )
            bundle = verify_calibration_bundle(cal_path)
            if not bundle["passed"]:
                raise PermissionError(
                    f"Calibration bundle verification failed for {cal_path}: {bundle['reasons']}"
                )

        self.profile = profile
        self.table_calib = table_calib
        self.retarget_mode = retarget_mode
        self.max_ik_nfev = int(max_ik_nfev)

        cal = default_calibration_metadata(calibration_dir)
        self.T_robot_table = pose7_to_mat(table_calib.T_robot_table_pose7)
        self.T_table_vive = pose7_to_mat(np.asarray(cal["T_table_vive"], dtype=np.float64))
        self.T_tracker_tcp = {
            s: pose7_to_mat(np.asarray(cal["T_tracker_tcp"][s], dtype=np.float64))
            for s in ("left", "right")
            if s in cal["T_tracker_tcp"]
        }
        self.T_pika_to_ee = pose7_to_mat(profile.pika_to_ee_pose7)
        self.table_z_robot = float(self.T_robot_table[2, 3])
        self.solver = BimanualIKSolver(profile)

        # Anchors initialized lazily on first healthy sample when retarget_mode == "relative-anchor"
        self._anchor_table_pos: dict[str, np.ndarray] | None = None
        self._anchor_src_rot: dict[str, np.ndarray] | None = None
        self._anchor_robot_ee: dict[str, np.ndarray] | None = None

    def sample_to_table_tcp_mats(self, sample: PikaDualPoseSample) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for side in self.profile.active_sides:
            src_side = side if side in sample.poses7 else "right"
            if src_side not in sample.poses7:
                raise KeyError(f"Missing Pika pose for side={side!r}")
            T_raw = pose7_to_mat(sample.poses7[src_side])
            if sample.pose_frame == "table_tcp":
                out[side] = T_raw
            else:
                T_tt = self.T_tracker_tcp.get(src_side, np.eye(4, dtype=np.float64))
                out[side] = self.T_table_vive @ T_raw @ T_tt
        return out

    def compute_ee_targets(
        self,
        sample: PikaDualPoseSample,
        q_current_full: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        table_tcp = self.sample_to_table_tcp_mats(sample)
        ee_targets: dict[str, np.ndarray] = {}
        widths_out: dict[str, float] = {}

        if self.retarget_mode == "absolute-table":
            for side in self.profile.active_sides:
                T_ee = self.T_robot_table @ table_tcp[side] @ self.T_pika_to_ee
                ee_targets[side] = mat_to_pose7(T_ee)
                src_side = side if side in sample.widths_m else "right"
                widths_out[side] = float(np.clip(sample.widths_m.get(src_side, 0.05), 0.0, 0.10))
            return ee_targets, widths_out

        # relative-anchor mode
        if self._anchor_table_pos is None:
            fk_cur = self.solver.fk_poses7(q_current_full)
            self._anchor_table_pos = {s: table_tcp[s][:3, 3].copy() for s in self.profile.active_sides}
            self._anchor_src_rot = {
                s: (self.T_robot_table @ table_tcp[s] @ self.T_pika_to_ee)[:3, :3].copy()
                for s in self.profile.active_sides
            }
            self._anchor_robot_ee = {s: pose7_to_mat(fk_cur[s]) for s in self.profile.active_sides}

        assert self._anchor_table_pos is not None
        assert self._anchor_src_rot is not None
        assert self._anchor_robot_ee is not None

        R_rt = self.T_robot_table[:3, :3]
        for side in self.profile.active_sides:
            T_cur = self.T_robot_table @ table_tcp[side] @ self.T_pika_to_ee
            dp_world = R_rt @ (table_tcp[side][:3, 3] - self._anchor_table_pos[side])
            R_rel = T_cur[:3, :3] @ self._anchor_src_rot[side].T
            T_base = self._anchor_robot_ee[side]
            T_tgt = np.eye(4, dtype=np.float64)
            T_tgt[:3, 3] = T_base[:3, 3] + dp_world
            T_tgt[:3, :3] = R_rel @ T_base[:3, :3]
            ee_targets[side] = mat_to_pose7(T_tgt)
            src_side = side if side in sample.widths_m else "right"
            widths_out[side] = float(np.clip(sample.widths_m.get(src_side, 0.05), 0.0, 0.10))
        return ee_targets, widths_out

    def compute_ik_residual(
        self,
        q_full: np.ndarray,
        ee_targets: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        """Return `(max_pos_err_m, max_rot_err_deg)` between FK(`q_full`) and `ee_targets`."""
        fk = self.solver.fk_poses7(q_full)
        max_pos_m = 0.0
        max_rot_deg = 0.0
        for side, tgt_p7 in ee_targets.items():
            if side not in fk:
                continue
            p_err, r_err = se3_world_residual(pose7_to_mat(fk[side]), pose7_to_mat(tgt_p7))
            max_pos_m = max(max_pos_m, float(np.linalg.norm(p_err)))
            max_rot_deg = max(max_rot_deg, float(math.degrees(np.linalg.norm(r_err))))
        return max_pos_m, max_rot_deg

    def solve_step(
        self,
        sample: PikaDualPoseSample,
        q_current_full: np.ndarray,
        *,
        max_nfev: int | None = None,
        use_prev_smoothness: bool = True,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
        ee_targets, pika_widths = self.compute_ee_targets(sample, q_current_full)
        # Keep apply_delta_cap=False so excessive IK jumps are caught by GuardedRobotController
        q_ik = self.solver.solve_frame_lm(
            q_current_full,
            ee_targets,
            q_prev_full=q_current_full if use_prev_smoothness else None,
            max_nfev=self.max_ik_nfev if max_nfev is None else int(max_nfev),
            apply_delta_cap=False,
        )
        robot_widths: dict[str, float] = {}
        for side in self.profile.active_sides:
            q_ik, w_rob = self.profile.map_gripper_width_to_q(q_ik, side, pika_widths[side])
            robot_widths[side] = w_rob
        return q_ik, robot_widths, ee_targets


def run_teleop_loop(
    backend: RealRobotBackend,
    source: PikaPoseSource,
    retargeter: PikaLiveRetargeter,
    *,
    fps: float = 30.0,
    steps: int | None = None,
    stale_timeout_s: float = 0.25,
    ramp_to_first: bool = True,
    ramp_speed_deg_s: float | None = None,
    enforce_clearance: bool = False,
    enforce_ik_residual: bool = False,
    max_ik_pos_err_m: float = 0.03,
    max_ik_rot_err_deg: float = 15.0,
    sleep_realtime: bool = True,
    dry_run: bool = False,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run guarded live/mock Dual Pika -> Dual Piper online retargeting loop."""
    prof = retargeter.profile
    period_s = 1.0 / max(float(fps), 1.0)
    guard = GuardedRobotController(
        profile=prof,
        max_step_rad=max(0.08, float(prof.ik_weights.get("max_joint_delta", 0.08)) * 2.0),
        table_z_robot=retargeter.table_z_robot if retargeter.retarget_mode == "absolute-table" else None,
        enforce_clearance=enforce_clearance,
    )

    history: list[dict[str, Any]] = []
    executed_steps = 0
    ramped = False

    source.start()
    backend.connect()
    try:
        q_cur, w_cur = backend.read_state()
        guard.seed_state(
            q_cur,
            w_cur,
            mode=ArmStateMode.DRY_RUN_MOCK if dry_run else ArmStateMode.ACTIVE_TRACKING,
        )

        # Wait briefly for initial healthy Pika sample before engaging motion
        first_sample: PikaDualPoseSample | None = None
        wait_deadline = time.monotonic() + max(stale_timeout_s * 4.0, 1.0)
        while time.monotonic() < wait_deadline:
            if stop_event is not None and stop_event.is_set():
                break
            cand = source.latest_sample()
            if cand is not None:
                first_sample = cand
                break
            time.sleep(0.01)

        if first_sample is None:
            guard.latch_hold_last_safe("no_initial_pika_pose_sample")
            backend.hold_last_safe()
        else:
            all_acc = all(
                bool(first_sample.accurate.get(s if s in first_sample.accurate else "right", False))
                for s in prof.active_sides
            )
            if not all_acc:
                guard.latch_hold_last_safe("initial_pika_tracking_inaccurate")
                backend.hold_last_safe()
            elif ramp_to_first:
                # Converge initial ramp target fully before entering real-time guarded tracking
                q_first, w_first, ee_first = retargeter.solve_step(
                    first_sample,
                    q_cur,
                    max_nfev=max(80, retargeter.max_ik_nfev),
                    use_prev_smoothness=False,
                )
                if enforce_ik_residual:
                    pos_err_m, rot_err_deg = retargeter.compute_ik_residual(q_first, ee_first)
                    if pos_err_m > max_ik_pos_err_m or rot_err_deg > max_ik_rot_err_deg:
                        guard.latch_hold_last_safe(
                            f"initial_ik_residual_exceeded:pos={pos_err_m:.4f}m,rot={rot_err_deg:.1f}deg"
                        )
                        backend.hold_last_safe()
                if guard.mode != ArmStateMode.HOLD_LAST_SAFE:
                    ramped = backend.ramp_to_q(
                        q_first,
                        w_first,
                        speed_deg_s=ramp_speed_deg_s,
                        stop_event=stop_event,
                    )
                    if not ramped:
                        guard.latch_hold_last_safe("teleop_ramp_to_first_failed")
                        backend.hold_last_safe()
                    else:
                        q_cur, w_cur = q_first, w_first
                        guard.seed_state(
                            q_cur,
                            w_cur,
                            mode=ArmStateMode.DRY_RUN_MOCK if dry_run else ArmStateMode.ACTIVE_TRACKING,
                        )

        deadline = time.monotonic()
        step_idx = 0
        while steps is None or step_idx < steps:
            if stop_event is not None and stop_event.is_set():
                break
            if guard.mode == ArmStateMode.HOLD_LAST_SAFE:
                backend.hold_last_safe()
                break

            try:
                backend.check_health()
            except BaseException as exc:
                guard.latch_hold_last_safe(f"backend_health_fault:{exc}")
                backend.hold_last_safe()
                break

            sample = source.latest_sample()
            now_mono = time.monotonic()
            if sample is None:
                guard.latch_hold_last_safe("missing_pika_pose_sample")
                backend.hold_last_safe()
                break

            age_s = now_mono - sample.received_monotonic_s
            all_acc = all(
                bool(sample.accurate.get(s if s in sample.accurate else "right", False))
                for s in prof.active_sides
            )
            if age_s > stale_timeout_s:
                guard.latch_hold_last_safe(f"stale_pika_pose_s:{age_s:.3f}>{stale_timeout_s:.3f}")
                backend.hold_last_safe()
                break
            if not all_acc:
                guard.latch_hold_last_safe("pika_tracking_inaccurate")
                backend.hold_last_safe()
                break

            q_ik, w_rob, ee_tgts = retargeter.solve_step(sample, q_cur)
            if enforce_ik_residual:
                pos_err_m, rot_err_deg = retargeter.compute_ik_residual(q_ik, ee_tgts)
                if pos_err_m > max_ik_pos_err_m or rot_err_deg > max_ik_rot_err_deg:
                    guard.latch_hold_last_safe(
                        f"ik_residual_exceeded:pos={pos_err_m:.4f}m,rot={rot_err_deg:.1f}deg"
                    )
                    backend.hold_last_safe()
                    break

            q_cmd, w_cmd, mode = guard.filter_command(q_ik, w_rob, input_healthy=True)
            if mode == ArmStateMode.HOLD_LAST_SAFE:
                backend.hold_last_safe()
                break

            backend.write_target(q_cmd, w_cmd)
            delta_rad = float(np.max(np.abs(q_cmd - q_cur)))
            q_cur, w_cur = q_cmd, w_cmd
            executed_steps += 1
            if len(history) < 200:
                history.append(
                    {
                        "step": step_idx,
                        "mode": mode.value,
                        "widths_cmd_m": w_cmd,
                        "max_joint_delta_rad": round(delta_rad, 6),
                    }
                )
            step_idx += 1

            if sleep_realtime and (steps is None or step_idx < steps):
                deadline = next_periodic_deadline(deadline, time.monotonic(), period_s)
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0.0:
                    if stop_event is not None:
                        if stop_event.wait(sleep_s):
                            break
                    else:
                        time.sleep(sleep_s)
    except KeyboardInterrupt:
        guard.latch_hold_last_safe("operator_keyboard_interrupt")
        backend.hold_last_safe()
    finally:
        source.close()
        backend.hold_last_safe()
        backend.disconnect()

    return {
        "robot": prof.name,
        "retarget_mode": retargeter.retarget_mode,
        "fps": fps,
        "executed_steps": executed_steps,
        "ramped_to_first": ramped,
        "dry_run": bool(dry_run),
        "final_mode": guard.mode.value,
        "fault_reasons": list(guard.fault_reasons),
        "history": history,
    }


def run_teleop_cli(args: argparse.Namespace) -> int:
    prof = load_robot_profile(args.robot)
    if args.live:
        tcal = require_live_authorization(
            robot=args.robot,
            live=args.live,
            operator_supported=args.operator_supported,
        )
    else:
        tcal = load_table_calibration(args.robot, require_physical=False)

    retargeter = PikaLiveRetargeter(
        prof,
        table_calib=tcal,
        calibration_dir=args.calibration_dir,
        retarget_mode=args.retarget_mode,
        require_session_calibration=bool(args.live and args.retarget_mode == "absolute-table"),
    )

    if args.source == "mock":
        source: PikaPoseSource = MockPikaPoseSource(fps=args.fps)
    elif args.source == "udp":
        source = UdpJsonPikaPoseSource(host=args.udp_host, port=args.udp_port)
    elif args.source == "docker-ros1":
        rig = load_rig_config(args.rig_config)
        container = str((rig.get("pika_sense") or {}).get("docker_container", "pika-ros"))
        source = DockerRos1PikaPoseSource(container_name=container)
    else:
        raise ValueError(f"Unsupported teleop source: {args.source!r}")

    if args.live:
        settings = load_piper_can_settings(
            prof,
            rig_config_path=args.rig_config,
            left_port=args.left_port,
            right_port=args.right_port,
            speed_percent=args.speed_percent,
            sdk_backend=getattr(args, "sdk", None),
            firmware=getattr(args, "firmware", None),
            enable_arms=bool(args.enable_arms),
        )
        backend: RealRobotBackend = PiperCanBackend(prof, settings)
        summary = run_teleop_loop(
            backend,
            source,
            retargeter,
            fps=args.fps,
            steps=args.steps,
            stale_timeout_s=args.stale_timeout_s,
            ramp_to_first=True,
            ramp_speed_deg_s=args.ramp_speed_deg_s,
            enforce_clearance=True,
            enforce_ik_residual=True,
            sleep_realtime=True,
            dry_run=False,
        )
    else:
        mock_backend = MockBimanualController(prof)
        summary = run_teleop_loop(
            mock_backend,
            source,
            retargeter,
            fps=args.fps,
            steps=args.steps if args.steps is not None else 10,
            stale_timeout_s=args.stale_timeout_s,
            ramp_to_first=True,
            ramp_speed_deg_s=args.ramp_speed_deg_s,
            enforce_clearance=False,
            enforce_ik_residual=False,
            sleep_realtime=False,
            dry_run=True,
        )
        summary["mock_robot"] = True

    if args.out:
        save_json_atomic(args.out, summary)
    print(
        f"[OK] teleop ({args.robot}, source={args.source}, dry_run={not args.live}): "
        f"steps={summary['executed_steps']} @ {args.fps} Hz, mode={summary['final_mode']}"
    )
    return 0
