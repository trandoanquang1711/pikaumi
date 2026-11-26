"""Multi-robot `SE(3)` VLA policy bridge with `piper_sdk` backend & `HOLD_LAST_SAFE` filter (`pika policy-bridge`)."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import threading
import time
from typing import Any, Callable, Protocol
import urllib.error
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.config import save_json_atomic
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


def build_umi_observation_state_11d(
    left_tcp_pose7: np.ndarray,
    right_tcp_pose7: np.ndarray,
    left_width_m: float,
    right_width_m: float,
) -> np.ndarray:
    """Build exact `GEMINI.md` 11D `observation.state` (Left-before-Right).

    Layout: Right TCP in Left TCP frame `[x, y, z (m), rot_col0 (3), rot_col1 (3), left_w_m, right_w_m]`.
    Matches `compute_umi_delta_tensors` in `pikaumi.dataset.umi_delta` bit-for-bit.
    """
    T_l = pose7_to_mat(left_tcp_pose7)
    T_r = pose7_to_mat(right_tcp_pose7)
    T_rel = np.linalg.inv(T_l) @ T_r
    state = np.zeros(11, dtype=np.float32)
    state[0:3] = T_rel[:3, 3]
    state[3:6] = T_rel[:3, 0]
    state[6:9] = T_rel[:3, 1]
    state[9] = float(np.clip(left_width_m, 0.0, 0.10))
    state[10] = float(np.clip(right_width_m, 0.0, 0.10))
    return state


def integrate_umi_delta_step(
    ee_poses7_cur: dict[str, np.ndarray],
    action_14d: np.ndarray,
    *,
    max_trans_step_m: float = 0.03,
    max_rot_step_rad: float = 0.20,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Integrate 14D TCP-local delta action `[dx,dy,dz,rx,ry,rz,next_width_m] x 2` safely."""
    act = np.asarray(action_14d, dtype=np.float64).reshape(14)
    if not np.all(np.isfinite(act)):
        raise ValueError("Policy action_14d contains NaN/Inf")
    out_poses: dict[str, np.ndarray] = {}
    out_widths: dict[str, float] = {}

    for side, offset in (("left", 0), ("right", 7)):
        if side not in ee_poses7_cur:
            continue
        T_cur = pose7_to_mat(ee_poses7_cur[side])
        dx = act[offset : offset + 3].copy()
        norm_dx = float(np.linalg.norm(dx))
        if norm_dx > max_trans_step_m:
            dx = dx * (max_trans_step_m / norm_dx)

        rvec = act[offset + 3 : offset + 6].copy()
        norm_r = float(np.linalg.norm(rvec))
        if norm_r > max_rot_step_rad:
            rvec = rvec * (max_rot_step_rad / norm_r)

        T_delta = np.eye(4, dtype=np.float64)
        T_delta[:3, 3] = dx
        T_delta[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        out_poses[side] = mat_to_pose7(T_cur @ T_delta)
        out_widths[side] = float(np.clip(act[offset + 6], 0.0, 0.10))

    return out_poses, out_widths


class PolicyClient(Protocol):
    """Pluggable VLA policy client returning a 14D action or `(H, 14)` action chunk."""

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray: ...


class MockPolicyClient:
    """Gentle deterministic synthetic policy (2 mm forward reach + smooth jaw closing)."""

    def __init__(self, chunk_size: int = 1) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self._step = 0

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        chunk = np.zeros((self.chunk_size, 14), dtype=np.float64)
        for k in range(self.chunk_size):
            idx = self._step + k
            w = max(0.01, 0.08 - 0.005 * idx)
            chunk[k, 0] = 0.002
            chunk[k, 6] = w
            chunk[k, 7] = 0.002
            chunk[k, 13] = w
        self._step += self.chunk_size
        return chunk


class HttpJsonPolicyClient:
    """HTTP JSON client for remote `pi05` / `vista` / LeRobot VLA policy servers.

    Sends POST JSON with `observation.state` (11 floats), `task`, `robot`, `step`, plus any
    optional camera payloads. Expects JSON response with `"action_chunk"` (`(H, 14)`) or
    `"action"` (`(14,)`). Any HTTP or timeout failure raises immediately so the bridge latches
    `HOLD_LAST_SAFE`.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 0.5,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Policy URL must start with http:// or https://, got {url!r}")
        self.url = url
        self.timeout_s = float(timeout_s)
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        body = json.dumps(observation, allow_nan=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Policy RPC failed ({self.url}): {exc}") from exc

        payload = json.loads(raw)
        if "action_chunk" in payload:
            arr = np.asarray(payload["action_chunk"], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, 14)
            elif arr.ndim != 2 or arr.shape[1] != 14 or arr.shape[0] < 1:
                raise ValueError(f"Invalid action_chunk shape {arr.shape}, expected (H, 14)")
        elif "action" in payload:
            arr = np.asarray(payload["action"], dtype=np.float64).reshape(1, 14)
        else:
            raise ValueError("Policy response JSON missing 'action_chunk' or 'action'")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Policy server returned NaN/Inf action")
        return arr


def _ee_to_pika_tcp_poses(profile: RobotProfile, ee_poses7: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    T_inv = np.linalg.inv(pose7_to_mat(profile.pika_to_ee_pose7))
    return {side: mat_to_pose7(pose7_to_mat(pose) @ T_inv) for side, pose in ee_poses7.items()}


def _pika_tcp_to_ee_poses(profile: RobotProfile, tcp_poses7: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    T_pe = pose7_to_mat(profile.pika_to_ee_pose7)
    return {side: mat_to_pose7(pose7_to_mat(pose) @ T_pe) for side, pose in tcp_poses7.items()}


def _max_ik_residual(
    solver: BimanualIKSolver,
    q_full: np.ndarray,
    ee_targets: dict[str, np.ndarray],
) -> tuple[float, float]:
    fk = solver.fk_poses7(q_full)
    max_pos_m = 0.0
    max_rot_deg = 0.0
    for side, tgt_p7 in ee_targets.items():
        if side not in fk:
            continue
        p_err, r_err = se3_world_residual(pose7_to_mat(fk[side]), pose7_to_mat(tgt_p7))
        max_pos_m = max(max_pos_m, float(np.linalg.norm(p_err)))
        max_rot_deg = max(max_rot_deg, float(math.degrees(np.linalg.norm(r_err))))
    return max_pos_m, max_rot_deg


def run_policy_bridge_loop(
    backend: RealRobotBackend,
    profile: RobotProfile,
    policy_client: PolicyClient,
    *,
    fps: float = 30.0,
    steps: int | None = 10,
    n_action_steps: int = 1,
    task: str = "",
    table_z_robot: float | None = None,
    enforce_clearance: bool = False,
    enforce_ik_residual: bool = False,
    max_ik_pos_err_m: float = 0.03,
    max_ik_rot_err_deg: float = 15.0,
    sleep_realtime: bool = False,
    dry_run: bool = True,
    image_provider: Callable[[], dict[str, Any]] | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run guarded VLA policy inference -> SE(3) delta integration -> IK -> `RealRobotBackend` loop."""
    solver = BimanualIKSolver(profile)
    guard = GuardedRobotController(
        profile=profile,
        max_step_rad=0.08,
        table_z_robot=table_z_robot,
        enforce_clearance=enforce_clearance,
    )
    period_s = 1.0 / max(float(fps), 1.0)
    action_queue: deque[np.ndarray] = deque()
    history: list[dict[str, Any]] = []
    executed_steps = 0
    rpc_queries = 0

    backend.connect()
    try:
        q_cur, w_cur = backend.read_state()
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

            q_cur, w_cur = backend.read_state()
            ee_cur = solver.fk_poses7(q_cur)
            tcp_cur = _ee_to_pika_tcp_poses(profile, ee_cur)

            if "left" in tcp_cur and "right" in tcp_cur:
                w_left_pika = profile.map_robot_width_to_pika("left", w_cur.get("left", 0.05))
                w_right_pika = profile.map_robot_width_to_pika("right", w_cur.get("right", 0.05))
                state_11d = build_umi_observation_state_11d(
                    tcp_cur["left"],
                    tcp_cur["right"],
                    w_left_pika,
                    w_right_pika,
                )
            else:
                only_side = profile.active_sides[0]
                w_only_pika = profile.map_robot_width_to_pika(only_side, w_cur.get(only_side, 0.05))
                state_11d = build_umi_observation_state_11d(
                    tcp_cur[only_side],
                    tcp_cur[only_side],
                    w_only_pika,
                    w_only_pika,
                )

            if not action_queue:
                obs: dict[str, Any] = {
                    "observation.state": state_11d.tolist(),
                    "task": task,
                    "robot": profile.name,
                    "step": step_idx,
                }
                if image_provider is not None:
                    obs.update(image_provider())
                try:
                    chunk = np.asarray(policy_client.predict_action_chunk(obs), dtype=np.float64)
                    if chunk.ndim == 1:
                        chunk = chunk.reshape(1, 14)
                    if chunk.ndim != 2 or chunk.shape[1] != 14 or len(chunk) == 0 or not np.all(np.isfinite(chunk)):
                        raise ValueError(f"Invalid policy chunk shape/values: {chunk.shape}")
                    take = min(len(chunk), max(1, int(n_action_steps)))
                    for k in range(take):
                        action_queue.append(chunk[k])
                    rpc_queries += 1
                except BaseException as exc:
                    guard.latch_hold_last_safe(f"policy_rpc_fault:{exc}")
                    backend.hold_last_safe()
                    break

            act_14d = action_queue.popleft()
            try:
                tcp_next, pika_w_next = integrate_umi_delta_step(tcp_cur, act_14d)
            except BaseException as exc:
                guard.latch_hold_last_safe(f"action_integration_fault:{exc}")
                backend.hold_last_safe()
                break

            ee_next = _pika_tcp_to_ee_poses(profile, tcp_next)
            # Keep apply_delta_cap=False so excessive IK jumps are caught by GuardedRobotController
            q_ik = solver.solve_frame_lm(
                q_cur,
                ee_next,
                q_prev_full=q_cur,
                max_nfev=20,
                apply_delta_cap=False,
            )
            if enforce_ik_residual:
                pos_err_m, rot_err_deg = _max_ik_residual(solver, q_ik, ee_next)
                if pos_err_m > max_ik_pos_err_m or rot_err_deg > max_ik_rot_err_deg:
                    guard.latch_hold_last_safe(
                        f"ik_residual_exceeded:pos={pos_err_m:.4f}m,rot={rot_err_deg:.1f}deg"
                    )
                    backend.hold_last_safe()
                    break

            w_robot_next: dict[str, float] = {}
            for s in profile.active_sides:
                q_ik, w_rob = profile.map_gripper_width_to_q(q_ik, s, pika_w_next[s])
                w_robot_next[s] = w_rob

            q_cmd, w_cmd, mode = guard.filter_command(q_ik, w_robot_next, input_healthy=True)
            if mode == ArmStateMode.HOLD_LAST_SAFE:
                backend.hold_last_safe()
                break

            backend.write_target(q_cmd, w_cmd)
            executed_steps += 1
            if len(history) < 200:
                history.append(
                    {
                        "step": step_idx,
                        "mode": mode.value,
                        "widths_cmd_m": w_cmd,
                        "max_joint_delta_rad": round(float(np.max(np.abs(q_cmd - q_cur))), 6),
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
        backend.hold_last_safe()
        backend.disconnect()

    return {
        "robot": profile.name,
        "fps": fps,
        "steps": executed_steps,
        "rpc_queries": rpc_queries,
        "dry_run": bool(dry_run),
        "mock_robot": isinstance(backend, MockBimanualController),
        "final_mode": guard.mode.value,
        "fault_reasons": list(guard.fault_reasons),
        "history": history,
    }


def run_mock_policy_bridge_loop(
    *,
    robot: str = "piper_pika",
    fps: int = 30,
    steps: int = 10,
) -> dict[str, Any]:
    """Run an offline mock VLA inference + IK + `HOLD_LAST_SAFE` control loop across any target robot."""
    prof = load_robot_profile(robot)
    ctrl = MockBimanualController(prof)
    client = MockPolicyClient(chunk_size=1)
    return run_policy_bridge_loop(
        ctrl,
        prof,
        client,
        fps=fps,
        steps=steps,
        n_action_steps=1,
        enforce_clearance=False,
        enforce_ik_residual=False,
        sleep_realtime=False,
        dry_run=True,
    )


def run_policy_bridge_cli(args: argparse.Namespace) -> int:
    prof = load_robot_profile(args.robot)
    policy_url = getattr(args, "policy_url", None)
    n_action_steps = int(getattr(args, "n_action_steps", 1))
    task = str(getattr(args, "task", ""))

    if policy_url:
        client: PolicyClient = HttpJsonPolicyClient(
            policy_url,
            timeout_s=float(getattr(args, "rpc_timeout_s", 0.5)),
        )
    else:
        client = MockPolicyClient(chunk_size=max(1, n_action_steps))

    if args.live:
        tcal = require_live_authorization(
            robot=args.robot,
            live=args.live,
            operator_supported=args.operator_supported,
        )
        if not policy_url:
            raise ValueError(
                "Live policy-bridge requires --policy-url (e.g. http://127.0.0.1:8000/act) to avoid running mock actions on hardware."
            )
        table_z_robot = float(pose7_to_mat(tcal.T_robot_table_pose7)[2, 3])
        settings = load_piper_can_settings(
            prof,
            rig_config_path=getattr(args, "rig_config", None),
            left_port=getattr(args, "left_port", None),
            right_port=getattr(args, "right_port", None),
            speed_percent=getattr(args, "speed_percent", None),
            sdk_backend=getattr(args, "sdk", None),
            firmware=getattr(args, "firmware", None),
            enable_arms=bool(getattr(args, "enable_arms", False)),
        )
        backend: RealRobotBackend = PiperCanBackend(prof, settings)
        summary = run_policy_bridge_loop(
            backend,
            prof,
            client,
            fps=args.fps,
            steps=args.steps if args.steps > 0 else None,
            n_action_steps=n_action_steps,
            task=task,
            table_z_robot=table_z_robot,
            enforce_clearance=True,
            enforce_ik_residual=True,
            sleep_realtime=True,
            dry_run=False,
        )
    else:
        tcal = load_table_calibration(args.robot, require_physical=False)
        table_z_robot = float(pose7_to_mat(tcal.T_robot_table_pose7)[2, 3])
        ctrl = MockBimanualController(prof)
        summary = run_policy_bridge_loop(
            ctrl,
            prof,
            client,
            fps=args.fps,
            steps=args.steps,
            n_action_steps=n_action_steps,
            task=task,
            table_z_robot=table_z_robot,
            enforce_clearance=False,
            enforce_ik_residual=False,
            sleep_realtime=False,
            dry_run=True,
        )

    if args.out:
        save_json_atomic(args.out, summary)
    print(
        f"[OK] policy-bridge ({args.robot}, dry_run={not args.live}): "
        f"{summary['steps']} steps @ {args.fps} Hz, mode={summary['final_mode']}"
    )
    return 0
