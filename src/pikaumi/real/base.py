"""Safety-first physical robot controller contract enforcing `HOLD_LAST_SAFE` (`GEMINI.md` S-V1.8-9)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Any, Protocol

import numpy as np

from pikaumi.calibration.deployment import TableCalibration, load_table_calibration
from pikaumi.robots.collision import evaluate_config_clearance
from pikaumi.robots.registry import RobotProfile


class ArmStateMode(str, Enum):
    DRY_RUN_MOCK = "DRY_RUN_MOCK"
    ACTIVE_TRACKING = "ACTIVE_TRACKING"
    HOLD_LAST_SAFE = "HOLD_LAST_SAFE"


class RealRobotBackend(Protocol):
    """Uniform real/mock robot backend interface shared across replay, teleop, and policy-bridge."""

    def connect(self) -> None: ...

    def read_state(self) -> tuple[np.ndarray, dict[str, float]]: ...

    def ramp_to_q(
        self,
        q_target_full: np.ndarray,
        widths_m: dict[str, float] | None = None,
        *,
        speed_deg_s: float | None = None,
        timeout_s: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> bool: ...

    def write_target(self, q_target_full: np.ndarray, widths_m: dict[str, float]) -> None: ...

    def check_health(self) -> None: ...

    def hold_last_safe(self) -> None: ...

    def disconnect(self) -> None: ...


def require_live_authorization(
    *,
    robot: str,
    live: bool,
    operator_supported: bool,
) -> TableCalibration:
    """Enforce mandatory safety gates before any physical hardware connection can be opened."""
    if not live:
        raise PermissionError("Live hardware motion is disabled (--dry-run active).")
    if not operator_supported:
        raise PermissionError(
            "Live hardware motion requires explicit on-site human confirmation (--operator-supported)."
        )
    return load_table_calibration(robot, require_physical=True)


@dataclass
class GuardedRobotController:
    """Base safety state machine that latches `HOLD_LAST_SAFE` on any tracking/kinematic fault.

    NOTE (`GEMINI.md`): Firmware `S-V1.8-9` torque-release commands cause gravity-damped arm drop.
    This controller NEVER releases torque automatically as a fault fallback; it strictly holds
    `_last_safe_q` and `_last_safe_widths`.
    """

    profile: RobotProfile
    max_step_rad: float = 0.08
    table_z_robot: float | None = None
    min_inter_arm_clearance_m: float = 0.005
    min_table_clearance_m: float = -0.005
    enforce_clearance: bool = False

    def __post_init__(self) -> None:
        self.mode = ArmStateMode.DRY_RUN_MOCK
        self._last_safe_q = self.profile.home_q.copy()
        self._last_safe_widths = {
            s: min(0.05, self.profile.gripper_max_width_m) for s in self.profile.active_sides
        }
        self.fault_reasons: list[str] = []

    def seed_state(
        self,
        q_init: np.ndarray,
        widths_init: dict[str, float],
        *,
        mode: ArmStateMode = ArmStateMode.ACTIVE_TRACKING,
    ) -> None:
        """Initialize the safety filter from measured hardware feedback before streaming."""
        self._last_safe_q = np.asarray(q_init, dtype=np.float64).copy()
        self._last_safe_widths = {
            s: float(np.clip(widths_init.get(s, 0.0), 0.0, self.profile.gripper_max_width_m))
            for s in self.profile.active_sides
        }
        self.mode = mode

    def latch_hold_last_safe(self, reason: str) -> None:
        self.mode = ArmStateMode.HOLD_LAST_SAFE
        self.fault_reasons.append(reason)

    def filter_command(
        self,
        q_target: np.ndarray,
        widths_target: dict[str, float],
        *,
        input_healthy: bool = True,
    ) -> tuple[np.ndarray, dict[str, float], ArmStateMode]:
        if self.mode == ArmStateMode.HOLD_LAST_SAFE:
            return self._last_safe_q.copy(), dict(self._last_safe_widths), self.mode

        widths_finite = all(
            np.isfinite(widths_target.get(s, 0.0)) for s in self.profile.active_sides
        )
        if not input_healthy or not np.all(np.isfinite(q_target)) or not widths_finite:
            self.latch_hold_last_safe("unhealthy_or_nonfinite_command")
            return self._last_safe_q.copy(), dict(self._last_safe_widths), self.mode

        # Slew-rate & joint-limit clamp
        dq = q_target - self._last_safe_q
        if float(np.max(np.abs(dq))) > self.max_step_rad * 4.0:
            # Excessive jump -> latch HOLD_LAST_SAFE rather than jerking physical arm
            self.latch_hold_last_safe(f"excessive_joint_jump_rad:{float(np.max(np.abs(dq))):.3f}")
            return self._last_safe_q.copy(), dict(self._last_safe_widths), self.mode

        dq_clamped = np.clip(dq, -self.max_step_rad, self.max_step_rad)
        q_next = np.clip(
            self._last_safe_q + dq_clamped,
            self.profile.urdf_model.lower_limits,
            self.profile.urdf_model.upper_limits,
        )

        if self.enforce_clearance:
            ee_links = tuple(self.profile.arms[s].ee_link for s in self.profile.active_sides)
            inter_clr, table_clr = evaluate_config_clearance(
                self.profile.urdf_model,
                q_next,
                table_z_in_robot_world=self.table_z_robot,
                ee_links=ee_links,
            )
            if inter_clr < self.min_inter_arm_clearance_m:
                self.latch_hold_last_safe(f"inter_arm_clearance_m:{inter_clr:.4f}")
                return self._last_safe_q.copy(), dict(self._last_safe_widths), self.mode
            if self.table_z_robot is not None and table_clr < self.min_table_clearance_m:
                self.latch_hold_last_safe(f"table_clearance_m:{table_clr:.4f}")
                return self._last_safe_q.copy(), dict(self._last_safe_widths), self.mode

        w_next = {
            s: float(np.clip(widths_target.get(s, 0.0), 0.0, self.profile.gripper_max_width_m))
            for s in self.profile.active_sides
        }
        self._last_safe_q = q_next
        self._last_safe_widths = w_next
        return q_next.copy(), dict(w_next), self.mode


class MockBimanualController(GuardedRobotController):
    """Zero-I/O mock robot controller & `RealRobotBackend` for deterministic offline testing."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._connected = False
        self.command_log: list[tuple[np.ndarray, dict[str, float]]] = []

    def __enter__(self) -> MockBimanualController:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.hold_last_safe()
        self.disconnect()

    def connect(self) -> None:
        self._connected = True

    def read_state(self) -> tuple[np.ndarray, dict[str, float]]:
        return self._last_safe_q.copy(), dict(self._last_safe_widths)

    def ramp_to_q(
        self,
        q_target_full: np.ndarray,
        widths_m: dict[str, float] | None = None,
        *,
        speed_deg_s: float | None = None,
        timeout_s: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> bool:
        if stop_event is not None and stop_event.is_set():
            return False
        self._last_safe_q = np.clip(
            np.asarray(q_target_full, dtype=np.float64).copy(),
            self.profile.urdf_model.lower_limits,
            self.profile.urdf_model.upper_limits,
        )
        if widths_m is not None:
            self._last_safe_widths = {
                s: float(np.clip(widths_m.get(s, 0.0), 0.0, self.profile.gripper_max_width_m))
                for s in self.profile.active_sides
            }
        self.command_log.append((self._last_safe_q.copy(), dict(self._last_safe_widths)))
        return True

    def write_target(self, q_target_full: np.ndarray, widths_m: dict[str, float]) -> None:
        self._last_safe_q = np.asarray(q_target_full, dtype=np.float64).copy()
        self._last_safe_widths = {
            s: float(np.clip(widths_m.get(s, 0.0), 0.0, self.profile.gripper_max_width_m))
            for s in self.profile.active_sides
        }
        self.command_log.append((self._last_safe_q.copy(), dict(self._last_safe_widths)))

    def check_health(self) -> None:
        return None

    def hold_last_safe(self) -> None:
        if self.mode != ArmStateMode.HOLD_LAST_SAFE:
            self.mode = ArmStateMode.HOLD_LAST_SAFE

    def disconnect(self) -> None:
        self._connected = False

