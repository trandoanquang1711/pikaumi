"""Direct AgileX Piper SocketCAN driver supporting **`pyAgxArm`** (primary, `S-V1.8-9`) and legacy **`piper_sdk`**.

STRICT FIRMWARE `S-V1.8-9` SAFETY INVARIANTS (`GEMINI.md` & `pika-piper-control-guard`):
1. `DisableArm` / `disable()` (`CAN 0x150 / 0x01`) releases holding torque into gravity-damped arm drop.
2. `ResetPiper` / `reset()` / `MotionCtrl_1(0x02)` is NOT a fixed-pose stop and drops holding torque.
3. Therefore, this module contains **ZERO** torque-release or auto-reset calls:
   - On teach/fault/MIT arm states during connect, raise `RuntimeError` demanding manual
     operator recovery with physical arm support.
   - `auto_enable` defaults to `False` — explicit `enable_arms=True` (`--enable-arms`) is required
     to energize motors, and initial joint targets are always seeded from measured feedback at low
     speed (`startup_speed_percent`) before raising speed.
   - On any fault, timeout, or disconnect, latch `HOLD_LAST_SAFE` (freeze target at current
     commanded pose and stop motion cleanly without releasing holding torque).
4. Supports both **`piper_pika`** (100 mm = `0.10 m` / `100,000 µm` Pika gripper, `physical-width`)
   and stock **`piper`** (70 mm = `0.07 m` / `70,000 µm` gripper, `normalized`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np

from pikaumi.config import load_rig_config
from pikaumi.real.streamer import AccelerationLimitedJointTrajectory, JointStreamer
from pikaumi.robots.registry import RobotProfile

RAD_TO_MDEG = 1000.0 * 180.0 / math.pi
MDEG_TO_RAD = math.pi / (180.0 * 1000.0)
PIPER_ARM_DOF = 6
VALID_SDK_BACKENDS = ("auto", "pyagxarm", "piper_sdk")


def q_to_piper_mdeg(q_rad: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """Convert 6-DoF arm joint angles in radians to legacy `piper_sdk` integer milli-degrees."""
    arr = np.asarray(q_rad, dtype=np.float64).reshape(-1)
    if arr.shape != (PIPER_ARM_DOF,):
        raise ValueError(f"Expected ({PIPER_ARM_DOF},) arm joint vector, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Arm joint angles contain NaN/Inf")
    vals = [int(round(float(v) * RAD_TO_MDEG)) for v in arr]
    return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def piper_mdeg_to_q(mdeg: tuple[int, int, int, int, int, int] | np.ndarray) -> np.ndarray:
    """Convert 6-DoF legacy `piper_sdk` milli-degree feedback into radians."""
    arr = np.asarray(mdeg, dtype=np.float64).reshape(-1)
    if arr.shape != (PIPER_ARM_DOF,):
        raise ValueError(f"Expected ({PIPER_ARM_DOF},) mdeg tuple, got {arr.shape}")
    return arr * MDEG_TO_RAD


def resolve_pyagxarm_firmware_profile(firmware_version: str) -> str:
    """Map AgileX Piper firmware string (e.g. `S-V1.8-9`) to `pyAgxArm` `PiperFW` profile name."""
    s = str(firmware_version or "").strip().lower()
    if s in {"v189", "v188", "v183", "default"}:
        return s
    m = re.search(r"(\d+)\.(\d+)[.-](\d+)", s)
    if not m:
        return "v189" if "189" in s else ("v188" if "188" in s else "default")
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    ver = (major, minor, patch)
    if ver >= (1, 8, 9):
        return "v189"
    if ver == (1, 8, 8):
        return "v188"
    if ver >= (1, 8, 3):
        return "v183"
    return "default"


@dataclass(frozen=True)
class PiperGripperRange:
    """Physical micrometer bounds (`[min_microm, max_microm]`) for a single Piper gripper."""

    min_microm: int = 0
    max_microm: int = 100_000  # 100 mm default for piper_pika; 70_000 for stock piper

    @property
    def min_m(self) -> float:
        return float(self.min_microm) / 1_000_000.0

    @property
    def max_m(self) -> float:
        return float(self.max_microm) / 1_000_000.0

    def width_m_to_microm(
        self,
        pika_width_m: float,
        *,
        retarget_mode: str = "physical-width",
        source_max_m: float = 0.10,
    ) -> int:
        w = float(pika_width_m)
        if not math.isfinite(w):
            raise ValueError(f"Non-finite gripper width: {pika_width_m!r}")
        span = max(0, int(self.max_microm) - int(self.min_microm))
        if retarget_mode == "normalized":
            opening = float(np.clip(w / max(source_max_m, 1e-6), 0.0, 1.0))
            return int(round(self.min_microm + opening * span))
        microm = int(round(w * 1_000_000.0))
        return int(np.clip(microm, self.min_microm, self.max_microm))

    def map_target_width_m(
        self,
        pika_width_m: float,
        *,
        retarget_mode: str = "physical-width",
        source_max_m: float = 0.10,
    ) -> float:
        """Map source Pika jaw width (m) into calibrated hardware gripper width (m)."""
        return self.microm_to_width_m(
            self.width_m_to_microm(
                pika_width_m,
                retarget_mode=retarget_mode,
                source_max_m=source_max_m,
            )
        )

    def microm_to_width_m(self, microm: int) -> float:
        clamped = int(np.clip(int(microm), self.min_microm, self.max_microm))
        return float(clamped) / 1_000_000.0

    def clamp_width_m(self, width_m: float) -> float:
        return float(np.clip(float(width_m), self.min_m, self.max_m))


@dataclass(frozen=True)
class PiperCanSettings:
    """Resolved CAN, SDK & safety parameters for Dual Piper (`piper_pika` or stock `piper`)."""

    robot_name: str
    ports: dict[str, str]
    sdk_backend: str = "auto"  # "auto" | "pyagxarm" | "piper_sdk"
    firmware: str = "S-V1.8-9"
    command_rate_hz: float = 200.0
    max_joint_speed_deg_s: float = 180.0
    max_joint_acceleration_deg_s2: float = 720.0
    home_max_joint_speed_deg_s: float = 20.0
    home_timeout_s: float = 30.0
    home_tolerance_deg: float = 3.0
    startup_speed_percent: int = 10
    speed_percent: int = 60
    gripper_effort: int = 1000
    gripper_force_n: float = 1.0
    gripper_ranges: dict[str, PiperGripperRange] = field(default_factory=dict)
    auto_enable: bool = False
    fault_mode: str = "HOLD_LAST_SAFE"
    stale_feedback_timeout_s: float = 0.25


def load_piper_can_settings(
    profile: RobotProfile,
    *,
    rig_config_path: str | Path | None = None,
    left_port: str | None = None,
    right_port: str | None = None,
    speed_percent: int | None = None,
    gripper_effort: int | None = None,
    gripper_force_n: float | None = None,
    sdk_backend: str | None = None,
    firmware: str | None = None,
    enable_arms: bool = False,
) -> PiperCanSettings:
    """Resolve `PiperCanSettings` from `RobotProfile` + `configs/rig.yaml` + CLI overrides."""
    if profile.kind not in {"piper_pika", "piper"}:
        raise ValueError(
            f"PiperCanBackend only supports piper_pika/piper profiles, got robot={profile.name!r} (kind={profile.kind!r})"
        )

    rig = load_rig_config(rig_config_path)
    robots_cfg = rig.get("robots") or {}
    rcfg = robots_cfg.get(profile.name) or robots_cfg.get(profile.kind) or {}
    can_cfg = rcfg.get("can") or {}
    real_cfg = profile.real_config or {}

    resolved_sdk = str(sdk_backend or can_cfg.get("sdk") or real_cfg.get("sdk") or "auto").strip().lower()
    if resolved_sdk not in VALID_SDK_BACKENDS:
        raise ValueError(f"Unsupported sdk_backend={resolved_sdk!r}; choose from {VALID_SDK_BACKENDS}")

    resolved_fw = str(firmware or can_cfg.get("firmware") or real_cfg.get("firmware") or "S-V1.8-9")

    ports: dict[str, str] = {}
    if "left" in profile.active_sides:
        ports["left"] = str(left_port or can_cfg.get("left_port") or "can0")
    if "right" in profile.active_sides:
        ports["right"] = str(right_port or can_cfg.get("right_port") or "can1")

    default_max_mm = float(can_cfg.get("gripper_max_width_mm", profile.gripper_max_width_m * 1000.0))
    default_max_microm = int(round(default_max_mm * 1000.0))

    gripper_ranges: dict[str, PiperGripperRange] = {}
    for side in profile.active_sides:
        min_um = int(can_cfg.get(f"{side}_gripper_min_microm", 0))
        max_um = int(can_cfg.get(f"{side}_gripper_max_microm", default_max_microm))
        gripper_ranges[side] = PiperGripperRange(min_microm=min_um, max_microm=max_um)

    resolved_speed = int(
        speed_percent
        if speed_percent is not None
        else can_cfg.get("speed_percent", real_cfg.get("speed_percent", 60))
    )
    resolved_startup_speed = int(
        can_cfg.get("startup_speed_percent", real_cfg.get("startup_speed_percent", 10))
    )
    resolved_effort = int(
        gripper_effort
        if gripper_effort is not None
        else can_cfg.get("gripper_effort", real_cfg.get("gripper_effort", 1000))
    )
    if gripper_force_n is not None:
        resolved_force_n = float(gripper_force_n)
    elif "gripper_force_n" in can_cfg:
        resolved_force_n = float(can_cfg["gripper_force_n"])
    elif "gripper_force_n" in real_cfg:
        resolved_force_n = float(real_cfg["gripper_force_n"])
    else:
        resolved_force_n = float(resolved_effort) / 1000.0

    return PiperCanSettings(
        robot_name=profile.name,
        ports=ports,
        sdk_backend=resolved_sdk,
        firmware=resolved_fw,
        command_rate_hz=float(real_cfg.get("command_rate_hz", 200.0)),
        max_joint_speed_deg_s=float(real_cfg.get("max_joint_speed_deg_s", 180.0)),
        max_joint_acceleration_deg_s2=float(real_cfg.get("max_joint_acceleration_deg_s2", 720.0)),
        home_max_joint_speed_deg_s=float(real_cfg.get("home_max_joint_speed_deg_s", 20.0)),
        home_timeout_s=float(real_cfg.get("home_timeout_s", 30.0)),
        home_tolerance_deg=float(real_cfg.get("home_tolerance_deg", 3.0)),
        startup_speed_percent=int(np.clip(resolved_startup_speed, 1, 100)),
        speed_percent=int(np.clip(resolved_speed, 1, 100)),
        gripper_effort=int(np.clip(resolved_effort, 0, 5000)),
        gripper_force_n=float(np.clip(resolved_force_n, 0.2, 5.0)),
        gripper_ranges=gripper_ranges,
        auto_enable=bool(enable_arms),
        fault_mode=str(real_cfg.get("fault_mode", "HOLD_LAST_SAFE")),
    )


class PiperArm(Protocol):
    """SDK-agnostic SI-unit protocol for a single 6-DoF AgileX Piper arm + gripper over CAN."""

    side: str
    can_port: str

    def connect(
        self,
        *,
        enable: bool,
        startup_speed_percent: int,
        speed_percent: int,
        timeout_s: float = 5.0,
    ) -> None: ...

    def read_q_rad(self) -> np.ndarray: ...

    def read_gripper_width_m(self) -> float: ...

    def send_q_rad(self, q_rad: np.ndarray) -> None: ...

    def send_gripper_width_m(self, width_m: float, force_n: float) -> None: ...

    def check_arm_health(self) -> None: ...

    def disconnect(self) -> None: ...


def _extract_status_payload(status_msg: Any) -> Any:
    """Unwrap either `pyAgxArm` `MessageAbstract.msg` or legacy `piper_sdk` `.arm_status`."""
    if status_msg is None:
        return None
    if hasattr(status_msg, "msg"):
        return status_msg.msg
    if hasattr(status_msg, "arm_status") and not isinstance(status_msg.arm_status, int):
        return status_msg.arm_status
    return status_msg


def _inspect_arm_status_for_unsafe_mode(status_msg: Any, can_port: str) -> str | None:
    """Return a non-None fault description if status is missing or the arm is in teach/MIT/fault state.

    We NEVER auto-reset on Firmware S-V1.8-9 because reset removes holding torque!
    """
    arm_st = _extract_status_payload(status_msg)
    if arm_st is None:
        return (
            f"Piper on {can_port} returned no arm status frame (status_msg is None). "
            "Check SocketCAN interface, power, and cabling before commanding motion."
        )
    ctrl_mode = getattr(arm_st, "ctrl_mode", 0x01)
    arm_status = getattr(arm_st, "arm_status", 0)
    teach_status = getattr(arm_st, "teach_status", 0)
    # pyAgxArm uses `mode_feedback` (0x04 / 0x06 for MIT on V188/V189); legacy piper_sdk uses `mode_feed` (0xAD)
    mode_feed = getattr(arm_st, "mode_feed", None)
    mode_feedback = getattr(arm_st, "mode_feedback", None)
    if ctrl_mode == 0x02 or teach_status not in (0, None):
        return (
            f"Piper on {can_port} is in TEACH mode (ctrl_mode={ctrl_mode!r}, teach_status={teach_status}). "
            "Firmware S-V1.8-9 forbids auto-reset (risk of arm drop). "
            "Physically support the arm, exit teach mode manually, then retry."
        )
    if mode_feed == 0xAD or mode_feedback in (0x04, 0x06, 0xAD):
        return (
            f"Piper on {can_port} is in MIT mode (mode_feed={mode_feed!r}, mode_feedback={mode_feedback!r}). "
            "Physically support the arm and switch back to standard CAN position mode manually."
        )
    if arm_status not in (0, None):
        return (
            f"Piper on {can_port} reported arm_status={arm_status} (non-zero fault). "
            "Firmware S-V1.8-9 forbids automatic reset. Physically support the arm and clear the fault manually."
        )
    return None


def _extract_msg_timestamp(msg: Any) -> float | None:
    """Extract SDK feedback timestamp (`timestamp` in pyAgxArm, `time_stamp` in piper_sdk)."""
    if msg is None:
        return None
    for attr in ("timestamp", "time_stamp"):
        val = getattr(msg, attr, None)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _extract_msg_hz(msg: Any) -> float | None:
    """Extract SDK feedback rate (`hz` in pyAgxArm, `Hz` in piper_sdk) when available."""
    if msg is None:
        return None
    for attr in ("hz", "Hz"):
        val = getattr(msg, attr, None)
        if isinstance(val, (int, float)):
            return float(val)
    return None


class _CanFeedbackFreshnessTracker:
    """Tracks monotonic staleness of SDK CAN feedback timestamps (latch fault after `timeout_s`)."""

    def __init__(self, can_port: str, stream_name: str, timeout_s: float = 0.25) -> None:
        self.can_port = can_port
        self.stream_name = stream_name
        self.timeout_s = max(0.01, float(timeout_s))
        self._last_ts: float | None = None
        self._last_change_mono: float | None = None

    def reset(self) -> None:
        self._last_ts = None
        self._last_change_mono = None

    def observe(self, msg: Any, *, now_mono: float | None = None) -> None:
        if msg is None:
            raise RuntimeError(f"Missing CAN {self.stream_name} feedback on {self.can_port}")
        now = time.monotonic() if now_mono is None else float(now_mono)
        hz = _extract_msg_hz(msg)
        if hz is not None and hz <= 0.0:
            raise RuntimeError(
                f"stale_can_feedback_hz_zero:{self.stream_name} on {self.can_port} (hz={hz:.2f})"
            )
        ts = _extract_msg_timestamp(msg)
        if ts is None:
            return
        if ts <= 0.0:
            raise RuntimeError(
                f"uninitialized_can_feedback_ts:{self.stream_name} on {self.can_port} (ts={ts})"
            )
        if self._last_ts is None or ts != self._last_ts:
            self._last_ts = ts
            self._last_change_mono = now
            return
        assert self._last_change_mono is not None
        stale_s = now - self._last_change_mono
        if stale_s > self.timeout_s:
            raise RuntimeError(
                f"stale_can_feedback_s:{self.stream_name} on {self.can_port} "
                f"({stale_s:.3f}>{self.timeout_s:.3f})"
            )


class PyAgxPiperArm:
    """Primary AgileX Piper arm wrapper around next-gen **`pyAgxArm`** (native SI units, `PiperFW.V189`).

    Follows exact `pyAgxArm` lifecycle:
      `create_agx_arm_config` -> `AgxArmFactory.create_arm` -> `init_effector(AGX_GRIPPER)` BEFORE `connect()`.
    Enforces strict `S-V1.8-9` invariants: zero `disable()`/`reset()`/`electronic_emergency_stop()` calls.
    """

    def __init__(
        self,
        side: str,
        can_port: str,
        *,
        firmware: str = "S-V1.8-9",
        interface: str = "socketcan",
        stale_feedback_timeout_s: float = 0.25,
        sdk_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.side = side
        self.can_port = can_port
        self.firmware = firmware
        self.firmware_profile = resolve_pyagxarm_firmware_profile(firmware)
        self.interface = interface
        self.stale_feedback_timeout_s = float(stale_feedback_timeout_s)
        self._sdk_factory = sdk_factory
        self._robot: Any | None = None
        self._effector: Any | None = None
        self._connected = False
        self._status_freshness = _CanFeedbackFreshnessTracker(
            can_port, "arm_status", timeout_s=self.stale_feedback_timeout_s
        )
        self._joint_freshness = _CanFeedbackFreshnessTracker(
            can_port, "joint_angles", timeout_s=self.stale_feedback_timeout_s
        )

    def _create_robot(self) -> Any:
        if self._sdk_factory is not None:
            return self._sdk_factory(self.can_port, self.firmware_profile)
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
        except ImportError as exc:  # pragma: no cover - tested via injected sdk_factory
            raise ImportError(
                'pyAgxArm is not installed (not published on PyPI). Install with: '
                '`uv pip install "git+https://github.com/agilexrobotics/pyAgxArm.git" python-can` '
                "or `uv pip install -e '.[piper]'` (or install `piper-sdk` from PyPI / pass `--sdk piper_sdk`)."
            ) from exc

        fw_map = {
            "v189": getattr(PiperFW, "V189", "v189"),
            "v188": getattr(PiperFW, "V188", "v188"),
            "v183": getattr(PiperFW, "V183", "v183"),
            "default": getattr(PiperFW, "DEFAULT", "default"),
        }
        cfg = create_agx_arm_config(
            robot=ArmModel.PIPER,
            comm="can",
            firmeware_version=fw_map[self.firmware_profile],
            interface=self.interface,
            channel=self.can_port,
        )
        return AgxArmFactory.create_arm(cfg)

    def connect(
        self,
        *,
        enable: bool,
        startup_speed_percent: int,
        speed_percent: int,
        timeout_s: float = 5.0,
    ) -> None:
        robot = self._create_robot()
        # CRITICAL pyAgxArm contract: init_effector MUST be called BEFORE connect(), once per instance
        effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        robot.connect()
        self._robot = robot
        self._effector = effector
        self._connected = True
        self._status_freshness.reset()
        self._joint_freshness.reset()

        deadline = time.monotonic() + max(timeout_s, 0.2)
        status_msg = None
        while time.monotonic() < deadline:
            cand = robot.get_arm_status()
            if cand is not None:
                ts = _extract_msg_timestamp(cand)
                hz = _extract_msg_hz(cand)
                if ts is None or ts > 0.0:
                    status_msg = cand
                    # pyAgxArm FPSManager starts at hz=0.0 until its first 0.1s calculation tick
                    if hz is None or hz > 0.0:
                        break
            time.sleep(0.01)

        unsafe_reason = _inspect_arm_status_for_unsafe_mode(status_msg, self.can_port)
        if unsafe_reason is not None:
            raise RuntimeError(unsafe_reason)
        self._status_freshness.observe(status_msg)

        # Wait for initial measured joint angles before any motion command
        q_init: np.ndarray | None = None
        while time.monotonic() < deadline:
            try:
                q_init = self.read_q_rad()
                break
            except RuntimeError:
                time.sleep(0.01)
        if q_init is None:
            raise RuntimeError(f"Timed out waiting for initial pyAgxArm joint feedback on {self.can_port}")

        if not enable:
            arm_st = _extract_status_payload(status_msg)
            ctrl_mode = getattr(arm_st, "ctrl_mode", None)
            if ctrl_mode != 0x01:
                raise RuntimeError(
                    f"Piper on {self.can_port} is not active in CAN control mode (ctrl_mode={ctrl_mode!r}) "
                    "and --enable-arms was not passed (auto_enable=False). "
                    "Verify workspace clearance, physically supervise the arms, and re-run with --enable-arms."
                )
            robot.set_speed_percent(int(startup_speed_percent))
            robot.move_j(q_init.tolist())
            robot.set_speed_percent(int(speed_percent))
            return

        # Explicit enable: CAN joint mode -> low speed -> enable() -> seed move_j(measured) -> target speed
        if hasattr(robot, "set_normal_mode"):
            robot.set_normal_mode()
        elif hasattr(robot, "set_motion_mode"):
            robot.set_motion_mode("j")
        robot.set_speed_percent(int(startup_speed_percent))
        enable_deadline = time.monotonic() + max(timeout_s, 0.2)
        enabled = False
        while time.monotonic() < enable_deadline:
            if bool(robot.enable()):
                enabled = True
                break
            time.sleep(0.02)
        if not enabled:
            raise RuntimeError(
                f"Timed out waiting for pyAgxArm Piper on {self.can_port} to enable within {timeout_s:.1f}s."
            )

        # Re-read measured pose and seed move_j immediately so arm NEVER jumps to zero
        q_init = self.read_q_rad()
        robot.move_j(q_init.tolist())
        robot.set_speed_percent(int(speed_percent))

    def _require_robot(self) -> tuple[Any, Any]:
        if self._robot is None or self._effector is None or not self._connected:
            raise RuntimeError(f"PyAgxPiperArm({self.side}, {self.can_port}) is not connected")
        return self._robot, self._effector

    def read_q_rad(self) -> np.ndarray:
        robot, _ = self._require_robot()
        msg_wrap = robot.get_joint_angles()
        if msg_wrap is None:
            raise RuntimeError(f"pyAgxArm joint feedback unavailable on {self.can_port}")
        ts = _extract_msg_timestamp(msg_wrap)
        if ts is not None and ts <= 0.0:
            raise RuntimeError(f"pyAgxArm joint feedback uninitialized on {self.can_port} (ts={ts})")
        self._joint_freshness.observe(msg_wrap)
        vals = getattr(msg_wrap, "msg", msg_wrap)
        arr = np.asarray(vals, dtype=np.float64).reshape(-1)
        if arr.shape != (PIPER_ARM_DOF,):
            raise ValueError(f"Expected {PIPER_ARM_DOF} joints from pyAgxArm, got {arr.shape}")
        return arr

    def read_gripper_width_m(self) -> float:
        _, effector = self._require_robot()
        gs = effector.get_gripper_status()
        if gs is None:
            return 0.0
        payload = getattr(gs, "msg", gs)
        return float(getattr(payload, "value", getattr(payload, "width", 0.0)))

    def send_q_rad(self, q_rad: np.ndarray) -> None:
        robot, _ = self._require_robot()
        arr = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if arr.shape != (PIPER_ARM_DOF,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"Invalid pyAgxArm joint command: {arr!r}")
        robot.move_j([float(v) for v in arr])

    def send_gripper_width_m(self, width_m: float, force_n: float) -> None:
        _, effector = self._require_robot()
        w = float(width_m)
        f = float(force_n)
        if not (math.isfinite(w) and math.isfinite(f)):
            raise ValueError(f"Non-finite gripper command: width_m={w!r}, force_n={f!r}")
        effector.move_gripper_m(value=w, force=f)

    def check_arm_health(self) -> None:
        robot, _ = self._require_robot()
        status_msg = robot.get_arm_status()
        unsafe_reason = _inspect_arm_status_for_unsafe_mode(status_msg, self.can_port)
        if unsafe_reason is not None:
            raise RuntimeError(unsafe_reason)
        self._status_freshness.observe(status_msg)
        joint_msg = robot.get_joint_angles()
        self._joint_freshness.observe(joint_msg)

    def disconnect(self) -> None:
        """Disconnect socket cleanly WITHOUT sending torque-release or reset frames."""
        if self._robot is not None and self._connected:
            disconnect_fn = getattr(self._robot, "disconnect", None)
            if callable(disconnect_fn):
                disconnect_fn()
        self._connected = False


class PiperSdkArm:
    """Legacy AgileX Piper arm wrapper around `piper_sdk.C_PiperInterface_V2`.

    NEVER calls torque-release or reset frames; enforces seed-from-measured-pose startup.
    """

    def __init__(
        self,
        side: str,
        can_port: str,
        *,
        stale_feedback_timeout_s: float = 0.25,
        sdk_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.side = side
        self.can_port = can_port
        self.stale_feedback_timeout_s = float(stale_feedback_timeout_s)
        self._sdk_factory = sdk_factory
        self._arm: Any | None = None
        self._connected = False
        self._last_feedback_monotonic: float = 0.0
        self._status_freshness = _CanFeedbackFreshnessTracker(
            can_port, "arm_status", timeout_s=self.stale_feedback_timeout_s
        )
        self._joint_freshness = _CanFeedbackFreshnessTracker(
            can_port, "joint_msgs", timeout_s=self.stale_feedback_timeout_s
        )

    def _create_sdk(self) -> Any:
        if self._sdk_factory is not None:
            return self._sdk_factory(self.can_port)
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError as exc:  # pragma: no cover - tested via injected sdk_factory
            raise ImportError(
                "piper_sdk is not installed. Recommended: install next-gen `pyAgxArm` (`pip install pyAgxArm`) "
                "or legacy `piper-sdk` (`uv pip install -e '.[piper-legacy]'`)."
            ) from exc
        return C_PiperInterface_V2(self.can_port)

    def connect(
        self,
        *,
        enable: bool,
        startup_speed_percent: int,
        speed_percent: int,
        timeout_s: float = 5.0,
    ) -> None:
        arm = self._create_sdk()
        arm.ConnectPort()
        self._arm = arm
        self._connected = True
        self._status_freshness.reset()
        self._joint_freshness.reset()

        deadline = time.monotonic() + max(timeout_s, 0.2)
        status_msg = None
        while time.monotonic() < deadline:
            cand = arm.GetArmStatus()
            if cand is not None:
                ts = _extract_msg_timestamp(cand)
                hz = _extract_msg_hz(cand)
                # C_PiperInterface_V2 returns all-zero struct (time_stamp == 0.0) before first CAN frame arrives
                if ts is None or ts > 0.0:
                    status_msg = cand
                    if hz is None or hz > 0.0:
                        break
            time.sleep(0.01)

        unsafe_reason = _inspect_arm_status_for_unsafe_mode(status_msg, self.can_port)
        if unsafe_reason is not None:
            raise RuntimeError(unsafe_reason)
        self._status_freshness.observe(status_msg)

        # Wait until live joint CAN feedback has arrived (non-zero time_stamp) so arm never jumps to zero
        init_mdeg: tuple[int, int, int, int, int, int] | None = None
        while time.monotonic() < deadline:
            try:
                init_mdeg = self.read_mdeg()
                break
            except RuntimeError:
                time.sleep(0.01)
        if init_mdeg is None:
            raise RuntimeError(
                f"Timed out waiting for live piper_sdk joint CAN feedback (time_stamp > 0) on {self.can_port}"
            )

        if not enable:
            arm_st = _extract_status_payload(status_msg)
            ctrl_mode = getattr(arm_st, "ctrl_mode", None)
            if ctrl_mode != 0x01:
                raise RuntimeError(
                    f"Piper on {self.can_port} is not active in CAN control mode (ctrl_mode={ctrl_mode!r}) "
                    "and --enable-arms was not passed (auto_enable=False). "
                    "Verify workspace clearance, physically supervise the arms, and re-run with --enable-arms."
                )
            arm.MotionCtrl_2(0x01, 0x01, int(startup_speed_percent), 0x00)
            arm.JointCtrl(*init_mdeg)
            arm.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
            return

        arm.MotionCtrl_2(0x01, 0x01, int(startup_speed_percent), 0x00)
        enable_deadline = time.monotonic() + max(timeout_s, 0.2)
        enabled = False
        while time.monotonic() < enable_deadline:
            if bool(arm.EnablePiper()):
                enabled = True
                break
            time.sleep(0.02)
        if not enabled:
            raise RuntimeError(
                f"Timed out waiting for Piper on {self.can_port} to enable within {timeout_s:.1f}s."
            )

        init_mdeg = self.read_mdeg()
        arm.JointCtrl(*init_mdeg)
        arm.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)

    def _require_arm(self) -> Any:
        if self._arm is None or not self._connected:
            raise RuntimeError(f"PiperSdkArm({self.side}, {self.can_port}) is not connected")
        return self._arm

    def read_mdeg(self) -> tuple[int, int, int, int, int, int]:
        arm = self._require_arm()
        msg = arm.GetArmJointMsgs()
        if msg is None:
            raise RuntimeError(f"piper_sdk joint feedback unavailable on {self.can_port}")
        ts = _extract_msg_timestamp(msg)
        if ts is not None and ts <= 0.0:
            raise RuntimeError(
                f"piper_sdk joint feedback uninitialized on {self.can_port} (time_stamp={ts})"
            )
        self._joint_freshness.observe(msg)
        js = getattr(msg, "joint_state", msg)
        vals = (
            int(js.joint_1),
            int(js.joint_2),
            int(js.joint_3),
            int(js.joint_4),
            int(js.joint_5),
            int(js.joint_6),
        )
        self._last_feedback_monotonic = time.monotonic()
        return vals

    def read_gripper_microm(self) -> int:
        arm = self._require_arm()
        msg = arm.GetArmGripperMsgs()
        gs = getattr(msg, "gripper_state", msg)
        return int(getattr(gs, "grippers_angle", 0))

    def send_joint_mdeg(self, mdeg: tuple[int, int, int, int, int, int]) -> None:
        arm = self._require_arm()
        arm.JointCtrl(mdeg[0], mdeg[1], mdeg[2], mdeg[3], mdeg[4], mdeg[5])

    def send_gripper_microm(self, opening_microm: int, effort: int) -> None:
        arm = self._require_arm()
        arm.GripperCtrl(int(opening_microm), int(effort), 0x01, 0x00)

    # SI-unit PiperArm protocol methods
    def read_q_rad(self) -> np.ndarray:
        return piper_mdeg_to_q(self.read_mdeg())

    def read_gripper_width_m(self) -> float:
        return float(self.read_gripper_microm()) / 1_000_000.0

    def send_q_rad(self, q_rad: np.ndarray) -> None:
        self.send_joint_mdeg(q_to_piper_mdeg(q_rad))

    def send_gripper_width_m(self, width_m: float, force_n: float) -> None:
        microm = int(round(float(width_m) * 1_000_000.0))
        effort = int(np.clip(round(float(force_n) * 1000.0), 0, 5000))
        self.send_gripper_microm(microm, effort)

    def check_arm_health(self) -> None:
        arm = self._require_arm()
        status_msg = arm.GetArmStatus()
        unsafe_reason = _inspect_arm_status_for_unsafe_mode(status_msg, self.can_port)
        if unsafe_reason is not None:
            raise RuntimeError(unsafe_reason)
        self._status_freshness.observe(status_msg)
        joint_msg = arm.GetArmJointMsgs()
        self._joint_freshness.observe(joint_msg)

    def disconnect(self) -> None:
        """Disconnect socket cleanly WITHOUT sending torque-release or reset frames."""
        if self._arm is not None and self._connected:
            disconnect_fn = getattr(self._arm, "DisconnectPort", None)
            if callable(disconnect_fn):
                disconnect_fn()
        self._connected = False


def resolve_active_sdk_backend(requested: str) -> str:
    """Resolve `'auto'` -> `'pyagxarm'` (preferred) or `'piper_sdk'` if only legacy SDK is installed."""
    req = str(requested or "auto").strip().lower()
    if req in {"pyagxarm", "piper_sdk"}:
        return req
    if importlib.util.find_spec("pyAgxArm") is not None:
        return "pyagxarm"
    if importlib.util.find_spec("piper_sdk") is not None:
        return "piper_sdk"
    return "pyagxarm"


def create_piper_arm(side: str, can_port: str, settings: PiperCanSettings) -> PiperArm:
    """Factory creating either `PyAgxPiperArm` (preferred) or legacy `PiperSdkArm`."""
    backend = resolve_active_sdk_backend(settings.sdk_backend)
    if backend == "piper_sdk":
        return PiperSdkArm(
            side,
            can_port,
            stale_feedback_timeout_s=settings.stale_feedback_timeout_s,
        )
    return PyAgxPiperArm(
        side,
        can_port,
        firmware=settings.firmware,
        stale_feedback_timeout_s=settings.stale_feedback_timeout_s,
    )


class MockPyAgxGripper:
    """In-memory fake `pyAgxArm` `AgxGripper` effector for offline unit/integration tests."""

    def __init__(self, initial_width_m: float = 0.05) -> None:
        self.width_m = float(initial_width_m)
        self.force_n = 1.0
        self.call_log: list[tuple[str, tuple[Any, ...]]] = []

    def move_gripper_m(self, value: float = 0.0, force: float = 1.0) -> None:
        self.width_m = float(value)
        self.force_n = float(force)
        self.call_log.append(("move_gripper_m", (float(value), float(force))))

    def get_gripper_status(self) -> Any:
        @dataclass
        class _Payload:
            value: float
            force: float
            mode: int = 0

        @dataclass
        class _MsgWrap:
            msg: _Payload
            hz: float = 200.0
            timestamp: float = 0.0

        return _MsgWrap(msg=_Payload(value=self.width_m, force=self.force_n), timestamp=time.monotonic())


class MockPyAgxArm:
    """In-memory deterministic fake `pyAgxArm` Piper arm matching exact `pyAgxArm` v189 contracts."""

    class OPTIONS:
        class EFFECTOR:
            AGX_GRIPPER = "agx_gripper"

    def __init__(
        self,
        can_port: str,
        *,
        firmware_profile: str = "v189",
        initial_q_rad: tuple[float, ...] = (0.0, 1.45, -1.35, 0.0, 0.45, 0.0),
        initial_gripper_m: float = 0.05,
        ctrl_mode: int = 0x01,
        arm_status: int = 0,
        teach_status: int = 0,
        mode_feedback: int = 0x01,
        enable_succeeds: bool = True,
        hz: float = 200.0,
        freeze_timestamps: bool = False,
        return_none_status: bool = False,
    ) -> None:
        self.can_port = can_port
        self.firmware_profile = firmware_profile
        self.joint_rad = [float(v) for v in initial_q_rad]
        self.ctrl_mode = int(ctrl_mode)
        self.arm_status = int(arm_status)
        self.teach_status = int(teach_status)
        self.mode_feedback = int(mode_feedback)
        self.enable_succeeds = bool(enable_succeeds)
        self.hz = float(hz)
        self.freeze_timestamps = bool(freeze_timestamps)
        self.return_none_status = bool(return_none_status)
        self._frozen_ts = time.monotonic()
        self.connected = False
        self.enabled = ctrl_mode == 0x01
        self.effector: MockPyAgxGripper | None = None
        self._initial_gripper_m = float(initial_gripper_m)
        self.call_log: list[tuple[str, tuple[Any, ...]]] = []

    def _current_ts(self) -> float:
        if self.freeze_timestamps:
            return self._frozen_ts
        self._frozen_ts = time.monotonic()
        return self._frozen_ts

    def init_effector(self, effector_type: str) -> MockPyAgxGripper:
        if self.connected:
            raise RuntimeError("pyAgxArm requires init_effector() BEFORE connect()")
        if self.effector is not None:
            raise RuntimeError("pyAgxArm init_effector() can only be called once per robot instance")
        self.effector = MockPyAgxGripper(initial_width_m=self._initial_gripper_m)
        self.call_log.append(("init_effector", (effector_type,)))
        return self.effector

    def connect(self) -> None:
        self.connected = True
        self.call_log.append(("connect", (self.can_port,)))

    def disconnect(self) -> None:
        self.connected = False
        self.call_log.append(("disconnect", (self.can_port,)))

    def set_normal_mode(self) -> None:
        self.call_log.append(("set_normal_mode", ()))

    def set_speed_percent(self, speed_percent: int) -> None:
        self.call_log.append(("set_speed_percent", (int(speed_percent),)))

    def enable(self) -> bool:
        self.call_log.append(("enable", ()))
        if self.enable_succeeds:
            self.enabled = True
            self.ctrl_mode = 0x01
            return True
        return False

    def move_j(self, joints: list[float]) -> None:
        self.joint_rad = [float(v) for v in joints]
        self.call_log.append(("move_j", tuple(self.joint_rad)))

    def get_arm_status(self) -> Any:
        if self.return_none_status:
            return None

        @dataclass
        class _StatusPayload:
            ctrl_mode: int
            arm_status: int
            teach_status: int
            mode_feedback: int

        @dataclass
        class _MsgWrap:
            msg: _StatusPayload
            hz: float = 200.0
            timestamp: float = 0.0

        return _MsgWrap(
            msg=_StatusPayload(
                ctrl_mode=self.ctrl_mode,
                arm_status=self.arm_status,
                teach_status=self.teach_status,
                mode_feedback=self.mode_feedback,
            ),
            hz=self.hz,
            timestamp=self._current_ts(),
        )

    def get_joint_angles(self) -> Any:
        @dataclass
        class _MsgWrap:
            msg: list[float]
            hz: float = 200.0
            timestamp: float = 0.0

        return _MsgWrap(msg=list(self.joint_rad), hz=self.hz, timestamp=self._current_ts())


class MockPiperSdkArm:
    """In-memory deterministic fake legacy `C_PiperInterface_V2` for offline verification & unit tests."""

    def __init__(
        self,
        can_port: str,
        *,
        initial_mdeg: tuple[int, int, int, int, int, int] = (0, 83079, -77349, 0, 25783, 0),
        initial_gripper_microm: int = 50_000,
        ctrl_mode: int = 0x01,
        arm_status: int = 0,
        teach_status: int = 0,
        mode_feed: int = 0,
        enable_succeeds: bool = True,
        hz: float = 200.0,
        freeze_timestamps: bool = False,
        uninitialized_frames: int = 0,
        return_none_status: bool = False,
    ) -> None:
        self.can_port = can_port
        self.joint_mdeg = list(initial_mdeg)
        self.gripper_microm = int(initial_gripper_microm)
        self.ctrl_mode = int(ctrl_mode)
        self.arm_status = int(arm_status)
        self.teach_status = int(teach_status)
        self.mode_feed = int(mode_feed)
        self.enable_succeeds = bool(enable_succeeds)
        self.hz = float(hz)
        self.freeze_timestamps = bool(freeze_timestamps)
        self._uninit_remaining = int(uninitialized_frames)
        self.return_none_status = bool(return_none_status)
        self._frozen_ts = time.monotonic()
        self.connected = False
        self.enabled = ctrl_mode == 0x01
        self.call_log: list[tuple[str, tuple[Any, ...]]] = []

    def _current_ts(self) -> float:
        if self._uninit_remaining > 0:
            self._uninit_remaining -= 1
            return 0.0
        if self.freeze_timestamps:
            return self._frozen_ts
        self._frozen_ts = time.monotonic()
        return self._frozen_ts

    def ConnectPort(self) -> None:
        self.connected = True
        self.call_log.append(("ConnectPort", (self.can_port,)))

    def DisconnectPort(self) -> None:
        self.connected = False
        self.call_log.append(("DisconnectPort", (self.can_port,)))

    def EnablePiper(self) -> bool:
        self.call_log.append(("EnablePiper", ()))
        if self.enable_succeeds:
            self.enabled = True
            self.ctrl_mode = 0x01
            return True
        return False

    def MotionCtrl_2(self, ctrl_mode: int, move_mode: int, speed_percent: int, is_mit: int) -> None:
        self.call_log.append(("MotionCtrl_2", (ctrl_mode, move_mode, speed_percent, is_mit)))

    def JointCtrl(self, j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> None:
        self.joint_mdeg = [int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)]
        self.call_log.append(("JointCtrl", tuple(self.joint_mdeg)))

    def GripperCtrl(self, opening_microm: int, effort: int, code: int, set_zero: int) -> None:
        self.gripper_microm = int(opening_microm)
        self.call_log.append(("GripperCtrl", (int(opening_microm), int(effort), int(code), int(set_zero))))

    def GetArmStatus(self) -> Any:
        if self.return_none_status:
            return None

        @dataclass
        class _Inner:
            ctrl_mode: int
            arm_status: int
            teach_status: int
            mode_feed: int

        @dataclass
        class _Outer:
            arm_status: _Inner
            time_stamp: float = 0.0
            Hz: float = 200.0

        return _Outer(
            arm_status=_Inner(
                ctrl_mode=self.ctrl_mode,
                arm_status=self.arm_status,
                teach_status=self.teach_status,
                mode_feed=self.mode_feed,
            ),
            time_stamp=self._current_ts(),
            Hz=self.hz,
        )

    def GetArmJointMsgs(self) -> Any:
        @dataclass
        class _JS:
            joint_1: int
            joint_2: int
            joint_3: int
            joint_4: int
            joint_5: int
            joint_6: int

        @dataclass
        class _Msg:
            joint_state: _JS
            time_stamp: float = 0.0
            Hz: float = 200.0

        return _Msg(joint_state=_JS(*self.joint_mdeg), time_stamp=self._current_ts(), Hz=self.hz)

    def GetArmGripperMsgs(self) -> Any:
        @dataclass
        class _GS:
            grippers_angle: int

        @dataclass
        class _Msg:
            gripper_state: _GS
            time_stamp: float = 0.0

        return _Msg(gripper_state=_GS(grippers_angle=self.gripper_microm), time_stamp=self._current_ts())


class PiperJointStreamer(JointStreamer):
    """Fixed-rate dual-arm SI-unit command streamer supporting both `pyAgxArm` and `piper_sdk`."""

    def __init__(
        self,
        arms: dict[str, PiperArm],
        settings: PiperCanSettings,
        initial_q_by_side: dict[str, np.ndarray],
        initial_gripper_m: dict[str, float] | None = None,
    ) -> None:
        self.arms = dict(arms)
        self.settings = settings
        self._trajs: dict[str, AccelerationLimitedJointTrajectory] = {}
        # Start with `None` targets so connecting NEVER slams grippers shut before an explicit command
        self._gripper_targets_m: dict[str, float | None] = {}
        self._last_sent_gripper_m: dict[str, float | None] = {}
        self._hold_latched = False

        max_vel_rad = math.radians(settings.max_joint_speed_deg_s)
        max_acc_rad = math.radians(settings.max_joint_acceleration_deg_s2)
        for side, arm_q in initial_q_by_side.items():
            self._trajs[side] = AccelerationLimitedJointTrajectory(
                arm_q,
                max_velocity_rad_s=max_vel_rad,
                max_acceleration_rad_s2=max_acc_rad,
            )
            self._gripper_targets_m[side] = None
            self._last_sent_gripper_m[side] = None

        super().__init__(
            rate_hz=settings.command_rate_hz,
            thread_name=f"piper-streamer-{settings.robot_name}",
            step_fn=self._step_once,
        )

    @property
    def hold_latched(self) -> bool:
        return self._hold_latched

    def set_motion_limits(
        self,
        *,
        max_joint_speed_deg_s: float | None = None,
        max_joint_acceleration_deg_s2: float | None = None,
    ) -> None:
        with self._lock:
            v_rad = math.radians(max_joint_speed_deg_s) if max_joint_speed_deg_s is not None else None
            a_rad = (
                math.radians(max_joint_acceleration_deg_s2)
                if max_joint_acceleration_deg_s2 is not None
                else None
            )
            for traj in self._trajs.values():
                traj.set_limits(max_velocity_rad_s=v_rad, max_acceleration_rad_s2=a_rad)

    def set_tracking_filter(self, cutoff_hz: float | None) -> None:
        """Enable/disable 3rd-order critically-damped tracking filter on all active arm trajectories."""
        with self._lock:
            for traj in self._trajs.values():
                traj.set_tracking_filter(cutoff_hz)

    @property
    def tracking_cutoff_hz(self) -> float | None:
        with self._lock:
            for traj in self._trajs.values():
                return traj.tracking_cutoff_hz
            return None

    def set_targets(
        self,
        q_by_side: dict[str, np.ndarray],
        gripper_m_by_side: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            if self._hold_latched:
                return
            for side, q_arm in q_by_side.items():
                if side in self._trajs:
                    self._trajs[side].set_target(q_arm)
            if gripper_m_by_side is not None:
                for side, w_m in gripper_m_by_side.items():
                    if side in self._gripper_targets_m:
                        rng = self.settings.gripper_ranges.get(side)
                        if rng is not None:
                            w_m = rng.clamp_width_m(w_m)
                        self._gripper_targets_m[side] = float(w_m)

    def hold_current_commands(self) -> dict[str, np.ndarray]:
        """Latch `HOLD_LAST_SAFE` immediately: freeze all targets at current commanded pose."""
        with self._lock:
            self._hold_latched = True
            return {side: traj.hold_current() for side, traj in self._trajs.items()}

    def commanded_q_by_side(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: traj.position for side, traj in self._trajs.items()}

    def wait_until_targets(
        self,
        target_q_by_side: dict[str, np.ndarray],
        *,
        tolerance_deg: float | None = None,
        timeout_s: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> bool:
        tol_rad = math.radians(
            self.settings.home_tolerance_deg if tolerance_deg is None else float(tolerance_deg)
        )
        deadline = time.monotonic() + (
            self.settings.home_timeout_s if timeout_s is None else float(timeout_s)
        )
        while time.monotonic() < deadline:
            self.check_health()
            if stop_event is not None and stop_event.is_set():
                self.hold_current_commands()
                return False
            cmd = self.commanded_q_by_side()
            max_err = max(
                float(np.max(np.abs(cmd[s] - np.asarray(target_q_by_side[s], dtype=np.float64))))
                for s in target_q_by_side
                if s in cmd
            )
            if max_err <= tol_rad:
                return True
            time.sleep(min(0.01, self.period_s))
        return False

    def _step_once(self, dt_s: float) -> None:
        with self._lock:
            for side, arm in self.arms.items():
                q_cmd = self._trajs[side].step(dt_s)
                arm.send_q_rad(q_cmd)
                g_tgt = self._gripper_targets_m[side]
                if g_tgt is None:
                    continue
                last = self._last_sent_gripper_m[side]
                if last is None or abs(last - g_tgt) > 1e-6:
                    arm.send_gripper_width_m(g_tgt, self.settings.gripper_force_n)
                    self._last_sent_gripper_m[side] = g_tgt


class PiperCanBackend:
    """High-level bimanual AgileX Piper SocketCAN backend (`pyAgxArm` primary, `piper_sdk` legacy).

    Exposes full-URDF `q_full` and physical robot gripper widths in meters (`widths_m`) so Flow 1
    (`replay-real`), Flow 2 (`teleop`), and Flow 3 (`policy-bridge`) share a single S-V1.8-9 safe driver.
    """

    def __init__(
        self,
        profile: RobotProfile,
        settings: PiperCanSettings,
        *,
        arm_factory: Callable[[str, str], PiperArm] | None = None,
    ) -> None:
        self.profile = profile
        self.settings = settings
        self._arm_factory = arm_factory or (lambda side, port: create_piper_arm(side, port, settings))
        self.arms: dict[str, PiperArm] = {}
        self.streamer: PiperJointStreamer | None = None
        self._connected = False
        self._hold_latched = False

    def __enter__(self) -> PiperCanBackend:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.hold_last_safe()
        finally:
            self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def hold_latched(self) -> bool:
        return self._hold_latched

    def connect(self) -> None:
        if self._connected:
            return
        arms: dict[str, PiperArm] = {}
        try:
            for side in self.profile.active_sides:
                port = self.settings.ports[side]
                arm = self._arm_factory(side, port)
                arm.connect(
                    enable=self.settings.auto_enable,
                    startup_speed_percent=self.settings.startup_speed_percent,
                    speed_percent=self.settings.speed_percent,
                )
                arms[side] = arm
        except BaseException:
            for arm in arms.values():
                try:
                    arm.disconnect()
                except Exception:
                    pass
            raise

        self.arms = arms
        init_q_by_side = {side: arm.read_q_rad() for side, arm in self.arms.items()}
        self.streamer = PiperJointStreamer(
            self.arms,
            self.settings,
            init_q_by_side,
        )
        self.streamer.start()
        self._connected = True
        self._hold_latched = False

    def extract_arm_q_by_side(self, q_full: np.ndarray) -> dict[str, np.ndarray]:
        arr = np.asarray(q_full, dtype=np.float64).reshape(-1)
        return {
            side: arr[list(self.profile.arms[side].arm_actuated_indices)].copy()
            for side in self.profile.active_sides
        }

    def clamp_robot_widths_m(self, robot_widths_m: dict[str, float]) -> dict[str, float]:
        """Clamp already-mapped physical robot gripper widths (m) to calibrated hardware bounds."""
        out: dict[str, float] = {}
        for side in self.profile.active_sides:
            if side not in robot_widths_m:
                continue
            w = float(robot_widths_m[side])
            if not math.isfinite(w):
                raise ValueError(f"Non-finite robot gripper width for side={side}: {robot_widths_m[side]!r}")
            rng = self.settings.gripper_ranges.get(
                side,
                PiperGripperRange(0, int(round(self.profile.gripper_max_width_m * 1_000_000.0))),
            )
            out[side] = rng.clamp_width_m(w)
        return out

    def map_widths_to_hardware_m(self, pika_widths_m: dict[str, float]) -> dict[str, float]:
        """Explicitly convert raw Pika Sense jaw widths (0..0.10 m) into hardware gripper widths (m)."""
        out: dict[str, float] = {}
        for side in self.profile.active_sides:
            if side not in pika_widths_m:
                continue
            rng = self.settings.gripper_ranges.get(
                side,
                PiperGripperRange(0, int(round(self.profile.gripper_max_width_m * 1_000_000.0))),
            )
            out[side] = rng.map_target_width_m(
                pika_widths_m[side],
                retarget_mode=self.profile.gripper_retarget_mode,
            )
        return out

    def widths_m_to_microm(self, pika_widths_m: dict[str, float]) -> dict[str, int]:
        out: dict[str, int] = {}
        for side in self.profile.active_sides:
            if side not in pika_widths_m:
                continue
            rng = self.settings.gripper_ranges.get(
                side,
                PiperGripperRange(0, int(round(self.profile.gripper_max_width_m * 1_000_000.0))),
            )
            out[side] = rng.width_m_to_microm(
                pika_widths_m[side],
                retarget_mode=self.profile.gripper_retarget_mode,
            )
        return out

    def read_state(self) -> tuple[np.ndarray, dict[str, float]]:
        """Read measured `(q_full, physical_widths_m)` from CAN feedback."""
        if not self._connected:
            raise RuntimeError("PiperCanBackend is not connected")
        q_full = self.profile.home_q.copy()
        widths_m: dict[str, float] = {}
        for side in self.profile.active_sides:
            arm = self.arms[side]
            q_arm = arm.read_q_rad()
            q_full[list(self.profile.arms[side].arm_actuated_indices)] = q_arm
            rng = self.settings.gripper_ranges.get(
                side,
                PiperGripperRange(0, int(round(self.profile.gripper_max_width_m * 1_000_000.0))),
            )
            w_m = rng.clamp_width_m(arm.read_gripper_width_m())
            widths_m[side] = w_m
            pika_equiv_m = self.profile.map_robot_width_to_pika(side, w_m)
            q_full, _ = self.profile.map_gripper_width_to_q(q_full, side, pika_equiv_m)
        return q_full, widths_m

    def ramp_to_q(
        self,
        q_target_full: np.ndarray,
        widths_m: dict[str, float] | None = None,
        *,
        speed_deg_s: float | None = None,
        timeout_s: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Slowly and safely ramp from current measured pose to `q_target_full`.

        `widths_m` must already be in physical robot gripper width meters (e.g. `0..0.07 m` for stock
        Piper, `0..0.10 m` for `piper_pika`); no second normalization is applied.
        """
        if not self._connected or self.streamer is None:
            raise RuntimeError("PiperCanBackend is not connected")
        ramp_speed = (
            self.settings.home_max_joint_speed_deg_s if speed_deg_s is None else float(speed_deg_s)
        )
        targets = self.extract_arm_q_by_side(q_target_full)
        g_m = self.clamp_robot_widths_m(widths_m) if widths_m is not None else None
        prev_tracking_hz = self.streamer.tracking_cutoff_hz
        self.streamer.set_tracking_filter(None)
        self.streamer.set_motion_limits(max_joint_speed_deg_s=ramp_speed)
        try:
            self.streamer.set_targets(targets, g_m)
            return self.streamer.wait_until_targets(
                targets,
                timeout_s=timeout_s,
                stop_event=stop_event,
            )
        finally:
            self.streamer.set_motion_limits(
                max_joint_speed_deg_s=self.settings.max_joint_speed_deg_s
            )
            self.streamer.set_tracking_filter(prev_tracking_hz)

    def set_tracking_filter(self, cutoff_hz: float | None) -> None:
        """Enable/disable smooth 3rd-order critically-damped tracking filter on the live CAN streamer."""
        if self.streamer is not None:
            self.streamer.set_tracking_filter(cutoff_hz)

    def write_target(self, q_target_full: np.ndarray, widths_m: dict[str, float]) -> None:
        """Push latest safe joint + physical robot gripper targets (m) to the fixed-rate CAN streamer."""
        if not self._connected or self.streamer is None:
            raise RuntimeError("PiperCanBackend is not connected")
        if self._hold_latched:
            return
        self.streamer.check_health()
        self.streamer.set_targets(
            self.extract_arm_q_by_side(q_target_full),
            self.clamp_robot_widths_m(widths_m),
        )

    def check_health(self) -> None:
        if self.streamer is not None:
            self.streamer.check_health()
        for arm in self.arms.values():
            arm.check_arm_health()

    def hold_last_safe(self) -> None:
        """Latch `HOLD_LAST_SAFE` on the streamer (keeps holding torque active)."""
        self._hold_latched = True
        if self.streamer is not None:
            self.streamer.hold_current_commands()

    def disconnect(self) -> None:
        """Stop streamer and disconnect CAN sockets cleanly WITHOUT releasing holding torque."""
        if self.streamer is not None:
            self.streamer.hold_current_commands()
            self.streamer.stop()
            self.streamer = None
        for arm in self.arms.values():
            try:
                arm.disconnect()
            except Exception:
                pass
        self._connected = False
