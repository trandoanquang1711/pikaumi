"""Layer 5 Read-Only 3-Point Tabletop Calibration Helper (`pika calibrate capture-table-robot`).

Captures 3 stationary tabletop contact points (`origin`, `x_axis`, `y_plane`) for Dual-Piper
(or any `RobotProfile`) in the global `{Robot}` (`world`) URDF frame and solves `T_robot_table`.

NON-NEGOTIABLE HARDWARE SAFETY (`GEMINI.md` / `pika-piper-control-guard`):
- `ReadOnlyPiperJointListener` is STRICTLY PASSIVE / READ-ONLY.
- It NEVER calls `enable`, `disable`, `reset`, `set_normal_mode`, `move_j`, `move_gripper_m`,
  `EnableArm`, `DisableArm`, `MotionCtrl_1`, `MotionCtrl_2`, `JointCtrl`, or `GripperCtrl`.
- It is explicitly designed to read joint feedback while a human operator hand-guides ONE arm
  in Teach Mode (or passive power-on) with physical supervision on-site.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.calibration.deployment import (
    save_physical_table_calibration,
    solve_table_from_three_contact_points,
)
from pikaumi.config import TABLE_LOCAL_CALIB_DIR, save_json_atomic
from pikaumi.real.piper_driver import (
    MDEG_TO_RAD,
    PIPER_ARM_DOF,
    load_piper_can_settings,
    resolve_active_sdk_backend,
    resolve_pyagxarm_firmware_profile,
)
from pikaumi.robots.registry import RobotProfile, load_robot_profile
from pikaumi.robots.utils import pose7_to_mat


def compute_tcp_position_from_arm_joints(
    profile: RobotProfile,
    side: str,
    q_arm_rad: np.ndarray,
) -> np.ndarray:
    """Compute 3D TCP position `[x, y, z]` (m) in global `{Robot}` (`world`) frame via URDF FK."""
    if side not in profile.arms:
        raise ValueError(
            f"Side {side!r} is not present in robot profile {profile.name!r} "
            f"(available: {sorted(profile.arms.keys())})"
        )
    arm_spec = profile.arms[side]
    q_arr = np.asarray(q_arm_rad, dtype=np.float64).reshape(-1)
    expected_dof = len(arm_spec.arm_actuated_indices)
    if q_arr.shape != (expected_dof,) or not np.all(np.isfinite(q_arr)):
        raise ValueError(f"Expected {expected_dof} finite arm joint radians for {side}, got {q_arr!r}")

    q_full = profile.home_q.copy()
    q_full[list(arm_spec.arm_actuated_indices)] = q_arr
    link_tf, _ = profile.urdf_model.forward_kinematics_all(q_full)
    T_world_ee = link_tf[arm_spec.ee_link]
    return np.asarray(T_world_ee[:3, 3], dtype=np.float64).copy()


def summarize_three_point_geometry(
    p0: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    T_robot_table_pose7: np.ndarray,
) -> dict[str, float]:
    """Compute human-friendly sanity metrics for a 3-point tabletop calibration."""
    p0 = np.asarray(p0, dtype=np.float64)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    vx = px - p0
    vy = py - p0
    x_len_mm = float(np.linalg.norm(vx) * 1000.0)
    y_len_mm = float(np.linalg.norm(vy) * 1000.0)
    cos_xy = float(np.clip(np.dot(vx, vy) / max(np.linalg.norm(vx) * np.linalg.norm(vy), 1e-12), -1.0, 1.0))
    xy_angle_deg = float(math.degrees(math.acos(cos_xy)))

    T = pose7_to_mat(T_robot_table_pose7)
    z_hat = T[:3, 2]
    tilt_deg = float(math.degrees(math.acos(float(np.clip(z_hat[2], -1.0, 1.0)))))
    rpy_deg = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return {
        "x_span_mm": round(x_len_mm, 2),
        "y_span_mm": round(y_len_mm, 2),
        "xy_touch_angle_deg": round(xy_angle_deg, 2),
        "table_tilt_deg": round(tilt_deg, 3),
        "table_yaw_deg": round(float(rpy_deg[2]), 2),
        "origin_x_m": round(float(p0[0]), 5),
        "origin_y_m": round(float(p0[1]), 5),
        "origin_z_m": round(float(p0[2]), 5),
    }


class ReadOnlyPiperJointListener:
    """Strictly passive SocketCAN joint-angle reader supporting `pyAgxArm` and `piper_sdk`.

    ZERO TX invariant: never sends enable, disable, reset, mode switch, joint, or gripper commands.
    """

    def __init__(
        self,
        can_port: str,
        *,
        sdk_backend: str = "auto",
        firmware: str = "S-V1.8-9",
        interface: str = "socketcan",
        sdk_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.can_port = str(can_port)
        self.sdk_backend = str(sdk_backend).strip().lower()
        self.firmware = str(firmware)
        self.firmware_profile = resolve_pyagxarm_firmware_profile(firmware)
        self.interface = str(interface)
        self._sdk_factory = sdk_factory
        self._resolved_sdk: str | None = None
        self._handle: Any | None = None
        self._connected = False

    def connect(self, *, timeout_s: float = 4.0) -> str:
        resolved = self.sdk_backend
        if resolved == "auto":
            if self._sdk_factory is not None:
                resolved = "pyagxarm"
            else:
                resolved = resolve_active_sdk_backend("auto")
        self._resolved_sdk = resolved

        if resolved == "pyagxarm":
            if self._sdk_factory is not None:
                robot = self._sdk_factory(self.can_port, self.firmware_profile)
            else:
                try:
                    from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
                except ImportError as exc:
                    raise ImportError(
                        'pyAgxArm is not installed (not on PyPI). Install with: '
                        '`uv pip install --python .venv-pika-dataset/bin/python '
                        '"git+https://github.com/agilexrobotics/pyAgxArm.git" python-can` '
                        '(or install `piper-sdk` from PyPI).'
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
                robot = AgxArmFactory.create_arm(cfg)
            robot.connect()
            self._handle = robot
        elif resolved == "piper_sdk":
            if self._sdk_factory is not None:
                arm = self._sdk_factory(self.can_port, self.firmware_profile)
            else:
                try:
                    from piper_sdk import C_PiperInterface_V2
                except ImportError as exc:
                    raise ImportError(
                        "piper-sdk is not installed. Install with: "
                        "`uv pip install --python .venv-pika-dataset/bin/python piper-sdk python-can`"
                    ) from exc

                arm = C_PiperInterface_V2(self.can_port)
            arm.ConnectPort()
            self._handle = arm
        else:
            raise ValueError(f"Unsupported sdk_backend={resolved!r}")

        self._connected = True
        deadline = time.monotonic() + max(timeout_s, 0.2)
        while time.monotonic() < deadline:
            try:
                _ = self.read_q_rad()
                return resolved
            except RuntimeError:
                time.sleep(0.02)
        self.disconnect()
        raise RuntimeError(
            f"Timed out waiting for read-only Piper joint feedback on {self.can_port} ({resolved}). "
            "Check that CAN interface is UP and the arm is powered on."
        )

    def read_q_rad(self) -> np.ndarray:
        if not self._connected or self._handle is None:
            raise RuntimeError("ReadOnlyPiperJointListener is not connected")
        if self._resolved_sdk == "pyagxarm":
            msg_wrap = self._handle.get_joint_angles()
            if msg_wrap is None:
                raise RuntimeError(f"Missing pyAgxArm joint feedback on {self.can_port}")
            ts = getattr(msg_wrap, "timestamp", None)
            if isinstance(ts, (int, float)) and ts <= 0.0:
                raise RuntimeError(f"Uninitialized pyAgxArm joint feedback on {self.can_port}")
            vals = getattr(msg_wrap, "msg", msg_wrap)
            arr = np.asarray(vals, dtype=np.float64).reshape(-1)
            if arr.shape != (PIPER_ARM_DOF,) or not np.all(np.isfinite(arr)):
                raise RuntimeError(f"Invalid pyAgxArm joint feedback: {arr!r}")
            return arr

        jm = self._handle.GetArmJointMsgs()
        if jm is None:
            raise RuntimeError(f"Missing piper_sdk joint feedback on {self.can_port}")
        ts = getattr(jm, "time_stamp", None)
        if isinstance(ts, (int, float)) and ts <= 0.0:
            raise RuntimeError(f"Uninitialized piper_sdk joint feedback on {self.can_port}")
        js = jm.joint_state
        arr = (
            np.array(
                [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
                dtype=np.float64,
            )
            * MDEG_TO_RAD
        )
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Non-finite piper_sdk joint feedback: {arr!r}")
        return arr

    def sample_stationary_q_rad(
        self,
        *,
        samples: int = 15,
        interval_s: float = 0.02,
        max_spread_deg: float = 1.5,
    ) -> tuple[np.ndarray, float]:
        """Collect `samples` consecutive joint readings and return `(median_q_rad, max_spread_deg)`."""
        buf: list[np.ndarray] = []
        for i in range(max(int(samples), 3)):
            buf.append(self.read_q_rad())
            if i + 1 < samples and interval_s > 0.0:
                time.sleep(interval_s)
        arr = np.asarray(buf, dtype=np.float64)
        med = np.median(arr, axis=0)
        spread_deg = float(np.degrees(np.max(np.abs(arr - med[None, :]))))
        if spread_deg > float(max_spread_deg):
            raise RuntimeError(
                f"Arm moved during capture (max joint spread={spread_deg:.2f} deg > {max_spread_deg:.2f} deg). "
                "Hold the TCP steady on the mark and retry."
            )
        return med, spread_deg

    def disconnect(self) -> None:
        if self._handle is not None and self._connected:
            fn_name = "disconnect" if self._resolved_sdk == "pyagxarm" else "DisconnectPort"
            fn = getattr(self._handle, fn_name, None)
            if callable(fn):
                fn()
        self._connected = False
        self._handle = None


def parse_manual_point_input(
    raw_text: str,
    profile: RobotProfile,
    side: str,
) -> tuple[np.ndarray, list[float] | None]:
    """Parse either `x y z` (meters in `{Robot}`) or `deg: j1 j2 j3 j4 j5 j6` / `rad: ...` (URDF FK)."""
    text = raw_text.strip()
    lower = text.lower()
    if lower.startswith("deg:") or lower.startswith("rad:"):
        is_deg = lower.startswith("deg:")
        body = text.split(":", 1)[1].replace(",", " ").strip()
        vals = [float(tok) for tok in body.split()]
        if len(vals) != len(profile.arms[side].arm_actuated_indices):
            raise ValueError(
                f"Expected {len(profile.arms[side].arm_actuated_indices)} joint values, got {len(vals)}"
            )
        q_rad = np.deg2rad(vals) if is_deg else np.asarray(vals, dtype=np.float64)
        xyz = compute_tcp_position_from_arm_joints(profile, side, q_rad)
        return xyz, q_rad.tolist()

    body = text.replace(",", " ").strip()
    vals = [float(tok) for tok in body.split()]
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        raise ValueError(
            "Enter either 3 coordinates `x y z` in meters (e.g. `0.34 0.00 -0.03`), "
            "or 6 arm joint angles `deg: j1 j2 j3 j4 j5 j6`."
        )
    return np.asarray(vals, dtype=np.float64), None


@dataclass(frozen=True)
class TableCaptureResult:
    contacts_json: Path
    table_yaml: Path
    T_robot_table_pose7: np.ndarray
    metrics: dict[str, float]
    verified: bool


def capture_table_robot_three_points(
    *,
    robot: str = "piper_pika",
    side: str = "left",
    source: str = "can",
    output_yaml: str | Path | None = None,
    contacts_json: str | Path = "data/calibration_std/contacts_piper.json",
    port: str | None = None,
    sdk_backend: str | None = None,
    firmware: str | None = None,
    rig_config_path: str | Path | None = None,
    samples_per_point: int = 15,
    max_spread_deg: float = 1.5,
    verified: bool = True,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    listener_factory: Callable[..., ReadOnlyPiperJointListener] | None = None,
) -> TableCaptureResult:
    """Interactively capture 3 tabletop TCP contacts and save physical `T_robot_table` YAML."""
    if source not in ("can", "manual", "mock"):
        raise ValueError(f"Unsupported source={source!r}; choose from ('can', 'manual', 'mock')")

    profile = load_robot_profile(robot)
    if side not in profile.arms:
        raise ValueError(f"Side {side!r} not in robot {robot!r} (available: {sorted(profile.arms)})")

    out_yaml = Path(output_yaml) if output_yaml is not None else (TABLE_LOCAL_CALIB_DIR / f"{robot}.yaml")
    out_contacts = Path(contacts_json)

    prompts = [
        ("origin", "1/3 [ORIGIN (0,0,0)]: Cham mui kep vao TAM MAT BAN (tam bang ChArUco)"),
        ("x_axis", "2/3 [+X AXIS]: Cham mui kep vao 1 diem doc TRUC +X (sang PHAI >= 50 mm, khuyen 150-200 mm)"),
        ("y_plane", "3/3 [+Y PLANE]: Cham mui kep vao 1 diem phia TRUC +Y (huong len phia truoc >= 100 mm)"),
    ]

    points: dict[str, np.ndarray] = {}
    joints_rad: dict[str, list[float]] = {}
    spreads_deg: dict[str, float] = {}
    source_label = f"contact_three_point_{source}"

    if source == "mock":
        # Deterministic synthetic tabletop points in {Robot} frame (zero CAN/hardware I/O)
        points = {
            "origin": np.array([0.34, 0.00, -0.03], dtype=np.float64),
            "x_axis": np.array([0.34, -0.18, -0.03], dtype=np.float64),
            "y_plane": np.array([0.52, 0.00, -0.03], dtype=np.float64),
        }
        verified = False
        output_fn(f"[INFO] Mock Layer 5 3-point capture for robot={robot!r} (offline, zero CAN I/O)")
    elif source == "manual":
        output_fn(
            f"[INFO] Manual Layer 5 3-point capture for robot={robot!r} (side={side!r}, URDF world frame)."
        )
        output_fn(
            "[INFO] Nhap `x y z` (met trong he goc chung giua 2 de robot) hoac `deg: j1 j2 j3 j4 j5 j6` (tu tinh FK)."
        )
        for key, label in prompts:
            while True:
                raw = input_fn(f"{label}\n  > ")
                try:
                    xyz, q_opt = parse_manual_point_input(raw, profile, side)
                    points[key] = xyz
                    if q_opt is not None:
                        joints_rad[key] = q_opt
                    output_fn(f"  [OK] {key} = [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] m")
                    break
                except Exception as exc:
                    output_fn(f"  [RETRY] {exc}")
    else:
        settings = load_piper_can_settings(
            profile,
            rig_config_path=rig_config_path,
            left_port=port if side == "left" else None,
            right_port=port if side == "right" else None,
            sdk_backend=sdk_backend,
            firmware=firmware,
            enable_arms=False,
        )
        can_port = port or settings.ports[side]
        listener = (
            listener_factory(
                can_port,
                sdk_backend=settings.sdk_backend,
                firmware=settings.firmware,
            )
            if listener_factory is not None
            else ReadOnlyPiperJointListener(
                can_port,
                sdk_backend=settings.sdk_backend,
                firmware=settings.firmware,
            )
        )
        resolved_sdk = listener.connect()
        source_label = f"contact_three_point_{resolved_sdk}_readonly"
        output_fn(
            f"[INFO] Connected READ-ONLY to {robot} ({side}) on {can_port} via {resolved_sdk} "
            "(ZERO motion/enable/disable TX)."
        )
        output_fn(
            "[INFO] Dat canh tay o che do Teach Mode (dat tay nhe nhang), cham mui kep vao tung diem va bam Enter."
        )
        try:
            for key, label in prompts:
                while True:
                    _ = input_fn(f"{label} -> Nhan Enter de doc... ")
                    try:
                        q_med, spread = listener.sample_stationary_q_rad(
                            samples=samples_per_point,
                            max_spread_deg=max_spread_deg,
                        )
                        xyz = compute_tcp_position_from_arm_joints(profile, side, q_med)
                        points[key] = xyz
                        joints_rad[key] = q_med.tolist()
                        spreads_deg[key] = round(spread, 3)
                        output_fn(
                            f"  [OK] {key} = [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] m "
                            f"(joint spread={spread:.2f} deg)"
                        )
                        break
                    except Exception as exc:
                        output_fn(f"  [RETRY] {exc}")
        finally:
            listener.disconnect()

    pose7 = solve_table_from_three_contact_points(
        points["origin"],
        points["x_axis"],
        points["y_plane"],
    )
    metrics = summarize_three_point_geometry(
        points["origin"],
        points["x_axis"],
        points["y_plane"],
        pose7,
    )

    contacts_payload: dict[str, Any] = {
        "schema_version": 1,
        "robot": robot,
        "side": side,
        "source": source_label,
        "origin": points["origin"].tolist(),
        "x_axis": points["x_axis"].tolist(),
        "y_plane": points["y_plane"].tolist(),
        "metrics": metrics,
    }
    if joints_rad:
        contacts_payload["joints_rad"] = joints_rad
    if spreads_deg:
        contacts_payload["joint_spread_deg"] = spreads_deg

    saved_contacts = save_json_atomic(out_contacts, contacts_payload)
    saved_yaml = save_physical_table_calibration(
        out_yaml,
        robot=robot,
        T_robot_table_pose7=pose7,
        verified=bool(verified),
        source=source_label,
        extra_metrics=metrics,
    )

    output_fn(
        f"[OK] Saved Layer 5 physical table calibration -> {saved_yaml} (verified={bool(verified)})\n"
        f"     contacts JSON : {saved_contacts}\n"
        f"     origin (Robot): x={metrics['origin_x_m']*100:.1f} cm, y={metrics['origin_y_m']*100:.1f} cm, z={metrics['origin_z_m']*100:.1f} cm\n"
        f"     span / angle  : |X|={metrics['x_span_mm']:.1f} mm, |Y|={metrics['y_span_mm']:.1f} mm, "
        f"angle(X,Y)={metrics['xy_touch_angle_deg']:.1f} deg\n"
        f"     table tilt/yaw: tilt={metrics['table_tilt_deg']:.2f} deg, yaw={metrics['table_yaw_deg']:.1f} deg"
    )
    return TableCaptureResult(
        contacts_json=saved_contacts,
        table_yaml=saved_yaml,
        T_robot_table_pose7=pose7,
        metrics=metrics,
        verified=bool(verified),
    )
