"""Layer 1 Calibration: SteamVR / libsurvive Vive Lighthouse base-station & tracker gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.robots.utils import normalize_quat_xyzw


@dataclass(frozen=True)
class ViveGateReport:
    passed: bool
    num_samples: int
    accurate_ratio: float
    position_std_mm: float
    orientation_std_deg: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_stationary_vive_window(
    poses7: np.ndarray,
    accurate_flags: np.ndarray | None = None,
    *,
    max_pos_std_mm: float = 1.5,
    max_ori_std_deg: float = 0.5,
    min_samples: int = 10,
) -> ViveGateReport:
    """Gate a stationary Vive tracker window for Lighthouse calibration stability."""
    arr = np.asarray(poses7, dtype=np.float64)
    reasons: list[str] = []
    if arr.ndim != 2 or arr.shape[1] != 7 or len(arr) < min_samples:
        return ViveGateReport(
            passed=False,
            num_samples=int(len(arr)) if arr.ndim == 2 else 0,
            accurate_ratio=0.0,
            position_std_mm=float("inf"),
            orientation_std_deg=float("inf"),
            reasons=[f"insufficient_samples:{len(arr) if arr.ndim == 2 else 0}<{min_samples}"],
        )

    if accurate_flags is not None:
        acc = np.asarray(accurate_flags, dtype=bool)
        acc_ratio = float(np.mean(acc))
    else:
        acc_ratio = 1.0

    if acc_ratio < 1.0:
        reasons.append(f"inaccurate_vive_frames:{acc_ratio:.3f}<1.0")

    pos_std_m = float(np.linalg.norm(np.std(arr[:, :3], axis=0)))
    pos_std_mm = pos_std_m * 1000.0
    if pos_std_mm > max_pos_std_mm:
        reasons.append(f"vive_jitter_mm:{pos_std_mm:.3f}>{max_pos_std_mm:.3f}")

    quats = normalize_quat_xyzw(arr[:, 3:7])
    rots = Rotation.from_quat(quats)
    mean_rot = rots.mean()
    angles_rad = (mean_rot.inv() * rots).magnitude()
    ori_rms_deg = float(np.degrees(np.sqrt(np.mean(angles_rad**2))))
    if ori_rms_deg > max_ori_std_deg:
        reasons.append(f"vive_ori_jitter_deg:{ori_rms_deg:.3f}>{max_ori_std_deg:.3f}")

    return ViveGateReport(
        passed=len(reasons) == 0,
        num_samples=int(len(arr)),
        accurate_ratio=acc_ratio,
        position_std_mm=pos_std_mm,
        orientation_std_deg=ori_rms_deg,
        reasons=reasons,
    )
