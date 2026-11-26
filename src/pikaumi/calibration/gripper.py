"""Layer 2 Calibration: Pika Sense 100 mm jaw encoder zero-point & stroke mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GripperEncoderCalibration:
    side: str
    closed_raw_m: float
    open_raw_m: float
    nominal_max_width_m: float = 0.10
    deadband_m: float = 0.0005
    passed: bool = True
    reasons: tuple[str, ...] = ()

    def map_width_m(self, raw_width_m: float | np.ndarray) -> np.ndarray | float:
        raw = np.asarray(raw_width_m, dtype=np.float64)
        span = max(self.open_raw_m - self.closed_raw_m, 1e-6)
        norm = (raw - self.closed_raw_m) / span
        scaled = np.clip(norm * self.nominal_max_width_m, 0.0, self.nominal_max_width_m)
        scaled = np.where(scaled < self.deadband_m, 0.0, scaled)
        if np.ndim(raw_width_m) == 0:
            return float(scaled)
        return scaled

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def calibrate_gripper_encoder(
    side: str,
    closed_samples_m: np.ndarray,
    open_samples_m: np.ndarray,
    *,
    nominal_max_width_m: float = 0.10,
    min_measured_span_m: float = 0.085,
    max_std_m: float = 0.0015,
) -> GripperEncoderCalibration:
    closed_arr = np.asarray(closed_samples_m, dtype=np.float64)
    open_arr = np.asarray(open_samples_m, dtype=np.float64)
    closed_med = float(np.median(closed_arr))
    open_med = float(np.median(open_arr))
    span = open_med - closed_med
    reasons: list[str] = []
    if span < min_measured_span_m:
        reasons.append(f"insufficient_jaw_span_m:{span:.4f}<{min_measured_span_m:.4f}")
    if float(np.std(closed_arr)) > max_std_m or float(np.std(open_arr)) > max_std_m:
        reasons.append("encoder_noise_exceeds_threshold")
    return GripperEncoderCalibration(
        side=side,
        closed_raw_m=closed_med,
        open_raw_m=open_med,
        nominal_max_width_m=nominal_max_width_m,
        passed=len(reasons) == 0,
        reasons=tuple(reasons),
    )
