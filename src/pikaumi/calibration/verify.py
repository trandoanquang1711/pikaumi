"""Static SHA-256 chain & physical two-tip touch / tabletop height verifier (`pika calibrate verify`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pikaumi.calibration.pika_tcp import compute_pika_tcp_sha256
from pikaumi.calibration.spatial import verify_fisheye_intrinsics_dict
from pikaumi.config import load_yaml, sha256_object
from pikaumi.robots.utils import mat_to_pose7, pose7_to_mat


def verify_calibration_bundle(
    calibration_dir: str | Path,
    *,
    touch_observations: dict[str, Any] | None = None,
    max_tip_touch_error_m: float = 0.015,
    max_table_z_error_m: float = 0.010,
) -> dict[str, Any]:
    """Validate standardized YAML bundle (`spatial.yaml`, `pika_tcp.yaml`, optional `session.yaml`) + physical checks."""
    cal_dir = Path(calibration_dir)
    reasons: list[str] = []
    warnings: list[str] = []
    hashes: dict[str, str] = {}

    spatial_path = cal_dir / "spatial.yaml"
    tcp_path = cal_dir / "pika_tcp.yaml"
    session_path = cal_dir / "session.yaml"

    spatial_cfg: dict[str, Any] = {}
    if spatial_path.is_file():
        spatial_cfg = load_yaml(spatial_path)
        hashes["spatial_sha256"] = sha256_object(spatial_cfg)
        cams = spatial_cfg.get("cameras") or {}
        for side in ("left", "right"):
            if side in cams:
                gate = verify_fisheye_intrinsics_dict(cams[side])
                if not gate.passed:
                    reasons.extend(f"intrinsics_{side}:{r}" for r in gate.reasons)
    else:
        warnings.append(f"missing_optional:{spatial_path.name}")

    tcp_poses_by_side: dict[str, np.ndarray] = {}
    if tcp_path.is_file():
        tcp_cfg = load_yaml(tcp_path)
        computed_tcp_sha = compute_pika_tcp_sha256(tcp_cfg)
        hashes["pika_tcp_sha256"] = computed_tcp_sha
        stored_tcp_sha = tcp_cfg.get("pika_tcp_sha256")
        if stored_tcp_sha and str(stored_tcp_sha) != computed_tcp_sha:
            reasons.append("sha256_mismatch:pika_tcp.yaml")
        if not bool(tcp_cfg.get("verified", False)):
            warnings.append(f"pika_tcp_unverified:{tcp_cfg.get('quality_status', 'provisional')}")
        for side in ("left", "right"):
            scfg = (tcp_cfg.get("sides") or {}).get(side)
            if scfg is None:
                reasons.append(f"missing_pika_tcp_side:{side}")
            else:
                tcp_poses_by_side[side] = np.concatenate(
                    [np.asarray(scfg["position"], dtype=np.float64), np.asarray(scfg["quaternion"], dtype=np.float64)]
                )
    else:
        reasons.append(f"missing_required:{tcp_path.name}")

    T_table_vive = np.eye(4, dtype=np.float64)
    has_session = False
    if session_path.is_file():
        has_session = True
        ses_cfg = load_yaml(session_path)
        expected_spatial_sha = ses_cfg.get("spatial_sha256")
        if expected_spatial_sha and "spatial_sha256" in hashes and expected_spatial_sha != hashes["spatial_sha256"]:
            reasons.append("sha256_mismatch:session.spatial_sha256!=spatial.yaml")
        expected_tcp_sha = ses_cfg.get("pika_tcp_sha256")
        if expected_tcp_sha and "pika_tcp_sha256" in hashes and expected_tcp_sha != hashes["pika_tcp_sha256"]:
            reasons.append("sha256_mismatch:session.pika_tcp_sha256!=pika_tcp.yaml")
        tv = ses_cfg.get("T_table_vive") or {}
        tv_pose7 = np.concatenate(
            [np.asarray(tv["position"], dtype=np.float64), np.asarray(tv["quaternion"], dtype=np.float64)]
        )
        T_table_vive = pose7_to_mat(tv_pose7)
        hashes["session_sha256"] = sha256_object(ses_cfg)

    physical_metrics: dict[str, Any] = {}
    if touch_observations is not None and "left" in tcp_poses_by_side and "right" in tcp_poses_by_side:
        T_trk_tcp_l = pose7_to_mat(tcp_poses_by_side["left"])
        T_trk_tcp_r = pose7_to_mat(tcp_poses_by_side["right"])

        if "tip_touch" in touch_observations:
            tt = touch_observations["tip_touch"]
            T_vl = pose7_to_mat(np.asarray(tt["left_tracker_pose7"], dtype=np.float64)) @ T_trk_tcp_l
            T_vr = pose7_to_mat(np.asarray(tt["right_tracker_pose7"], dtype=np.float64)) @ T_trk_tcp_r
            tip_dist_m = float(np.linalg.norm(T_vl[:3, 3] - T_vr[:3, 3]))
            physical_metrics["two_tip_touch_distance_m"] = tip_dist_m
            if tip_dist_m > max_tip_touch_error_m:
                reasons.append(f"two_tip_touch_m:{tip_dist_m:.4f}>{max_tip_touch_error_m:.4f}")

        if "table_contacts" in touch_observations and has_session:
            z_errs: list[float] = []
            for item in touch_observations["table_contacts"]:
                side = str(item.get("side", "left"))
                T_vt = pose7_to_mat(np.asarray(item["tracker_pose7"], dtype=np.float64))
                T_tcp_in_table = T_table_vive @ T_vt @ pose7_to_mat(tcp_poses_by_side[side])
                pose_table = mat_to_pose7(T_tcp_in_table)
                z_errs.append(abs(float(pose_table[2])))
            max_z = float(max(z_errs)) if z_errs else 0.0
            physical_metrics["max_table_contact_abs_z_m"] = max_z
            if max_z > max_table_z_error_m:
                reasons.append(f"table_contact_abs_z_m:{max_z:.4f}>{max_table_z_error_m:.4f}")

    return {
        "passed": len(reasons) == 0,
        "calibration_dir": str(cal_dir),
        "hashes": hashes,
        "physical_metrics": physical_metrics,
        "warnings": warnings,
        "reasons": reasons,
    }
