"""CLI handlers for `pika calibrate {import-legacy, tcp-pivot, tcp-axes, session, verify}`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pikaumi.calibration.deployment import (
    save_physical_table_calibration,
    solve_table_from_three_contact_points,
)
from pikaumi.calibration.gripper import calibrate_gripper_encoder
from pikaumi.calibration.pika_tcp import (
    compute_pika_tcp_sha256,
    load_snapshots_directory,
    solve_tcp_axes_svd,
    solve_tcp_pivot_translation,
)
from pikaumi.calibration.spatial import solve_hand_eye_mount, solve_session_table_from_vive
from pikaumi.calibration.verify import verify_calibration_bundle
from pikaumi.calibration.vive import evaluate_stationary_vive_window
from pikaumi.config import (
    DEFAULT_DATA_CALIB_DIR,
    load_yaml,
    save_json_atomic,
    save_yaml_atomic,
    sha256_object,
)
from pikaumi.robots.utils import mat_to_pose7


def import_legacy_calibration_dir(
    legacy_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Import existing `data/calibration/{left,right}_{fisheye_intrinsics,pika_tcp,piper_tcp}.json` into standardized YAMLs."""
    src = Path(legacy_dir) if legacy_dir is not None else DEFAULT_DATA_CALIB_DIR
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    cameras: dict[str, Any] = {}
    for side in ("left", "right"):
        intr_file = src / f"{side}_fisheye_intrinsics.json"
        if intr_file.is_file():
            cameras[side] = json.loads(intr_file.read_text(encoding="utf-8"))

    spatial_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pikaumi_spatial_calibration",
        "source": str(src),
        "cameras": cameras,
        "mounts": {
            "left": {"T_tracker_cam": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "provisional": True},
            "right": {"T_tracker_cam": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "provisional": True},
        },
    }
    spatial_path = save_yaml_atomic(dst / "spatial.yaml", spatial_payload)

    sides_tcp: dict[str, Any] = {}
    all_verified = True
    for side in ("left", "right"):
        tcp_file = src / f"{side}_pika_tcp.json"
        if tcp_file.is_file():
            raw_tcp = json.loads(tcp_file.read_text(encoding="utf-8"))
            T = np.asarray(raw_tcp["transform"], dtype=np.float64)
            pose7 = mat_to_pose7(T)
            q_status = str(raw_tcp.get("quality_status", "provisional"))
            if q_status != "verified":
                all_verified = False
            sides_tcp[side] = {
                "position": pose7[:3].tolist(),
                "quaternion": pose7[3:7].tolist(),
                "rms_error_m": float(raw_tcp.get("rms_error_m", 0.0)),
                "max_error_m": float(raw_tcp.get("max_error_m", 0.0)),
                "rotation_estimated": bool(raw_tcp.get("rotation_estimated", False)),
                "quality_status": q_status,
            }
        else:
            all_verified = False

    pika_tcp_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pikaumi_tracker_tcp_calibration",
        "device": "vive_pika_sense_v1",
        "gripper_max_width_m": 0.10,
        "axes_convention": "+X wrist-to-fingertip, +Y left jaw opening, +Z top face",
        "verified": all_verified and ("left" in sides_tcp and "right" in sides_tcp),
        "quality_status": "verified" if all_verified else "provisional_imported",
        "sides": sides_tcp,
    }
    pika_tcp_payload["pika_tcp_sha256"] = compute_pika_tcp_sha256(pika_tcp_payload)
    tcp_path = save_yaml_atomic(dst / "pika_tcp.yaml", pika_tcp_payload)

    return {
        "spatial_yaml": str(spatial_path),
        "pika_tcp_yaml": str(tcp_path),
        "spatial_sha256": sha256_object(spatial_payload),
        "pika_tcp_sha256": pika_tcp_payload["pika_tcp_sha256"],
    }


def dispatch_calibrate(args: argparse.Namespace) -> int:
    if args.calib_cmd == "import-legacy":
        summary = import_legacy_calibration_dir(args.legacy_dir, args.output_dir)
        print(f"[OK] Imported legacy calibration into {args.output_dir}")
        print(f"     spatial_sha256  = {summary['spatial_sha256']}")
        print(f"     pika_tcp_sha256 = {summary['pika_tcp_sha256']}")
        return 0

    if args.calib_cmd == "vive":
        data = json.loads(Path(args.poses_json).read_text(encoding="utf-8"))
        poses = np.asarray(data["poses7"] if isinstance(data, dict) else data, dtype=np.float64)
        acc = (
            np.asarray(data["accurate"], dtype=bool)
            if isinstance(data, dict) and "accurate" in data
            else None
        )
        rep = evaluate_stationary_vive_window(
            poses,
            accurate_flags=acc,
            max_pos_std_mm=args.max_pos_std_mm,
            max_ori_std_deg=args.max_ori_std_deg,
        )
        payload = rep.to_dict()
        if args.output:
            save_yaml_atomic(args.output, payload)
        print(
            f"[{'OK' if rep.passed else 'FAIL'}] Vive stationary gate pos_std={rep.position_std_mm:.3f}mm "
            f"ori_rms={rep.orientation_std_deg:.3f}deg acc={rep.accurate_ratio:.3f}"
        )
        return 0 if rep.passed else 1

    if args.calib_cmd == "gripper":
        data = json.loads(Path(args.samples_json).read_text(encoding="utf-8"))
        cal = calibrate_gripper_encoder(
            args.side,
            np.asarray(data["closed_m"], dtype=np.float64),
            np.asarray(data["open_m"], dtype=np.float64),
            nominal_max_width_m=args.max_width_m,
        )
        payload = cal.to_dict()
        if args.output:
            save_yaml_atomic(args.output, payload)
        print(
            f"[{'OK' if cal.passed else 'FAIL'}] {args.side} gripper encoder closed={cal.closed_raw_m*1000:.2f}mm "
            f"open={cal.open_raw_m*1000:.2f}mm"
        )
        return 0 if cal.passed else 1

    if args.calib_cmd == "capture-charuco":
        from pikaumi.calibration.capture import (
            capture_charuco_observations,
            load_board_spec_from_configs,
        )

        spec = load_board_spec_from_configs(
            spatial_yaml=args.spatial_yaml,
            side=args.side,
            rig_config_path=args.rig_config,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length_m=args.square_m,
            marker_length_m=args.marker_m,
            dictionary=args.dictionary,
        )
        capture_charuco_observations(
            mode=args.mode,
            side=args.side,
            spatial_yaml=args.spatial_yaml,
            output_json=args.output,
            board_spec=spec,
            target_views=args.views,
            min_corners=args.min_corners,
            max_reprojection_px=args.max_reprojection_px,
            max_pose_gap_ms=args.max_pose_gap_ms,
            min_rot_deg=args.min_rot_deg,
            min_trans_mm=args.min_trans_mm,
            max_motion_mm=args.max_motion_mm,
            max_motion_deg=args.max_motion_deg,
            stationary_window_ms=args.stationary_window_ms,
            preview=not bool(args.no_preview),
            container_name=args.container,
        )
        if args.mode == "hand-eye":
            print(
                f"[NEXT] ./pika calibrate hand-eye --observations-json {args.output} "
                f"--side {args.side} --spatial-yaml {args.spatial_yaml} "
                f"--output {Path(args.spatial_yaml).parent / f'hand_eye_{args.side}.yaml'}"
            )
        else:
            print(
                f"[NEXT] ./pika calibrate session --observations-json {args.output} "
                f"--spatial-yaml {args.spatial_yaml} "
                f"--output {Path(args.spatial_yaml).parent / 'session.yaml'}"
            )
        return 0

    if args.calib_cmd == "hand-eye":
        obs_path = Path(args.observations_json)
        if not obs_path.is_file():
            raise FileNotFoundError(
                f"Observations JSON not found: {obs_path}. Capture it first with:\n"
                f"  ./pika calibrate capture-charuco --mode hand-eye --side {args.side} "
                f"--spatial-yaml {args.spatial_yaml or 'data/calibration_std/spatial.yaml'} "
                f"--output {obs_path}"
            )
        obs = json.loads(obs_path.read_text(encoding="utf-8"))
        res = solve_hand_eye_mount(
            np.asarray(obs["T_vive_tracker"], dtype=np.float64),
            np.asarray(obs["T_cam_board"], dtype=np.float64),
            max_rms_m=args.max_rms_m,
        )
        res["side"] = args.side
        save_yaml_atomic(args.output, res)
        if args.spatial_yaml:
            sp_path = Path(args.spatial_yaml)
            sp = load_yaml(sp_path) if sp_path.is_file() else {"schema_version": 1, "kind": "pikaumi_spatial_calibration", "mounts": {}}
            sp.setdefault("mounts", {})[args.side] = {
                "T_tracker_cam": res["T_tracker_cam"],
                "rms_error_m": res["rms_error_m"],
                "provisional": not bool(res["passed"]),
            }
            save_yaml_atomic(sp_path, sp)
        print(
            f"[{'OK' if res['passed'] else 'FAIL'}] {args.side} hand-eye rms={res['rms_error_m']*1000:.2f}mm "
            f"max={res['max_error_m']*1000:.2f}mm"
        )
        return 0 if res["passed"] else 1

    if args.calib_cmd == "table-robot":
        data = json.loads(Path(args.contacts_json).read_text(encoding="utf-8"))
        pose7 = solve_table_from_three_contact_points(
            np.asarray(data["origin"], dtype=np.float64),
            np.asarray(data["x_axis"], dtype=np.float64),
            np.asarray(data["y_plane"], dtype=np.float64),
        )
        out_path = save_physical_table_calibration(
            args.output,
            robot=args.robot,
            T_robot_table_pose7=pose7,
            verified=bool(args.verified),
            source="contact_three_point",
        )
        print(
            f"[OK] Saved physical table calibration for {args.robot} to {out_path} "
            f"(verified={bool(args.verified)})"
        )
        return 0

    if args.calib_cmd == "capture-table-robot":
        from pikaumi.calibration.table_capture import capture_table_robot_three_points

        capture_table_robot_three_points(
            robot=args.robot,
            side=args.side,
            source=args.source,
            output_yaml=args.output,
            contacts_json=args.contacts_json,
            port=args.port,
            sdk_backend=args.sdk,
            firmware=args.firmware,
            rig_config_path=args.rig_config,
            samples_per_point=args.samples,
            max_spread_deg=args.max_spread_deg,
            verified=not bool(args.unverified),
        )
        return 0

    if args.calib_cmd == "tcp-pivot":
        data = json.loads(Path(args.poses_json).read_text(encoding="utf-8"))
        poses = np.asarray(data["poses7"] if isinstance(data, dict) else data, dtype=np.float64)
        res = solve_tcp_pivot_translation(poses)
        res["side"] = args.side
        save_yaml_atomic(args.output, res)
        print(
            f"[{'OK' if res['passed'] else 'FAIL'}] {args.side} pivot rms={res['rms_error_m']*1000:.2f}mm "
            f"max={res['max_error_m']*1000:.2f}mm cond={res['condition_number']:.1f}"
        )
        return 0 if res["passed"] else 1

    if args.calib_cmd == "tcp-axes":
        snaps = load_snapshots_directory(args.directory)
        res = solve_tcp_axes_svd(snaps, expected_distance_m=args.distance_mm / 1000.0)
        save_yaml_atomic(args.output, res)
        print(
            f"[{'OK' if res['passed'] else 'FAIL'}] TCP axes max_orth_err={res['max_orthogonality_error_deg']:.2f}deg"
        )
        return 0 if res["passed"] else 1

    if args.calib_cmd == "session":
        obs_path = Path(args.observations_json)
        if not obs_path.is_file():
            raise FileNotFoundError(
                f"Observations JSON not found: {obs_path}. Capture it first with:\n"
                f"  ./pika calibrate capture-charuco --mode session --side left "
                f"--spatial-yaml {args.spatial_yaml} --output {obs_path}"
            )
        obs = json.loads(obs_path.read_text(encoding="utf-8"))
        spatial_cfg = load_yaml(args.spatial_yaml)
        side = str(obs.get("side", "left"))
        mount = np.asarray(
            spatial_cfg["mounts"][side]["T_tracker_cam"],
            dtype=np.float64,
        )
        res = solve_session_table_from_vive(
            np.asarray(obs["T_vive_tracker"], dtype=np.float64),
            np.asarray(obs["T_cam_table"], dtype=np.float64),
            mount,
            spatial_payload_for_hash=spatial_cfg,
        )
        save_yaml_atomic(args.output, res)
        print(
            f"[{'OK' if res['verified'] else 'FAIL'}] Session T_table_vive rms="
            f"{res['metrics']['rms_error_m']*1000:.2f}mm"
        )
        return 0 if res["verified"] else 1

    if args.calib_cmd == "verify":
        touch = (
            json.loads(Path(args.touch_json).read_text(encoding="utf-8"))
            if args.touch_json
            else None
        )
        rep = verify_calibration_bundle(args.calibration_dir, touch_observations=touch)
        if args.json_out:
            save_json_atomic(args.json_out, rep)
        print(f"[{'OK' if rep['passed'] else 'FAIL'}] Calibration verification ({args.calibration_dir})")
        for w in rep["warnings"]:
            print(f"  [WARN] {w}")
        for r in rep["reasons"]:
            print(f"  [FAIL] {r}")
        return 0 if rep["passed"] else 1

    raise ValueError(f"Unknown calibrate subcommand: {args.calib_cmd}")
