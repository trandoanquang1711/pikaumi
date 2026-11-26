"""Live ChArUco + Vive Tracker observation capture from `pika-ros` Docker for Layer 3 calibration.

Supports both:
- `mode="hand-eye"` -> writes `{"T_vive_tracker": [...], "T_cam_board": [...]}` for `pika calibrate hand-eye`
- `mode="session"`  -> writes `{"side": ..., "T_vive_tracker": [...], "T_cam_table": [...]}` for `pika calibrate session`

Note on board-to-table frame (`T_board_table`):
  OpenCV `CharucoBoard` places origin at the top-left corner with `+X` right, `+Y` down along the
  printed page, and `+Z` pointing INTO the table. `board_to_table_transform()` shifts the origin to
  the geometric center of the board and rotates 180 deg around `+X` so `{Table}` matches `quang_ws`:
  `+X` right, `+Y` forward (toward top of printed page), `+Z` straight UP out of the table.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pikaumi.config import load_rig_config, load_yaml, save_json_atomic
from pikaumi.robots.utils import mat_to_pose7, normalize_quat_xyzw, pose7_to_mat


@dataclass(frozen=True)
class CharucoBoardSpec:
    """Physical geometry & ArUco dictionary of the calibration board."""

    squares_x: int = 4
    squares_y: int = 5
    square_length_m: float = 0.045
    marker_length_m: float = 0.035
    dictionary: str = "DICT_4X4_50"

    def board_to_table_transform(self) -> np.ndarray:
        """Return 4x4 `T_board_table` mapping `{Table}` (+X right, +Y top, +Z up, center origin) into OpenCV `{Board}`."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = np.array(
            [
                0.5 * float(self.squares_x) * float(self.square_length_m),
                0.5 * float(self.squares_y) * float(self.square_length_m),
                0.0,
            ],
            dtype=np.float64,
        )
        return T

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ChArUco",
            "squares_x": int(self.squares_x),
            "squares_y": int(self.squares_y),
            "square_length_m": float(self.square_length_m),
            "marker_length_m": float(self.marker_length_m),
            "dictionary": str(self.dictionary),
        }


@dataclass(frozen=True)
class CharucoDetectionResult:
    """Single fisheye ChArUco PnP detection result."""

    T_cam_board_pose7: np.ndarray  # (7,) pose7 in OpenCV board frame
    T_cam_table_pose7: np.ndarray  # (7,) pose7 in {Table} frame (+X right, +Y forward, +Z up)
    num_corners: int
    reprojection_rms_px: float
    charuco_corners: np.ndarray  # (N, 1, 2)
    charuco_ids: np.ndarray  # (N, 1)


def load_board_spec_from_configs(
    *,
    spatial_yaml: str | Path | None = None,
    side: str = "left",
    rig_config_path: str | Path | None = None,
    squares_x: int | None = None,
    squares_y: int | None = None,
    square_length_m: float | None = None,
    marker_length_m: float | None = None,
    dictionary: str | None = None,
) -> CharucoBoardSpec:
    """Resolve ChArUco board specification with precedence: CLI > spatial.yaml > rig.yaml > defaults."""
    spec = CharucoBoardSpec()
    sx, sy, sq, mk, dic = (
        spec.squares_x,
        spec.squares_y,
        spec.square_length_m,
        spec.marker_length_m,
        spec.dictionary,
    )

    rig = load_rig_config(rig_config_path)
    rig_board = (rig.get("spatial_calibration") or {}).get("board") or {}
    if rig_board:
        sx = int(rig_board.get("squares_x", sx))
        sy = int(rig_board.get("squares_y", sy))
        sq = float(rig_board.get("square_length_m", sq))
        mk = float(rig_board.get("marker_length_m", mk))
        dic = str(rig_board.get("dictionary", dic))

    if spatial_yaml is not None and Path(spatial_yaml).is_file():
        sp = load_yaml(spatial_yaml)
        cam_board = ((sp.get("cameras") or {}).get(side) or {}).get("board") or {}
        if cam_board:
            sx = int(cam_board.get("squares_x", sx))
            sy = int(cam_board.get("squares_y", sy))
            sq = float(cam_board.get("square_length_m", sq))
            mk = float(cam_board.get("marker_length_m", mk))
            dic = str(cam_board.get("dictionary", dic))

    if squares_x is not None:
        sx = int(squares_x)
    if squares_y is not None:
        sy = int(squares_y)
    if square_length_m is not None:
        sq = float(square_length_m)
    if marker_length_m is not None:
        mk = float(marker_length_m)
    if dictionary is not None:
        dic = str(dictionary)

    return CharucoBoardSpec(
        squares_x=sx,
        squares_y=sy,
        square_length_m=sq,
        marker_length_m=mk,
        dictionary=dic,
    )


def load_fisheye_intrinsics_from_spatial(
    spatial_yaml: str | Path,
    side: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    """Load Kannala-Brandt `(K, D, (width, height))` for `side` from `spatial.yaml` or raw JSON."""
    path = Path(spatial_yaml)
    if not path.is_file():
        raise FileNotFoundError(
            f"Spatial calibration file not found: {path}. "
            "Run `./pika calibrate import-legacy --legacy-dir data/calibration --output-dir data/calibration_std` first."
        )
    if path.suffix.lower() == ".json":
        cam = json.loads(path.read_text(encoding="utf-8"))
    else:
        sp = load_yaml(path)
        cameras = sp.get("cameras") or {}
        if side not in cameras:
            raise KeyError(
                f"Side {side!r} fisheye intrinsics missing from {path}. "
                f"Available sides: {sorted(cameras.keys())}"
            )
        cam = cameras[side]

    K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(cam["D"], dtype=np.float64).reshape(4)
    if not (np.isfinite(K).all() and np.isfinite(D).all() and K[0, 0] > 10.0 and K[1, 1] > 10.0):
        raise ValueError(f"Invalid fisheye K/D intrinsics for side={side!r} in {path}")
    size = None
    if "image_width" in cam and "image_height" in cam:
        size = (int(cam["image_width"]), int(cam["image_height"]))
    return K, D, size


def _build_opencv_charuco(cv2: Any, spec: CharucoBoardSpec) -> tuple[Any, Any, np.ndarray]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco module is unavailable. Use `.venv-pika-dataset` or install `opencv-contrib-python`."
        )
    dict_id = getattr(cv2.aruco, spec.dictionary, None)
    if dict_id is None:
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {spec.dictionary!r}")
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            spec.squares_x,
            spec.squares_y,
            spec.square_length_m,
            spec.marker_length_m,
            dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_length_m,
            spec.marker_length_m,
            dictionary,
        )
    corners_3d = (
        board.chessboardCorners
        if hasattr(board, "chessboardCorners")
        else board.getChessboardCorners()
    )
    return dictionary, board, np.asarray(corners_3d, dtype=np.float64)


def detect_charuco_fisheye_pose(
    image_bgr: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    board_spec: CharucoBoardSpec,
    *,
    min_corners: int = 6,
    max_reprojection_px: float = 2.5,
) -> CharucoDetectionResult | None:
    """Detect ChArUco corners and solve Kannala-Brandt fisheye `T_cam_board` & `T_cam_table`."""
    import cv2

    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else np.asarray(image_bgr, dtype=np.uint8)
    )
    dictionary, board, corners_3d = _build_opencv_charuco(cv2, board_spec)

    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    else:
        params = (
            cv2.aruco.DetectorParameters_create()
            if hasattr(cv2.aruco, "DetectorParameters_create")
            else cv2.aruco.DetectorParameters()
        )
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if marker_ids is None or len(marker_ids) == 0:
            return None
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < int(min_corners):
        return None

    ids_flat = charuco_ids.reshape(-1).astype(int)
    obj_pts = corners_3d[ids_flat].astype(np.float64)
    obs_pts = charuco_corners.reshape(-1, 2).astype(np.float64)

    # Undistort observed pixel coordinates using exact Kannala-Brandt fisheye model before PnP
    undist_pts = cv2.fisheye.undistortPoints(
        obs_pts.reshape(-1, 1, 2),
        K,
        D.reshape(4, 1),
        P=K,
    ).reshape(-1, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        undist_pts,
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    if float(tvec[2, 0]) <= 0.02:
        return None

    proj, _ = cv2.fisheye.projectPoints(
        obj_pts.reshape(-1, 1, 3),
        rvec,
        tvec,
        K,
        D.reshape(4, 1),
    )
    reproj_rms = float(
        np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - obs_pts) ** 2, axis=1)))
    )
    if not np.isfinite(reproj_rms) or reproj_rms > float(max_reprojection_px):
        return None

    R_cb, _ = cv2.Rodrigues(rvec)
    T_cb = np.eye(4, dtype=np.float64)
    T_cb[:3, :3] = R_cb
    T_cb[:3, 3] = tvec.reshape(3)
    T_ct = T_cb @ board_spec.board_to_table_transform()

    return CharucoDetectionResult(
        T_cam_board_pose7=mat_to_pose7(T_cb),
        T_cam_table_pose7=mat_to_pose7(T_ct),
        num_corners=int(len(ids_flat)),
        reprojection_rms_px=reproj_rms,
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
    )


def interpolate_pose7_pair(
    t0_ns: int,
    pose0: np.ndarray,
    t1_ns: int,
    pose1: np.ndarray,
    target_ns: int,
    *,
    max_gap_ns: int = 25_000_000,
) -> np.ndarray | None:
    """SLERP + linear interpolate between two timestamped pose7 samples, rejecting stale gaps."""
    p0 = np.asarray(pose0, dtype=np.float64).reshape(7)
    p1 = np.asarray(pose1, dtype=np.float64).reshape(7)
    if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
        return None
    if t0_ns == t1_ns:
        if abs(target_ns - t0_ns) > max_gap_ns:
            return None
        out = p0.copy()
        out[3:7] = normalize_quat_xyzw(out[3:7])
        return out
    if t1_ns < t0_ns:
        t0_ns, t1_ns = t1_ns, t0_ns
        p0, p1 = p1, p0
    if (t1_ns - t0_ns) > max_gap_ns:
        return None
    if target_ns < t0_ns - max_gap_ns or target_ns > t1_ns + max_gap_ns:
        return None

    alpha = float(np.clip((target_ns - t0_ns) / float(t1_ns - t0_ns), 0.0, 1.0))
    pos = (1.0 - alpha) * p0[:3] + alpha * p1[:3]
    q0 = normalize_quat_xyzw(p0[3:7])
    q1 = normalize_quat_xyzw(p1[3:7])
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    slerp = Slerp([0.0, 1.0], Rotation.from_quat([q0, q1]))
    q_interp = normalize_quat_xyzw(slerp([alpha]).as_quat()[0])
    return np.concatenate([pos, q_interp])


def is_view_diverse(
    pose7: np.ndarray,
    existing_poses7: Iterable[np.ndarray],
    *,
    min_rot_deg: float = 6.0,
    min_trans_m: float = 0.025,
) -> bool:
    """Return True if `pose7` differs from every previously accepted view by enough rotation or translation."""
    T_new = pose7_to_mat(pose7)
    R_new = T_new[:3, :3]
    p_new = T_new[:3, 3]
    for prev in existing_poses7:
        T_prev = pose7_to_mat(prev)
        dp = float(np.linalg.norm(p_new - T_prev[:3, 3]))
        R_rel = T_prev[:3, :3].T @ R_new
        cos_ang = float(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))
        d_deg = float(math.degrees(math.acos(cos_ang)))
        if d_deg < float(min_rot_deg) and dp < float(min_trans_m):
            return False
    return True


_DOCKER_ROS1_CHARUCO_STREAM_SCRIPT = r"""
import base64, collections, json, math, sys, time
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from data_msgs.msg import LocalizationStatus

pose_topic = sys.argv[1]
status_topic = sys.argv[2]
image_topic = sys.argv[3]
stream_hz = float(sys.argv[4])
window_ns = int(round(float(sys.argv[5]) * 1e6)) if len(sys.argv) > 5 else 220000000

bridge = CvBridge()
pose_buf = collections.deque(maxlen=200)
status_state = {"accurate": False, "seen": False}
latest_img = {"stamp_ns": 0, "msg": None, "seq": 0}

def _stamp_ns(header):
    if header is not None and header.stamp.to_nsec() > 0:
        return int(header.stamp.to_nsec())
    return int(rospy.Time.now().to_nsec())

def _on_pose(msg):
    p = msg.pose.position
    q = msg.pose.orientation
    pose_buf.append((_stamp_ns(msg.header), [p.x, p.y, p.z, q.x, q.y, q.z, q.w]))

def _on_status(msg):
    status_state["accurate"] = bool(msg.accurate)
    status_state["seen"] = True

def _on_image(msg):
    latest_img["stamp_ns"] = _stamp_ns(msg.header)
    latest_img["msg"] = msg
    latest_img["seq"] += 1

def _window_motion(poses, ref_ns, ref_pose):
    win = [p for ts, p in poses if (ref_ns - window_ns) <= ts <= (ref_ns + 40000000)]
    if len(win) < 3:
        return 999.0, 999.0, len(win)
    arr = np.asarray(win, dtype=np.float64)
    p_ref = np.asarray(ref_pose[:3], dtype=np.float64)
    q_ref = np.asarray(ref_pose[3:7], dtype=np.float64)
    q_norm = float(np.linalg.norm(q_ref))
    if q_norm < 1e-9:
        return 999.0, 999.0, len(win)
    q_ref = q_ref / q_norm
    pos_mm = float(np.max(np.linalg.norm(arr[:, :3] - p_ref[None, :], axis=1)) * 1000.0)
    qs = arr[:, 3:7]
    q_norms = np.maximum(np.linalg.norm(qs, axis=1, keepdims=True), 1e-9)
    qs = qs / q_norms
    dots = np.clip(np.abs(qs @ q_ref), 0.0, 1.0)
    rot_deg = float(np.max(2.0 * np.degrees(np.arccos(dots))))
    return pos_mm, rot_deg, len(win)

rospy.init_node("pikaumi_charuco_calib_stream", anonymous=True, disable_signals=True)
rospy.Subscriber(pose_topic, PoseStamped, _on_pose, queue_size=20)
rospy.Subscriber(status_topic, LocalizationStatus, _on_status, queue_size=5)
rospy.Subscriber(image_topic, Image, _on_image, queue_size=2)

rate = rospy.Rate(max(stream_hz, 1.0))
last_seq = 0
while not rospy.is_shutdown():
    if latest_img["msg"] is not None and latest_img["seq"] != last_seq and len(pose_buf) >= 1:
        last_seq = latest_img["seq"]
        img_ns = latest_img["stamp_ns"]
        msg = latest_img["msg"]
        poses = list(pose_buf)
        # Find bracketing or nearest pose around image timestamp
        p_prev = poses[0]
        p_next = poses[-1]
        for i in range(len(poses)):
            if poses[i][0] <= img_ns:
                p_prev = poses[i]
            if poses[i][0] >= img_ns:
                p_next = poses[i]
                break
        if poses[-1][0] < img_ns:
            p_prev = poses[-1]
            p_next = poses[-1]
        elif poses[0][0] > img_ns:
            p_prev = poses[0]
            p_next = poses[0]
        pos_mm, rot_deg, win_n = _window_motion(poses, img_ns, p_prev[1])
        try:
            bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                payload = {
                    "image_stamp_ns": img_ns,
                    "pose_prev_ns": p_prev[0],
                    "pose_prev": p_prev[1],
                    "pose_next_ns": p_next[0],
                    "pose_next": p_next[1],
                    "stationary_pos_mm": pos_mm,
                    "stationary_rot_deg": rot_deg,
                    "stationary_samples": win_n,
                    "accurate": bool(status_state["accurate"]),
                    "status_seen": bool(status_state["seen"]),
                    "jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
                }
                sys.stdout.write(json.dumps(payload) + "\n")
                sys.stdout.flush()
        except Exception:
            pass
    rate.sleep()
"""


class DockerRos1CharucoCaptureSource:
    """Streams paired fisheye frames and Vive poses for one Pika side out of `pika-ros`."""

    def __init__(
        self,
        *,
        side: str,
        container_name: str = "pika-ros",
        stream_hz: float = 8.0,
        stationary_window_ms: float = 220.0,
        pose_topic: str | None = None,
        status_topic: str | None = None,
        image_topic: str | None = None,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Expected side in ('left', 'right'), got {side!r}")
        suffix = "l" if side == "left" else "r"
        self.side = side
        self.container_name = container_name
        self.stream_hz = float(stream_hz)
        self.stationary_window_ms = float(stationary_window_ms)
        self.pose_topic = pose_topic or f"/pika_pose_{suffix}"
        self.status_topic = status_topic or f"/pika_localization_status_{suffix}"
        self.image_topic = image_topic or f"/camera_fisheye_{suffix}/color/image_raw"
        self._proc: subprocess.Popen[str] | None = None

    def build_exec_command(self) -> list[str]:
        inner = (
            "source /opt/ros/noetic/setup.bash && "
            "source /root/pika_ros/install/setup.bash && "
            f"python3 -u -c {shlex.quote(_DOCKER_ROS1_CHARUCO_STREAM_SCRIPT)} "
            f"{shlex.quote(self.pose_topic)} {shlex.quote(self.status_topic)} "
            f"{shlex.quote(self.image_topic)} {self.stream_hz:.2f} {self.stationary_window_ms:.1f}"
        )
        return ["docker", "exec", "-i", self.container_name, "bash", "-lc", inner]

    def __enter__(self) -> "DockerRos1CharucoCaptureSource":
        self._proc = subprocess.Popen(
            self.build_exec_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def iter_frames(self) -> Iterable[dict[str, Any]]:
        assert self._proc is not None and self._proc.stdout is not None
        import cv2

        yielded = 0
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                jpg_bytes = base64.b64decode(obj["jpeg_b64"])
                arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                obj["image_bgr"] = bgr
                yielded += 1
                yield obj
            except Exception:
                continue
        rc = self._proc.poll()
        if rc is None:
            try:
                rc = self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                rc = None
        if rc not in (None, 0) and yielded == 0:
            err_txt = ""
            if self._proc.stderr is not None:
                try:
                    err_txt = self._proc.stderr.read().strip()
                except Exception:
                    err_txt = ""
            raise RuntimeError(
                f"Docker ROS 1 capture stream exited with code {rc}"
                + (f":\n{err_txt}" if err_txt else "")
            )

    def close(self) -> None:
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


def capture_charuco_observations(
    *,
    mode: str,
    side: str,
    spatial_yaml: str | Path,
    output_json: str | Path,
    board_spec: CharucoBoardSpec | None = None,
    target_views: int | None = None,
    min_corners: int | None = None,
    max_reprojection_px: float = 2.5,
    max_pose_gap_ms: float = 25.0,
    min_interval_s: float = 0.4,
    min_rot_deg: float | None = None,
    min_trans_mm: float | None = None,
    max_motion_mm: float = 4.0,
    max_motion_deg: float = 2.0,
    stationary_window_ms: float = 220.0,
    require_accurate: bool = True,
    preview: bool = True,
    container_name: str = "pika-ros",
    frame_iterable: Iterable[dict[str, Any]] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Capture paired `(T_vive_tracker, T_cam_board/T_cam_table)` views and write observations JSON."""
    if mode not in ("hand-eye", "session"):
        raise ValueError(f"Unsupported capture mode: {mode!r} (expected 'hand-eye' or 'session')")

    K, D, expected_size = load_fisheye_intrinsics_from_spatial(spatial_yaml, side)
    spec = board_spec or load_board_spec_from_configs(spatial_yaml=spatial_yaml, side=side)
    n_target = int(target_views if target_views is not None else (30 if mode == "hand-eye" else 15))
    n_corners = int(min_corners if min_corners is not None else (8 if mode == "hand-eye" else 6))
    rot_thresh = float(min_rot_deg if min_rot_deg is not None else (6.0 if mode == "hand-eye" else 3.0))
    trans_thresh_m = float(
        (min_trans_mm if min_trans_mm is not None else (25.0 if mode == "hand-eye" else 15.0)) / 1000.0
    )
    max_gap_ns = int(round(float(max_pose_gap_ms) * 1e6))

    use_gui = bool(preview and os.environ.get("DISPLAY") and frame_iterable is None)
    cv2_mod = None
    if use_gui:
        try:
            import cv2 as _cv2

            cv2_mod = _cv2
        except Exception:
            use_gui = False

    vive_poses: list[np.ndarray] = []
    cam_poses: list[np.ndarray] = []
    reproj_errors: list[float] = []
    last_accept_t = 0.0

    def _emit(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)
        else:
            print(msg, flush=True)

    _emit(
        f"[INFO] Capturing {n_target} {mode} ChArUco views for side={side!r} "
        f"(board={spec.squares_x}x{spec.squares_y}, sq={spec.square_length_m*1000:.1f}mm, "
        f"mk={spec.marker_length_m*1000:.1f}mm, dict={spec.dictionary}, "
        f"stationary<={max_motion_mm:.1f}mm/{max_motion_deg:.1f}deg)"
    )
    if frame_iterable is None:
        _emit("[INFO] Pause ~0.3s at each distinct angle/position facing the board so camera lag is zero. Press Ctrl+C to stop early.")

    last_status_t = 0.0

    def _consume(frames: Iterable[dict[str, Any]]) -> None:
        nonlocal last_accept_t, last_status_t, use_gui
        for pkt in frames:
            if len(vive_poses) >= n_target:
                break
            bgr = pkt["image_bgr"]
            if expected_size is not None and (bgr.shape[1], bgr.shape[0]) != expected_size:
                raise ValueError(
                    f"Fisheye frame size {(bgr.shape[1], bgr.shape[0])} does not match "
                    f"intrinsics resolution {expected_size}"
                )

            accurate = bool(pkt.get("accurate", False))
            has_motion = ("stationary_pos_mm" in pkt) or ("stationary_rot_deg" in pkt)
            motion_mm = float(pkt.get("stationary_pos_mm", 0.0))
            motion_deg = float(pkt.get("stationary_rot_deg", 0.0))
            motion_samples = int(pkt.get("stationary_samples", 10))
            is_stationary = (
                not has_motion
                or (
                    motion_samples >= 3
                    and motion_mm <= float(max_motion_mm)
                    and motion_deg <= float(max_motion_deg)
                )
            )
            pose_interp = interpolate_pose7_pair(
                int(pkt["pose_prev_ns"]),
                np.asarray(pkt["pose_prev"], dtype=np.float64),
                int(pkt["pose_next_ns"]),
                np.asarray(pkt["pose_next"], dtype=np.float64),
                int(pkt["image_stamp_ns"]),
                max_gap_ns=max_gap_ns,
            )
            det = detect_charuco_fisheye_pose(
                bgr,
                K,
                D,
                spec,
                min_corners=n_corners,
                max_reprojection_px=max_reprojection_px,
            )

            status_text = "waiting..."
            captured_now = False
            now = time.monotonic()
            if require_accurate and not accurate:
                status_text = "Vive NOT accurate"
            elif pose_interp is None:
                status_text = f"Pose/Image gap > {max_pose_gap_ms:.0f}ms"
            elif not is_stationary:
                status_text = (
                    f"Hold still... (motion={motion_mm:.1f}mm / {motion_deg:.1f}deg; "
                    f"need <= {max_motion_mm:.1f}mm / {max_motion_deg:.1f}deg)"
                )
            elif det is None:
                status_text = f"Board not detected (<{n_corners} corners or reproj>{max_reprojection_px:.1f}px)"
            elif (now - last_accept_t) < min_interval_s and frame_iterable is None:
                status_text = "Hold/move to next angle..."
            elif not is_view_diverse(
                pose_interp,
                vive_poses,
                min_rot_deg=rot_thresh,
                min_trans_m=trans_thresh_m,
            ):
                status_text = "Move to a new angle/position"
            else:
                vive_poses.append(pose_interp)
                cam_poses.append(
                    det.T_cam_board_pose7 if mode == "hand-eye" else det.T_cam_table_pose7
                )
                reproj_errors.append(det.reprojection_rms_px)
                last_accept_t = now
                captured_now = True
                status_text = f"CAPTURED {len(vive_poses)}/{n_target} (reproj={det.reprojection_rms_px:.2f}px)"
                motion_note = f" motion={motion_mm:.1f}mm/{motion_deg:.1f}deg" if has_motion else ""
                _emit(
                    f"  [{len(vive_poses):02d}/{n_target:02d}] corners={det.num_corners} "
                    f"reproj={det.reprojection_rms_px:.2f}px{motion_note}"
                )

            if use_gui and cv2_mod is not None:
                try:
                    vis = bgr.copy()
                    if det is not None:
                        cv2_mod.aruco.drawDetectedCornersCharuco(vis, det.charuco_corners, det.charuco_ids)
                    color = (0, 220, 0) if captured_now else (0, 200, 255)
                    cv2_mod.putText(
                        vis,
                        f"{mode} ({side}) {len(vive_poses)}/{n_target} | {status_text}",
                        (12, 28),
                        cv2_mod.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                    cv2_mod.imshow(f"pika calibrate capture ({mode} - {side})", vis)
                    key = cv2_mod.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                except Exception:
                    use_gui = False
                    _emit(
                        "[WARN] OpenCV GUI (cv2.imshow) is unavailable in this headless build; "
                        "continuing with live terminal status (press Ctrl+C to stop early)."
                    )

            if not use_gui and not captured_now and frame_iterable is None and (now - last_status_t) >= 1.0:
                last_status_t = now
                _emit(f"  [WAIT {len(vive_poses):02d}/{n_target:02d}] {status_text}")

    try:
        if frame_iterable is not None:
            _consume(frame_iterable)
        else:
            with DockerRos1CharucoCaptureSource(
                side=side,
                container_name=container_name,
                stationary_window_ms=stationary_window_ms,
            ) as src:
                _consume(src.iter_frames())
    except KeyboardInterrupt:
        _emit("[INFO] Capture interrupted by user.")
    finally:
        if use_gui and cv2_mod is not None:
            try:
                cv2_mod.destroyAllWindows()
            except Exception:
                pass

    min_required = 6 if mode == "hand-eye" else 3
    if len(vive_poses) < min_required:
        raise RuntimeError(
            f"Captured only {len(vive_poses)} usable {mode} views (minimum required: {min_required}). "
            "Check that `./pika sensors` is running, Vive tracking is accurate, and the ChArUco board is clearly visible."
        )

    cam_key = "T_cam_board" if mode == "hand-eye" else "T_cam_table"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "side": side,
        "board": spec.to_dict(),
        "num_views": len(vive_poses),
        "mean_reprojection_px": float(np.mean(reproj_errors)),
        "max_reprojection_px": float(np.max(reproj_errors)),
        "T_vive_tracker": [p.tolist() for p in vive_poses],
        cam_key: [p.tolist() for p in cam_poses],
        "reprojection_errors_px": [float(e) for e in reproj_errors],
    }
    out_path = save_json_atomic(output_json, payload)
    _emit(
        f"[OK] Saved {len(vive_poses)} {mode} observations to {out_path} "
        f"(mean reproj={payload['mean_reprojection_px']:.2f}px)"
    )
    return payload
