#!/usr/bin/env python3
"""Solve camera-to-reported-pose extrinsics in the Pika ROS 1 Docker container."""
import argparse
import bisect
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

BOARD = (4, 5, .045, .035)  # squares x/y, square/marker length (metres)


def quat_to_rotation(x, y, z, w):
    q = np.array([x, y, z, w], dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid Vive quaternion')
    return Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()


def interpolate_pose(poses, pose_times, ns, max_gap_ns):
    index = bisect.bisect_left(pose_times, ns)
    if index < len(poses) and pose_times[index] == ns:
        return poses[index][1:]
    if index == 0 or index == len(poses):
        return None
    ta, ra, pa = poses[index - 1]
    tb, rb, pb = poses[index]
    if tb - ta > max_gap_ns:
        return None
    alpha = (ns - ta) / (tb - ta)
    rotation = Slerp([0., 1.], Rotation.from_matrix([ra, rb]))([alpha]).as_matrix()[0]
    return rotation, (1 - alpha) * pa + alpha * pb


def board_in_base(base_tool_r, base_tool_t, camera_to_tool_r, camera_to_tool_t,
                  board_camera_r, board_camera_t):
    return (base_tool_r @ camera_to_tool_r @ board_camera_r,
            base_tool_r @ (camera_to_tool_r @ board_camera_t
                           + np.asarray(camera_to_tool_t).reshape(3)) + base_tool_t)


def rotation_error_deg(a, b):
    cosine = np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def validate_excitation(rotations):
    relative = Rotation.from_matrix(np.asarray(rotations)[0].T @ np.asarray(rotations)).as_rotvec()
    singular_values = np.linalg.svd(relative, compute_uv=False)
    if len(singular_values) < 2 or singular_values[1] < 1e-3:
        raise ValueError('Hand-eye needs rotation about at least two non-parallel axes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('--fisheye')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--side', choices=['left', 'right'], default='right')
    parser.add_argument('--pose-frame', required=True,
                        help='Measured pose child frame; do not label tracker pose as TCP')
    parser.add_argument('--pose-topic', default='/pika_pose')
    parser.add_argument('--color-topic', default='/camera/color/image_raw')
    parser.add_argument('--info-topic', default='/camera/color/camera_info')
    parser.add_argument('--fisheye-topic', default='/camera_fisheye/color/image_raw')
    parser.add_argument('--camera', choices=['d405', 'fisheye', 'both'], default='both')
    parser.add_argument('--image-stride', type=int, default=15)
    parser.add_argument('--max-pose-gap-ms', type=float, default=20)
    parser.add_argument('--max-reprojection-px', type=float, default=2.)
    parser.add_argument('--validate-existing-dir', type=Path,
                        help='Evaluate saved extrinsics on this separate bag without refitting')
    parser.add_argument('--validation-every', type=int, default=5,
                        help='reserve every Nth usable observation for validation (default: 5)')
    args = parser.parse_args()
    if args.validation_every < 3:
        raise ValueError('--validation-every must be at least 3')
    if args.image_stride < 1 or args.max_pose_gap_ms <= 0 or args.max_reprojection_px <= 0:
        raise ValueError('Stride and error/time limits must be positive')
    if args.camera in ('fisheye', 'both') and not args.fisheye:
        parser.error('--fisheye is required for a fisheye camera')
    # ROS 1 imports stay inside main so geometry helpers can be tested on the host.
    import rosbag
    from cv_bridge import CvBridge

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = (aruco.CharucoBoard_create(*BOARD, dictionary)
             if hasattr(aruco, 'CharucoBoard_create')
             else aruco.CharucoBoard((4, 5), .045, .035, dictionary))
    chessboard = (board.chessboardCorners if hasattr(board, 'chessboardCorners')
                  else board.getChessboardCorners())

    bridge = CvBridge()
    poses, images, camera_info = [], {'d405': [], 'fisheye': []}, None
    topics = [args.pose_topic]
    if args.camera in ('d405', 'both'):
        topics.extend([args.color_topic, args.info_topic])
    if args.camera in ('fisheye', 'both'):
        topics.append(args.fisheye_topic)
    receipt_fallbacks, invalid_poses = 0, 0
    with rosbag.Bag(args.bag) as bag:
        for topic, message, stamp in bag.read_messages(topics=topics):
            header = getattr(message, 'header', None)
            ns = header.stamp.to_nsec() if header is not None else 0
            if ns <= 0:
                ns = stamp.to_nsec()
                receipt_fallbacks += 1
            if topic == args.pose_topic:
                pose = message.pose
                try:
                    rotation = quat_to_rotation(pose.orientation.x, pose.orientation.y,
                                                pose.orientation.z, pose.orientation.w)
                    translation = np.array([pose.position.x, pose.position.y, pose.position.z])
                    if not np.isfinite(translation).all():
                        raise ValueError('Non-finite translation')
                except ValueError:
                    invalid_poses += 1
                    continue
                poses.append((ns, rotation, translation))
            elif topic == args.info_topic:
                camera_info = message
            elif topic in (args.color_topic, args.fisheye_topic):
                stream = 'fisheye' if topic == args.fisheye_topic else 'd405'
                images[stream].append((ns, bridge.imgmsg_to_cv2(message, 'bgr8')))
    if len(poses) < 2:
        raise RuntimeError(f'Missing valid poses on {args.pose_topic}')
    poses.sort(key=lambda p: p[0])
    pose_times = [x[0] for x in poses]
    if any(b <= a for a, b in zip(pose_times, pose_times[1:])):
        raise ValueError('Duplicate Vive timestamps in bag')
    cameras = {}
    if args.camera in ('d405', 'both'):
        if camera_info is None:
            raise RuntimeError(f'Missing {args.info_topic}')
        cameras['d405'] = (np.array(camera_info.K).reshape(3, 3), np.array(camera_info.D), False)
    if args.camera in ('fisheye', 'both'):
        fisheye = json.loads(Path(args.fisheye).read_text())
        matrix, distortion = np.asarray(fisheye['K'], float), np.asarray(fisheye['D'], float)
        if (matrix.shape != (3, 3) or distortion.size != 4 or not np.isfinite(matrix).all()
                or not np.isfinite(distortion).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
            raise ValueError('Invalid measured fisheye K/D')
        cameras['fisheye'] = (matrix, distortion, True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (matrix, distortion, is_fisheye) in cameras.items():
        samples = []
        reprojection_errors = []
        for ns, image in sorted(images[name], key=lambda x: x[0])[::args.image_stride]:
            paired = interpolate_pose(poses, pose_times, ns, int(args.max_pose_gap_ms * 1e6))
            if paired is None:
                continue
            base_to_tool_rotation, base_to_tool_translation = paired
            if is_fisheye and (image.shape[1], image.shape[0]) != (
                    fisheye['image_width'], fisheye['image_height']):
                raise ValueError('Fisheye image and calibration resolutions differ')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)
            if ids is None:
                continue
            count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if charuco_ids is None or count < 8:
                continue
            object_points = chessboard[charuco_ids.reshape(-1)].astype(np.float64)
            observed_points = charuco_corners.reshape(-1, 2).astype(np.float64)
            image_points = observed_points
            if is_fisheye:
                image_points = cv2.fisheye.undistortPoints(
                    image_points.reshape(-1, 1, 2), matrix, distortion, P=matrix).reshape(-1, 2)
                pnp_distortion = None
            else:
                pnp_distortion = distortion
            ok, rotation_vector, translation = cv2.solvePnP(
                object_points, image_points, matrix, pnp_distortion)
            if ok and np.isfinite(translation).all() and translation[2, 0] > 0:
                if is_fisheye:
                    projected, _ = cv2.fisheye.projectPoints(
                        object_points.reshape(-1, 1, 3), rotation_vector, translation,
                        matrix, distortion)
                else:
                    projected, _ = cv2.projectPoints(object_points, rotation_vector,
                                                     translation, matrix, distortion)
                error = float(np.sqrt(np.mean(np.sum(
                    (projected.reshape(-1, 2) - observed_points) ** 2, axis=1))))
                if not np.isfinite(error) or error > args.max_reprojection_px:
                    continue
                reprojection_errors.append(error)
                samples.append((base_to_tool_rotation, base_to_tool_translation,
                                cv2.Rodrigues(rotation_vector)[0], translation.reshape(3)))

        required_pairs = 5 if args.validate_existing_dir else 25
        if len(samples) < required_pairs:
            raise RuntimeError(f'{name}: only {len(samples)} usable pairs; need {required_pairs}')
        train = [sample for i, sample in enumerate(samples) if (i + 1) % args.validation_every]
        test = [sample for i, sample in enumerate(samples) if not (i + 1) % args.validation_every]
        if len(test) < 5 and not args.validate_existing_dir:
            raise RuntimeError(f'{name}: only {len(test)} held-out pairs; record a longer bag')

        # OpenCV requires tool -> base and board -> camera transforms.
        rotations_tool_to_base = [r for r, _, _, _ in train]
        translations_tool_to_base = [t for _, t, _, _ in train]
        rotations_board_to_camera = [r for _, _, r, _ in train]
        translations_board_to_camera = [t for _, _, _, t in train]
        if args.validate_existing_dir:
            saved = json.loads((args.validate_existing_dir /
                                f'{args.side}_{name}_to_pose.json').read_text())
            if saved['parent_frame'] != args.pose_frame:
                raise ValueError('Saved extrinsic and --pose-frame differ')
            rotation_camera_to_tool = np.asarray(saved['rotation_matrix'])
            translation_camera_to_tool = np.asarray(saved['translation_m'])
            test, train = samples, []
        else:
            validate_excitation(rotations_tool_to_base)
            rotation_camera_to_tool, translation_camera_to_tool = cv2.calibrateHandEye(
                rotations_tool_to_base, translations_tool_to_base,
                rotations_board_to_camera, translations_board_to_camera,
                method=cv2.CALIB_HAND_EYE_ANDREFF)
        translation_camera_to_tool = np.asarray(translation_camera_to_tool).reshape(3)
        determinant = np.linalg.det(rotation_camera_to_tool)
        if (not np.isfinite(rotation_camera_to_tool).all()
                or not np.isfinite(translation_camera_to_tool).all()
                or abs(determinant - 1.0) > 1e-3
                or not np.allclose(rotation_camera_to_tool.T @ rotation_camera_to_tool,
                                   np.eye(3), atol=1e-3)):
            raise RuntimeError(f'{name}: invalid hand-eye rotation (det={determinant:.6f})')

        # Held-out observations must infer one consistent fixed board pose in base frame.
        base_board = [
            board_in_base(base_tool_r, base_tool_t, rotation_camera_to_tool,
                          translation_camera_to_tool, board_camera_r, board_camera_t)
            for base_tool_r, base_tool_t, board_camera_r, board_camera_t in test
        ]
        translations = np.array([translation for _, translation in base_board])
        median_translation = np.median(translations, axis=0)
        medoid_index = min(
            range(len(base_board)),
            key=lambda i: sum(rotation_error_deg(base_board[i][0], other[0]) for other in base_board))
        median_rotation = base_board[medoid_index][0]
        position_errors_mm = np.linalg.norm(translations - median_translation, axis=1) * 1000
        rotation_errors_deg = np.array([
            rotation_error_deg(median_rotation, rotation) for rotation, _ in base_board])
        validation = {
            'held_out_pair_count': len(test),
            'base_board_translation_median_m': median_translation.tolist(),
            'position_error_mm': {'mean': float(position_errors_mm.mean()),
                                  'p95': float(np.percentile(position_errors_mm, 95)),
                                  'max': float(position_errors_mm.max())},
            'rotation_error_deg': {'mean': float(rotation_errors_deg.mean()),
                                   'p95': float(np.percentile(rotation_errors_deg, 95)),
                                   'max': float(rotation_errors_deg.max())},
            'independent_bag': bool(args.validate_existing_dir),
            'note': ('Saved extrinsic evaluated without refitting.' if args.validate_existing_dir
                     else 'Internal hold-out only; evaluate on a separate bag without refitting.'),
        }
        result = {
            'parent_frame': args.pose_frame,
            'child_frame': f'pika_{args.side}_{name}_optical_frame',
            'translation_m': translation_camera_to_tool.reshape(-1).tolist(),
            'rotation_matrix': rotation_camera_to_tool.tolist(),
            'method': 'ChArUco + Vive pose hand-eye calibration (Andreff)',
            'accepted_pair_count': len(samples), 'training_pair_count': len(train),
            'timestamp_source': 'header.stamp, receipt fallback if unavailable',
            'receipt_timestamp_fallbacks': receipt_fallbacks,
            'invalid_pose_count': invalid_poses,
            'reprojection_rms_px_p95': float(np.percentile(reprojection_errors, 95)),
            'max_pose_gap_ms': args.max_pose_gap_ms,
            'board': {'squares_x': 4, 'squares_y': 5, 'square_length_m': .045,
                      'marker_length_m': .035, 'dictionary': 'DICT_4X4_50'},
            'validation': validation,
        }
        suffix = '_validation' if args.validate_existing_dir else ''
        destination = output_dir / f'{args.side}_{name}_to_pose{suffix}.json'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite calibration: {destination}')
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(name, 'train=', len(train), 'held-out=', len(test), destination)
        print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
