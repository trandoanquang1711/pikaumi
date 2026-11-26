# Onboarding a New Robot Embodiment in `pikaumi`

`pikaumi` decouples **Dual Pika Sense** human data collection (`pika_raw`, 16D Vive + 100 mm jaw width) from target robot kinematics. Adding a 10th robot embodiment requires **zero changes to captured datasets** and only 3 declarative steps.

---

## Step 1 — Add Self-Contained URDF & Visual Meshes under `assets/<robot>/`

Place the robot's URDF and STL/OBJ/DAE meshes in `assets/<robot>/`:
- All `<mesh filename="...">` paths must resolve either relatively from the URDF directory or via `package://...` under `pkg_root`.
- Both **6-DoF**, **7-DoF**, **single-arm**, and **bimanual** trees with arbitrary `<axis xyz="...">` vectors, locked torso joints, and `<mimic>` gripper joints are natively supported by [`load_urdf_model()`](../src/pikaumi/robots/loader.py).

---

## Step 2 — Create `configs/robots/<robot>.yaml`

Define the embodiment profile in `configs/robots/<robot>.yaml`:

```yaml
kind: my_robot
urdf: assets/my_robot/my_robot_bimanual.urdf
pkg_root: assets/my_robot

arms:
  left:
    ee_link: left_tcp_link
    joint_names: [left_j1, left_j2, left_j3, left_j4, left_j5, left_j6, left_finger]
    gripper_joints:
      - name: left_finger
        closed: 0.0
        open: 0.045
  right:
    ee_link: right_tcp_link
    joint_names: [right_j1, right_j2, right_j3, right_j4, right_j5, right_j6, right_finger]
    gripper_joints:
      - name: right_finger
        closed: 0.0
        open: 0.045

home_q: [0.0, 1.2, -1.1, 0.0, 0.4, 0.0, 0.0,  0.0, 1.2, -1.1, 0.0, 0.4, 0.0, 0.0]
posture_q: [0.0, 1.3, -1.2, null, null, null, null,  0.0, 1.3, -1.2, null, null, null, null]

gripper_max_width_m: 0.090
# Pika TCP uses +X approach, +Y jaw left, +Z top.
# Use [0.0, 0.0, 0.0] if ee_link has +X approach (e.g. piper_pika, r1lite),
# or [0.0, 1.5707963, 0.0] if ee_link has +Z approach (e.g. openarmv1, franka_emika_panda).
pika_to_ee_rpy: [0.0, 1.5707963, 0.0]

ik_weights:
  pos: 125.0
  ori: 15.0
  rest: 8.0
  max_joint_delta: 0.06
  max_reach: 0.75

replay:
  max_joint_delta: 0.30
  gripper_retarget: physical-width   # or "normalized" when stroke << 100 mm
  ik_weights:
    ori: 6.0
    rest: 4.0
    posture: 2.0
    limit_margin: 5.0

real:
  fault_mode: HOLD_LAST_SAFE
  command_rate_hz: 200
  max_joint_speed_deg_s: 180
  max_joint_acceleration_deg_s2: 720
```

---

## Step 3 — Add Table Calibration (`configs/calibration/table/sim/<robot>.yaml`)

Create `configs/calibration/table/sim/<robot>.yaml` with `scope: simulation` and `verified: false` for offline screening, simulation preview, and retargeting benchmarks:

```yaml
schema_version: 1
kind: pikaumi_robot_table_calibration
robot: my_robot
scope: simulation
source: cad_nominal
verified: false
calibration:
  frame_convention: pose7=[x,y,z,qx,qy,qz,qw], meters, xyzw quaternion
  robot_from_table:
    position: [0.35, 0.0, 0.0]
    quaternion: [0.0, 0.0, -0.70710678, 0.70710678]
```

> **Safety Gate:** Physical replay (`--live`) automatically refuses `scope: simulation` or `verified: false`. Before physical deployment, measure 3 tabletop contact points and save `configs/calibration/table/local/my_robot.yaml` (`gitignored`) with `scope: physical` and `verified: true`.

---

## Step 4 — Verify Kinematics, Screening & Benchmark

```bash
./pika robots info my_robot
./pika dataset screen <pika_raw_dir> --robots my_robot --solver cstro
./pika convert <pika_raw_dir> --output-dir outputs/my_robot_joints --robot my_robot --solver cstro
./pika replay <pika_raw_dir> --episode 0 --robot my_robot --output-dir outputs/preview
./pika benchmark-retarget --robots my_robot --output-dir outputs/bench_my_robot
```
