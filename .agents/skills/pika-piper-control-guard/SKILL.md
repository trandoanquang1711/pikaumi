---
name: pika-piper-control-guard
description: Enforces hardware safety invariants for Dual Pika Sense capture, AgileX Piper Firmware S-V1.8-9 CAN/ROS2 control, and multi-robot retargeting/replay in quang_ws (`pikaumi` & `Pika_Data`). Always consult before touching robot drivers, CAN interfaces, policy bridges, real-robot replay, or sensor launch scripts.
---

# Pika & Multi-Robot Hardware Safety Guard (`pika-piper-control-guard`)

## Non-Negotiable Safety Rules

1. **Zero Unauthorized Live Hardware Actuation**
   - **Never** run `ros2 launch piper ...`, `can_activate.sh`, `ip link set can*`, `pika_piper_hold_probe.py --execute`, `./pika_data.sh policy-bridge --live-ros2`, `pika replay-real --live`, or `pika policy-bridge --live` without explicit human authorization and physical arm support on-site.
   - Default all CLI checks and tests to `--dry-run` / `--mock-robot`.

2. **AgileX Piper Firmware `S-V1.8-9` Fault Latching (`HOLD_LAST_SAFE`)**
   - `DisableArm` (`CAN 0x150`, byte 0 = `0x01`) immediately releases holding torque into damped gravitational descent.
   - `MotionCtrl_1(0x02)` (`Reset`) is **not** a fixed-pose stop.
   - On any fault (RPC timeout, NaN/Inf, stale feedback `> 250 ms`, workspace AABB violation, step jump, or IK residual violation), always latch **`HOLD_LAST_SAFE`** (hold last verified safe command or halt new motion TX while keeping holding torque active). Never publish `Bool(False)` to `/enable_*` automatically on error.

3. **Physical Deployment Calibration Gate**
   - Physical replay or policy execution (`scope: physical`) must reject `provisional` or `verified: false` calibrations unless `--allow-provisional-calibration` is passed strictly in `--dry-run --mock-robot` mode.
   - Target `piper_pika` arms use **100 mm-stroke Pika grippers** (`[0.0, 0.10] m` / `100,000 µm`, `~194/195 mm` axial TCP offset), distinct from stock `66-70 mm` Piper grippers (`piper` profile).

4. **Single Sensor & Vive Tracker Lock**
   - Run at most one `pika sensors` / `./pika_data.sh sensors` instance (`/tmp/pika-sensors.lock`), and never run `sensors` concurrently with `survive-cli --force-calibrate`.
   - Bind Vive Trackers strictly by `LHR-*` serial codes, never by connection order (`T20`/`T21`).
