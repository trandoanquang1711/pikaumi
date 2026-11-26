"""Real-time fixed-rate acceleration-limited joint command streamer for physical robots."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

import numpy as np


def next_periodic_deadline(deadline: float, now: float, period_s: float) -> float:
    """Advance a periodic loop deadline without accumulating backlog after overruns."""
    if period_s <= 0.0:
        raise ValueError(f"period_s must be positive, got {period_s}")
    nxt = deadline + period_s
    if nxt <= now:
        missed = math.floor((now - deadline) / period_s) + 1
        nxt = deadline + missed * period_s
    return nxt


class AccelerationLimitedJointTrajectory:
    """Online latest-target joint interpolator enforcing hard velocity & acceleration bounds.

    Each tick updates `position` toward `target` using braking-distance clamping so the joint
    decelerates smoothly without overshoot even when `target` is updated asynchronously.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        *,
        max_velocity_rad_s: float | np.ndarray,
        max_acceleration_rad_s2: float | np.ndarray,
        tracking_cutoff_hz: float | None = None,
    ) -> None:
        pos = np.asarray(initial_position, dtype=np.float64).reshape(-1).copy()
        self._position = pos
        self._velocity = np.zeros_like(pos)
        self._target = pos.copy()
        self._filtered_target = pos.copy()
        self._tracking_cutoff_hz: float | None = None
        self.set_tracking_filter(tracking_cutoff_hz)
        self._max_velocity = self._broadcast_positive(max_velocity_rad_s, pos.shape, "max_velocity_rad_s")
        self._max_acceleration = self._broadcast_positive(
            max_acceleration_rad_s2, pos.shape, "max_acceleration_rad_s2"
        )

    @staticmethod
    def _broadcast_positive(val: float | np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        arr = np.broadcast_to(np.asarray(val, dtype=np.float64), shape).copy()
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} must be finite and strictly positive, got {val!r}")
        return arr

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def tracking_cutoff_hz(self) -> float | None:
        return self._tracking_cutoff_hz

    def set_tracking_filter(self, cutoff_hz: float | None) -> None:
        """Enable/disable 3rd-order critically-damped tracking filter (`None` or `<= 0` restores trapezoidal mode)."""
        if cutoff_hz is None or float(cutoff_hz) <= 0.0:
            self._tracking_cutoff_hz = None
            return
        val = float(cutoff_hz)
        if not math.isfinite(val):
            raise ValueError(f"tracking_cutoff_hz must be finite, got {cutoff_hz!r}")
        if self._tracking_cutoff_hz is None:
            self._filtered_target = self._position.copy()
        self._tracking_cutoff_hz = val

    def set_target(self, target: np.ndarray) -> None:
        arr = np.asarray(target, dtype=np.float64).reshape(-1)
        if arr.shape != self._position.shape:
            raise ValueError(f"Expected target shape {self._position.shape}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Joint trajectory target must be finite")
        self._target = arr.copy()

    def set_limits(
        self,
        *,
        max_velocity_rad_s: float | np.ndarray | None = None,
        max_acceleration_rad_s2: float | np.ndarray | None = None,
    ) -> None:
        if max_velocity_rad_s is not None:
            self._max_velocity = self._broadcast_positive(
                max_velocity_rad_s, self._position.shape, "max_velocity_rad_s"
            )
        if max_acceleration_rad_s2 is not None:
            self._max_acceleration = self._broadcast_positive(
                max_acceleration_rad_s2, self._position.shape, "max_acceleration_rad_s2"
            )

    def hold_current(self) -> np.ndarray:
        """Immediately latch `HOLD_LAST_SAFE` at current commanded position and zero velocity."""
        self._target = self._position.copy()
        self._filtered_target = self._position.copy()
        self._velocity.fill(0.0)
        return self._position.copy()

    def reset(self, position: np.ndarray) -> None:
        arr = np.asarray(position, dtype=np.float64).reshape(-1)
        if arr.shape != self._position.shape:
            raise ValueError(f"Expected position shape {self._position.shape}, got {arr.shape}")
        self._position = arr.copy()
        self._target = arr.copy()
        self._filtered_target = arr.copy()
        self._velocity.fill(0.0)

    def step(self, dt_s: float) -> np.ndarray:
        if dt_s <= 0.0:
            return self._position.copy()
        if self._tracking_cutoff_hz is not None:
            # 3rd-order critically-damped cascade (C^2 smooth acceleration, bounded jerk) for continuous tracking
            wn = 2.0 * math.pi * self._tracking_cutoff_hz
            alpha = 1.0 - math.exp(-wn * dt_s)
            self._filtered_target = self._filtered_target + alpha * (self._target - self._filtered_target)
            a_cmd = (wn * wn) * (self._filtered_target - self._position) - 2.0 * wn * self._velocity
            a_cmd = np.clip(a_cmd, -self._max_acceleration, self._max_acceleration)
            v_next = np.clip(self._velocity + a_cmd * dt_s, -self._max_velocity, self._max_velocity)
            self._position = self._position + 0.5 * (self._velocity + v_next) * dt_s
            self._velocity = v_next
            return self._position.copy()

        err = self._target - self._position
        dist = np.abs(err)
        a = self._max_acceleration
        dv_max = a * dt_s

        # Exact discrete-time trapezoidal braking velocity so velocity reaches 0 at target without overshoot
        half_dv = 0.5 * dv_max
        v_brake = np.maximum(0.0, -half_dv + np.sqrt(half_dv * half_dv + 2.0 * a * dist))
        v_desired = np.sign(err) * np.minimum(self._max_velocity, np.minimum(v_brake, dist / dt_s))

        dv = np.clip(v_desired - self._velocity, -dv_max, dv_max)
        v_next = np.clip(self._velocity + dv, -self._max_velocity, self._max_velocity)

        step_pos = 0.5 * (self._velocity + v_next) * dt_s
        # Snap cleanly to target ONLY when we can decelerate to zero within max_acceleration * dt
        can_stop = np.abs(self._velocity) <= dv_max + 1e-12
        within_step = dist <= np.abs(step_pos) + 0.5 * dv_max * dt_s + 1e-12
        snap = can_stop & within_step & ((err * (err - step_pos)) <= 1e-12)
        pos_next = np.where(snap, self._target, self._position + step_pos)
        v_next = np.where(snap, 0.0, v_next)

        self._position = pos_next
        self._velocity = v_next
        return self._position.copy()


class JointStreamer:
    """Background fixed-rate command loop with thread-safe error propagation and clean stop."""

    def __init__(
        self,
        *,
        rate_hz: float,
        thread_name: str,
        step_fn: Callable[[float], None],
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        self.rate_hz = float(rate_hz)
        self.period_s = 1.0 / self.rate_hz
        self._thread_name = thread_name
        self._step_fn = step_fn
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._ticks = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def tick_count(self) -> int:
        return self._ticks

    def check_health(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Streamer {self._thread_name!r} faulted: {self._error}") from self._error

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run_loop, name=self._thread_name, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def _run_loop(self) -> None:
        deadline = time.monotonic()
        last_t = deadline
        while not self._stop_event.is_set():
            now = time.monotonic()
            dt = max(1e-4, min(now - last_t, self.period_s * 4.0))
            last_t = now
            try:
                self._step_fn(dt)
                self._ticks += 1
            except BaseException as exc:
                self._error = exc
                self._stop_event.set()
                return
            deadline = next_periodic_deadline(deadline, time.monotonic(), self.period_s)
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0.0:
                self._stop_event.wait(sleep_s)
