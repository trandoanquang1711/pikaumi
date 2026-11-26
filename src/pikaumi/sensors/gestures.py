"""Hands-free Pika Sense jaw double-clap recording/discard gesture detector (`[0, 0.10] m` encoder)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DoubleClapDetector:
    """Schmitt-trigger double-clap detector on Pika jaw width (`0.00 m` closed .. `0.10 m` open)."""

    close_thresh_m: float = 0.015
    open_thresh_m: float = 0.055
    min_inter_clap_s: float = 0.10
    max_inter_clap_s: float = 0.70
    cooldown_s: float = 0.80

    _armed: bool = False
    _is_closed: bool = False
    _last_clap_ts_s: float | None = None
    _cooldown_until_s: float = -1e9

    def step(self, ts_s: float, width_m: float) -> bool:
        t = float(ts_s)
        w = float(width_m)
        if not self._armed:
            if w >= self.open_thresh_m:
                self._armed = True
                self._is_closed = False
            return False

        if t < self._cooldown_until_s:
            if w >= self.open_thresh_m:
                self._is_closed = False
            return False

        if not self._is_closed and w <= self.close_thresh_m:
            self._is_closed = True
            if self._last_clap_ts_s is not None:
                dt = t - self._last_clap_ts_s
                if self.min_inter_clap_s <= dt <= self.max_inter_clap_s:
                    self._last_clap_ts_s = None
                    self._cooldown_until_s = t + self.cooldown_s
                    return True
            self._last_clap_ts_s = t
        elif self._is_closed and w >= self.open_thresh_m:
            self._is_closed = False

        if self._last_clap_ts_s is not None and (t - self._last_clap_ts_s) > self.max_inter_clap_s:
            self._last_clap_ts_s = None
        return False


class BilateralClapArbiter:
    """Arbitrates unilateral vs. bilateral Pika jaw double-claps.

    - Single-hand double-clap (after `bilateral_window_s` settle) -> `"toggle_record"`
    - Simultaneous left + right double-clap (`<= bilateral_window_s`) -> `"discard_episode"`
    """

    def __init__(self, bilateral_window_s: float = 0.28) -> None:
        self.bilateral_window_s = float(bilateral_window_s)
        self.left = DoubleClapDetector()
        self.right = DoubleClapDetector()
        self._pending_side: str | None = None
        self._pending_ts_s: float = 0.0

    def step(self, ts_s: float, left_width_m: float, right_width_m: float) -> str | None:
        t = float(ts_s)
        ev_l = self.left.step(t, left_width_m)
        ev_r = self.right.step(t, right_width_m)

        expired_event: str | None = None
        if self._pending_side is not None and (t - self._pending_ts_s) > self.bilateral_window_s:
            self._pending_side = None
            expired_event = "toggle_record"

        if ev_l and ev_r:
            self._pending_side = None
            return "discard_episode"

        for side, fired in (("left", ev_l), ("right", ev_r)):
            if fired:
                if self._pending_side is not None and self._pending_side != side:
                    if (t - self._pending_ts_s) <= self.bilateral_window_s:
                        self._pending_side = None
                        return "discard_episode"
                self._pending_side = side
                self._pending_ts_s = t

        return expired_event
