"""Tracking quality as a function of distance to the base stations.

Lighthouse bases measure angles, so position noise grows with distance, and the
laser gets dimmer, so tracking is lost sooner far away. OpenVR exposes no signal
strength, but we know where every base and device is, so we measure the effect:
jitter and tracking loss binned by device-to-base distance.

Two kinds of series:
  * per base  - jitter while that base is one the device is using
  * "nearest" - jitter and tracking loss against distance to the closest base
                (a dropout cannot be pinned on one base, so loss lives only here)
"""
from __future__ import annotations

import math

from .model import Cause, DeviceMetrics, Finding, Sample, Vec3

BIN_M = 0.5
MAX_M = 12.0
NEAREST = "nearest"
MIN_BIN_SAMPLES = 50          # ~5 s of one device at the 10 Hz feed rate

# finding thresholds
NEAR_M, FAR_M = 3.5, 4.5
MIN_SIDE_SAMPLES = 300


class _Bin:
    __slots__ = ("n", "ok", "jit_sum", "jit_n")

    def __init__(self) -> None:
        self.n = 0
        self.ok = 0
        self.jit_sum = 0.0
        self.jit_n = 0


class RangeProfile:
    def __init__(self) -> None:
        self.series: dict[str, dict[int, _Bin]] = {}
        self._last_ok_pos: dict[str, Vec3] = {}

    def _bin(self, key: str, dist: float) -> _Bin | None:
        if not 0.0 <= dist < MAX_M:
            return None
        bins = self.series.setdefault(key, {})
        i = int(dist / BIN_M)
        b = bins.get(i)
        if b is None:
            b = bins[i] = _Bin()
        return b

    def add(self, sample: Sample, metrics: DeviceMetrics, base_pos: dict[str, Vec3],
            seen: set[str] | None) -> None:
        """Feed one device sample (call at ~10 Hz per device, not every frame).

        base_pos: live base positions by serial. seen: serials of the bases the driver
        says this device is using, or None when that is unknown (then all count).
        """
        if not base_pos:
            return
        if sample.ok:
            pos = self._last_ok_pos[sample.serial] = sample.pos
        else:
            pos = self._last_ok_pos.get(sample.serial)   # where it was when it was last tracked
            if pos is None:
                return
        nearest = math.inf
        for serial, bp in base_pos.items():
            d = math.dist(pos, bp)
            nearest = min(nearest, d)
            if sample.ok and (seen is None or serial in seen):
                b = self._bin(serial, d)
                if b is not None:
                    b.n += 1
                    b.ok += 1
                    b.jit_sum += metrics.jitter_mm
                    b.jit_n += 1
        b = self._bin(NEAREST, nearest)
        if b is not None:
            b.n += 1
            if sample.ok:
                b.ok += 1
                b.jit_sum += metrics.jitter_mm
                b.jit_n += 1

    def snapshot(self) -> dict:
        out = {}
        for key, bins in self.series.items():
            rows = []
            for i in sorted(bins):
                b = bins[i]
                if b.n < MIN_BIN_SAMPLES:
                    continue
                rows.append({"d": round((i + 0.5) * BIN_M, 2), "n": b.n,
                             "jitter_mm": round(b.jit_sum / b.jit_n, 3) if b.jit_n else None,
                             "ok_frac": round(b.ok / b.n, 4)})
            if rows:
                out[key] = rows
        return {"bin": BIN_M, "nearest_key": NEAREST, "warn_m": 5.0, "series": out}

    def _side(self, lo: float, hi: float) -> tuple[int, float, float | None, float]:
        """(samples, ok fraction, mean jitter, mean distance) of 'nearest' bins in [lo, hi)."""
        n = ok = jit_n = 0
        jit = dsum = 0.0
        for i, b in self.series.get(NEAREST, {}).items():
            d = (i + 0.5) * BIN_M
            if lo <= d < hi:
                n += b.n
                ok += b.ok
                jit += b.jit_sum
                jit_n += b.jit_n
                dsum += d * b.n
        return n, (ok / n if n else 1.0), (jit / jit_n if jit_n else None), (dsum / n if n else 0.0)

    def finding(self) -> Finding | None:
        """A session-level conclusion when tracking is clearly worse far from every base."""
        n_near, ok_near, jit_near, d_near = self._side(0.0, NEAR_M)
        n_far, ok_far, jit_far, d_far = self._side(FAR_M, MAX_M)
        if n_near < MIN_SIDE_SAMPLES or n_far < MIN_SIDE_SAMPLES:
            return None
        loss_near, loss_far = 1.0 - ok_near, 1.0 - ok_far
        parts, score = [], 0.0
        if loss_far > 0.05 and loss_far > 3.0 * loss_near + 0.02:
            score = max(score, min(1.0, 0.4 + 2.0 * (loss_far - loss_near)))
            parts.append(f"tracking is lost {loss_far:.0%} of the time beyond {FAR_M:g} m from the "
                         f"nearest base, against {loss_near:.0%} within {NEAR_M:g} m")
        if jit_near and jit_far and d_near > 0:
            expected = jit_near * d_far / d_near       # noise scales about linearly with distance
            if jit_far > 2.0 * expected and jit_far > 1.0:
                score = max(score, min(1.0, 0.3 + 0.15 * jit_far / expected))
                parts.append(f"jitter is {jit_far:.1f} mm at ~{d_far:.1f} m against {jit_near:.1f} mm "
                             f"at ~{d_near:.1f} m, more than distance alone explains")
        if not parts:
            return None
        return Finding(
            cause=Cause.RANGE, score=round(score, 2),
            title="Tracking gets worse far from the base stations",
            detail="Measured by distance to the nearest base station: " + "; ".join(parts) + ".",
            fix=("Add a base station near that part of the room or move one closer, so every spot "
                 "you use is within about 5 m of at least two bases. Also check the far bases are "
                 "aimed at the play space and their front glass is clean."))
