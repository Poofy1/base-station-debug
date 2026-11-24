"""Voxel heatmaps of tracking health, plus predicted base-station coverage.

Also hosts the small scalar geometry helpers (quaternion rotation, base field
of view, segment/capsule tests) shared with the simulator and attribution.
"""
from __future__ import annotations

import math

import numpy as np

from .model import DeviceMetrics, Incident, IncidentKind, Quat, Sample, Vec3

# Lighthouse 2.0 sweep envelope. A base looks along its local -Z.
FOV_H_DEG = 150.0
FOV_V_DEG = 110.0
RANGE_M = 6.0

_TAN_H = math.tan(math.radians(FOV_H_DEG / 2))
_TAN_V = math.tan(math.radians(FOV_V_DEG / 2))

METRICS = ("samples", "dropout", "jitter", "jumps", "update_rate", "incidents")


# ---- geometry helpers -------------------------------------------------------
def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate v by unit quaternion q (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def quat_inv_rotate(q: Quat, v: Vec3) -> Vec3:
    w, x, y, z = q
    return quat_rotate((w, -x, -y, -z), v)


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def base_view(base_pos: Vec3, base_quat: Quat, p: Vec3) -> tuple[float, float, float]:
    """(distance m, azimuth deg, elevation deg) of point p in the base's frame.

    Azimuth/elevation are 0 straight ahead (local -Z); azimuth is 180 behind.
    """
    lx, ly, lz = quat_inv_rotate(base_quat, (p[0] - base_pos[0], p[1] - base_pos[1], p[2] - base_pos[2]))
    depth = -lz
    dist = math.sqrt(lx * lx + ly * ly + lz * lz)
    az = math.degrees(math.atan2(lx, depth))
    el = math.degrees(math.atan2(ly, math.hypot(lx, depth)))
    return dist, az, el


def base_sees(base_pos: Vec3, base_quat: Quat, p: Vec3) -> bool:
    """True if p is inside the base's sweep envelope and useful range."""
    dist, az, el = base_view(base_pos, base_quat, p)
    return dist <= RANGE_M and abs(az) <= FOV_H_DEG / 2 and abs(el) <= FOV_V_DEG / 2


def segment_distance(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> float:
    """Minimum distance between segments p1-q1 and p2-q2."""
    d1 = (q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2])
    d2 = (q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2])
    r = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
    e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
    f = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2]
    eps = 1e-12
    if a <= eps and e <= eps:
        s = t = 0.0
    elif a <= eps:
        s = 0.0
        t = min(1.0, max(0.0, f / e))
    else:
        c = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2]
        if e <= eps:
            t = 0.0
            s = min(1.0, max(0.0, -c / a))
        else:
            b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - c) / a))
    dx = p1[0] + d1[0] * s - (p2[0] + d2[0] * t)
    dy = p1[1] + d1[1] * s - (p2[1] + d2[1] * t)
    dz = p1[2] + d1[2] * s - (p2[2] + d2[2] * t)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def sight_blocked(device: Vec3, base: Vec3, capsules: list[tuple[Vec3, Vec3, float]],
                  clearance: float = 0.25) -> bool:
    """Does any body capsule (a, b, radius) block the line from device to base?

    The first `clearance` metres of the ray are ignored so that the body part
    the device is strapped to does not count as blocking it.
    """
    dx, dy, dz = base[0] - device[0], base[1] - device[1], base[2] - device[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n <= clearance:
        return False
    k = clearance / n
    start = (device[0] + dx * k, device[1] + dy * k, device[2] + dz * k)
    for a, b, radius in capsules:
        if segment_distance(start, base, a, b) < radius:
            return True
    return False


# ---- heatmap ----------------------------------------------------------------
class Heatmap:
    """Per-device voxel accumulators; the combined view is merged on snapshot.

    Cell record: [n, n_bad, jitter_sum_mm, update_hz_sum, jumps, incidents].
    Samples that are not ok are booked at the device's last good position,
    since the reported pose is meaningless while tracking is lost.
    """

    def __init__(self, voxel: float = 0.2) -> None:
        self.voxel = float(voxel)
        self._inv = 1.0 / self.voxel
        self._cells: dict[str, dict[tuple[int, int, int], list]] = {}
        self._last_pos: dict[str, Vec3] = {}

    def _key(self, pos: Vec3) -> tuple[int, int, int]:
        inv = self._inv
        return (math.floor(pos[0] * inv), math.floor(pos[1] * inv), math.floor(pos[2] * inv))

    def add(self, sample: Sample, metrics: DeviceMetrics) -> None:
        serial = sample.serial
        ok = sample.connected and sample.valid and sample.result == 200
        if ok:
            pos = sample.pos
            self._last_pos[serial] = pos
        else:
            pos = self._last_pos.get(serial)
            if pos is None:
                return
        inv = self._inv
        key = (math.floor(pos[0] * inv), math.floor(pos[1] * inv), math.floor(pos[2] * inv))
        cells = self._cells.get(serial)
        if cells is None:
            cells = self._cells[serial] = {}
        c = cells.get(key)
        if c is None:
            c = cells[key] = [0, 0, 0.0, 0.0, 0, 0]
        c[0] += 1
        if ok:
            c[2] += metrics.jitter_mm
            c[3] += metrics.update_hz
        else:
            c[1] += 1

    def add_incident(self, inc: Incident) -> None:
        if inc.kind in (IncidentKind.BASE_SHIFT, IncidentKind.UNIVERSE_SHIFT):
            return
        cells = self._cells.setdefault(inc.serial, {})
        key = self._key(inc.pos)
        c = cells.get(key)
        if c is None:
            c = cells[key] = [0, 0, 0.0, 0.0, 0, 0]
        c[5] += 1
        if inc.kind is IncidentKind.JUMP:
            c[4] += max(1, int(inc.metrics.get("count", 1)))

    def _merged(self, serial: str | None) -> dict[tuple[int, int, int], list]:
        if serial is not None:
            return self._cells.get(serial, {})
        out: dict[tuple[int, int, int], list] = {}
        for cells in self._cells.values():
            for key, c in cells.items():
                o = out.get(key)
                if o is None:
                    out[key] = list(c)
                else:
                    for i in range(6):
                        o[i] += c[i]
        return out

    def snapshot(self, metric: str, serial: str | None = None) -> dict:
        if metric not in METRICS:
            raise ValueError(f"unknown heatmap metric {metric!r}")
        out = []
        for (ix, iy, iz), c in self._merged(serial).items():
            n, bad = c[0], c[1]
            if metric == "samples":
                if n == 0:
                    continue
                v = float(n)
            elif metric == "dropout":
                if n == 0:
                    continue
                v = bad / n
            elif metric == "jitter":
                if n - bad <= 0:
                    continue
                v = c[2] / (n - bad)
            elif metric == "update_rate":
                if n - bad <= 0:
                    continue
                v = c[3] / (n - bad)
            elif metric == "jumps":
                if c[4] == 0:
                    continue
                v = float(c[4])
            else:
                if c[5] == 0:
                    continue
                v = float(c[5])
            out.append([ix, iy, iz, round(v, 4), n])
        return _result(self.voxel, metric, out)

    def coverage(self, bases: list[tuple[Vec3, Quat]], bounds=None) -> dict:
        """Predicted number of bases that can see each voxel centre.

        bounds = ((xmin, ymin, zmin), (xmax, ymax, zmax)) in metres. Default:
        the volume the devices have visited (padded), or failing that the box
        spanned by the bases down to the floor. `n` in each cell is the number
        of bases evaluated.
        """
        if not bases:
            return _result(self.voxel, "coverage", [])
        v = self.voxel
        if bounds is None:
            keys = [k for cells in self._cells.values() for k in cells]
            if keys:
                a = np.array(keys)
                lo = (a.min(axis=0) - 2) * v
                hi = (a.max(axis=0) + 3) * v
                lo[1] = min(lo[1], 0.0)
            else:
                bp = np.array([b[0] for b in bases], dtype=float)
                lo = bp.min(axis=0) - 0.2
                hi = bp.max(axis=0) + 0.2
                lo[1] = 0.0
            bounds = (tuple(lo), tuple(hi))
        (x0, y0, z0), (x1, y1, z1) = bounds
        e = 1e-9
        ix = np.arange(math.floor(x0 / v + e), max(math.floor(x0 / v + e) + 1, math.ceil(x1 / v - e)))
        iy = np.arange(math.floor(y0 / v + e), max(math.floor(y0 / v + e) + 1, math.ceil(y1 / v - e)))
        iz = np.arange(math.floor(z0 / v + e), max(math.floor(z0 / v + e) + 1, math.ceil(z1 / v - e)))
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        pts = (idx + 0.5) * v
        count = np.zeros(len(pts), dtype=np.int32)
        for pos, quat in bases:
            # base axes in world space; local coordinate = axis . (p - pos)
            ax = np.asarray(quat_rotate(quat, (1.0, 0.0, 0.0)))
            ay = np.asarray(quat_rotate(quat, (0.0, 1.0, 0.0)))
            az = np.asarray(quat_rotate(quat, (0.0, 0.0, 1.0)))
            rel = pts - np.asarray(pos, dtype=float)
            lx = rel @ ax
            ly = rel @ ay
            depth = -(rel @ az)
            dist = np.sqrt((rel * rel).sum(axis=1))
            vis = ((depth > 0) & (dist <= RANGE_M)
                   & (np.abs(lx) <= depth * _TAN_H)
                   & (np.abs(ly) <= np.hypot(lx, depth) * _TAN_V))
            count += vis
        nb = len(bases)
        cells = [[a, b, c, float(k), nb] for (a, b, c), k in zip(idx.tolist(), count.tolist())]
        return _result(self.voxel, "coverage", cells)


def _result(voxel: float, metric: str, cells: list) -> dict:
    vals = [c[3] for c in cells]
    return {"voxel": voxel, "metric": metric,
            "min": float(min(vals)) if vals else 0.0,
            "max": float(max(vals)) if vals else 0.0,
            "cells": cells}
