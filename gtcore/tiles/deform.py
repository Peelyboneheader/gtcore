"""Deformable tile fit WITHOUT a cavity surface (plan Step 3).

Collagen bends but barely stretches, so an implanted tile is (close to) a
developable sheet.  This module fits a bent tile to an observed seed quad
with a compact physical model and returns the pose, the bending parameters,
the residual and the bending energy -- the deformed-fit "explains this quad
with low bending energy" test that replaces chord windows with physics, and
that natively covers a crumpled (folded) tile.

Model
-----
The *seed sheet* (the surface the seed centres lie on) is a double-arc
patch: in bending axes ``(a, b)`` rotated by ``psi`` from the tile axes
``(u, v)``, arc length ``a`` along the first bending axis maps to the
circular arc of curvature ``kappa1`` and ``b`` to the arc of ``kappa2``
(heights add; exact for a cylinder, second-order exact for a sphere).
Positive curvature bends the sheet toward the model normal ``n`` -- for a
tile on a cavity wall that is toward the cavity, the side the seeds are
offset to.

The seeds sit on the manufactured 10 mm grid of the *tissue face*.  The
seed sheet is that face's parallel surface 3 mm toward the centre of
curvature (``gtcore.geometry.SEED_PLANE_OFFSET_MM``), whose metric is
scaled by ``1 / (1 + 3 kappa)`` along each bending axis -- this is what
contracts a conformed tile's chords (radius 17 mm cavity: 10 mm -> 8.2 mm)
and squeezes a folded tile's sides to ~5 mm, with the seeds' geodesic
spacing on the collagen staying exactly 10 mm (the isometry constraint).

Parameters: pose (6) + ``kappa1, kappa2, psi`` (3) = 9, against 12 seed
coordinates plus 4 seed axes (tangent to the sheet along ``u``).  A
single-crease hinge is the limit ``kappa2 = 0`` with ``kappa1`` large;
``bending_energy = kappa1^2 + kappa2^2`` (mm^-2) is what the configuration
search penalises.  Curvatures are box-constrained to
``KAPPA_RANGE`` (slightly convex to a 2 mm fold radius).

Fitting is a bounded Levenberg-Marquardt (``scipy.optimize.least_squares``)
from a few structured starts (the similarity Kabsch fit, its flipped normal,
and hinge starts about either tile axis); the correspondence comes from
:func:`gtcore.tiles.model.fit_rigid`.  Half tiles (2 seeds) cannot support
a bending fit and fall back to the rigid pose with zero curvature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .. import geometry as _geom
from .model import RigidTile, TilePose6, fit_rigid

__all__ = ["DeformParams", "DeformableFit", "fit_deformable",
           "deformed_points", "deformed_seed_points", "deformed_footprint",
           "deformed_surface_grid", "KAPPA_RANGE"]

_OFF = _geom.SEED_PLANE_OFFSET_MM                 # 3.0
KAPPA_RANGE = (-0.08, 0.5)                        # 1/mm on the seed sheet
_W_AXIS_MM_PER_RAD = 4.0        # axis misalignment weight (mm per radian)
_W_BEND_MM_MM = 1.5             # regulariser: mm of residual per (1/mm) kappa


@dataclass
class DeformParams:
    """Bending parameters of the seed sheet."""

    kappa1: float = 0.0             # 1/mm along bending axis a
    kappa2: float = 0.0             # 1/mm along bending axis b
    psi: float = 0.0                # rad, bending axes rotated from (u, v)

    @property
    def bending_energy(self) -> float:
        return float(self.kappa1 ** 2 + self.kappa2 ** 2)

    @property
    def fold_deg(self) -> float:
        """Total bend angle across the 20 mm tile in the stiffer direction."""
        k = max(abs(self.kappa1), abs(self.kappa2))
        return float(np.degrees(k * _geom.TILE_SIZE_MM))


@dataclass
class DeformableFit:
    pose: TilePose6                 # R = [t1 | t2 | n], t = tile centre
    params: DeformParams
    rms_mm: float
    residuals_mm: np.ndarray        # (k,) per observed seed
    axis_err_deg: float
    assignment: Tuple[int, ...]     # observed i <-> canonical assignment[i]
    n_evals: int = 0

    @property
    def bending_energy(self) -> float:
        return self.params.bending_energy

    def seed_points(self) -> np.ndarray:
        return deformed_seed_points(self.pose, self.params)

    def footprint(self, offset_mm: float = -_OFF) -> np.ndarray:
        return deformed_footprint(self.pose, self.params, offset_mm)


# ------------------------------------------------------------------ geometry
def _arc(kappa, s):
    """(chord along the tangent, height toward the normal) of arc length s."""
    s = np.asarray(s, dtype=float)
    if abs(kappa) < 1e-9:
        return s, 0.5 * kappa * s * s
    return np.sin(kappa * s) / kappa, (1.0 - np.cos(kappa * s)) / kappa


def _sheet_local(uv, params: DeformParams, offset_mm=0.0):
    """Map tissue-face geodesic coords ``uv`` (k, 2) to the seed sheet (or
    its parallel surface ``offset_mm`` further along the local normal) in
    the tile frame.  Returns ``(points (k, 3), normals (k, 3))``."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    c, s = np.cos(params.psi), np.sin(params.psi)
    a = c * uv[:, 0] + s * uv[:, 1]
    b = -s * uv[:, 0] + c * uv[:, 1]
    # parallel-surface metric: geodesic length on the seed sheet
    fa = 1.0 / (1.0 + _OFF * params.kappa1)
    fb = 1.0 / (1.0 + _OFF * params.kappa2)
    xa, za = _arc(params.kappa1, a * fa)
    yb, zb = _arc(params.kappa2, b * fb)
    # tangents of each arc (unit)
    ta = np.stack([np.cos(params.kappa1 * a * fa), np.zeros_like(a),
                   np.sin(params.kappa1 * a * fa)], axis=1)
    tb = np.stack([np.zeros_like(b), np.cos(params.kappa2 * b * fb),
                   np.sin(params.kappa2 * b * fb)], axis=1)
    nrm = np.cross(ta, tb)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    pts = np.stack([xa, yb, za + zb], axis=1) + offset_mm * nrm
    # rotate bending axes back to the tile frame
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pts @ Rz.T, nrm @ Rz.T


def deformed_points(pose: TilePose6, params: DeformParams, uv,
                    offset_mm: float = 0.0):
    """World points and normals of tissue-face coords ``uv`` on the deformed
    tile; ``offset_mm`` shifts along the local normal (``-3`` = tissue face,
    ``+1`` = cavity face of the 4 mm tile, ``0`` = seed sheet)."""
    loc, nrm = _sheet_local(uv, params, offset_mm)
    return loc @ pose.R.T + pose.t, nrm @ pose.R.T


def deformed_seed_points(pose, params, kind: Optional[str] = None):
    kind = kind or pose.kind
    pts, _ = deformed_points(pose, params, RigidTile(kind).seed_uv)
    return pts


def deformed_seed_axes(pose, params, kind: Optional[str] = None):
    """Unit seed axes: sheet tangent along ``u`` at each seed."""
    kind = kind or pose.kind
    _pts, tu = _sheet_and_tangent(RigidTile(kind).seed_uv, params.kappa1,
                                  params.kappa2, params.psi)
    return tu @ pose.R.T


def deformed_footprint(pose, params, offset_mm: float = -_OFF,
                       kind: Optional[str] = None):
    """The 4 collagen corners on the tissue face (default) of the bent tile."""
    kind = kind or pose.kind
    pts, _ = deformed_points(pose, params, RigidTile(kind).footprint_uv,
                             offset_mm)
    return pts


def deformed_surface_grid(pose, params, n: int = 7, offset_mm: float = -_OFF,
                          kind: Optional[str] = None):
    """``(n*n, 3)`` points sampling the bent collagen sheet (for rendering /
    overlap tests), row-major over ``v`` then ``u``."""
    kind = kind or pose.kind
    fp = RigidTile(kind).footprint_uv
    us = np.linspace(fp[:, 0].min(), fp[:, 0].max(), n)
    vs = np.linspace(fp[:, 1].min(), fp[:, 1].max(), n)
    uu, vv = np.meshgrid(us, vs)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    pts, _ = deformed_points(pose, params, uv, offset_mm)
    return pts


# ----------------------------------------------------------------------- fit
def _unit_rows(a):
    a = np.asarray(a, dtype=float).reshape(-1, 3).copy()
    n = np.linalg.norm(a, axis=1)
    n[n == 0.0] = 1.0
    return a / n[:, None]


def _rodrigues(w):
    """Rotation matrix of rotation vector ``w`` (numpy-only, fast)."""
    th = float(np.sqrt(w @ w))
    if th < 1e-12:
        K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]])
        return np.eye(3) + K
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _sheet_and_tangent(uv, k1, k2, psi):
    """Seed-sheet points and unit tangents along ``u`` in the tile frame
    (analytic; the fast path used inside the optimiser)."""
    c, s = np.cos(psi), np.sin(psi)
    a = c * uv[:, 0] + s * uv[:, 1]
    b = -s * uv[:, 0] + c * uv[:, 1]
    fa = 1.0 / (1.0 + _OFF * k1)
    fb = 1.0 / (1.0 + _OFF * k2)
    xa, za = _arc(k1, a * fa)
    yb, zb = _arc(k2, b * fb)
    pts_b = np.stack([xa, yb, za + zb], axis=1)
    # d/du = c * d/da - s * d/db  (bending frame)
    ta = fa * np.stack([np.cos(k1 * a * fa), np.zeros_like(a),
                        np.sin(k1 * a * fa)], axis=1)
    tb = fb * np.stack([np.zeros_like(b), np.cos(k2 * b * fb),
                        np.sin(k2 * b * fb)], axis=1)
    tu = c * ta - s * tb
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = pts_b @ Rz.T
    tu = tu @ Rz.T
    tu /= np.linalg.norm(tu, axis=1, keepdims=True)
    return pts, tu


def _unpack(x, R0):
    R = R0 @ _rodrigues(x[:3])
    t = x[3:6]
    params = DeformParams(kappa1=float(x[6]), kappa2=float(x[7]),
                          psi=float(x[8]))
    return R, t, params


def _residuals(x, R0, P, A, uv, w_axis, w_bend):
    R = R0 @ _rodrigues(x[:3])
    pts, tu = _sheet_and_tangent(uv, x[6], x[7], x[8])
    world = pts @ R.T + x[3:6]
    res = [(world - P).ravel()]
    if A is not None:
        ax = tu @ R.T
        # sin(angle) between undirected axes
        res.append(w_axis * np.linalg.norm(np.cross(ax, A), axis=1))
    res.append(w_bend * x[6:8])
    return np.concatenate(res)


def _bowl_sign(P, A, pose, uv):
    """+1 if the observed axis tilts say the sheet bows toward ``pose.normal``
    (seeds at +u tilt toward +n), -1 if away, 0 if there is no axis evidence.

    Four seeds on a square lie on a circle, hence on a plane, whatever the
    curvature: the positions alone cannot tell a bowl from a dome, only
    the tangent tilt of the seed axes can."""
    if A is None:
        return 0.0
    t1, n = pose.t1, pose.normal
    aligned = A * np.where(A @ t1 < 0.0, -1.0, 1.0)[:, None]
    return float(np.sum(np.sign(uv[:, 0]) * (aligned @ n)))


def fit_deformable(seed_pts, seed_axes=None, kind: Optional[str] = None,
                   kappa_range=KAPPA_RANGE, w_axis=_W_AXIS_MM_PER_RAD,
                   w_bend=_W_BEND_MM_MM, hinge_starts=True) -> DeformableFit:
    """Fit the bent-tile model to 4 observed seeds (2 -> rigid fallback).

    Returns the best of several starts by total cost; ``rms_mm`` is the
    seed-position residual alone (the axis and bending terms are not in it),
    so it is directly comparable to :func:`fit_rigid`'s.
    """
    P = np.asarray(seed_pts, dtype=float).reshape(-1, 3)
    k = P.shape[0]
    A = None if seed_axes is None else _unit_rows(seed_axes)
    rigid = fit_rigid(P, A, kind=kind, allow_scale=True,
                      scale_range=(0.6, 1.1))
    kind = rigid.pose.kind
    if k == 2:
        pose = TilePose6(rigid.pose.R, rigid.pose.t, kind, 1.0)
        pts = deformed_seed_points(pose, DeformParams())
        res = np.linalg.norm(P - pts[list(rigid.assignment)], axis=1)
        return DeformableFit(pose=pose, params=DeformParams(),
                             rms_mm=float(np.sqrt(np.mean(res ** 2))),
                             residuals_mm=res, axis_err_deg=rigid.axis_err_deg,
                             assignment=rigid.assignment)

    uv = RigidTile(kind).seed_uv[list(rigid.assignment)]
    # curvature implied by the similarity scale: s = 1 / (1 + 3 kappa)
    k_iso = float(np.clip((1.0 / rigid.pose.scale - 1.0) / _OFF,
                          kappa_range[0], kappa_range[1]))
    lo = np.array([-np.inf] * 6 + [kappa_range[0]] * 2 + [-np.pi])
    hi = np.array([np.inf] * 6 + [kappa_range[1]] * 2 + [np.pi])
    k_hinge = min(0.35, kappa_range[1])

    # Both normal signs are tried: the positions cannot tell a bowl from a
    # dome (see _bowl_sign), only the axis tilts can.  Turning the tile over
    # (n -> -n, t2 -> -t2) mirrors the labelling, so the flipped start uses
    # v -> -v on the correspondence.  Hinge starts go with the sign the axis
    # tilts favour (both when the evidence is weak).
    R_up = rigid.pose.R
    uv_up = uv
    R_dn = rigid.pose.flipped_normal().R
    uv_dn = uv * np.array([1.0, -1.0])
    ev = _bowl_sign(P, A, rigid.pose, uv)
    if ev < 0.0:
        R_up, R_dn = R_dn, R_up
        uv_up, uv_dn = uv_dn, uv_up
    starts = [(R_up, uv_up, [k_iso, k_iso, 0.0]),
              (R_dn, uv_dn, [k_iso, k_iso, 0.0])]
    hinges = []
    if hinge_starts:
        sides = [(R_up, uv_up)] if abs(ev) >= 0.3 else \
            [(R_up, uv_up), (R_dn, uv_dn)]
        for R0, uv0 in sides:
            hinges.append((R0, uv0, [k_hinge, 0.0, 0.0]))
            hinges.append((R0, uv0, [k_hinge, 0.0, 0.5 * np.pi]))

    best = None
    n_evals = 0

    def _run(R0, uv0, kap):
        nonlocal best, n_evals
        x0 = np.concatenate([np.zeros(3), rigid.pose.t, kap])
        try:
            sol = least_squares(_residuals, x0, bounds=(lo, hi),
                                args=(R0, P, A, uv0, w_axis, w_bend),
                                method="trf", xtol=1e-7, ftol=1e-7,
                                x_scale=np.array([0.1] * 3 + [1.0] * 3
                                                 + [0.05, 0.05, 0.3]),
                                max_nfev=200)
        except Exception:
            return
        n_evals += int(sol.nfev)
        if best is None or sol.cost < best[0]:
            best = (sol.cost, sol.x, R0, uv0)

    for R0, uv0, kap in starts:
        _run(R0, uv0, kap)
    # the hinge (fold) starts only matter when the smooth fit is poor: a
    # well-explained quad (cost ~ 4 x 0.4^2 / 2) never needs them
    if best is None or best[0] > 0.5 * 4 * 0.45 ** 2:
        for R0, uv0, kap in hinges:
            _run(R0, uv0, kap)

    _cost, x, R0, uv0 = best
    R, t, params = _unpack(x, R0)
    pose = TilePose6(R, t, kind, 1.0)
    pts, tu = _sheet_and_tangent(uv0, params.kappa1, params.kappa2, params.psi)
    res = np.linalg.norm(P - (pts @ R.T + t), axis=1)
    aerr = 0.0
    if A is not None:
        ax = tu @ R.T
        c = np.clip(np.abs((ax * A).sum(axis=1)), 0.0, 1.0)
        aerr = float(np.degrees(np.arccos(c)).mean())
    # correspondence actually used (may be the mirrored labelling)
    canon = RigidTile(kind).seed_uv
    assignment = tuple(int(np.argmin(np.linalg.norm(canon - row, axis=1)))
                       for row in uv0)
    return DeformableFit(pose=pose, params=params,
                         rms_mm=float(np.sqrt(np.mean(res ** 2))),
                         residuals_mm=res, axis_err_deg=aerr,
                         assignment=assignment, n_evals=n_evals)
