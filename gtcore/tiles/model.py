"""Rigid nominal GammaTile model and its Kabsch pose fit.

A manufactured tile (see :mod:`gtcore.geometry`) carries its seeds on an
exact 10 mm grid.  This module expresses that grid in a canonical tile frame
and fits an observed seed group to it with a closed-form Kabsch alignment,
searching the correspondence over the symmetries of the square.  The
residual it returns (``rms_mm``) is the "how far from a manufactured tile"
metric that the count-free configuration search (:mod:`gtcore.tiles.auto`)
and the deformable fits (:mod:`gtcore.tiles.deform`) build on.

Canonical tile frame
--------------------
Origin at the tile centre, ``t1 = +x`` along the seed long axes, ``t2 = +y``
across them, ``n = +z`` the tile normal.  Seeds sit in the ``z = 0`` plane:

* full tile: ``(+-5, +-5)``; footprint corners ``(+-10, +-10)``;
* half tile: one column, ``(0, +-5)``; footprint ``(+-5, +-10)``.

A pose is ``x_ras = R @ x_tile + t`` with ``R = [t1 | t2 | n]``.

Symmetries
----------
The 4 seed points of a full tile are invariant under the 8 symmetries of the
square (4 rotations x reflection).  Reflections correspond to flipping the
tile over, so all 8 are physically admissible; the observed seed *axes* then
resolve the 90-degree ambiguity (axes run along ``t1``, not ``t2``).  What
remains -- a 180-degree turn about ``n`` and a flip about ``t1`` -- leaves
the seed set, the undirected axes and the footprint unchanged, so it is not
an ambiguity for anything downstream.  Orientation of ``n`` (into the wall
vs into the cavity) is fixed later from the cavity centre, exactly as
:func:`gtcore.tiles.fit.fit_tiles` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Optional, Tuple

import numpy as np

from .. import geometry as _geom

__all__ = ["RigidTile", "TilePose6", "RigidFit", "fit_rigid", "kabsch"]

_HALF_PITCH = _geom.SEED_PITCH_MM / 2.0          # 5.0
_HALF_SIZE = _geom.TILE_HALF_SIZE_MM             # 10.0

# Axis-alignment weight inside the Kabsch covariance (mm^2 per axis): keeps
# the point term (~4 x 50 mm^2) dominant on a full tile while still deciding
# the 90-degree symmetry, and fully determines the roll of a 2-seed pair.
_W_AXIS_COV = 10.0
# Objective weight for choosing among correspondences: mm of point residual
# equivalent to one radian of axis misalignment (20 deg ~ 1.7 mm).
_W_AXIS_OBJ_MM_PER_RAD = 5.0


# ------------------------------------------------------------------- model
class RigidTile:
    """Canonical seed grid and footprint of a manufactured tile."""

    def __init__(self, kind: str = "full"):
        if kind not in ("full", "half"):
            raise ValueError("kind must be 'full' or 'half', got %r" % (kind,))
        self.kind = kind

    @property
    def n_seeds(self) -> int:
        return 4 if self.kind == "full" else 2

    @property
    def seed_uv(self) -> np.ndarray:
        """(k, 2) seed positions in the tile plane (u along t1, v along t2)."""
        h = _HALF_PITCH
        if self.kind == "full":
            return np.array([[-h, -h], [-h, h], [h, -h], [h, h]])
        return np.array([[0.0, -h], [0.0, h]])

    @property
    def seed_xyz(self) -> np.ndarray:
        """(k, 3) canonical seed positions."""
        uv = self.seed_uv
        return np.hstack([uv, np.zeros((uv.shape[0], 1))])

    @property
    def footprint_uv(self) -> np.ndarray:
        """(4, 2) footprint corners in loop order."""
        cu = _HALF_SIZE if self.kind == "full" else _HALF_PITCH
        cv = _HALF_SIZE
        return np.array([[-cu, -cv], [cu, -cv], [cu, cv], [-cu, cv]])

    @property
    def footprint_xyz(self) -> np.ndarray:
        uv = self.footprint_uv
        return np.hstack([uv, np.zeros((4, 1))])


@dataclass
class TilePose6:
    """SE(3) pose of a tile: ``x_ras = R @ x_tile + t``."""

    R: np.ndarray                   # (3, 3), columns t1 | t2 | n
    t: np.ndarray                   # (3,)
    kind: str = "full"
    scale: float = 1.0              # in-plane similarity scale (1 = rigid)

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)

    @property
    def t1(self) -> np.ndarray:
        return self.R[:, 0]

    @property
    def t2(self) -> np.ndarray:
        return self.R[:, 1]

    @property
    def normal(self) -> np.ndarray:
        return self.R[:, 2]

    @property
    def center(self) -> np.ndarray:
        return self.t

    def apply(self, xyz_tile) -> np.ndarray:
        xyz = np.asarray(xyz_tile, dtype=float).reshape(-1, 3)
        return (self.scale * xyz) @ self.R.T + self.t

    def seed_points(self) -> np.ndarray:
        return self.apply(RigidTile(self.kind).seed_xyz)

    def footprint(self) -> np.ndarray:
        return self.apply(RigidTile(self.kind).footprint_xyz)

    def flipped_normal(self) -> "TilePose6":
        """Same tile turned over: n -> -n, t2 -> -t2 (a 180 turn about t1).
        Seed set, footprint and undirected axes are unchanged."""
        R = self.R.copy()
        R[:, 1] *= -1.0
        R[:, 2] *= -1.0
        return TilePose6(R, self.t.copy(), self.kind, self.scale)


@dataclass
class RigidFit:
    """Result of :func:`fit_rigid`."""

    pose: TilePose6
    rms_mm: float                   # RMS seed-position residual
    residuals_mm: np.ndarray        # (k,) per observed seed
    axis_err_deg: float             # mean undirected angle(observed axis, t1)
    assignment: Tuple[int, ...]     # observed seed i <-> canonical seed assignment[i]


# ------------------------------------------------------------------ kabsch
def kabsch(P, Q, axes=None, axis_dir=(1.0, 0.0, 0.0), w_axis=_W_AXIS_COV,
           allow_scale=False, scale_range=(0.7, 1.1)):
    """Least-squares ``R, t, s`` with ``P ~= s * R @ Q + t`` (rows are points).

    ``axes`` (optional, same row count as ``P``) are unit observed directions
    that should align with ``R @ axis_dir``; their signs must already be
    consistent with that direction.  Returns ``(R, t, s)`` with ``R`` a
    proper rotation.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    Pc = P - pc
    Qc = Q - qc
    H = Qc.T @ Pc
    if axes is not None:
        A = np.asarray(axes, dtype=float).reshape(-1, 3)
        d = np.asarray(axis_dir, dtype=float).reshape(3)
        H = H + w_axis * np.outer(d, A.sum(axis=0))
    U, _S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ D @ U.T
    s = 1.0
    if allow_scale:
        denom = float((Qc * Qc).sum())
        if denom > 0.0:
            s = float((Pc * (Qc @ R.T)).sum()) / denom
        s = float(np.clip(s, scale_range[0], scale_range[1]))
    t = pc - s * (R @ qc)
    return R, t, s


def _unit_rows(a):
    a = np.asarray(a, dtype=float).reshape(-1, 3).copy()
    n = np.linalg.norm(a, axis=1)
    n[n == 0.0] = 1.0
    return a / n[:, None]


def _axis_err_deg(axes, t1):
    if axes is None or axes.shape[0] == 0:
        return 0.0
    c = np.clip(np.abs(axes @ t1), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)).mean())


# --------------------------------------------------------------------- fit
def fit_rigid(seed_pts, seed_axes=None, kind: Optional[str] = None,
              allow_scale: bool = False,
              scale_range=(0.7, 1.1)) -> RigidFit:
    """Fit the rigid nominal tile to 4 (full) or 2 (half) observed seeds.

    Parameters
    ----------
    seed_pts : (k, 3) array, k in {2, 4}
        Observed seed centres, RAS mm, in any order.
    seed_axes : (k, 3) array, optional
        Observed seed long axes (sign arbitrary).  Strongly recommended: they
        resolve the 90-degree symmetry of a full tile and are *required* to
        determine the roll of a 2-seed half tile (without them the half-tile
        normal is an arbitrary direction perpendicular to the pair).
    kind : "full" | "half", optional
        Defaults from ``k``.
    allow_scale : bool
        Fit an in-plane similarity scale (clipped to ``scale_range``).  A
        wall-conformed tile's chords contract because the seed plane rides
        3 mm inside the concave wall, so the scale is a cheap proxy for that
        deformation when the true developable fit is not wanted.

    Returns
    -------
    RigidFit
        Best correspondence by ``rms^2 + (5 mm/rad * axis_rms)^2``.
        The pose normal has arbitrary sign; orient it downstream.
    """
    P = np.asarray(seed_pts, dtype=float).reshape(-1, 3)
    k = P.shape[0]
    if kind is None:
        kind = {4: "full", 2: "half"}.get(k)
        if kind is None:
            raise ValueError("fit_rigid needs 2 or 4 seeds, got %d" % k)
    model = RigidTile(kind)
    if k != model.n_seeds:
        raise ValueError("%s tile needs %d seeds, got %d"
                         % (kind, model.n_seeds, k))
    A = None if seed_axes is None else _unit_rows(seed_axes)
    if A is not None and A.shape[0] != k:
        raise ValueError("seed_axes must match seed_pts")
    Q = model.seed_xyz
    xhat = np.array([1.0, 0.0, 0.0])

    best = None
    for perm in permutations(range(k)):
        # observed i <-> canonical perm[i]
        Qp = Q[list(perm)]
        if A is None:
            R, t, s = kabsch(P, Qp, allow_scale=allow_scale,
                             scale_range=scale_range)
            cands = [(R, t, s)]
        elif k == 4:
            # points fix the frame up to the square symmetries; align axis
            # signs to that frame, then refine jointly
            R0, _t0, _s0 = kabsch(P, Qp, allow_scale=allow_scale,
                                  scale_range=scale_range)
            sign = np.where(A @ R0[:, 0] < 0.0, -1.0, 1.0)
            R, t, s = kabsch(P, Qp, axes=A * sign[:, None], axis_dir=xhat,
                             allow_scale=allow_scale, scale_range=scale_range)
            cands = [(R, t, s)]
        else:
            # 2 points leave the roll free: the axes decide it, and their
            # global sign is a genuine 2-way choice (both are tried)
            Ac = A.copy()
            if float(Ac[1] @ Ac[0]) < 0.0:
                Ac[1] *= -1.0
            cands = []
            for g in (1.0, -1.0):
                cands.append(kabsch(P, Qp, axes=g * Ac, axis_dir=xhat,
                                    allow_scale=allow_scale,
                                    scale_range=scale_range))
        for R, t, s in cands:
            fitted = (s * Qp) @ R.T + t
            res = np.linalg.norm(P - fitted, axis=1)
            rms = float(np.sqrt(np.mean(res ** 2)))
            aerr = _axis_err_deg(A, R[:, 0])
            J = rms ** 2 + (_W_AXIS_OBJ_MM_PER_RAD * np.deg2rad(aerr)) ** 2
            if best is None or J < best[0] - 1e-12:
                best = (J, R, t, s, res, rms, aerr, perm)

    _J, R, t, s, res, rms, aerr, perm = best
    pose = TilePose6(R=R, t=t, kind=kind, scale=s)
    return RigidFit(pose=pose, rms_mm=rms, residuals_mm=res,
                    axis_err_deg=aerr, assignment=tuple(perm))
