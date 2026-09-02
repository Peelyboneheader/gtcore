"""Stick-to-surface tile fit: deformation tier 2 (plan Step 4).

When a cavity mesh exists, an implanted tile's tissue face lies ON the wall
and its seed plane 3 mm off it, into the cavity.  The tile's shape is then
no longer free: it is the collagen sheet draped onto the known surface,
exactly what the planner's :func:`gtcore.interact.conform_tile` computes
for a placed tile.  Fitting a seed quad therefore reduces to a 3-dof
problem -- where on the wall the tile is anchored (2 tangent offsets) and
how it is turned about the local normal (1 angle) -- solved here by
Nelder-Mead over the conformer, with the seed correspondence fixed by
assignment at the start.

Cross-feed with the surface-free fit (:mod:`gtcore.tiles.deform`):

* ``agreement_mm`` -- rms between the free fit's seed positions and the
  surface-constrained ones.  Small = both models tell the same story, a
  strong confidence signal.
* ``detachment_mm`` -- how far the OBSERVED seeds sit from the 3 mm
  offset surface.  Large = either the cavity segmentation is wrong here
  or the tile has lifted off the wall (a resorbing / floating tile);
  both are clinically interesting, so the flag is surfaced, not hidden.

The result also localizes the tile's footprint on the wall
(``placed.corners_ras``) for the planner and the coverage metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment, minimize

from .. import geometry as _geom
from ..interact import (
    PlacedTile,
    _rodrigues,
    _tangent_frame,
    conform_tile,
    snap_to_wall,
)
from .deform import DeformableFit
from .model import TilePose6, fit_rigid

__all__ = ["SurfaceFit", "fit_on_surface", "DETACHED_MM", "DISAGREE_MM"]

_OFF = _geom.SEED_PLANE_OFFSET_MM
DETACHED_MM = 1.5           # mean |wall distance - 3 mm| beyond -> detached
DISAGREE_MM = 1.5           # free-vs-surface seed rms beyond -> inconsistent
_W_AXIS_MM_PER_RAD = 3.0


@dataclass
class SurfaceFit:
    placed: PlacedTile
    rms_mm: float                       # observed vs conformed seed positions
    residuals_mm: np.ndarray
    axis_err_deg: float
    assignment: Tuple[int, ...]         # observed i <-> conformed seed assignment[i]
    anchor_ras: np.ndarray
    angle_rad: float
    wall_distance_mm: np.ndarray        # observed seeds' distance from the mesh
    detachment_mm: float
    free_rms_mm: Optional[float] = None
    agreement_mm: Optional[float] = None
    n_evals: int = 0

    @property
    def attached(self) -> bool:
        return self.detachment_mm <= DETACHED_MM

    @property
    def consistent(self) -> Optional[bool]:
        if self.agreement_mm is None:
            return None
        return self.agreement_mm <= DISAGREE_MM

    def verdict(self) -> str:
        if not self.attached:
            return ("detached: seeds sit %.1f mm off the 3 mm wall offset "
                    "(segmentation error or tile lift-off)" % self.detachment_mm)
        if self.consistent is False:
            return ("inconsistent: free and surface fits disagree by %.1f mm"
                    % self.agreement_mm)
        return "attached, consistent (rms %.2f mm)" % self.rms_mm


def _unit_rows(a):
    a = np.asarray(a, dtype=float).reshape(-1, 3).copy()
    n = np.linalg.norm(a, axis=1)
    n[n == 0.0] = 1.0
    return a / n[:, None]


def _match(model_pts, P):
    D = np.linalg.norm(model_pts[:, None, :] - P[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(D)
    assign = np.empty(P.shape[0], dtype=int)
    assign[cols] = rows            # observed col -> model row
    return tuple(int(a) for a in assign)


def fit_on_surface(mesh, seed_pts, seed_axes=None, kind: Optional[str] = None,
                   init=None, max_iter: int = 120,
                   w_axis: float = _W_AXIS_MM_PER_RAD) -> SurfaceFit:
    """Fit a tile conformed to ``mesh`` to observed seeds.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Cavity wall (either winding; normals are re-oriented internally).
    seed_pts, seed_axes : (k, 3)
        Observed seed centres / long axes (k = 4 full, 2 half).
    init : TilePose6 | DeformableFit | None
        Starting pose (centre and t1).  A :class:`DeformableFit` also gives
        the cross-check fields ``free_rms_mm`` / ``agreement_mm``.
    """
    P = np.asarray(seed_pts, dtype=float).reshape(-1, 3)
    k = P.shape[0]
    A = None if seed_axes is None else _unit_rows(seed_axes)
    free = None
    if isinstance(init, DeformableFit):
        free = init
        pose0 = init.pose
    elif isinstance(init, TilePose6):
        pose0 = init
    else:
        pose0 = fit_rigid(P, A, kind=kind, allow_scale=True,
                          scale_range=(0.6, 1.1)).pose
    kind = kind or pose0.kind

    # wall anchor under the tile centre, local frame, initial correspondence
    surf0, n0 = snap_to_wall(mesh, pose0.center)
    n0, t1_0, t2_0 = _tangent_frame(n0, pose0.t1)
    n_evals = 0

    def _place(x):
        anchor = surf0 + x[0] * t1_0 + x[1] * t2_0
        surf, n_in = snap_to_wall(mesh, anchor)
        hint = _rodrigues(t1_0, n0, float(x[2]))
        return conform_tile(mesh, surf, n_in, hint, kind=kind)

    tile0 = _place(np.zeros(3))
    assign = _match(tile0.seed_centers, P)

    def _cost(x):
        nonlocal n_evals
        n_evals += 1
        try:
            tile = _place(x)
        except Exception:
            return 1e6
        model = tile.seed_centers[list(assign)]
        c = float(((model - P) ** 2).sum())
        if A is not None:
            ax = tile.seed_axes[list(assign)]
            s = np.linalg.norm(np.cross(ax, A), axis=1)
            c += float((w_axis * s) @ (w_axis * s))
        return c

    x0 = np.zeros(3)
    sol = minimize(_cost, x0, method="Nelder-Mead",
                   options=dict(maxiter=max_iter, xatol=0.05, fatol=1e-3,
                                initial_simplex=np.array(
                                    [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0],
                                     [0, 0, np.deg2rad(10.0)]], float)))
    x = sol.x
    tile = _place(x)
    # re-match at the optimum (the start could have crossed a symmetry)
    assign = _match(tile.seed_centers, P)
    model = tile.seed_centers[list(assign)]
    res = np.linalg.norm(model - P, axis=1)
    aerr = 0.0
    if A is not None:
        ax = tile.seed_axes[list(assign)]
        c = np.clip(np.abs((ax * A).sum(axis=1)), 0.0, 1.0)
        aerr = float(np.degrees(np.arccos(c)).mean())

    _s, dist, _t = trimesh.proximity.closest_point(mesh, P)
    dist = np.asarray(dist, dtype=float)
    detachment = float(np.mean(np.abs(dist - _OFF)))

    free_rms = agreement = None
    if free is not None:
        free_rms = free.rms_mm
        fp = free.seed_points()
        D = np.linalg.norm(fp[:, None, :] - tile.seed_centers[None, :, :], axis=2)
        r, c = linear_sum_assignment(D)
        agreement = float(np.sqrt(np.mean(D[r, c] ** 2)))

    return SurfaceFit(placed=tile, rms_mm=float(np.sqrt(np.mean(res ** 2))),
                      residuals_mm=res, axis_err_deg=aerr, assignment=assign,
                      anchor_ras=tile.anchor_ras, angle_rad=float(x[2]),
                      wall_distance_mm=dist, detachment_mm=detachment,
                      free_rms_mm=free_rms, agreement_mm=agreement,
                      n_evals=n_evals)
