"""Count-free tile configuration search (``n_full_tiles="auto"``).

When the implant count is unknown or untrusted, the number of tiles has to be
*inferred* from the seed cloud.  Candidate tiles come from two tiers:

* **standard** -- the gate-passing quads and pairs :mod:`gtcore.tiles.fit`
  enumerates (chord windows, planarity, axis coherence);
* **deformable** -- quads that FAIL the chord windows but that the bent-tile
  model of :mod:`gtcore.tiles.deform` explains with a small residual and a
  bounded bending energy (a crumpled / folded tile).  These carry
  ``degraded=True`` like the count-constrained completion they replace.

Every full-tile candidate is scored with the deformable fit::

    score = QUAD_BASE - 2.0 * rms_def
                      - 30 * max(0, bending_energy - 0.02)
                      - 2.0 * max(0, axis_err - 15 deg) [rad]

and the configuration is chosen by model selection::

    objective(config) = sum(tile scores) - lambda_full * n_full
                                         - lambda_half * n_half

maximised over *disjoint* selections.  ``lambda`` is a BIC-like complexity
penalty: a tile only earns its place when its geometric evidence exceeds the
penalty, so a configuration cannot inflate ``n`` with mediocre quads carved
out of clutter.  The exact optimum for every tile count is computed as well
(``best_score(n)`` for ``n = 0 .. n_max``) and returned as a score curve, so
the UI can show *why* ``n`` was chosen ("evidence supports n=4; a 5th tile
adds only X, below the penalty").  The chosen ``n`` is where the penalised
curve peaks -- equivalently, where the marginal gain of one more tile drops
below ``lambda`` (the saturation we observed on the real post-op scan, where
requesting 5 or 6 tiles kept returning 4).

Calibration (2026-09, plan Steps 2-3).  Deformable-fit residuals: synthetic
wall-conformed tiles 0.2-0.5 mm, the 8-tile printed phantom 0.3-1.5 mm
(1 mm slices; its crumpled tile 0.46 mm at bending energy 0.066), the
post-op case 0.4-0.7 mm; junk quads that pass the loose gates sit at
0.8-2.8 mm (median 1.6) with axis errors of 30 deg and more.  Every real
tile scores >= 4.9 under the formula above; ``LAMBDA_FULL = 3.5`` sits
below that with margin.  A half tile pays the SAME penalty: it has the same
pose parameters as a full tile but explains half the data, and with a
smaller penalty the optimiser happily splits a genuine (deformed) quad into
two near-perfect pairs.

Half tiles in auto mode
-----------------------
Two seeds carry little evidence: on the real post-op scan, clutter pairs
(bone / streak candidates 7-10 mm apart with roughly parallel axes) score
as well as the synthetic phantom's true half tiles, so geometry alone
cannot tell them apart.  Auto mode therefore does NOT count half tiles
unless ``allow_half=True`` (the OR team confirms halves were cut); it still
enumerates the pairs left over after the full tiles and reports them as
``half_candidates`` for the UI / the surface-constrained validation of
plan Step 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional

import numpy as np

from .deform import DeformableFit, fit_deformable
from .fit import (
    DIAG_MAX_MM,
    QUAD_BASE,
    SIDE_MIN_MM,
    TileFitResult,
    TilePose,
    _axis_angle_deg,
    _enumerate_pairs,
    _enumerate_quads,
    _full_pose,
    _half_pose,
    _normalize_axes,
    _orient_normal,
    _project_in_plane,
)
from .model import fit_rigid

__all__ = ["LAMBDA_FULL", "LAMBDA_HALF", "ScorePoint", "AutoFitResult",
           "fit_tiles_auto", "deformable_score", "to_placed_tiles"]

LAMBDA_FULL = 3.5
LAMBDA_HALF = LAMBDA_FULL

# deformable-fit scoring of a full-tile candidate
DEF_W_RMS = 2.0                 # score per mm of seed residual
DEF_E_FREE = 0.02               # bending energy (1/mm^2) with no penalty
DEF_W_E = 30.0                  # score per unit of bending energy beyond it
DEF_AXIS_SOFT_DEG = 15.0
DEF_W_AXIS = 2.0                # per radian beyond the soft angle

# admission of quads that fail the standard chord gates (crumpled tiles)
LOOSE_CHORD_MM = (3.5, 16.5)
LOOSE_AXIS_HARD_DEG = 55.0      # 3rd-smallest pairwise axis angle
LOOSE_MIN_EXTENT_MM = 1.5       # 2nd singular value: not a clip line
LOOSE_SIM_RMS_MAX_MM = 2.0      # closed-form similarity-fit prefilter
LOOSE_RMS_MAX_MM = 0.8
LOOSE_E_MAX = 0.10
LOOSE_AXIS_MAX_DEG = 25.0

_SEARCH_NODE_CAP = 200000
_OFF_MM = 3.0                   # geometry.SEED_PLANE_OFFSET_MM


def deformable_score(fit: DeformableFit) -> float:
    """Tile score of a quad from its deformable fit (see module docstring)."""
    axis_pen = DEF_W_AXIS * float(np.deg2rad(
        max(0.0, fit.axis_err_deg - DEF_AXIS_SOFT_DEG)))
    bend_pen = DEF_W_E * max(0.0, fit.bending_energy - DEF_E_FREE)
    return QUAD_BASE - DEF_W_RMS * fit.rms_mm - bend_pen - axis_pen


@dataclass
class ScorePoint:
    """One point of the model-selection curve: the best configuration with
    exactly ``n`` tiles."""

    n: int
    score: float                # best total unpenalised score with n tiles
    penalized: float            # score - sum of per-tile penalties
    marginal: float             # score(n) - score(n - 1)
    n_full: int
    n_half: int
    feasible: bool = True


@dataclass
class AutoFitResult(TileFitResult):
    """``TileFitResult`` plus the evidence behind the chosen tile count."""

    score_curve: List[ScorePoint] = field(default_factory=list)
    n_selected: int = 0
    lambda_full: float = LAMBDA_FULL
    lambda_half: float = LAMBDA_HALF
    auto: bool = True
    # (score, (i, j)) of gate-passing pairs among the unassigned candidates,
    # descending; only *selected* when allow_half=True
    half_candidates: List[tuple] = field(default_factory=list)

    def summary(self) -> str:
        """One-line evidence statement for the chosen count."""
        n = self.n_selected
        nxt = [p for p in self.score_curve if p.n == n + 1 and p.feasible]
        if not nxt:
            return "evidence supports n=%d (no further tile candidate)" % n
        return ("evidence supports n=%d (tile %d would add only %.2f, below "
                "the %.1f penalty)" % (n, n + 1, nxt[0].marginal,
                                       self.lambda_full))


# ------------------------------------------------------------- exact search
class _PerCountSelector:
    """Best disjoint selection for EVERY tile count.

    Items are ``(score, indices, penalty)`` sorted by descending score.
    Depth-first over include/exclude with a bound per count: from position
    ``pos`` with ``c`` items chosen the most that can still be added is the
    suffix's best ``m``-item sum, so a branch dies when it cannot improve
    ``best[c + m]`` for any ``m``.  Greedy is explored first; a node cap
    keeps pathological inputs finite (the incumbent is then at least greedy).
    """

    def __init__(self, items, n_max):
        self.items = items
        self.n_max = n_max
        self.best = [-np.inf] * (n_max + 1)
        self.best[0] = 0.0
        self.best_sets = [None] * (n_max + 1)
        self.best_sets[0] = []
        self.nodes = 0
        self.suffix = self._suffix([it[0] for it in items], n_max)

    @staticmethod
    def _suffix(scores, max_slots):
        n = len(scores)
        out = [None] * (n + 1)
        out[n] = [0.0]
        for i in range(n - 1, -1, -1):
            prev = out[i + 1]
            cur = [0.0]
            for m in range(1, min(len(prev) + 1, max_slots + 1)):
                take = scores[i] + (prev[m - 1] if m - 1 < len(prev) else 0.0)
                skip = prev[m] if m < len(prev) else -np.inf
                cur.append(max(take, skip))
            out[i] = cur
        return out

    def run(self):
        self._dfs(0, [], frozenset(), 0.0)
        return self.best, self.best_sets

    def _dfs(self, pos, chosen, used, score):
        self.nodes += 1
        if self.nodes > _SEARCH_NODE_CAP:
            return
        c = len(chosen)
        if score > self.best[c]:
            self.best[c] = score
            self.best_sets[c] = list(chosen)
        if c == self.n_max or pos == len(self.items):
            return
        row = self.suffix[pos]
        improvable = False
        for m in range(1, min(len(row), self.n_max - c + 1)):
            if score + row[m] > self.best[c + m] + 1e-12:
                improvable = True
                break
        if not improvable:
            return
        s, idx, _pen = self.items[pos]
        if not (used & idx):
            self._dfs(pos + 1, chosen + [pos], used | idx, score + s)
        self._dfs(pos + 1, chosen, used, score)


# ------------------------------------------------------- deformable tier
def _enumerate_loose_quads(centers, axes, dist, exclude):
    """Quads outside the standard chord gates that the bent-tile model still
    explains: loose chord window, quad topology, axis coherence, planar
    extent, a closed-form similarity prefilter, then the deformable fit.
    Returns ``[(idx, DeformableFit)]``."""
    n = centers.shape[0]
    lo, hi = LOOSE_CHORD_MM
    link = (dist >= lo) & (dist <= hi)
    out = []
    for idx in combinations(range(n), 4):
        i, j, k, l = idx
        if not (link[i, j] and link[i, k] and link[i, l]
                and link[j, k] and link[j, l] and link[k, l]):
            continue
        if frozenset(idx) in exclude:
            continue
        pair_ids = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        d6 = np.array([dist[a, b] for a, b in pair_ids])
        order = np.argsort(d6)
        d1, d2 = (pair_ids[int(o)] for o in order[4:])
        if len({d1[0], d1[1], d2[0], d2[1]}) != 4:
            continue
        angles = sorted(_axis_angle_deg(axes[a], axes[b]) for a, b in pair_ids)
        if angles[2] > LOOSE_AXIS_HARD_DEG:
            continue
        pts = centers[list(idx)]
        sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        if float(sv[1]) < LOOSE_MIN_EXTENT_MM:
            continue
        sim = fit_rigid(pts, axes[list(idx)], allow_scale=True,
                        scale_range=(0.5, 1.1))
        if sim.rms_mm > LOOSE_SIM_RMS_MAX_MM:
            continue
        fit = fit_deformable(pts, axes[list(idx)])
        if (fit.rms_mm <= LOOSE_RMS_MAX_MM and fit.bending_energy <= LOOSE_E_MAX
                and fit.axis_err_deg <= LOOSE_AXIS_MAX_DEG):
            out.append((idx, fit))
    return out


def _deformed_pose(tile_id, idx, centers, fit, cavity_center, degraded):
    """TilePose from a deformable fit (normal oriented like fit.py's)."""
    pts = centers[list(idx)]
    center = pts.mean(axis=0)
    normal = _orient_normal(fit.pose.normal.copy(), center, cavity_center)
    t1 = _project_in_plane(fit.pose.t1, normal)
    return TilePose(
        tile_id=tile_id, kind="full", seed_indices=list(idx),
        center_ras=center, normal_ras=normal, axis_ras=t1,
        residual_mm=fit.rms_mm, degraded=degraded, deform=fit,
    )


# ------------------------------------------------------------ planner feed
def to_placed_tiles(result, centers_ras=None, axes_ras=None):
    """Turn recovered tiles into planner :class:`~gtcore.interact.PlacedTile`
    objects (the "suggest tiles" feed): the surface-conformed tile when a
    cavity mesh was available, otherwise a tile built from the bent-tile
    fit (seed sheet corners, normal toward the cavity) or, for a plain
    pose, from the observed seeds.  Every returned tile is an ordinary
    placed tile: it can be dragged, rotated and deleted like a dropped one.
    """
    from ..interact import PlacedTile
    from .deform import deformed_footprint

    out = []
    for pose in result.tiles:
        if pose.surface is not None:
            out.append(pose.surface.placed)
            continue
        idx = list(pose.seed_indices)
        if centers_ras is not None and axes_ras is not None:
            seed_c = np.asarray(centers_ras, float)[idx]
            seed_a = np.asarray(axes_ras, float)[idx]
        else:
            seed_c = None
            seed_a = None
        if pose.deform is not None:
            fit = pose.deform
            n = fit.pose.normal                      # toward the cavity
            corners = deformed_footprint(fit.pose, fit.params, offset_mm=0.0)
            if seed_c is None:
                seed_c = fit.seed_points()
                from .deform import deformed_seed_axes
                seed_a = deformed_seed_axes(fit.pose, fit.params)
            anchor = fit.pose.center - _OFF_MM * n
            axis = fit.pose.t1
        else:
            n = -pose.normal_ras                     # fit.py: away from cavity
            axis = pose.axis_ras
            t2 = np.cross(n, axis)
            cu = 10.0 if pose.kind == "full" else 5.0
            corners = np.array([pose.center_ras + su * cu * axis + sv * 10.0 * t2
                                for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if seed_c is None:
                raise ValueError("centers_ras/axes_ras needed for a plain pose")
            anchor = pose.center_ras - _OFF_MM * n
        out.append(PlacedTile(kind=pose.kind, center_ras=seed_c.mean(axis=0),
                              normal_ras=n, axis_ras=axis, seed_centers=seed_c,
                              seed_axes=seed_a, corners_ras=corners,
                              anchor_ras=anchor))
    return out


# --------------------------------------------------------------------- main
def fit_tiles_auto(centers_ras, axes_ras, cavity_center_ras=None,
                   allow_half=False, lambda_full=LAMBDA_FULL,
                   lambda_half=LAMBDA_HALF, max_tiles=None,
                   deformable=True, mesh=None) -> AutoFitResult:
    """Infer the tile configuration from the seed cloud alone.

    Parameters
    ----------
    centers_ras, axes_ras : (N, 3) arrays
        Seed candidates (see :func:`gtcore.tiles.fit.fit_tiles`).
    cavity_center_ras : (3,), optional
        Orients tile normals away from the cavity centre.
    allow_half : bool
        Also *select* surgeon-cut 2-seed half tiles (default False: see the
        module docstring; pairs are always reported in ``half_candidates``).
    lambda_full, lambda_half : float
        Complexity penalty per full / half tile (see module docstring).
    max_tiles : int, optional
        Upper bound on the number of tiles considered (default ``N // 2``).
    deformable : bool
        Score full tiles with the bent-tile fit and admit crumpled tiles
        that fail the chord gates (default).  ``False`` falls back to the
        standard geometric scores only.
    mesh : trimesh.Trimesh, optional
        Cavity wall.  When given, every selected tile is additionally
        conformed onto it (:func:`gtcore.tiles.surface.fit_on_surface`,
        plan Step 4) and ``TilePose.surface`` carries the wall footprint
        and the attached / consistent verdict.  Selection itself is NOT
        changed by the mesh: the surface fit is a cross-check.

    Returns
    -------
    AutoFitResult
        ``tiles`` of the penalised optimum (fulls first, then halves, each by
        descending score), ``score_curve`` for ``n = 0 .. n_max``,
        ``n_selected`` / ``n_expected`` = the chosen count, ``all_assigned``
        always True (there is no external count to fall short of).
    """
    centers = np.asarray(centers_ras, dtype=float).reshape(-1, 3)
    axes = _normalize_axes(axes_ras) if centers.size else \
        np.zeros((0, 3), dtype=float)
    if axes.shape[0] != centers.shape[0]:
        raise ValueError(
            "centers_ras and axes_ras disagree: %d vs %d candidates"
            % (centers.shape[0], axes.shape[0]))
    cavity_center = None if cavity_center_ras is None else \
        np.asarray(cavity_center_ras, dtype=float).reshape(3)
    n = centers.shape[0]
    n_max = n // 2 if max_tiles is None else int(max_tiles)

    result = AutoFitResult(lambda_full=float(lambda_full),
                           lambda_half=float(lambda_half))
    result.score_curve = [ScorePoint(0, 0.0, 0.0, 0.0, 0, 0)]
    result.rejected_indices = list(range(n))
    result.all_assigned = True
    if n < 2 or n_max < 1:
        return result

    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    link = (dist >= SIDE_MIN_MM) & (dist <= DIAG_MAX_MM)
    quads = _enumerate_quads(centers, axes, dist, link) if n >= 4 else []
    pairs = _enumerate_pairs(centers, axes, dist)
    leftover_pairs = []
    if not allow_half:
        leftover_pairs, pairs = pairs, []

    # items: (score, frozenset, penalty, kind, idx, payload, degraded)
    items = []
    if deformable:
        std = set()
        for _s, idx, _r in quads:
            fit = fit_deformable(centers[list(idx)], axes[list(idx)])
            score = deformable_score(fit)
            std.add(frozenset(idx))
            if score > 0.0:
                items.append((score, frozenset(idx), float(lambda_full),
                              "full", idx, fit, False))
        if n >= 4:
            for idx, fit in _enumerate_loose_quads(centers, axes, dist, std):
                score = deformable_score(fit)
                if score > 0.0:
                    items.append((score, frozenset(idx), float(lambda_full),
                                  "full", idx, fit, True))
    else:
        items += [(s, frozenset(idx), float(lambda_full), "full", idx, r,
                   False) for s, idx, r in quads]
    items += [(s, frozenset(idx), float(lambda_half), "half", idx, r, False)
              for s, idx, r in pairs]
    # descending score, kind then index tuple as the deterministic tie-break
    items.sort(key=lambda it: (-it[0], it[3], it[4]))
    if not items:
        result.half_candidates = sorted(
            [(s, idx) for s, idx, _r in leftover_pairs],
            key=lambda p: (-p[0], p[1]))
        return result

    n_max = min(n_max, len(items))
    best, best_sets = _PerCountSelector(
        [(it[0], it[1], it[2]) for it in items], n_max).run()

    curve = []
    prev = 0.0
    for c in range(n_max + 1):
        feasible = np.isfinite(best[c])
        if not feasible:
            curve.append(ScorePoint(c, float("nan"), float("nan"),
                                    float("nan"), 0, 0, feasible=False))
            continue
        chosen = best_sets[c]
        n_full = sum(1 for i in chosen if items[i][3] == "full")
        n_half = len(chosen) - n_full
        pen = best[c] - lambda_full * n_full - lambda_half * n_half
        curve.append(ScorePoint(c, float(best[c]), float(pen),
                                float(best[c] - prev), n_full, n_half))
        prev = best[c]
    result.score_curve = curve

    # penalised optimum; ties -> fewer tiles (parsimony)
    feas = [p for p in curve if p.feasible]
    n_sel = max(feas, key=lambda p: (p.penalized, -p.n)).n
    chosen = best_sets[n_sel]

    tiles = []
    assigned = set()
    for kind in ("full", "half"):
        for i in chosen:
            s, _fs, _pen, k, idx, payload, degraded = items[i]
            if k != kind:
                continue
            if kind == "full" and isinstance(payload, DeformableFit):
                pose = _deformed_pose(len(tiles), idx, centers, payload,
                                      cavity_center, degraded)
            elif kind == "full":
                pose = _full_pose(len(tiles), idx, centers, axes, payload,
                                  cavity_center)
            else:
                pose = _half_pose(len(tiles), idx, centers, axes, payload,
                                  cavity_center)
            tiles.append(pose)
            assigned.update(idx)
    if mesh is not None and len(getattr(mesh, "faces", [])) > 0:
        from .surface import fit_on_surface

        for pose in tiles:
            try:
                pose.surface = fit_on_surface(
                    mesh, centers[pose.seed_indices], axes[pose.seed_indices],
                    kind=pose.kind, init=pose.deform)
            except Exception:
                pose.surface = None
    result.tiles = tiles
    result.rejected_indices = [i for i in range(n) if i not in assigned]
    result.half_candidates = sorted(
        [(s, idx) for s, idx, _r in leftover_pairs
         if not (assigned & set(idx))],
        key=lambda p: (-p[0], p[1]))
    result.n_selected = n_sel
    result.n_expected = n_sel
    result.all_assigned = True
    return result
