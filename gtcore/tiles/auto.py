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

Cover pass (explain every seed of the implant)
----------------------------------------------
The penalised optimum above is *conservative by design*: its gates and the
3.5 penalty are calibrated on <= 1 mm scans, so on a coarse or gappy export
(2 mm slices, ~1 mm z error per seed) real tiles fail the chord gates or
score below the penalty and their seeds are dropped silently.  The cover
pass therefore runs AFTER the confident selection, only on the seeds left
inside the implant region (within ``CLUSTER_RADIUS_MM`` of a supported
tile), with two lower-confidence tiers that never touch the supported set:

* **relaxed quads** -- the loose (deformable) tier again, its residual /
  similarity / axis limits scaled by the slice-spacing factor ``tol`` and
  a small per-tile penalty ``LAMBDA_COVER``;
* **triplet completion** -- 3 detected seeds forming an L of the tile
  square (two ~10 mm arms at ~90 deg) are completed to a FULL tile by the
  manufactured geometry; the 4th seed is *inferred* and reported as such.
  A 1- or 3-seed tile does not exist physically; a triplet is always read
  as a 4-seed tile with one detection miss.

Tiles from the cover pass carry ``confidence="tentative"`` and live in
``tentative_tiles`` (``tiles`` keeps the supported set, so the calibrated
count ``n_selected`` is unchanged).  Whatever still remains inside the
implant region is listed in ``unassigned_indices``; candidates far from
the implant are ``clutter_indices`` (bone, clips, streaks).  Half tiles are
never created by the cover pass: without the OR's word the safe prior is
"no half tiles".
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

__all__ = ["LAMBDA_FULL", "LAMBDA_HALF", "LAMBDA_COVER", "AMBIGUOUS_MARGIN",
           "ScorePoint", "AutoFitResult", "ImplantPrior", "fit_tiles_auto",
           "fit_tiles_prior", "spacing_tolerance", "deformable_score",
           "to_placed_tiles"]

LAMBDA_FULL = 3.5
LAMBDA_HALF = LAMBDA_FULL

# deformable-fit scoring of a full-tile candidate
DEF_W_RMS = 2.0                 # score per mm of seed residual
DEF_E_FREE = 0.02               # bending energy (1/mm^2) with no penalty
DEF_W_E = 30.0                  # score per unit of bending energy beyond it
DEF_AXIS_SOFT_DEG = 15.0
DEF_W_AXIS = 2.0                # per radian beyond the soft angle

# Partition margin (plan-localization stage 6): a selected tile whose best
# same-count alternative scores within this of the chosen configuration is
# flagged ambiguous.  The bent-tile score falls by DEF_W_RMS = 2 per mm of
# seed RMS, so 1.0 score unit = 0.5 mm of bent-tile RMS on ONE tile: two
# groupings closer than that differ by less than the per-tile residual
# spread of real tiles (printed phantom 0.26-0.87 mm), i.e. the seed cloud
# does not decide between them.  Same number on the counted chord path,
# where W_RESIDUAL = 1.5/mm makes it ~0.67 mm of chord RMS.
AMBIGUOUS_MARGIN = 0.5 * DEF_W_RMS

# admission of quads that fail the standard chord gates (crumpled tiles)
LOOSE_CHORD_MM = (3.5, 16.5)
LOOSE_AXIS_HARD_DEG = 55.0      # 3rd-smallest pairwise axis angle
LOOSE_MIN_EXTENT_MM = 1.5       # 2nd singular value: not a clip line
LOOSE_SIM_RMS_MAX_MM = 2.0      # closed-form similarity-fit prefilter
LOOSE_RMS_MAX_MM = 0.8
LOOSE_E_MAX = 0.10
LOOSE_AXIS_MAX_DEG = 25.0

_SEARCH_NODE_CAP = 200000
_SEARCH_NODE_CAP_SMALL = 2000000  # <= _SMALL_ITEMS candidates: exact search
_SMALL_ITEMS = 400
_OFF_MM = 3.0                   # geometry.SEED_PLANE_OFFSET_MM

# cover pass (tentative tier; see module docstring).  ``tol`` is the slice
# spacing in mm clamped at >= 1: partial volume and interpolation put ~dz/2
# of z error on every seed, which the thin-cut calibration never saw.
LAMBDA_COVER = 1.0
CLUSTER_RADIUS_MM = 25.0        # "inside the implant": near a supported tile
COVER_RMS_MAX_MM = 1.2          # x tol
COVER_E_MAX = 0.20
COVER_AXIS_MAX_DEG = 35.0
COVER_SIM_RMS_MAX_MM = 3.0      # x tol
# triplet completion
TRIPLET_ARM_MM = (6.0, 12.5)    # widened by (tol - 1) mm on each side
TRIPLET_HYPOT_MM = (9.0, 16.5)
TRIPLET_CORNER_DEG = (65.0, 115.0)
TRIPLET_RMS_MAX_MM = 1.0        # x tol, deformable rms with the inferred seed
TRIPLET_PENALTY = 1.5           # score handicap vs a 4-seed quad
TRIPLET_MIN_GAP_MM = 3.5        # inferred seed must not sit on a detected one
# Above this slice spacing the per-seed PCA axis is degenerate (a capsule
# spanning one slice is a pancake whose axis is whatever the voxel grid
# says), so the cover pass neither gates nor fits on axes there.
AXES_RELIABLE_DZ_MM = 1.2


def spacing_tolerance(spacing_mm) -> float:
    """Gate-scaling factor for a scan: 1 for thin cuts (<= 1 mm), else the
    largest voxel dimension in mm (2 mm slices -> 2)."""
    if spacing_mm is None:
        return 1.0
    return float(max(1.0, np.max(np.asarray(spacing_mm, dtype=float))))


@dataclass
class ImplantPrior:
    """What the OR team told us about the implant (all optional).

    ``n_full`` / ``n_half`` are trusted counts when given.  When unknown the
    safe assumption is *no* half tiles (they cannot be told from clutter
    pairs by geometry alone) and every tile carries 4 seeds.  ``n_seeds``
    is the number of seeds implanted (4 per full, 2 per half): it is a
    consistency check on detection and drives the coarse-scan threshold
    search in the pipeline.
    """

    n_full: Optional[int] = None
    n_half: int = 0
    n_seeds: Optional[int] = None

    def __post_init__(self):
        if self.n_full is not None:
            self.n_full = int(self.n_full)
            if self.n_full < 0:
                raise ValueError("n_full must be >= 0")
        self.n_half = int(self.n_half or 0)
        if self.n_half < 0:
            raise ValueError("n_half must be >= 0")
        if self.n_seeds is None and self.n_full is not None:
            self.n_seeds = 4 * self.n_full + 2 * self.n_half
        elif self.n_seeds is not None:
            self.n_seeds = int(self.n_seeds)

    @property
    def count_known(self) -> bool:
        return self.n_full is not None

    def describe(self) -> str:
        if self.count_known:
            txt = "%d full + %d half (given)" % (self.n_full, self.n_half)
            if self.n_seeds is not None:
                txt += ", %d seeds" % self.n_seeds
            return txt
        txt = "unknown -- assuming 0 half tiles, 4 seeds per tile"
        if self.n_seeds is not None:
            txt += ", %d seeds given" % self.n_seeds
        return txt


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
    # cover pass (module docstring): lower-confidence tiles that explain
    # seeds the supported set left inside the implant region
    tentative_tiles: List[TilePose] = field(default_factory=list)
    unassigned_indices: List[int] = field(default_factory=list)  # in-implant
    clutter_indices: List[int] = field(default_factory=list)     # far away
    spacing_tol: float = 1.0
    capped: bool = False            # the exact search hit its node cap
    score_rule: str = "deformable"  # bent-tile scores (chord: deformable=False)
    n_requested: Optional[int] = None  # OR count when one was given
    prior: Optional["ImplantPrior"] = None

    @property
    def all_tiles(self) -> List[TilePose]:
        """Supported tiles first, then tentative ones."""
        return list(self.tiles) + list(self.tentative_tiles)

    @property
    def n_tentative(self) -> int:
        return len(self.tentative_tiles)

    @property
    def n_inferred_seeds(self) -> int:
        return sum(1 for p in self.all_tiles if p.inferred_seed_ras is not None)

    def summary(self) -> str:
        """One-line evidence statement for the chosen count."""
        n = self.n_selected
        if self.n_requested is not None:
            base = "asked for %d full tile(s): %d supported" % (
                self.n_requested, n)
            if n < self.n_requested:
                base += " (geometry supports only %d)" % n
        else:
            nxt = [p for p in self.score_curve if p.n == n + 1 and p.feasible]
            if not nxt:
                base = "evidence supports n=%d (no further tile candidate)" % n
            else:
                base = ("evidence supports n=%d (tile %d would add only %.2f, "
                        "below the %.1f penalty)" % (n, n + 1, nxt[0].marginal,
                                                    self.lambda_full))
        extra = []
        if self.tentative_tiles:
            extra.append("%d tentative" % len(self.tentative_tiles))
        if self.n_inferred_seeds:
            extra.append("%d inferred seed%s" % (
                self.n_inferred_seeds, "" if self.n_inferred_seeds == 1 else "s"))
        if self.unassigned_indices:
            extra.append("%d seed%s unassigned" % (
                len(self.unassigned_indices),
                "" if len(self.unassigned_indices) == 1 else "s"))
        if self.capped:
            extra.append("search capped")
        if extra:
            base += "; " + ", ".join(extra)
        return base


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
        self.node_cap = (_SEARCH_NODE_CAP_SMALL if len(items) <= _SMALL_ITEMS
                         else _SEARCH_NODE_CAP)
        self.capped = False
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
        if self.nodes > self.node_cap:
            self.capped = True
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


def _per_count_margins(items, chosen, n_sel, best_n_sel):
    """Partition margin of every selected item (plan-localization stage 6):
    re-run the per-count search with that one item removed (its seeds stay
    available to every other grouping) and compare the best total at the
    SAME count ``n_sel``.  ``inf`` when no other grouping of ``n_sel``
    tiles exists.  Returns ``({pos: (margin, alt positions or None)},
    capped)``."""
    out, capped = {}, False
    for p in chosen:
        keep = [j for j in range(len(items)) if j != p]
        sel = _PerCountSelector([(items[j][0], items[j][1], items[j][2])
                                 for j in keep], n_sel)
        best, sets = sel.run()
        capped |= sel.capped
        if not np.isfinite(best[n_sel]):
            out[p] = (float("inf"), None)
        else:
            out[p] = (float(best_n_sel - best[n_sel]),
                      [keep[j] for j in sets[n_sel]])
    return out, capped


# ------------------------------------------------------- deformable tier
def _enumerate_loose_quads(centers, axes, dist, exclude,
                           rms_max=LOOSE_RMS_MAX_MM, e_max=LOOSE_E_MAX,
                           axis_max=LOOSE_AXIS_MAX_DEG,
                           sim_max=LOOSE_SIM_RMS_MAX_MM, subset=None,
                           use_axes=True):
    """Quads outside the standard chord gates that the bent-tile model still
    explains: loose chord window, quad topology, axis coherence, planar
    extent, a closed-form similarity prefilter, then the deformable fit.
    Returns ``[(idx, DeformableFit)]``.  ``subset`` restricts the candidate
    indices (the cover pass runs on the leftovers only); the thresholds are
    overridable for the same reason."""
    n = centers.shape[0]
    lo, hi = LOOSE_CHORD_MM
    link = (dist >= lo) & (dist <= hi)
    out = []
    pool = range(n) if subset is None else sorted(subset)
    for idx in combinations(pool, 4):
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
        if use_axes:
            angles = sorted(_axis_angle_deg(axes[a], axes[b])
                            for a, b in pair_ids)
            if angles[2] > LOOSE_AXIS_HARD_DEG:
                continue
        pts = centers[list(idx)]
        sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        if float(sv[1]) < LOOSE_MIN_EXTENT_MM:
            continue
        ax = axes[list(idx)] if use_axes else None
        sim = fit_rigid(pts, ax, allow_scale=True, scale_range=(0.5, 1.1))
        if sim.rms_mm > sim_max:
            continue
        fit = fit_deformable(pts, ax)
        if (fit.rms_mm <= rms_max and fit.bending_energy <= e_max
                and (not use_axes or fit.axis_err_deg <= axis_max)):
            out.append((idx, fit))
    return out


# ------------------------------------------------------------ cover pass
def _select_penalised(items, lam, n_max):
    """Best disjoint selection under a per-item penalty ``lam``; returns
    ``(chosen item positions, capped)``.  ``items`` are
    ``(score, frozenset, ...)`` sorted by descending score."""
    if not items or n_max < 1:
        return [], False
    n_max = min(int(n_max), len(items))
    sel = _PerCountSelector([(it[0], it[1], lam) for it in items], n_max)
    best, best_sets = sel.run()
    best_c, best_v = 0, 0.0
    for c in range(n_max + 1):
        if not np.isfinite(best[c]):
            continue
        v = best[c] - lam * c
        if v > best_v + 1e-12:
            best_c, best_v = c, v
    return list(best_sets[best_c] or []), sel.capped


def _mean_axis(axes):
    ref = axes[0]
    signed = np.array([a if a @ ref >= 0 else -a for a in axes])
    m = signed.mean(axis=0)
    nrm = np.linalg.norm(m)
    return ref if nrm < 1e-9 else m / nrm


def _enumerate_triplets(centers, axes, dist, pool, tol, forbidden_pts,
                        use_axes=True):
    """L-shaped triplets completed to a full tile by the manufactured
    geometry.  Returns ``[(idx3, fourth_ras, DeformableFit, score)]``.

    The corner seed is the one off the longest chord; the missing seed is
    the parallelogram completion ``a + c - corner``.  The completed quad is
    then fitted with the bent-tile model (the inferred seed gets the mean
    axis of its mates) so the residual measures how square the three real
    seeds are, and the tile pose comes from the same model as every other
    tile.  A detected candidate already sitting where the 4th seed would go
    means the quad was judged (and rejected) upstream, so the triplet is
    skipped rather than used to smuggle it back in.
    """
    slack = tol - 1.0
    arm_lo, arm_hi = TRIPLET_ARM_MM[0] - slack, TRIPLET_ARM_MM[1] + slack
    hyp_lo, hyp_hi = TRIPLET_HYPOT_MM[0] - slack, TRIPLET_HYPOT_MM[1] + slack
    out = []
    for idx in combinations(sorted(pool), 3):
        i, j, k = idx
        chords = [(dist[i, j], k), (dist[i, k], j), (dist[j, k], i)]
        chords.sort(key=lambda c: c[0])
        (d_a, _), (d_b, _), (d_h, corner) = chords
        if not (arm_lo <= d_a <= arm_hi and arm_lo <= d_b <= arm_hi
                and hyp_lo <= d_h <= hyp_hi):
            continue
        arms = [m for m in idx if m != corner]
        u = centers[arms[0]] - centers[corner]
        v = centers[arms[1]] - centers[corner]
        cosang = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
        if not (TRIPLET_CORNER_DEG[0] <= ang <= TRIPLET_CORNER_DEG[1]):
            continue
        # axis coherence: at least two of the three axes agree (a coarse
        # scan hands single-slice seeds an arbitrary axis)
        if use_axes:
            angles = sorted(_axis_angle_deg(axes[a], axes[b])
                            for a, b in combinations(idx, 2))
            if angles[0] > LOOSE_AXIS_HARD_DEG:
                continue
        fourth = centers[arms[0]] + centers[arms[1]] - centers[corner]
        if len(forbidden_pts):
            gap = np.linalg.norm(forbidden_pts - fourth[None, :], axis=1).min()
            if gap < TRIPLET_MIN_GAP_MM:
                continue
        ax = None
        if use_axes:
            ax = np.vstack([axes[list(idx)],
                            _mean_axis(axes[list(idx)])[None, :]])
        fit = None
        for _ in range(3):
            # the parallelogram point is a flat-tile guess; a wall-conformed
            # tile bends, so re-place the 4th seed where the bent-tile model
            # fitted to the three real seeds predicts it (fixed-point, 3
            # rounds are plenty: the update is a fraction of a millimetre)
            pts = np.vstack([centers[list(idx)], fourth[None, :]])
            fit = fit_deformable(pts, ax)
            pred = fit.seed_points()[fit.assignment[3]]
            if np.linalg.norm(pred - fourth) < 0.05:
                break
            fourth = pred
        # the inferred seed sits ON the model by construction, so judge and
        # report the residual over the three real seeds only
        rms3 = float(np.sqrt(np.mean(np.asarray(fit.residuals_mm)[:3] ** 2)))
        if rms3 > TRIPLET_RMS_MAX_MM * tol or fit.bending_energy > COVER_E_MAX:
            continue
        score = (deformable_score(fit) - DEF_W_RMS * (rms3 - fit.rms_mm)
                 - TRIPLET_PENALTY)
        if score > 0.0:
            out.append((idx, fourth, fit, score, rms3))
    return out


def _cover_pass(centers, axes, dist, pool, cavity_center, tol, lam, n_max,
                next_tile_id):
    """Explain the leftover seeds ``pool`` with tentative tiles: relaxed
    quads first, then triplet completion on what is still left.  Returns
    ``(poses, assigned_indices, capped)``."""
    poses, assigned, capped = [], set(), False
    if n_max < 1 or len(pool) < 3:
        return poses, assigned, capped
    use_axes = tol <= AXES_RELIABLE_DZ_MM
    quads = _enumerate_loose_quads(
        centers, axes, dist, exclude=set(),
        rms_max=COVER_RMS_MAX_MM * tol, e_max=COVER_E_MAX,
        axis_max=COVER_AXIS_MAX_DEG, sim_max=COVER_SIM_RMS_MAX_MM * tol,
        subset=pool, use_axes=use_axes) if len(pool) >= 4 else []
    items = []
    for idx, fit in quads:
        score = deformable_score(fit)
        if score > 0.0:
            items.append((score, frozenset(idx), idx, fit))
    items.sort(key=lambda it: (-it[0], it[2]))
    chosen, c1 = _select_penalised(items, lam, n_max)
    capped |= c1
    for pos in chosen:
        _s, _fs, idx, fit = items[pos]
        pose = _deformed_pose(next_tile_id + len(poses), idx, centers, fit,
                              cavity_center, degraded=True)
        pose.confidence = "tentative"
        poses.append(pose)
        assigned.update(idx)

    rest = [i for i in pool if i not in assigned]
    room = n_max - len(poses)
    if room >= 1 and len(rest) >= 3:
        # the inferred seed must not land on ANY detected candidate
        forbidden = centers
        trips = _enumerate_triplets(centers, axes, dist, rest, tol, forbidden,
                                    use_axes=use_axes)
        items = [(score, frozenset(idx), idx, fourth, fit, rms3)
                 for idx, fourth, fit, score, rms3 in trips]
        items.sort(key=lambda it: (-it[0], it[2]))
        chosen, c2 = _select_penalised(items, lam, room)
        capped |= c2
        for pos in chosen:
            _s, _fs, idx, fourth, fit, rms3 = items[pos]
            pose = _deformed_pose(next_tile_id + len(poses), idx, centers,
                                  fit, cavity_center, degraded=True)
            pose.confidence = "tentative"
            pose.inferred_seed_ras = fourth
            pose.residual_mm = rms3
            # centre over all 4 seeds, the inferred one included
            pose.center_ras = np.vstack([centers[list(idx)],
                                         fourth[None, :]]).mean(axis=0)
            poses.append(pose)
            assigned.update(idx)
    return poses, assigned, capped


def _split_leftovers(centers, leftovers, tile_centers):
    """Leftover candidates inside the implant region (near a tile) vs far
    clutter.  With no tile at all there is no region: everything is
    clutter, nothing is 'unassigned'."""
    if not len(tile_centers) or not leftovers:
        return [], sorted(int(i) for i in leftovers)
    tc = np.asarray(tile_centers, dtype=float).reshape(-1, 3)
    near, far = [], []
    for i in leftovers:
        d = np.linalg.norm(tc - centers[i][None, :], axis=1).min()
        (near if d <= CLUSTER_RADIUS_MM else far).append(int(i))
    return sorted(near), sorted(far)


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
    poses = result.all_tiles if hasattr(result, "all_tiles") else result.tiles
    for pose in poses:
        if pose.surface is not None:
            out.append(pose.surface.placed)
            continue
        if centers_ras is not None and axes_ras is not None:
            seed_c, seed_a = pose.seed_points(centers_ras, axes_ras)
        else:
            seed_c = None
            seed_a = None
        if pose.deform is not None:
            fit = pose.deform
            n = np.asarray(fit.pose.normal, dtype=float)   # toward the cavity
            # near-flat fits (kappa ~ 0) leave the sheet normal's sign free:
            # make it agree with the pose normal (which points AWAY from the
            # cavity, fit.py convention) so placed tiles always face inward
            if float(n @ np.asarray(pose.normal_ras, dtype=float)) > 0.0:
                n = -n
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
                   deformable=True, mesh=None, spacing_mm=None,
                   cover=True, lambda_cover=LAMBDA_COVER,
                   margins=False) -> AutoFitResult:
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
    spacing_mm : sequence of 3, optional
        Voxel spacing of the scan; scales the cover pass gates
        (:func:`spacing_tolerance`).  The supported selection is NOT
        affected (its calibration is thin-cut).
    cover : bool
        Run the cover pass (module docstring) on the seeds the supported
        selection leaves inside the implant region.  Default True.
    lambda_cover : float
        Per-tile penalty of the cover pass.
    margins : bool
        Partition margins (plan-localization stage 6): for every selected
        tile, re-run the per-count search without it at the chosen count;
        ``partition_margins[tile_id]`` = best total minus the best total
        without that tile (``inf`` when no other grouping of that count
        exists), ``partition_alternatives[tile_id]`` = that alternative's
        seed groups, ``ambiguous_tiles`` = margin < :data:`AMBIGUOUS_MARGIN`.
        One extra selector run per tile; default off.

    Returns
    -------
    AutoFitResult
        ``tiles`` of the penalised optimum (fulls first, then halves, each by
        descending score), ``score_curve`` for ``n = 0 .. n_max``,
        ``n_selected`` / ``n_expected`` = the chosen count;
        ``tentative_tiles`` / ``unassigned_indices`` / ``clutter_indices``
        from the cover pass; ``all_assigned`` is True iff no seed inside the
        implant region is left unexplained.
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

    tol = spacing_tolerance(spacing_mm)
    result = AutoFitResult(lambda_full=float(lambda_full),
                           lambda_half=float(lambda_half), spacing_tol=tol,
                           score_rule="deformable" if deformable else "chord")
    result.score_curve = [ScorePoint(0, 0.0, 0.0, 0.0, 0, 0)]
    result.rejected_indices = list(range(n))
    result.clutter_indices = list(range(n))
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
        if margins:
            result.partition_margins, result.partition_alternatives = {}, {}
        return result  # no supported tile -> no implant region to cover

    n_max = min(n_max, len(items))
    selector = _PerCountSelector([(it[0], it[1], it[2]) for it in items], n_max)
    best, best_sets = selector.run()
    result.capped = selector.capped

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
    tile_of = {}
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
            tile_of[i] = pose.tile_id
            tiles.append(pose)
            assigned.update(idx)
    result.tiles = tiles
    if margins:
        per_item, c2 = _per_count_margins(items, chosen, n_sel, best[n_sel])
        result.capped = result.capped or c2
        result.partition_margins, result.partition_alternatives = {}, {}
        for i, (margin, alt) in per_item.items():
            tid = tile_of[i]
            result.partition_margins[tid] = margin
            result.partition_alternatives[tid] = None if alt is None else \
                sorted(tuple(int(v) for v in items[j][4]) for j in alt)
            if margin < AMBIGUOUS_MARGIN:
                result.ambiguous_tiles.append(tid)
        result.ambiguous_tiles.sort()
    result.rejected_indices = [i for i in range(n) if i not in assigned]
    result.half_candidates = sorted(
        [(s, idx) for s, idx, _r in leftover_pairs
         if not (assigned & set(idx))],
        key=lambda p: (-p[0], p[1]))
    result.n_selected = n_sel
    result.n_expected = n_sel
    room = (int(max_tiles) - len(tiles)) if max_tiles is not None \
        else (n - len(assigned)) // 3
    _finish(result, centers, axes, dist, assigned, cavity_center, mesh,
            cover=cover, tol=tol, lam=lambda_cover, n_max=room)
    return result


def _finish(result, centers, axes, dist, assigned, cavity_center, mesh,
            cover, tol, lam, n_max):
    """Shared tail of the auto and count-prior paths: cover pass, leftover
    classification, surface cross-check, ``all_assigned``."""
    n = centers.shape[0]
    assigned = set(assigned)
    leftovers = [i for i in range(n) if i not in assigned]
    tile_centers = [p.center_ras for p in result.tiles]
    near, far = _split_leftovers(centers, leftovers, tile_centers)
    if cover and near and n_max >= 1:
        poses, extra, capped = _cover_pass(
            centers, axes, dist, near, cavity_center, tol, lam, n_max,
            next_tile_id=len(result.tiles))
        result.tentative_tiles = poses
        result.capped = result.capped or capped
        assigned |= extra
        leftovers = [i for i in range(n) if i not in assigned]
        tile_centers += [p.center_ras for p in poses]
        near, far = _split_leftovers(centers, leftovers, tile_centers)
    result.unassigned_indices = near
    result.clutter_indices = far
    # pose uncertainty (plan-localization stage 6) for the reported tiles
    # only -- never inside the search.  A triplet-completed tile is skipped:
    # its 4th "seed" was placed ON the model, so s^2 and J^T J would both be
    # optimistic.
    for pose in result.all_tiles:
        if pose.deform is not None and pose.inferred_seed_ras is None:
            pose.deform.compute_uncertainty()
    if mesh is not None and len(getattr(mesh, "faces", [])) > 0:
        from .surface import fit_on_surface

        for pose in result.all_tiles:
            try:
                c, a = pose.seed_points(centers, axes)
                pose.surface = fit_on_surface(mesh, c, a, kind=pose.kind,
                                              init=pose.deform)
            except Exception:
                pose.surface = None
    if result.n_requested is not None:
        result.all_assigned = (result.n_selected >= result.n_requested
                               and not near)
    else:
        result.all_assigned = not near


def fit_tiles_prior(centers_ras, axes_ras, prior: ImplantPrior,
                    cavity_center_ras=None, mesh=None, spacing_mm=None,
                    cover=True, lambda_cover=LAMBDA_COVER,
                    complete_degraded=True, margins=False) -> AutoFitResult:
    """Tile inference driven by what the OR team knows (:class:`ImplantPrior`).

    * count known -> the count-constrained exact fit
      (:func:`gtcore.tiles.fit.fit_tiles`, degraded completion on) scored
      with the bent-tile rule auto mode uses (``score="deformable"``,
      plan-localization stage 4), then -- only if it fell short of the
      count -- the cover pass on the leftovers inside the implant region,
      capped at the shortfall;
    * count unknown -> :func:`fit_tiles_auto` (no half tiles, cover pass).

    ``margins`` forwards to either path (partition margins, stage 6).
    Always returns an :class:`AutoFitResult` so callers read one shape.
    """
    if prior is None or not prior.count_known:
        res = fit_tiles_auto(centers_ras, axes_ras,
                             cavity_center_ras=cavity_center_ras,
                             allow_half=False, mesh=mesh, spacing_mm=spacing_mm,
                             cover=cover, lambda_cover=lambda_cover,
                             margins=margins)
        res.prior = prior
        return res

    from .fit import fit_tiles

    centers = np.asarray(centers_ras, dtype=float).reshape(-1, 3)
    axes = _normalize_axes(axes_ras) if centers.size else \
        np.zeros((0, 3), dtype=float)
    cavity_center = None if cavity_center_ras is None else \
        np.asarray(cavity_center_ras, dtype=float).reshape(3)
    counted = fit_tiles(centers, axes, prior.n_full, prior.n_half,
                        cavity_center_ras=cavity_center,
                        complete_degraded=complete_degraded,
                        score="deformable", margins=margins)
    tol = spacing_tolerance(spacing_mm)
    result = AutoFitResult(spacing_tol=tol, auto=False, prior=prior,
                           n_requested=int(prior.n_full),
                           capped=counted.capped,
                           score_rule=counted.score_rule,
                           partition_margins=counted.partition_margins,
                           partition_alternatives=counted.partition_alternatives,
                           ambiguous_tiles=list(counted.ambiguous_tiles))
    result.tiles = list(counted.tiles)
    for pose in result.tiles:
        if pose.deform is None and pose.kind == "full":
            try:
                c, a = pose.seed_points(centers, axes)
                pose.deform = fit_deformable(c, a, kind=pose.kind)
            except Exception:
                pose.deform = None
        if pose.deform is not None:
            # report the pose of the model that is drawn: to_placed_tiles
            # builds the board tile from the bent-tile fit, so the normal and
            # in-plane axis come from it too (oriented as _deformed_pose does);
            # the centre (seed mean), ids and residual are unchanged
            normal = _orient_normal(np.asarray(pose.deform.pose.normal, float).copy(),
                                    pose.center_ras, cavity_center)
            pose.normal_ras = normal
            pose.axis_ras = _project_in_plane(pose.deform.pose.t1, normal)
    n_full_found = sum(1 for p in result.tiles if p.kind == "full")
    result.n_selected = n_full_found
    result.n_expected = int(prior.n_full) + int(prior.n_half)
    result.score_curve = [ScorePoint(0, 0.0, 0.0, 0.0, 0, 0)]
    result.rejected_indices = list(counted.rejected_indices)
    assigned = {i for p in result.tiles for i in p.seed_indices}
    n = centers.shape[0]
    if n:
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
    else:
        dist = np.zeros((0, 0))
    shortfall = max(0, int(prior.n_full) - n_full_found)
    _finish(result, centers, axes, dist, assigned, cavity_center, mesh,
            cover=cover and shortfall > 0, tol=tol, lam=lambda_cover,
            n_max=shortfall)
    return result
