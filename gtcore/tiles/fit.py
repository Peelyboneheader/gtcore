"""Tile-configuration inference: group seed candidates into GammaTiles.

The OR team knows exactly how many full (4-seed) and half (2-seed) tiles went
into the cavity; those counts are trusted inputs.  This module takes the seed
*candidates* found by :mod:`gtcore.seeds.detect` -- which deliberately
over-detects (surgical clips, dense-bone spikes, stray metal all pass) -- and
finds the combination of ``n_full`` disjoint 4-seed quads and ``n_half``
disjoint 2-seed pairs that best matches the manufactured tile geometry.
Everything left over lands in ``rejected_indices``: this global assignment is
the mechanism that discards false-positive candidates, because a clip has no
partners at tile spacing while true seeds do.

Geometry model
--------------
A GammaTile holds 4 Cs-131 seeds on the corners of a 10 mm square, seed axes
parallel within the tile.  The collagen tile *conforms* to the curved cavity
wall, so an implanted tile is a warped quad: sides contract to roughly
8-10.5 mm chords, diagonals to roughly 11.5-14.5 mm, coplanarity is only
approximate, and wall curvature fans the seed axes apart by up to ~30 deg.
A surgeon-cut half tile carries one column of the grid: 2 seeds at the same
spacing.  All windows below carry slack for that deformation plus ~0.5 mm of
localization noise per seed.

Scoring is consistency-first (sides against their own mean, diagonals against
``mean_side * sqrt(2)``) rather than against the nominal 10 mm grid, so a
strongly conformed tile is not penalized for the contraction itself.

Selection is an exact branch-and-bound over candidate quads then candidate
pairs (candidate counts are small, N <= ~40), maximizing tiles placed first
and total geometric score second, with a node cap that degrades gracefully to
the greedy answer on pathological inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List

import numpy as np

__all__ = ["TilePose", "TileFitResult", "fit_tiles"]

# ---------------------------------------------------------------- tolerances
# Chord windows (mm).  Sides of a wall-conformed full tile and the seed pair
# of a half tile share the same manufactured 10 mm spacing, hence one window.
# The lower bound leaves room for one member seed being displaced ~1 mm by a
# merged-blob split (adjacent tiles touching); scoring, not the gate, is what
# separates real tiles from junk.
SIDE_MIN_MM = 6.3
SIDE_MAX_MM = 11.8
DIAG_MIN_MM = 9.8
DIAG_MAX_MM = 16.0

# Hard geometric gates on a candidate quad.
MAX_GRID_RESIDUAL_MM = 2.0      # RMS chord deviation from the fitted grid
MAX_PLANE_RMS_MM = 1.8          # RMS out-of-plane deviation of the 4 seeds
# Axis parallelism: nominal tolerance is 25 deg, but wall conforming alone
# fans truth axes up to ~32 deg and PCA on a bloomed capsule adds ~10 deg.  A
# merged-blob split can hand ONE seed an essentially random axis (near 90 deg
# observed), which corrupts the 3 pairwise angles involving it -- so a quad is
# *gated* on the 3rd-smallest of its 6 pairwise angles (unaffected by a single
# bad axis) but *penalized* on all 6: the corrupted axis lands somewhere in
# every competing grouping, and the grouping that keeps it with its true
# tile-mates diverges least in total.
AXIS_SOFT_DEG = 25.0            # penalty-free axis divergence
AXIS_HARD_DEG = 55.0            # reject beyond this (outlier-trimmed for quads)
PAIR_AXIS_HARD_DEG = 70.0       # 2 seeds give no outlier redundancy

# Score weights.  Bases keep every accepted tile's score positive so that
# "place more tiles" and "raise the score" never fight; a pair's base is half
# a quad's (2 seeds vs 4).
QUAD_BASE = 8.0
PAIR_BASE = 4.0
W_RESIDUAL = 1.5
W_PLANE = 0.8
W_AXIS = 2.0                    # per radian beyond AXIS_SOFT_DEG
W_PAIR_DIST = 0.2               # per mm from the expected pair chord
PAIR_EXPECT_MM = 8.6            # wall-conformed 10 mm pair chord (~8.3-9)
# In a manufactured (half) tile the seed axes run parallel or perpendicular
# to the seed-pair separation; a pair crossing two adjacent tiles sits near
# 45 deg.  Penalize deviation from the nearest of {0, 90} deg.
PAIR_ALIGN_FREE_DEG = 15.0
W_PAIR_ALIGN = 2.0              # per radian beyond PAIR_ALIGN_FREE_DEG

_SEARCH_NODE_CAP = 200000


# ------------------------------------------------------------------- results
@dataclass
class TilePose:
    """One recovered tile: which candidates it owns and its fitted pose."""

    tile_id: int
    kind: str                       # "full" or "half"
    seed_indices: List[int]         # indices into the candidate arrays
    center_ras: np.ndarray          # (3,) mm -- mean of the member seeds
    normal_ras: np.ndarray          # (3,) unit; away from the cavity centre
                                    # when one was supplied
    axis_ras: np.ndarray            # (3,) unit tile t1 axis, in-plane
    residual_mm: float              # RMS chord deviation from the ideal grid
    degraded: bool = False          # True when recovered only by the implant
                                    # count constraint (crumpled/folded tile
                                    # outside the normal geometry gates)

    def __post_init__(self):
        self.center_ras = np.asarray(self.center_ras, dtype=float).reshape(3)
        self.normal_ras = np.asarray(self.normal_ras, dtype=float).reshape(3)
        self.axis_ras = np.asarray(self.axis_ras, dtype=float).reshape(3)


@dataclass
class TileFitResult:
    """Outcome of the global candidate-to-tile assignment.

    ``all_assigned`` is True when exactly ``n_full`` full and ``n_half`` half
    tiles were recovered; leftover candidates (clips, bone spikes, decoys) in
    ``rejected_indices`` do not clear it.
    """

    tiles: List[TilePose] = field(default_factory=list)
    rejected_indices: List[int] = field(default_factory=list)
    n_expected: int = 0
    all_assigned: bool = False


# ------------------------------------------------------------------ helpers
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else np.array([0.0, 0.0, 1.0])


def _normalize_axes(axes):
    axes = np.asarray(axes, dtype=float).reshape(-1, 3).copy()
    for i in range(axes.shape[0]):
        axes[i] = _unit(axes[i])
    return axes


def _axis_angle_deg(a, b):
    """Angle between two undirected axes (sign of each is arbitrary)."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def _axis_penalty(angles_deg):
    """Soft parallelism penalty: radians of divergence beyond the free zone."""
    excess = [max(0.0, a - AXIS_SOFT_DEG) for a in angles_deg]
    return W_AXIS * float(np.deg2rad(np.mean(excess))) if excess else 0.0


def _mean_axis(axes):
    """Mean of undirected unit axes, signs aligned to the first."""
    ref = axes[0]
    acc = np.zeros(3)
    for a in axes:
        acc += a if float(a @ ref) >= 0.0 else -a
    return _unit(acc)


def _plane_fit(pts):
    """(normal, rms) of the best-fit plane through ``pts`` (N, 3)."""
    d = pts - pts.mean(axis=0)
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    normal = _unit(vt[-1])
    rms = float(s[-1]) / np.sqrt(pts.shape[0])
    return normal, rms


def _orient_normal(normal, center, cavity_center):
    if cavity_center is not None:
        if float(normal @ (center - cavity_center)) < 0.0:
            normal = -normal
    return normal


def _project_in_plane(v, normal):
    return _unit(np.asarray(v, float) - float(v @ normal) * normal)


# ------------------------------------------------------- candidate generation
def _quad_geometry(idx, centers, axes, dist):
    """Score one 4-candidate combination as a deformed tile quad.

    Returns ``(score, residual_mm)`` or ``None`` when the combination cannot
    be a tile.
    """
    i, j, k, l = idx
    pair_ids = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
    d6 = np.array([dist[a, b] for a, b in pair_ids])
    order = np.argsort(d6)
    sides = d6[order[:4]]
    diags = d6[order[4:]]

    if sides.min() < SIDE_MIN_MM or sides.max() > SIDE_MAX_MM:
        return None
    if diags.min() < DIAG_MIN_MM or diags.max() > DIAG_MAX_MM:
        return None

    # Quad topology: the two longest chords are the diagonals, and in a true
    # quadrilateral they share no vertex (together they cover all 4 seeds).
    d1, d2 = (pair_ids[int(o)] for o in order[4:])
    if len({d1[0], d1[1], d2[0], d2[1]}) != 4:
        return None

    # Deformation-tolerant grid residual: sides against their own mean,
    # diagonals against mean_side * sqrt(2).
    s = float(sides.mean())
    ideal = np.array([s] * 4 + [s * np.sqrt(2.0)] * 2)
    fitted = np.concatenate([sides, diags])
    residual = float(np.sqrt(np.mean((fitted - ideal) ** 2)))
    if residual > MAX_GRID_RESIDUAL_MM:
        return None

    _, plane_rms = _plane_fit(centers[list(idx)])
    if plane_rms > MAX_PLANE_RMS_MM:
        return None

    # Gate outlier-trimmed (one corrupted axis inflates exactly 3 of the 6
    # pairwise angles, leaving the 3rd-smallest clean); penalize untrimmed.
    angles = sorted(_axis_angle_deg(axes[a], axes[b]) for a, b in pair_ids)
    if angles[2] > AXIS_HARD_DEG:
        return None

    score = (
        QUAD_BASE
        - W_RESIDUAL * residual
        - W_PLANE * plane_rms
        - _axis_penalty(angles)
    )
    if score <= 0.0:
        return None
    return score, residual


def _enumerate_quads(centers, axes, dist, link):
    """All 4-candidate combinations that pass the quad gates, scored."""
    n = centers.shape[0]
    quads = []
    for idx in combinations(range(n), 4):
        # Every chord of a tile quad lies inside [SIDE_MIN, DIAG_MAX]; prune
        # on the precomputed link matrix before any arithmetic.
        i, j, k, l = idx
        if not (link[i, j] and link[i, k] and link[i, l]
                and link[j, k] and link[j, l] and link[k, l]):
            continue
        geom = _quad_geometry(idx, centers, axes, dist)
        if geom is not None:
            score, residual = geom
            quads.append((score, idx, residual))
    return quads


def _enumerate_pairs(centers, axes, dist):
    """All 2-candidate combinations that pass the half-tile gates, scored."""
    n = centers.shape[0]
    pairs = []
    for i, j in combinations(range(n), 2):
        d = dist[i, j]
        if d < SIDE_MIN_MM or d > SIDE_MAX_MM:
            continue
        angle = _axis_angle_deg(axes[i], axes[j])
        if angle > PAIR_AXIS_HARD_DEG:
            continue
        # Manufactured alignment: each seed axis runs parallel or
        # perpendicular to the pair separation (which of the two depends on
        # how the surgeon cut the tile); crossings of adjacent tiles land
        # near 45 deg and are penalized hard.
        sep = _unit(centers[j] - centers[i])
        align = []
        for a in (axes[i], axes[j]):
            to_sep = _axis_angle_deg(a, sep)
            align.append(min(to_sep, 90.0 - to_sep))
        align_pen = W_PAIR_ALIGN * float(np.deg2rad(
            np.mean([max(0.0, a - PAIR_ALIGN_FREE_DEG) for a in align])))
        residual = abs(d - PAIR_EXPECT_MM)  # deviation from the ideal chord
        score = (
            PAIR_BASE
            - W_PAIR_DIST * abs(d - PAIR_EXPECT_MM)
            - _axis_penalty([angle])
            - align_pen
        )
        if score <= 0.0:
            continue
        pairs.append((score, (i, j), residual))
    return pairs


# ---------------------------------------------------------- global selection
class _Selector:
    """Exact-count disjoint selection of quads then pairs.

    Branch-and-bound maximizing ``(tiles placed, total score)`` -- encoded as
    ``count * BIG + score``, valid because every accepted tile score lies in
    ``(0, QUAD_BASE]``.  Items arrive sorted by descending score so the greedy
    solution is explored first and the bound prunes hard; a node cap keeps
    adversarial inputs finite (the incumbent is then at least the greedy
    answer).
    """

    _BIG = 1000.0

    def __init__(self, quads, pairs, n_full, n_half):
        self.quads = quads
        self.pairs = pairs
        self.n_full = n_full
        self.n_half = n_half
        self.best_value = -1.0
        self.best = ([], [])
        self.nodes = 0
        # Suffix maxima for optimistic score bounds. Rows are capped at the
        # number of usable slots: without the cap the table is O(M^2) and a
        # dense junk cluster (coarse scans emit thousands of gated quads)
        # hangs in the table build before the node cap can ever help.
        self.quad_suffix = self._suffix([q[0] for q in quads], n_full)
        self.pair_suffix = self._suffix([p[0] for p in pairs], n_half)

    @staticmethod
    def _suffix(scores, max_slots):
        """suffix[i][m]: best possible sum of m items from position i on,
        for m up to ``max_slots`` (more can never be used)."""
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

    def _bound_tail(self, suffix, pos, want):
        row = suffix[pos]
        m = min(want, len(row) - 1)
        return m, row[m]

    def run(self):
        self._quad_dfs(0, [], frozenset(), 0.0)
        return self.best

    def _quad_dfs(self, pos, chosen, used, score):
        self.nodes += 1
        if self.nodes > _SEARCH_NODE_CAP:
            return
        if len(chosen) == self.n_full or pos == len(self.quads):
            self._pair_dfs(0, chosen, [], used, score)
            return
        # Optimistic bound: fill remaining quad and pair slots from here.
        mq, sq = self._bound_tail(self.quad_suffix, pos,
                                  self.n_full - len(chosen))
        mp, sp = self._bound_tail(self.pair_suffix, 0, self.n_half)
        bound = (len(chosen) + mq + mp) * self._BIG + score + sq + sp
        if bound <= self.best_value:
            return
        q_score, q_idx, _ = self.quads[pos]
        if not (used & frozenset(q_idx)):
            self._quad_dfs(pos + 1, chosen + [pos], used | frozenset(q_idx),
                           score + q_score)
        self._quad_dfs(pos + 1, chosen, used, score)

    def _pair_dfs(self, pos, quads_chosen, pairs_chosen, used, score):
        self.nodes += 1
        if self.nodes > _SEARCH_NODE_CAP:
            return
        if len(pairs_chosen) == self.n_half or pos == len(self.pairs):
            value = (len(quads_chosen) + len(pairs_chosen)) * self._BIG + score
            if value > self.best_value:
                self.best_value = value
                self.best = (list(quads_chosen), list(pairs_chosen))
            return
        mp, sp = self._bound_tail(self.pair_suffix, pos,
                                  self.n_half - len(pairs_chosen))
        bound = (len(quads_chosen) + len(pairs_chosen) + mp) * self._BIG \
            + score + sp
        if bound <= self.best_value:
            return
        p_score, p_idx, _ = self.pairs[pos]
        if not (used & frozenset(p_idx)):
            self._pair_dfs(pos + 1, quads_chosen, pairs_chosen + [pos],
                           used | frozenset(p_idx), score + p_score)
        self._pair_dfs(pos + 1, quads_chosen, pairs_chosen, used, score)


# ------------------------------------------------------------- pose building
def _full_pose(tile_id, idx, centers, axes, residual, cavity_center):
    pts = centers[list(idx)]
    center = pts.mean(axis=0)
    normal, _ = _plane_fit(pts)
    normal = _orient_normal(normal, center, cavity_center)
    t1 = _project_in_plane(_mean_axis([axes[i] for i in idx]), normal)
    return TilePose(
        tile_id=tile_id, kind="full", seed_indices=list(idx),
        center_ras=center, normal_ras=normal, axis_ras=t1,
        residual_mm=residual,
    )


def _half_pose(tile_id, idx, centers, axes, residual, cavity_center):
    """Pose of a 2-seed half tile.

    Two seeds under-determine a plane: the only measured in-plane direction
    is the pair separation.  A conformed tile lies flat on the cavity wall,
    whose local normal is well approximated by the outward radial direction
    from the cavity centre -- so when that centre is available it IS the
    normal estimate (on a lumpy wall the seed-pair chord is not exactly
    tangent, so constraining perpendicularity to it would tilt the estimate
    by the wall slope).  The noisy per-seed PCA axes only serve as a
    fallback via their cross product when no cavity geometry was supplied.
    """
    i, j = idx
    center = 0.5 * (centers[i] + centers[j])
    t2 = _unit(centers[j] - centers[i])           # pair separation direction
    t1 = _mean_axis([axes[i], axes[j]])           # seed long axis
    normal = np.zeros(3)
    if cavity_center is not None:
        normal = center - cavity_center
    if np.linalg.norm(normal) < 1e-6:
        normal = np.cross(t1, t2)
    if np.linalg.norm(normal) < 0.2:
        # Axes nearly parallel to the pair direction and no usable cavity
        # geometry: pick any direction perpendicular to the pair.
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(helper @ t2)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        normal = np.cross(t2, helper)
    normal = _orient_normal(_unit(normal), center, cavity_center)
    t1 = _project_in_plane(t1, normal)
    return TilePose(
        tile_id=tile_id, kind="half", seed_indices=list(idx),
        center_ras=center, normal_ras=normal, axis_ras=t1,
        residual_mm=residual,
    )


# --------------------------------------------------------------------- main
def _complete_degraded_quads(centers, axes, dist, leftovers, missing_full,
                             tiles, assigned, cavity_center):
    """Count-constrained completion for tiles the geometry gates reject.

    A physically crumpled tile (observed on the 8-tile printed phantom: a
    folded collagen tile with two sides squeezed to ~5 mm) fails every chord
    gate, yet the implant count says a tile is missing and the leftover
    candidates form a compact, axis-coherent group -- there, the count
    constraint itself is the evidence. Requirements per group of 4: all six
    pairwise chords in [3.5, 18] mm and mean pairwise axis |dot| >= 0.6.
    Groups are taken best-first by (coherence, compactness); accepted tiles
    carry ``degraded=True`` and a consistency residual that will be large.
    """
    while missing_full > 0 and len(leftovers) >= 4:
        best = None
        for combo in combinations(sorted(leftovers), 4):
            chords = [dist[a, b] for a, b in combinations(combo, 2)]
            if min(chords) < 3.5 or max(chords) > 18.0:
                continue
            dots = [abs(float(np.dot(axes[a], axes[b])))
                    for a, b in combinations(combo, 2)]
            if float(np.mean(dots)) < 0.6:  # coherence is a GATE, not a rank
                continue
            # non-degenerate: a line of clips is compact and perfectly
            # coherent, but has no planar extent -- require the 2nd singular
            # value of the centred points to show a real 2D footprint
            pts = centers[list(combo)]
            sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
            if float(sv[1]) < 1.5:
                continue
            sides = sorted(chords)[:4]
            diags = sorted(chords)[4:]
            ms = float(np.mean(sides))
            resid = float(np.sqrt(np.mean(
                [(s - ms) ** 2 for s in sides]
                + [(d - ms * np.sqrt(2.0)) ** 2 for d in diags])))
            # rank by grid-likeness, then compactness; smallest-index
            # combo as the deterministic tie-break (selector convention)
            key = (resid, float(np.sum(chords)), combo)
            if best is None or key < best[0]:
                best = (key, combo, resid)
        if best is None:
            break
        _, combo, resid = best
        pose = _full_pose(len(tiles), list(combo), centers, axes, resid,
                         cavity_center)
        pose.degraded = True
        tiles.append(pose)
        assigned.update(combo)
        leftovers.difference_update(combo)
        missing_full -= 1
    return missing_full


def fit_tiles(centers_ras, axes_ras, n_full, n_half=0, cavity_center_ras=None,
              complete_degraded=False):
    """Assign seed candidates to ``n_full`` full and ``n_half`` half tiles.

    Parameters
    ----------
    centers_ras : (N, 3) array
        Candidate seed centres in RAS mm (:class:`SeedCandidates.centers_ras`).
    axes_ras : (N, 3) array
        Candidate seed long axes; per-axis sign is arbitrary.
    n_full, n_half : int
        Implanted tile counts reported by the OR team.
    cavity_center_ras : (3,) array, optional
        Resection-cavity centre; when given, every tile normal is flipped to
        point away from it (out of the cavity, into the wall). Without it the
        normal's sign is whatever the SVD plane fit returns -- do not rely on
        its orientation in that case.
    complete_degraded : bool
        Count-constrained completion for FULL tiles whose geometry fails the
        normal gates (physically crumpled/folded tiles): when fewer than
        ``n_full`` quads are found and the leftover pool is about the size the
        missing tiles explain, compact, axis-coherent, non-collinear 4-groups
        are accepted with ``degraded=True``. Missing HALF tiles are never
        completed. Default False.

    Returns
    -------
    TileFitResult
        ``tiles`` ordered full-then-half by descending fit score, with any
        degraded-completion fulls appended LAST (after the halves);
        ``rejected_indices`` holds every unassigned candidate (ascending);
        ``all_assigned`` is True iff the requested tile counts were met.
        Deterministic: identical inputs give identical output.
    """
    centers = np.asarray(centers_ras, dtype=float).reshape(-1, 3)
    axes = _normalize_axes(axes_ras) if centers.size else \
        np.zeros((0, 3), dtype=float)
    if axes.shape[0] != centers.shape[0]:
        raise ValueError(
            "centers_ras and axes_ras disagree: %d vs %d candidates"
            % (centers.shape[0], axes.shape[0])
        )
    n_full = int(n_full)
    n_half = int(n_half)
    if n_full < 0 or n_half < 0:
        raise ValueError("tile counts must be non-negative")
    cavity_center = None if cavity_center_ras is None else \
        np.asarray(cavity_center_ras, dtype=float).reshape(3)

    n = centers.shape[0]
    n_expected = n_full + n_half
    if n_expected == 0 or n == 0:
        return TileFitResult(
            tiles=[], rejected_indices=list(range(n)),
            n_expected=n_expected, all_assigned=(n_expected == 0),
        )

    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    link = (dist >= SIDE_MIN_MM) & (dist <= DIAG_MAX_MM)

    quads = _enumerate_quads(centers, axes, dist, link) if n_full else []
    pairs = _enumerate_pairs(centers, axes, dist) if n_half else []
    # Descending score, index tuple as the deterministic tie-break.
    quads.sort(key=lambda q: (-q[0], q[1]))
    pairs.sort(key=lambda p: (-p[0], p[1]))

    chosen_q, chosen_p = _Selector(quads, pairs, n_full, n_half).run()

    tiles = []
    assigned = set()
    for qi in chosen_q:
        score, idx, residual = quads[qi]
        tiles.append(_full_pose(len(tiles), idx, centers, axes, residual,
                                cavity_center))
        assigned.update(idx)
    for pi in chosen_p:
        score, idx, residual = pairs[pi]
        tiles.append(_half_pose(len(tiles), idx, centers, axes, residual,
                                cavity_center))
        assigned.update(idx)

    n_full_found = len(chosen_q)
    if complete_degraded and n_full_found < n_full:
        leftovers = set(range(n)) - assigned
        # only fire when the leftover pool is about the size the missing
        # tiles explain -- a sea of unexplained candidates means something
        # else is wrong and silent completion would hide it
        if len(leftovers) <= 4 * (n_full - n_full_found) + 4:
            missing = _complete_degraded_quads(
                centers, axes, dist, leftovers, n_full - n_full_found,
                tiles, assigned, cavity_center)
            n_full_found = n_full - missing

    rejected = [i for i in range(n) if i not in assigned]
    return TileFitResult(
        tiles=tiles,
        rejected_indices=rejected,
        n_expected=n_expected,
        all_assigned=(n_full_found == n_full and len(chosen_p) == n_half),
    )
