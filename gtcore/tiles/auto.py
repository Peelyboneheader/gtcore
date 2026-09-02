"""Count-free tile configuration search (``n_full_tiles="auto"``).

When the implant count is unknown or untrusted, the number of tiles has to be
*inferred* from the seed cloud.  The candidate tiles are the same
gate-passing quads and pairs :mod:`gtcore.tiles.fit` enumerates (each with a
positive geometric score in ``(0, QUAD_BASE]`` / ``(0, PAIR_BASE]``); this
module replaces the exact-count assignment with model selection:

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

Penalty calibration (2026-09, plan Step 2): true tiles score >= 4.8 on the
8-tile printed phantom (including a strongly axis-fanned one), >= 5.9 on the
post-op case and >= 7.1 on the synthetic sweep; disjoint junk quads that
pass the gates are rare and score low.  ``LAMBDA_FULL = 3.5`` sits below
every observed real tile with margin.  A half tile pays the SAME penalty:
it has the same pose parameters as a full tile but explains half the data,
and with a smaller penalty the optimiser happily splits a genuine
(deformed) quad into two near-perfect pairs.

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
from typing import List, Optional

import numpy as np

from .fit import (
    DIAG_MAX_MM,
    SIDE_MIN_MM,
    TileFitResult,
    _enumerate_pairs,
    _enumerate_quads,
    _full_pose,
    _half_pose,
    _normalize_axes,
)

__all__ = ["LAMBDA_FULL", "LAMBDA_HALF", "ScorePoint", "AutoFitResult",
           "fit_tiles_auto"]

LAMBDA_FULL = 3.5
LAMBDA_HALF = LAMBDA_FULL

_SEARCH_NODE_CAP = 200000


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
        # bound: can any count c + m still be improved from here?
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


# --------------------------------------------------------------------- main
def fit_tiles_auto(centers_ras, axes_ras, cavity_center_ras=None,
                   allow_half=False, lambda_full=LAMBDA_FULL,
                   lambda_half=LAMBDA_HALF, max_tiles=None) -> AutoFitResult:
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

    items = [(s, frozenset(idx), float(lambda_full), "full", idx, r)
             for s, idx, r in quads]
    items += [(s, frozenset(idx), float(lambda_half), "half", idx, r)
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
            s, _fs, _pen, k, idx, resid = items[i]
            if k != kind:
                continue
            if kind == "full":
                pose = _full_pose(len(tiles), idx, centers, axes, resid,
                                  cavity_center)
            else:
                pose = _half_pose(len(tiles), idx, centers, axes, resid,
                                  cavity_center)
            tiles.append(pose)
            assigned.update(idx)
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
