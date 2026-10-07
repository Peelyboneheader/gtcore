"""Independent reference implementation for reviewing ``gtcore.plan`` (A7).

Written from the problem statement in ``docs/plan-tile-optimize.md`` §2
ONLY -- deliberately without reading the optimizer implementation -- so that
the adversarial tests in ``tests/test_plan_review.py`` check the builders'
code against an independent derivation rather than against itself.

Everything here is intentionally minimal and slow-but-obvious:

* metrics come straight from ``gtcore.dose.engine.dose_at_points(exact=True)``
  (dose is additive over seeds, TG-43U1S2 line source in water, no
  interseed attenuation -- §2 "Dose");
* the conflict test IS ``gtcore.interact.find_overlapping_tiles`` on the pair
  (§1: "The optimizer's conflict definition must agree with this");
* the default target is the +5 mm shell with one-third-adjacent-face-area
  vertex weights (§2 "target sample set T");
* ``brute_force_best`` enumerates every feasible n-subset of a tiny
  candidate list and is therefore an exact reference for V100 on that
  discretized instance.

This is an importable helper, not a test module (no ``test_`` prefix).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gtcore.dose.dvh import outward_normals
from gtcore.dose.engine import dose_at_points
from gtcore.interact import PlacedTile, find_overlapping_tiles

__all__ = [
    "weighted_quantile",
    "reference_metrics",
    "reference_conflict",
    "shell_target",
    "tiles_dose",
    "metrics_for_tiles",
    "brute_force_best",
]


# ------------------------------------------------------------------ metrics
def weighted_quantile(values, weights, q: float) -> float:
    """Weighted ``q``-quantile (``q`` in [0, 1]) of ``values``.

    Convention (stated precisely because D90 depends on it): sort the
    samples ascending, form the cumulative weight ``C_k = sum_{j<=k} w_j``
    normalised by the total weight, and return the first sorted value whose
    normalised cumulative weight is ``>= q``.  This is the weighted analogue
    of ``numpy.percentile(..., method="inverted_cdf")``: it is a value that
    actually occurs in the sample (no interpolation between neighbours), and
    for equal weights it reduces to the ``inverted_cdf`` sample quantile.

    For D90 (``q = 0.10``) this returns the smallest dose ``d`` such that at
    least 10 % of the target weight receives ``<= d`` -- equivalently, at
    least 90 % of the weight receives ``>= d``.  Interpolating conventions
    (numpy's default ``linear``) can differ from this by up to one sample
    spacing in the lower tail, which is why the tests compare D90 with a
    tolerance of 1 % of rx rather than exactly.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0:
        return 0.0
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must not all be zero")
    order = np.argsort(v, kind="stable")
    v_sorted = v[order]
    cum = np.cumsum(w[order]) / total
    idx = int(np.searchsorted(cum, float(q), side="left"))
    idx = min(idx, v_sorted.size - 1)
    return float(v_sorted[idx])


def _weighted_fraction_at_least(doses, weights, level) -> float:
    d = np.asarray(doses, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    return float(w[d >= float(level)].sum() / total)


def reference_metrics(seed_centers, seed_axes, target_points, target_weights,
                      rx_cgy: float, **engine_kwargs) -> Dict[str, float]:
    """§2 metrics on a weighted target from the exact engine.

    Returns ``{"V100", "V150", "V200", "D90", "Dmin", "Dmax", "n"}`` with

    * ``V100`` = sum_m w_m [D_m >= rx] / sum_m w_m  (likewise V150 / V200 at
      1.5 rx / 2 rx; the comparison is ``>=`` as in ``gtcore.dose.dvh``);
    * ``D90`` = weighted 10th percentile of {D_m} per :func:`weighted_quantile`
      (the dose received by at least 90 % of the target weight).

    ``engine_kwargs`` are forwarded to ``dose_at_points`` (e.g.
    ``sk_per_seed_u``, ``elapsed_hours``); ``exact=True`` is always forced.
    Zero seeds gives zero dose everywhere.
    """
    pts = np.asarray(target_points, dtype=float).reshape(-1, 3)
    w = np.asarray(target_weights, dtype=float).reshape(-1)
    if pts.shape[0] != w.shape[0]:
        raise ValueError("target_points and target_weights length mismatch")
    doses = tiles_dose(seed_centers, seed_axes, pts, **engine_kwargs)
    rx = float(rx_cgy)
    return {
        "V100": _weighted_fraction_at_least(doses, w, rx),
        "V150": _weighted_fraction_at_least(doses, w, 1.5 * rx),
        "V200": _weighted_fraction_at_least(doses, w, 2.0 * rx),
        "D90": weighted_quantile(doses, w, 0.10),
        "Dmin": float(doses.min()) if doses.size else 0.0,
        "Dmax": float(doses.max()) if doses.size else 0.0,
        "n": float(pts.shape[0]),
    }


def tiles_dose(seed_centers, seed_axes, points, **engine_kwargs) -> np.ndarray:
    """Exact engine dose [cGy] at ``points`` from the given seeds (0 if none)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centers = np.asarray(seed_centers, dtype=float).reshape(-1, 3)
    axes = np.asarray(seed_axes, dtype=float).reshape(-1, 3)
    if centers.shape[0] == 0 or pts.shape[0] == 0:
        return np.zeros(pts.shape[0])
    engine_kwargs = dict(engine_kwargs)
    engine_kwargs["exact"] = True
    return np.asarray(dose_at_points(centers, axes, pts, **engine_kwargs),
                      dtype=float).reshape(-1)


def _stack_tiles(tiles: Sequence[PlacedTile]):
    tiles = list(tiles)
    if not tiles:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (np.vstack([t.seed_centers for t in tiles]),
            np.vstack([t.seed_axes for t in tiles]))


def metrics_for_tiles(tiles: Sequence[PlacedTile], target_points,
                      target_weights, rx_cgy: float,
                      **engine_kwargs) -> Dict[str, float]:
    """:func:`reference_metrics` for a list of ``PlacedTile``."""
    c, a = _stack_tiles(tiles)
    return reference_metrics(c, a, target_points, target_weights, rx_cgy,
                             **engine_kwargs)


# ---------------------------------------------------------------- conflicts
def reference_conflict(tile_a: PlacedTile, tile_b: PlacedTile,
                       threshold_mm: float = 1.0) -> bool:
    """True when the planner's overlap test flags the pair (§1 contract)."""
    return len(find_overlapping_tiles([tile_a, tile_b],
                                      threshold_mm=threshold_mm)) > 0


def is_feasible(tiles: Sequence[PlacedTile], threshold_mm: float = 1.0) -> bool:
    """No pair conflicts (pairwise non-overlap is the §2 hard constraint)."""
    return len(find_overlapping_tiles(list(tiles),
                                      threshold_mm=threshold_mm)) == 0


# ------------------------------------------------------------------- target
def shell_target(mesh, offset_mm: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Default §2 target: +offset shell vertices with vertex-area weights.

    Points are the mesh vertices pushed ``offset_mm`` outward (away from the
    centroid, i.e. into tissue, same orientation rule as
    ``gtcore.dose.dvh.outward_normals``).  Each vertex's weight is one third
    of the summed area of the faces incident to it (so the weights sum to
    the mesh area).  Vertices with no incident face get zero weight.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    areas = np.asarray(mesh.area_faces, dtype=float)
    weights = np.zeros(verts.shape[0])
    for k in range(3):
        np.add.at(weights, faces[:, k], areas / 3.0)
    points = verts + float(offset_mm) * outward_normals(mesh)
    return points, weights


# ------------------------------------------------------------- brute force
def brute_force_best(candidate_tiles: Sequence[PlacedTile], n: int,
                     target, rx_cgy: float,
                     threshold_mm: float = 1.0,
                     max_candidates: int = 12, max_n: int = 3,
                     **engine_kwargs):
    """Exhaustive best feasible ``n``-subset by V100 (tiny instances only).

    ``target`` is ``(points (M,3), weights (M,))``.  Every ``n``-subset of
    the candidates is enumerated; subsets with any pairwise conflict
    (:func:`reference_conflict`) are discarded; among the rest the one with
    the largest V100 wins, ties broken toward the larger D90 and then the
    lexicographically smallest index tuple (so the result is deterministic).

    Returns ``(best_indices, best_metrics, table)`` where ``table`` is the
    list of ``(indices, metrics)`` for every *feasible* subset (useful for
    proving that a construction is greedy-suboptimal).  Returns
    ``(None, None, [])`` when no feasible subset exists.
    """
    cands = list(candidate_tiles)
    if len(cands) > max_candidates or n > max_n:
        raise ValueError("brute_force_best is for <= %d candidates, n <= %d"
                         % (max_candidates, max_n))
    pts, w = target
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    w = np.asarray(w, dtype=float).reshape(-1)

    # cache per-candidate dose rows: dose is additive over seeds (§2)
    rows = [tiles_dose(t.seed_centers, t.seed_axes, pts, **engine_kwargs)
            for t in cands]
    # cache pairwise conflicts
    conflict = {}
    for i, j in combinations(range(len(cands)), 2):
        conflict[(i, j)] = reference_conflict(cands[i], cands[j], threshold_mm)

    rx = float(rx_cgy)
    table: List[Tuple[Tuple[int, ...], Dict[str, float]]] = []
    best: Optional[Tuple[int, ...]] = None
    best_m: Optional[Dict[str, float]] = None
    for subset in combinations(range(len(cands)), n):
        if any(conflict[(i, j)] for i, j in combinations(subset, 2)):
            continue
        d = np.sum([rows[i] for i in subset], axis=0) if subset else \
            np.zeros(pts.shape[0])
        m = {
            "V100": _weighted_fraction_at_least(d, w, rx),
            "V150": _weighted_fraction_at_least(d, w, 1.5 * rx),
            "V200": _weighted_fraction_at_least(d, w, 2.0 * rx),
            "D90": weighted_quantile(d, w, 0.10),
        }
        table.append((tuple(subset), m))
        if best_m is None or (m["V100"], m["D90"]) > (best_m["V100"], best_m["D90"]):
            best, best_m = tuple(subset), m
    return best, best_m, table


def greedy_forward_reference(candidate_tiles: Sequence[PlacedTile], n: int,
                             target, rx_cgy: float, threshold_mm: float = 1.0,
                             **engine_kwargs):
    """Plain forward greedy on V100 (my own, to certify a construction).

    At each step add the feasible candidate with the largest V100 of the
    partial configuration (ties -> lowest index); stop when ``n`` tiles are
    placed or nothing feasible remains.  Used only to show that a
    constructed instance is greedy-suboptimal *independently* of the
    builders' greedy; the tests then check the builders' greedy as well.
    """
    cands = list(candidate_tiles)
    pts, w = target
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    w = np.asarray(w, dtype=float).reshape(-1)
    rows = [tiles_dose(t.seed_centers, t.seed_axes, pts, **engine_kwargs)
            for t in cands]
    chosen: List[int] = []
    acc = np.zeros(pts.shape[0])
    rx = float(rx_cgy)
    while len(chosen) < n:
        best_i, best_v = None, -1.0
        for i in range(len(cands)):
            if i in chosen:
                continue
            if any(reference_conflict(cands[i], cands[j], threshold_mm)
                   for j in chosen):
                continue
            v = _weighted_fraction_at_least(acc + rows[i], w, rx)
            if v > best_v:
                best_i, best_v = i, v
        if best_i is None:
            break
        chosen.append(best_i)
        acc = acc + rows[best_i]
    return chosen, _weighted_fraction_at_least(acc, w, rx)
