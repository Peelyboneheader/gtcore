"""Conflict graph (section 3 C). Owner: A1 Geometry, branch ``plan/candidates``.

Implements ``build_conflicts(candidates, gap_mm) -> ConflictGraph``:

- pairwise conflicts defined exactly as ``gtcore.interact.find_overlapping_tiles``
  defines overlap: for every pair the result equals
  ``len(find_overlapping_tiles([tile_i, tile_j], threshold_mm=1.0 + gap_mm)) > 0``
  (same footprint sampling, same bounding-sphere skip, same directed hit
  test with the pair's own slack).  ``gap_mm`` composes ADDITIVELY with the
  planner's 1 mm threshold (Open decision 4 in ``docs/optimize-notes.md``);
- a sparse symmetric boolean matrix, diagonal False;
- clique constraints for the MILP: per sampled anchor, a maximal clique of
  the pairwise graph grown greedily (nearest anchors first) from the
  anchor's first candidate inside the anchor neighbourhood (candidates whose
  anchors lie within their own kind's tile diagonal), plus the clique of all
  spins / kinds of the anchor itself.  Every listed clique is verified to be
  a true clique of the pairwise graph, so the clique form can never forbid
  a pairwise-feasible selection.

Tests in ``tests/test_plan_conflicts.py``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ..interact import (
    _OVERLAP_GRID_N,
    _OVERLAP_NORMAL_DOT,
    _footprint_surface,
    _grid_triangles,
)
from . import CONFLICT_GAP_MM, CandidateSet, ConflictGraph
from .candidates import tile_diagonal_mm

PLANNER_THRESHOLD_MM = 1.0
# find_overlapping_tiles' default threshold: footprints closer than this
# are flagged by the planner (edge-to-edge abutment stays legal).


def conflict_threshold_mm(gap_mm: float = CONFLICT_GAP_MM) -> float:
    """Footprint distance below which two candidates conflict:
    the planner's 1 mm plus the optional minimum edge gap."""
    gap = float(gap_mm)
    if gap < 0.0:
        raise ValueError("gap_mm must be >= 0, got %r" % (gap_mm,))
    return PLANNER_THRESHOLD_MM + gap


# ------------------------------------------------------------- pairwise
def _footprints(tiles):
    """Per-tile footprint samples, normals, bounding spheres and slack --
    the quantities ``find_overlapping_tiles`` derives per tile."""
    pts, nrm, centers, radii, slack = [], [], [], [], []
    for t in tiles:
        p, nn = _footprint_surface(t)
        pts.append(p)
        nrm.append(nn)
        c = p.mean(axis=0)
        centers.append(c)
        radii.append(float(np.linalg.norm(p - c[None, :], axis=1).max()))
        slack.append(float(np.linalg.norm(p[_OVERLAP_GRID_N + 1] - p[0])))
    return (pts, nrm, np.asarray(centers, dtype=float),
            np.asarray(radii, dtype=float), np.asarray(slack, dtype=float))


def _point_tri_dist_batched(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """``(N, T)`` min distance from point ``P[n]`` to triangle ``tri[n, t]``.

    Element-for-element port of ``interact._point_triangle_dist`` (Ericson
    5.1.5) with a leading batch axis instead of the all-pairs broadcast, so
    every value equals the planner's for the same point / triangle.
    """
    A = tri[:, :, 0, :]
    B = tri[:, :, 1, :]
    C = tri[:, :, 2, :]
    Pp = P[:, None, :]
    ab = B - A
    ac = C - A
    ap = Pp - A
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = Pp - B
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = Pp - C
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _safe_div(num, den):
        return num / np.where(np.abs(den) < 1e-30, 1e-30, den)

    close = np.empty(A.shape, dtype=float)
    done = np.zeros(d1.shape, dtype=bool)

    def _take(mask, value):
        m = mask & ~done
        if m.any():
            close[m] = np.broadcast_to(value, close.shape)[m]
            done[m] = True

    _take((d1 <= 0) & (d2 <= 0), A)
    _take((d3 >= 0) & (d4 <= d3), B)
    _take((d6 >= 0) & (d5 <= d6), C)
    t_ab = _safe_div(d1, d1 - d3)[..., None]
    _take((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + t_ab * ab)
    t_ac = _safe_div(d2, d2 - d6)[..., None]
    _take((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + t_ac * ac)
    t_bc = _safe_div(d4 - d3, (d4 - d3) + (d5 - d6))[..., None]
    _take((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), B + t_bc * (C - B))
    denom = _safe_div(np.ones_like(va), va + vb + vc)
    _take(np.ones_like(done), A + (vb * denom)[..., None] * ab + (vc * denom)[..., None] * ac)
    return np.linalg.norm(Pp - close, axis=-1)


PAIR_CHUNK = 512
# Candidate pairs per vectorized sample-distance batch: (chunk, 49, 49, 3)
# doubles = 30 MB of transient.

ITEM_CHUNK = 65536
# (sample, triangle) pairs per batched point-triangle pass: ~20 temporaries
# of (chunk, 3) doubles stay around 30 MB.


def _exact_any(P: np.ndarray, D_items: np.ndarray, items_pair: np.ndarray,
               items_sample: np.ndarray, src: np.ndarray, dst: np.ndarray,
               tri_idx: np.ndarray, emax: np.ndarray, threshold: float,
               n_pairs: int) -> np.ndarray:
    """``(n_pairs,)`` bool: for the listed (pair, sample) items, whether the
    sample of tile ``src[pair]`` lies within ``threshold`` of any footprint
    triangle of tile ``dst[pair]`` (the exact stage of ``_directed_hit``).

    ``D_items[n]`` holds the item's distances to the 49 samples of its
    destination tile.  A triangle can be within ``threshold`` of the point
    only if every vertex is within ``threshold + longest edge`` of it (a
    point of a triangle is at most one longest edge from any vertex), so
    only those (item, triangle) pairs go through the exact distance.
    """
    out = np.zeros(n_pairs, dtype=bool)
    if items_pair.size == 0:
        return out
    reach = (threshold + emax[dst[items_pair]])[:, None]
    tmask = (D_items <= reach)[:, tri_idx].all(axis=2)          # (n_items, 72)
    ip, it = np.nonzero(tmask)
    for lo in range(0, ip.size, ITEM_CHUNK):
        a = ip[lo:lo + ITEM_CHUNK]
        t = it[lo:lo + ITEM_CHUNK]
        kp = items_pair[a]
        pts_a = P[src[kp], items_sample[a]]                      # (n, 3)
        tri_b = P[dst[kp][:, None], tri_idx[t]]                  # (n, 3, 3)
        d = _point_tri_dist_batched(pts_a, tri_b[:, None])[:, 0]
        np.logical_or.at(out, kp, d <= threshold)
    return out


def pairwise_conflicts(tiles, threshold_mm: float) -> Tuple[np.ndarray, int]:
    """``(P, 2)`` int array of conflicting pairs ``i < j`` among ``tiles``
    and the number of pairs that passed the bounding-sphere stage.

    Identical pair-by-pair to ``find_overlapping_tiles([t_i, t_j],
    threshold_mm)``: the same footprint surfaces, the same bounding-sphere
    skip, and the same two-stage directed hit test with that pair's slack
    (``max`` of the two tiles' grid-cell diagonals, as the planner computes
    it for a two-tile call), vectorized over pair chunks:

    - stage 1 (sample clouds): per sample of A its nearest sample of B
      (``cKDTree.query`` with an exclusive upper bound ``threshold + slack``)
      and the normal agreement with that sample; a pair with no sample pair
      inside the bound is clear, a pair with an agreeing sample pair within
      ``threshold`` conflicts outright;
    - stage 2 (exact): the agreeing in-band samples against the other
      footprint's triangles through :func:`_point_tri_dist_batched`.
    """
    tiles = list(tiles)
    c = len(tiles)
    if c < 2:
        return np.zeros((0, 2), dtype=int), 0
    pts, nrm, centers, radii, slack = _footprints(tiles)
    P = np.stack(pts)                                   # (C, 49, 3)
    N = np.stack(nrm)                                   # (C, 49, 3)
    tri_idx = _grid_triangles(_OVERLAP_GRID_N)          # (72, 3)
    thr = float(threshold_mm)
    tri_pts = P[:, tri_idx]                             # (C, 72, 3, 3)
    edges = np.stack([tri_pts[:, :, 1] - tri_pts[:, :, 0],
                      tri_pts[:, :, 2] - tri_pts[:, :, 1],
                      tri_pts[:, :, 0] - tri_pts[:, :, 2]], axis=2)
    emax = np.linalg.norm(edges, axis=-1).max(axis=(1, 2))  # longest edge per tile

    reach = 2.0 * float(radii.max()) + thr
    cand = cKDTree(centers).query_pairs(reach, output_type="ndarray")
    if cand.size == 0:
        return np.zeros((0, 2), dtype=int), 0
    cand = cand[np.lexsort((cand[:, 1], cand[:, 0]))]
    gap = np.linalg.norm(centers[cand[:, 0]] - centers[cand[:, 1]], axis=1)
    cand = cand[gap <= radii[cand[:, 0]] + radii[cand[:, 1]] + thr]
    n_close = int(cand.shape[0])
    conflict = np.zeros(n_close, dtype=bool)

    for lo in range(0, n_close, PAIR_CHUNK):
        cc = cand[lo:lo + PAIR_CHUNK]
        k = cc.shape[0]
        I = cc[:, 0]
        J = cc[:, 1]
        diff = P[I][:, :, None, :] - P[J][:, None, :, :]      # (k, 49, 49, 3)
        D = np.sqrt((diff * diff).sum(axis=-1))               # (k, 49, 49)
        bound = (thr + np.maximum(slack[I], slack[J]))[:, None]

        # A -> B: nearest B sample per A sample
        idx_ab = D.argmin(axis=2)
        d_ab = np.take_along_axis(D, idx_ab[:, :, None], axis=2)[:, :, 0]
        nb = np.take_along_axis(N[J], idx_ab[:, :, None], axis=1)
        agree_ab = (N[I] * nb).sum(-1) > _OVERLAP_NORMAL_DOT
        cand_ab = d_ab < bound
        none = ~cand_ab.any(axis=1)
        def_ab = (cand_ab & agree_ab & (d_ab <= thr)).any(axis=1)
        sub_ab = cand_ab & agree_ab

        # B -> A
        idx_ba = D.argmin(axis=1)
        d_ba = np.take_along_axis(D, idx_ba[:, None, :], axis=1)[:, 0, :]
        na = np.take_along_axis(N[I], idx_ba[:, :, None], axis=1)
        agree_ba = (N[J] * na).sum(-1) > _OVERLAP_NORMAL_DOT
        cand_ba = d_ba < bound
        def_ba = (cand_ba & agree_ba & (d_ba <= thr)).any(axis=1)
        sub_ba = cand_ba & agree_ba

        hit_ab = def_ab.copy()
        need_ab = ~none & ~def_ab & sub_ab.any(axis=1)
        if need_ab.any():
            kp, ks = np.nonzero(sub_ab & need_ab[:, None])
            hit_ab |= _exact_any(P, D[kp, ks, :], kp, ks, I, J, tri_idx, emax, thr, k)
        hit_ba = def_ba.copy()
        need_ba = ~none & ~hit_ab & ~def_ba & sub_ba.any(axis=1)
        if need_ba.any():
            kp, ks = np.nonzero(sub_ba & need_ba[:, None])
            hit_ba |= _exact_any(P, D[kp, :, ks], kp, ks, J, I, tri_idx, emax, thr, k)
        conflict[lo:lo + k] = ~none & (hit_ab | hit_ba)

    return cand[conflict].astype(int).reshape(-1, 2), n_close


def _pairs_matrix(n: int, pairs: np.ndarray) -> sp.csr_matrix:
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return sp.csr_matrix((n, n), dtype=bool)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    m = sp.coo_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=(n, n))
    m = m.tocsr()
    m.setdiag(False)
    m.eliminate_zeros()
    m.sort_indices()
    return m.astype(bool)


# -------------------------------------------------------------- cliques
def _greedy_clique(order: np.ndarray, seed: int, adj: sp.csr_matrix) -> np.ndarray:
    """Maximal clique of ``adj`` restricted to ``order`` (a priority-ordered
    id array) containing ``seed``: scan ``order`` and keep every candidate
    adjacent to all members so far.  O(|order| * clique size)."""
    order = np.asarray(order, dtype=int)
    sub = adj[order][:, order].toarray()          # (k, k) bool
    pos = {int(c): k for k, c in enumerate(order)}
    s = pos[int(seed)]
    ok = sub[s].copy()                             # compatible with every member
    members = [s]
    for k in range(order.size):
        if k == s or not ok[k]:
            continue
        members.append(k)
        ok &= sub[k]
    return np.sort(order[np.asarray(members, dtype=int)])


def _is_clique(ids: np.ndarray, adj: sp.csr_matrix) -> bool:
    ids = np.asarray(ids, dtype=int)
    k = ids.size
    if k < 2:
        return False
    sub = adj[ids][:, ids].toarray()
    return bool(sub[~np.eye(k, dtype=bool)].all())


def anchor_cliques(candidates: CandidateSet, pairs: sp.csr_matrix) -> List[np.ndarray]:
    """Clique constraints from anchor neighbourhoods (section 3 C).

    For every sampled anchor ``a``: (1) all candidates with ``anchor_id ==
    a`` (every spin and kind of one anchor), reduced greedily to a true
    clique if the geometry ever disagrees; (2) the neighbourhood
    ``N_a`` = candidates whose anchor lies within their own kind's tile
    diagonal of ``a``'s anchor, ordered by anchor distance (ties by id),
    grown greedily from ``a``'s first candidate into a maximal clique of
    the pairwise graph within ``N_a``.  Only true cliques of size >= 2 are
    returned, deduplicated, sorted by first id.
    """
    n = len(candidates)
    anchor_ids = np.asarray(candidates.anchor_ids, dtype=int)
    anchors = np.asarray(candidates.anchors, dtype=float)
    kinds = np.asarray(candidates.kinds, dtype=object)
    diag = np.asarray([tile_diagonal_mm(k) for k in kinds], dtype=float)
    r_max = float(diag.max()) if n else 0.0

    # first candidate per anchor (lowest id = kinds[0], spin 0 when accepted)
    first_of: Dict[int, int] = {}
    for c in range(n):
        first_of.setdefault(int(anchor_ids[c]), c)
    tree = cKDTree(anchors)

    seen = set()
    out: List[np.ndarray] = []

    def push(q: np.ndarray):
        q = np.sort(np.asarray(q, dtype=int))
        key = tuple(q.tolist())
        if q.size >= 2 and key not in seen and _is_clique(q, pairs):
            seen.add(key)
            out.append(q)

    for a in sorted(first_of):
        seed = first_of[a]
        same = np.flatnonzero(anchor_ids == a)
        if same.size >= 2:
            if _is_clique(same, pairs):
                push(same)
            else:
                push(_greedy_clique(same, seed, pairs))
        pos = anchors[seed]
        near = np.asarray(tree.query_ball_point(pos, r_max), dtype=int)
        d = np.linalg.norm(anchors[near] - pos[None, :], axis=1)
        near = near[d <= diag[near] + 1e-9]
        if near.size < 2:
            continue
        d = np.linalg.norm(anchors[near] - pos[None, :], axis=1)
        order = near[np.lexsort((near, d))]
        push(_greedy_clique(order, seed, pairs))

    out.sort(key=lambda q: (int(q[0]), q.size, tuple(q.tolist())))
    return out


# ------------------------------------------------------------------ build
def build_conflicts(candidates: CandidateSet, gap_mm: float = CONFLICT_GAP_MM
                    ) -> ConflictGraph:
    """Pairwise conflicts (the planner's overlap rule at threshold
    ``1 mm + gap_mm``) and anchor-neighbourhood cliques for ``candidates``.

    O(C log C) bounding-sphere candidates, exact footprint tests only for
    close pairs, O(A * neighbourhood) clique growth.
    """
    if not isinstance(candidates, CandidateSet):
        raise TypeError("build_conflicts expects a CandidateSet")
    n = len(candidates)
    thr = conflict_threshold_mm(gap_mm)
    pairs_arr, _n_exact = pairwise_conflicts(candidates.tiles, thr)
    pairs = _pairs_matrix(n, pairs_arr)
    cliques = anchor_cliques(candidates, pairs) if n else []
    return ConflictGraph(n=n, pairs=pairs, cliques=cliques, gap_mm=float(gap_mm))


__all__ = ["PLANNER_THRESHOLD_MM", "PAIR_CHUNK", "ITEM_CHUNK", "conflict_threshold_mm",
           "pairwise_conflicts", "anchor_cliques", "build_conflicts"]
