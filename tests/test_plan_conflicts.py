"""Tests for ``gtcore.plan.conflicts`` (A1 Geometry, branch plan/candidates).

Section 3 C: the conflict definition equals the planner's own
``find_overlapping_tiles`` pair by pair; symmetry; no self loops; same
anchor / other spin always conflicts; anchors > 30 mm apart on a flat wall
never conflict; every clique is a true clique and the clique and pairwise
forms agree on feasibility; the toy instance's matrix is reproduced; a
positive ``gap_mm`` adds conflicts monotonically.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import gtcore.plan as plan
import plan_fixtures as pf
from gtcore.interact import _point_triangle_dist, conform_tile, find_overlapping_tiles, snap_to_wall
from gtcore.plan import CandidateSet, ConflictGraph
from gtcore.plan import candidates as C
from gtcore.plan import conflicts as K


@pytest.fixture(scope="module")
def toy():
    return pf.toy_instance()


@pytest.fixture(scope="module")
def flat():
    mesh = pf.flat_wall_mesh(size_mm=90.0)
    return mesh, pf.flat_wall_top_faces(mesh)


@pytest.fixture(scope="module")
def flat_graph(flat):
    mesh, top = flat
    # no fallback points at all: on a flat wall every footprint fit is then
    # exact (a single fallback corner makes interact._footprint_surface's
    # quadratic bulge by tens of mm -- see the A1 notes)
    cs = C.build_candidates(mesh, h_mm=7.0, n_spins=3, kinds=("full", "half"),
                            eligible_faces=top, min_fraction_on_wall=1.0)
    return cs, K.build_conflicts(cs)


FOOTPRINT_SANE_MM = 20.0
# A conformed footprint's samples lie within ~14.1 mm (corner radius) of its
# centre; interact._footprint_surface's quadratic fit is ill-conditioned at
# the |u| == |v| fit points and can balloon to hundreds of mm on curved
# walls.  Tests that state geometric facts skip those tiles and assert the
# exceptions carry this signature.


def _footprint_radii(cs):
    return K._footprints(cs.tiles)[3]


@pytest.fixture(scope="module")
def cavity_graph():
    from gtcore.phantom import make_head_phantom
    from gtcore.segment import mask_to_mesh
    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    cs = C.build_candidates(mesh, h_mm=7.0, n_spins=2)
    return mesh, cs, K.build_conflicts(cs)


def _public(cs, i, j, gap=0.0):
    return len(find_overlapping_tiles([cs.tiles[i], cs.tiles[j]],
                                      threshold_mm=1.0 + gap)) > 0


# ------------------------------------------------------------ structure
def test_symmetric_no_self_loops(flat_graph, cavity_graph):
    for cs, g in (flat_graph, cavity_graph[1:]):
        assert isinstance(g, ConflictGraph) and g.n == len(cs)
        assert sp.issparse(g.pairs) and g.pairs.dtype == bool
        assert (g.pairs != g.pairs.T).nnz == 0
        assert not g.pairs.diagonal().any()
        assert g.count_pairs() > 0
        assert g.gap_mm == 0.0


def test_same_anchor_other_spin_or_kind_conflicts(flat_graph, cavity_graph):
    for cs, g in (flat_graph, cavity_graph[1:]):
        radii = _footprint_radii(cs)
        n_checked = 0
        for a in np.unique(cs.anchor_ids):
            ids = np.flatnonzero(cs.anchor_ids == a)
            if ids.size < 2:
                continue
            sub = g.pairs[ids][:, ids].toarray()
            off = ~np.eye(ids.size, dtype=bool)
            if np.all(radii[ids] < FOOTPRINT_SANE_MM):
                assert sub[off].all(), (a, ids)
                n_checked += 1
                # ... and that same-anchor set is covered by a listed clique
                assert any(set(ids.tolist()) <= set(q.tolist()) for q in g.cliques)
            elif not sub[off].all():
                # the planner's own rule misses it only for ballooned footprints
                bad = np.argwhere(~sub & off)
                assert all(max(radii[ids[x]], radii[ids[y]]) >= FOOTPRINT_SANE_MM
                           for x, y in bad)
        assert n_checked >= 10


def test_far_apart_on_flat_wall_never_conflict(flat, flat_graph):
    cs, g = flat_graph
    d = np.linalg.norm(cs.anchors[:, None] - cs.anchors[None], axis=2)
    far = np.triu(d > 30.0, 1)
    assert far.any()
    assert not g.pairs.toarray()[far].any()
    # explicit pair: two tiles 35 mm apart, every spin combination
    mesh, _top = flat
    tiles, spins = [], []
    for x in (-17.5, 17.5):
        surf, n_in = snap_to_wall(mesh, np.array([x, 0.0, 0.0]))
        for th in (0.0, 30.0, 60.0):
            hint = np.array([np.cos(np.radians(th)), np.sin(np.radians(th)), 0.0])
            tiles.append(conform_tile(mesh, surf, n_in, hint, kind="full"))
            spins.append(th)
    two = CandidateSet.from_tiles(tiles, spins_deg=spins, anchor_ids=[0, 0, 0, 1, 1, 1])
    g2 = K.build_conflicts(two)
    assert g2.pairs[:3][:, 3:].count_nonzero() == 0
    assert g2.pairs[:3][:, :3].count_nonzero() == 6         # the three spins
    assert find_overlapping_tiles(tiles, threshold_mm=1.0) == [(0, 1), (0, 2), (1, 2),
                                                               (3, 4), (3, 5), (4, 5)]


# ------------------------------------------------------- planner agreement
def test_pairwise_equals_planner_pair_call(flat_graph, cavity_graph):
    rng = np.random.default_rng(0)
    for cs, g in (flat_graph, cavity_graph[1:]):
        n = len(cs)
        d = np.linalg.norm(cs.anchors[:, None] - cs.anchors[None], axis=2)
        near = np.argwhere(np.triu((d > 0) & (d < 32.0), 1))
        far = np.argwhere(np.triu(d >= 32.0, 1))
        pick = [tuple(p) for p in near[rng.choice(len(near), min(250, len(near)), replace=False)]]
        pick += [tuple(p) for p in far[rng.choice(len(far), min(80, len(far)), replace=False)]]
        pick += [tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(60)]
        for i, j in pick:
            assert g.conflicts(i, j) == _public(cs, i, j), (i, j)


def test_pairwise_equals_planner_multi_tile_call(cavity_graph):
    # the multi-tile planner call on a subset: identical on the cavity mesh
    _mesh, cs, g = cavity_graph
    rng = np.random.default_rng(1)
    sub = np.sort(rng.choice(len(cs), 60, replace=False))
    pub = {(int(sub[i]), int(sub[j]))
           for i, j in find_overlapping_tiles(cs.tiles_of(sub), threshold_mm=1.0)}
    mine = {(int(sub[i]), int(sub[j]))
            for i, j in np.argwhere(np.triu(g.pairs[sub][:, sub].toarray(), 1))}
    assert pub == mine


def test_batched_point_triangle_distance_is_bit_identical():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(200, 3)) * 5.0
    tri = rng.normal(size=(200, 9, 3, 3)) * 5.0
    mine = K._point_tri_dist_batched(P, tri)
    ref = np.vstack([_point_triangle_dist(P[n:n + 1], tri[n]) for n in range(200)])
    assert np.array_equal(mine, ref)


def test_toy_instance_matrix_reproduced(toy):
    g = K.build_conflicts(toy["candidates"])
    ref = toy["conflicts"]
    assert g.n == ref.n
    assert (g.pairs != ref.pairs).nnz == 0
    assert g.count_pairs() == ref.count_pairs()
    # every reference clique (a true clique by construction) is covered by
    # the pairwise graph and ours are true cliques of the same graph
    for q in ref.cliques:
        assert K._is_clique(q, g.pairs)
    for q in g.cliques:
        assert K._is_clique(q, ref.pairs)
    via_init = plan.build_conflicts(toy["candidates"])
    assert (via_init.pairs != ref.pairs).nnz == 0


# ----------------------------------------------------------------- gaps
def test_gap_adds_conflicts_monotonically(flat_graph, cavity_graph):
    rng = np.random.default_rng(5)
    for cs, g0 in (flat_graph, cavity_graph[1:]):
        g1 = K.build_conflicts(cs, gap_mm=1.0)
        g3 = K.build_conflicts(cs, gap_mm=3.0)
        a0, a1, a3 = (g.pairs.astype(np.int8) for g in (g0, g1, g3))
        assert (a1 - a0).min() >= 0 and (a3 - a1).min() >= 0
        assert g1.count_pairs() > g0.count_pairs()
        assert g3.count_pairs() > g1.count_pairs()
        assert g3.gap_mm == 3.0
        # still the planner's rule at the widened threshold
        n = len(cs)
        for i, j in [tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(60)]:
            assert g3.conflicts(i, j) == _public(cs, i, j, gap=3.0)
    assert K.conflict_threshold_mm(0.0) == 1.0 and K.conflict_threshold_mm(2.5) == 3.5
    with pytest.raises(ValueError):
        K.conflict_threshold_mm(-1.0)


# ---------------------------------------------------------------- cliques
def test_every_clique_is_a_true_clique(flat_graph, cavity_graph):
    for cs, g in (flat_graph, cavity_graph[1:]):
        assert len(g.cliques) > 0
        seen = set()
        for q in g.cliques:
            assert q.dtype.kind == "i" and q.size >= 2
            assert np.all(np.diff(q) > 0)                      # sorted, unique ids
            assert 0 <= q.min() and q.max() < g.n
            key = tuple(q.tolist())
            assert key not in seen                             # deduplicated
            seen.add(key)
            sub = g.pairs[q][:, q].toarray()
            assert sub[~np.eye(q.size, dtype=bool)].all()
        # anchor neighbourhood: a clique's anchors lie within one tile
        # diagonal of some sampled anchor (the one it was grown around)
        diag = np.array([C.tile_diagonal_mm(k) for k in cs.kinds])
        for q in g.cliques:
            centre_ok = False
            for a in np.unique(cs.anchor_ids[q]):
                pos = cs.anchors[np.flatnonzero(cs.anchor_ids == a)[0]]
                d = np.linalg.norm(cs.anchors[q] - pos[None, :], axis=1)
                if np.all(d <= diag[q] + 1e-9):
                    centre_ok = True
                    break
            assert centre_ok


def test_clique_and_pairwise_forms_agree_on_random_selections(flat_graph, cavity_graph):
    rng = np.random.default_rng(11)
    for cs, g in (flat_graph, cavity_graph[1:]):
        n = g.n
        n_feasible = 0
        for _ in range(300):
            k = int(rng.integers(2, 7))
            sel = np.sort(rng.choice(n, k, replace=False))
            pair_ok = g.is_feasible(sel)
            clique_ok = all(np.isin(q, sel).sum() <= 1 for q in g.cliques)
            # a clique may never forbid a pairwise-feasible selection
            if pair_ok:
                assert clique_ok
            # and a clique violation is always a pairwise violation
            if not clique_ok:
                assert not pair_ok
            n_feasible += pair_ok
        assert n_feasible > 0
        # greedy independent sets built from the pairwise graph satisfy every clique
        for seed in range(5):
            order = rng.permutation(n)
            sel = []
            for c in order:
                if all(not g.conflicts(c, s) for s in sel):
                    sel.append(int(c))
            assert g.is_feasible(sel)
            assert all(np.isin(q, sel).sum() <= 1 for q in g.cliques)


def test_cliques_tighten_the_pairwise_graph(cavity_graph):
    # cliques are large on the cavity (dense local conflicts), so the MILP
    # tightening has something to work with
    _mesh, cs, g = cavity_graph
    sizes = np.array([q.size for q in g.cliques])
    assert sizes.max() >= 6
    assert len(g.cliques) >= np.unique(cs.anchor_ids).size


def test_empty_and_single_candidate_sets(toy):
    one = toy["candidates"].subset([0])
    g = K.build_conflicts(one)
    assert g.n == 1 and g.count_pairs() == 0 and g.cliques == []
    with pytest.raises(TypeError):
        K.build_conflicts(toy["candidates"].tiles)
