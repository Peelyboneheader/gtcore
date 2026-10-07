"""Adversarial review tests for the opt-in placement optimizer ``gtcore.plan``.

Reviewer agent A7 (docs/plan-tile-optimize.md §6).  Written from the §2
problem statement, the §0 goals and the Phase-0 *interface* (public
signatures, docstrings and dataclass fields of ``gtcore.plan``) only --
without reading §3 or any ``gtcore/plan/*`` implementation.  Every number
compared against comes from ``tests/review_reference.py`` (independent
metrics, conflict, target and brute-force enumeration).

Design choice: the solver probes feed the solvers an ``Objective`` whose
``InfluenceMatrix`` and ``ConflictGraph`` are built HERE from the reference
(exact engine rows, ``find_overlapping_tiles`` pairs, one 2-clique per
conflicting pair).  That checks the solvers in isolation from the builders'
influence / conflict code, which get their own probes (c) and (b).

Every probe ``pytest.skip``s -- never fails -- while the builder function it
needs still raises ``NotImplementedError``, so this file is green on every
branch at every time.

Probes
------
(a) ``test_greedy_provably_suboptimal_flat_wall`` -- constructed flat-wall
    instance where forward greedy is provably suboptimal at N = 2;
    SA / MILP must reach the brute-force optimum.
(b) ``test_clique_must_not_overconstrain_pairwise_feasible_triple`` and
    ``test_cliques_are_cliques_of_the_pairwise_graph`` -- §2 makes PAIRWISE
    non-overlap the hard constraint; cliques are only a MILP tightening and
    must not forbid a pairwise-feasible selection.
(c) ``test_influence_rows_match_exact_engine`` -- influence rows vs
    ``dose_at_points(exact=True)``.
(d) ``test_sa_seed_reproducible_and_no_worse_than_greedy``.
(e) ``test_reported_metrics_rederived_on_full_shell`` and
    ``test_objective_metrics_match_reference``.
(f) ``test_optimizer_output_never_flagged_by_planner``.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
import scipy.sparse as sp
import trimesh

from gtcore.interact import (PlacedTile, conform_tile, find_overlapping_tiles,
                             snap_to_wall)

from review_reference import (brute_force_best, greedy_forward_reference,
                              is_feasible, metrics_for_tiles,
                              reference_conflict, reference_metrics,
                              shell_target, tiles_dose)

RX_CGY = 4000.0          # phantom probes: 6000 cGy at +5 mm needs ~10 tiles on
                         # this 42 cm^2 cavity; at 4 tiles V100 would be ~0 and
                         # the objective flat.  4000 cGy gives V100 ~ 0.5 at
                         # N = 4 (V200 = 0), a discriminating regime.
PHANTOM_N = 4            # tiles requested in the phantom probes
P1_LAMBDA_HOT = 0.5      # §2 defaults, used to re-derive the P1 objective
P1_V200_TOL = 0.10
SOLVER_ROLES = {"greedy": "solve_greedy", "sa": "solve_sa", "milp": "solve_milp"}


# ================================================================ plumbing
def _plan():
    try:
        import gtcore.plan as plan_mod
    except ImportError:
        pytest.skip("gtcore.plan has not landed on this branch")
    return plan_mod


def _fn(name):
    fn = getattr(_plan(), name, None)
    if not callable(fn):
        pytest.skip("gtcore.plan.%s is not exposed" % name)
    return fn


def _try(fn, *args, **kwargs):
    """Call a builder function; skip (not fail) while it is a stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip("%s not implemented yet: %s"
                    % (getattr(fn, "__name__", fn), exc))


def _candidate_set(tiles):
    plan_mod = _plan()
    return _try(plan_mod.CandidateSet.from_tiles, list(tiles),
                np.zeros(len(tiles)), method="review")


def _target_set(pts, w):
    return _try(_plan().TargetSet.from_points, pts, w, name="review")


def _reference_conflict_graph(tiles, cliques="edges"):
    """ConflictGraph from ``find_overlapping_tiles`` on every pair.

    ``cliques="edges"``: one 2-clique per conflicting pair (a valid, minimal
    tightening); ``"none"``: no cliques at all (the solvers must then still
    honour ``pairs``).
    """
    plan_mod = _plan()
    tiles = list(tiles)
    n = len(tiles)
    pairs = sp.lil_matrix((n, n), dtype=bool)
    edges = find_overlapping_tiles(tiles)
    for i, j in edges:
        pairs[i, j] = True
        pairs[j, i] = True
    cl = [np.array([i, j], dtype=int) for i, j in edges] \
        if cliques == "edges" else []
    return _try(plan_mod.ConflictGraph, n=n, pairs=pairs.tocsr(), cliques=cl,
                gap_mm=0.0)


def _reference_influence(tiles, pts, w, rx):
    plan_mod = _plan()
    rows = np.array([tiles_dose(t.seed_centers, t.seed_axes, pts)
                     for t in tiles], dtype=np.float32)
    return _try(plan_mod.InfluenceMatrix, dose=rows, target=_target_set(pts, w),
                target_index=np.arange(pts.shape[0]), rx_cgy=float(rx),
                sk_per_seed_u=3.5, kernel="exact")


def _reference_objective(tiles, pts, w, rx, cliques="edges"):
    """(Objective, CandidateSet) built from the reference only."""
    plan_mod = _plan()
    infl = _reference_influence(tiles, pts, w, rx)
    graph = _reference_conflict_graph(tiles, cliques=cliques)
    cset = _candidate_set(tiles)
    obj = _try(plan_mod.Objective, influence=infl, conflicts=graph,
               rx_cgy=float(rx))
    return obj, cset


def _solve(role, objective, cset, n, **kw):
    """Run solver ``role`` -> (tiles, SolverResult); skip while stubbed."""
    fn = _fn(SOLVER_ROLES[role])
    res = _try(fn, objective, n, **kw)
    sel = np.asarray(res.selection, dtype=int).reshape(-1)
    return cset.tiles_of(sel), res


def _p1_objective(m):
    """§2 P1 objective without OARs from a metrics dict."""
    return m["V100"] - P1_LAMBDA_HOT * max(0.0, m["V200"] - P1_V200_TOL)


# ================================================================ geometry
def flat_wall_mesh(extent=140.0, depth=60.0, subdiv=3):
    """Closed box whose TOP face (z = 0) is the 'cavity wall'.

    ``snap_to_wall`` orients normals toward the mesh centroid (z = -depth/2),
    so a tile conformed at z = 0 has its seeds at z = -3 mm and the +5 mm
    target shell (outward = away from the centroid) sits at z = +5 mm.  The
    box is subdivided so the smooth vertex normals on the face interior are
    exactly (0, 0, -1) and the seed grid is exactly +-5 mm.
    """
    box = trimesh.creation.box(extents=(extent, extent, depth))
    box.apply_translation([0.0, 0.0, -depth / 2.0])
    for _ in range(subdiv):
        box = box.subdivide()
    return box


def tile_on_flat_wall(mesh, x, y=0.0, kind="full", axis=(1.0, 0.0, 0.0)):
    sp_, n_in = snap_to_wall(mesh, np.array([x, y, 0.0]))
    return conform_tile(mesh, sp_, n_in, np.asarray(axis, dtype=float), kind)


@pytest.fixture(scope="module")
def flat_wall():
    return flat_wall_mesh()


def _row_target(xs, half_width=6.0, n_per=5, z=5.0):
    ys = np.linspace(-half_width, half_width, n_per)
    return np.vstack([np.c_[np.full_like(ys, x), ys, np.full_like(ys, z)]
                      for x in xs])


@pytest.fixture(scope="module")
def greedy_trap(flat_wall):
    """The constructed greedy-suboptimal instance (probe a).

    Candidates on the flat wall, full tiles, spin 0:
        A at x = -15, B at x = +15 (A-B pitch 30 mm: feasible together),
        C at x = 0 (conflicts with both A and B),
        D at x = +45 (decoy: feasible with everything, covers nothing).
    Target: three rows of 5 points on the +5 mm shell above A, C and B with
    group weights 0.3 / 0.4 / 0.3 and rx = 3000 cGy.  A single tile delivers
    ~4500 cGy straight above itself at +5 mm and ~1600 cGy 15 mm to the
    side, so C alone covers the middle group (0.4), A or B alone covers one
    side group (0.3): forward greedy must pick C first, after which only D is
    feasible and the final V100 is 0.4.  A + B together cover both side
    groups and -- summing 2 x ~1600 -- most of the middle group as well
    (0.84).  V200 is zero throughout (no dose reaches 6000 cGy), so the P1
    objective equals V100 and no hot-spot penalty can rescue greedy.  The
    instance is certified by the reference brute force and an independent
    reference greedy before the builders' greedy is asked.
    """
    A = tile_on_flat_wall(flat_wall, -15.0)
    B = tile_on_flat_wall(flat_wall, 15.0)
    C = tile_on_flat_wall(flat_wall, 0.0)
    D = tile_on_flat_wall(flat_wall, 45.0)
    cands = [A, B, C, D]
    pts = _row_target([-15.0, 0.0, 15.0])
    w = np.concatenate([np.full(5, 0.3 / 5), np.full(5, 0.4 / 5),
                        np.full(5, 0.3 / 5)])
    return dict(cands=cands, pts=pts, w=w, rx=3000.0)


@pytest.fixture(scope="module")
def clique_triple(flat_wall):
    """Probe (b): pairwise-feasible triple inside one tile diagonal.

    Full tiles, spin 0, anchors P = (0, 0), Q = (23, 0), R = (11.5, 23).
    Footprints are 20 x 20 mm axis-aligned squares, so P-Q are 3 mm apart
    edge to edge, and R clears both P and Q by 3 mm in y.  All three
    pairwise anchor distances (23.0, 25.7, 25.7 mm) are below one tile
    diagonal (28.3 mm), so a clique built from an anchor neighbourhood of
    radius >= 25.7 mm (or any "within a tile diagonal" rule) would contain
    all three and ``sum x <= 1`` over it would wrongly forbid selecting them
    together.  Target: points above each tile so that all three is the
    unique optimum at N = 3.
    """
    P = tile_on_flat_wall(flat_wall, 0.0, 0.0)
    Q = tile_on_flat_wall(flat_wall, 23.0, 0.0)
    R = tile_on_flat_wall(flat_wall, 11.5, 23.0)
    cands = [P, Q, R]
    pts = np.vstack([_row_target([0.0, 23.0], half_width=4.0, n_per=3),
                     np.c_[np.linspace(7.5, 15.5, 3), np.full(3, 23.0),
                           np.full(3, 5.0)]])
    w = np.full(pts.shape[0], 1.0 / pts.shape[0])
    return dict(cands=cands, pts=pts, w=w, rx=3000.0)


@pytest.fixture(scope="module")
def dense_grid(flat_wall):
    """5 x 5 anchors at 11.5 mm pitch, spin 0: many true conflicts AND many
    pairwise-feasible pairs whose anchors lie inside one tile diagonal."""
    xs = np.arange(-2, 3) * 11.5
    return [tile_on_flat_wall(flat_wall, x, y) for y in xs for x in xs]


# ----------------------------------------------------------- phantom case
@pytest.fixture(scope="module")
def phantom_case():
    """Synthetic cavity: make_head_phantom(1.0 mm, 3 tiles, seed 1)."""
    from gtcore.phantom.generate import make_head_phantom
    from gtcore.segment.surface import mask_to_mesh

    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    pts, w = shell_target(mesh, 5.0)
    return dict(vol=vol, truth=truth, mesh=mesh, pts=pts, w=w)


def _farthest_point_anchors(mesh, n, seed=0):
    verts = np.asarray(mesh.vertices, dtype=float)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(verts.shape[0]))]
    d = np.linalg.norm(verts - verts[chosen[0]], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(verts - verts[k], axis=1))
    return verts[chosen]


def _phantom_candidates(mesh, n=24, seed=0):
    """My own candidate list: farthest-point anchors, spin 0, full tiles."""
    tiles = []
    for anchor in _farthest_point_anchors(mesh, n, seed):
        sp_, n_in = snap_to_wall(mesh, anchor)
        hint = np.array([1.0, 0.0, 0.0])
        if abs(float(hint @ n_in)) > 0.9:
            hint = np.array([0.0, 1.0, 0.0])
        tiles.append(conform_tile(mesh, sp_, n_in, hint, "full"))
    return tiles


@pytest.fixture(scope="module")
def phantom_candidates(phantom_case):
    return _phantom_candidates(phantom_case["mesh"])


@pytest.fixture(scope="module")
def phantom_objective(phantom_case, phantom_candidates):
    """Reference objective on the phantom (24 candidates, full +5 mm shell)."""
    return _reference_objective(phantom_candidates, phantom_case["pts"],
                                phantom_case["w"], RX_CGY)


# ============================================================ (a) greedy
def test_construction_is_certified_greedy_suboptimal(greedy_trap):
    """Reference-only part of probe (a): runs on every branch."""
    cands, pts, w, rx = (greedy_trap[k] for k in ("cands", "pts", "w", "rx"))
    A, B, C, D = cands
    # conflict structure exactly as designed
    assert reference_conflict(A, C) and reference_conflict(B, C)
    assert not reference_conflict(A, B)
    assert not any(reference_conflict(D, t) for t in (A, B, C))
    # C has the best single-tile gain -> any forward greedy picks it first
    single = [metrics_for_tiles([t], pts, w, rx) for t in cands]
    v1 = [m["V100"] for m in single]
    assert v1[2] > max(v1[0], v1[1]) > v1[3]
    assert all(m["V200"] == 0.0 for m in single)      # P1 == V100 here
    # exhaustive reference: {A, B} is the unique optimum and beats every
    # feasible pair containing C
    best, best_m, table = brute_force_best(cands, 2, (pts, w), rx)
    assert best == (0, 1)
    assert best_m["V100"] > 0.8
    assert best_m["V200"] == 0.0
    for subset, m in table:
        if 2 in subset:
            assert m["V100"] <= 0.4 + 1e-12
    # independent forward greedy lands in the trap
    chosen, v_greedy = greedy_forward_reference(cands, 2, (pts, w), rx)
    assert chosen[0] == 2
    assert v_greedy <= 0.4 + 1e-12 < best_m["V100"]


def test_greedy_provably_suboptimal_flat_wall(greedy_trap):
    """Probe (a): the builders' greedy must fall into the trap; brute wins."""
    cands, pts, w, rx = (greedy_trap[k] for k in ("cands", "pts", "w", "rx"))
    best, best_m, _ = brute_force_best(cands, 2, (pts, w), rx)
    obj, cset = _reference_objective(cands, pts, w, rx)
    tiles, res = _solve("greedy", obj, cset, 2)
    assert is_feasible(tiles), "greedy returned a conflicting pair"
    assert len(tiles) <= 2
    v_greedy = metrics_for_tiles(tiles, pts, w, rx)["V100"]
    print("probe (a): builders' greedy selection %s V100 = %.3f (status %r); "
          "brute-force optimum %s V100 = %.3f"
          % (list(res.selection), v_greedy, res.status, best, best_m["V100"]))
    assert best_m["V100"] > v_greedy + 0.3, (
        "forward greedy was expected to be trapped at V100 <= 0.4 on this "
        "instance (brute force 0.84); got %.3f -- if the builders' greedy "
        "has look-ahead or swap moves, say so in the notes" % v_greedy)
    assert 2 in list(res.selection), "greedy did not pick the central tile"


@pytest.mark.parametrize("role", ["sa", "milp"])
def test_sa_and_milp_reach_brute_force_optimum(greedy_trap, role):
    cands, pts, w, rx = (greedy_trap[k] for k in ("cands", "pts", "w", "rx"))
    best, best_m, _ = brute_force_best(cands, 2, (pts, w), rx)
    obj, cset = _reference_objective(cands, pts, w, rx)
    kw = {"seed": 0} if role == "sa" else {}
    tiles, res = _solve(role, obj, cset, 2, **kw)
    assert is_feasible(tiles)
    assert len(tiles) == 2
    v = metrics_for_tiles(tiles, pts, w, rx)["V100"]
    print("probe (a) %s: selection %s V100 = %.3f vs brute %.3f (status %r)"
          % (role, list(res.selection), v, best_m["V100"], res.status))
    assert v >= best_m["V100"] - 1e-9, (
        "%s returned V100 %.3f, brute-force optimum is %.3f (%s)"
        % (role, v, best_m["V100"], best))


def test_milp_honours_pairs_without_cliques(greedy_trap):
    """A ConflictGraph with ``cliques=[]`` still carries the hard pairwise
    constraint in ``pairs``; the MILP must not select a conflicting pair."""
    cands, pts, w, rx = (greedy_trap[k] for k in ("cands", "pts", "w", "rx"))
    obj, cset = _reference_objective(cands, pts, w, rx, cliques="none")
    tiles, res = _solve("milp", obj, cset, 2)
    assert is_feasible(tiles), (
        "MILP selected a conflicting pair %s when the graph carried no "
        "cliques: pairwise constraints are the hard §2 constraint"
        % list(res.selection))


# ============================================================ (b) cliques
def test_clique_triple_is_pairwise_feasible(clique_triple):
    """Reference-only part of probe (b): runs on every branch."""
    cands, pts, w, rx = (clique_triple[k] for k in ("cands", "pts", "w", "rx"))
    assert find_overlapping_tiles(cands) == []
    anchors = np.array([t.anchor_ras for t in cands])
    diag = 2.0 * 10.0 * np.sqrt(2.0)
    dists = [float(np.linalg.norm(anchors[i] - anchors[j]))
             for i, j in combinations(range(3), 2)]
    assert max(dists) < diag
    # the triple is worth selecting in full: all 3 beats every pair
    best, best_m, _ = brute_force_best(cands, 3, (pts, w), rx)
    assert best == (0, 1, 2)
    _, pair_m, _ = brute_force_best(cands, 2, (pts, w), rx)
    assert best_m["V100"] > pair_m["V100"]


def test_clique_must_not_overconstrain_pairwise_feasible_triple(clique_triple):
    """Probe (b): §2 makes PAIRWISE non-overlap the hard constraint.

    "no two tiles conflict (hard)" -- a clique inequality ``sum x <= 1`` is
    only a valid MILP tightening when every pair inside the clique actually
    conflicts under ``find_overlapping_tiles``.  Any clique built from a
    geometric neighbourhood (anchors within a tile diagonal, say) that
    contains two of P, Q, R is an over-constraint: it forbids a selection
    that the hard constraint permits.  Structural part: the builders'
    ``build_conflicts`` graph.  Behavioural part: every available solver,
    run on the builders' graph, must return all three at N = 3.
    """
    cands, pts, w, rx = (clique_triple[k] for k in ("cands", "pts", "w", "rx"))
    plan_mod = _plan()
    cset = _candidate_set(cands)
    graph = _try(_fn("build_conflicts"), cset)

    # --- structural
    pairs = sp.csr_matrix(graph.pairs)
    assert pairs.nnz == 0, (
        "build_conflicts flags pairs %r although find_overlapping_tiles "
        "flags nothing" % list(zip(*pairs.nonzero())))
    for i, j in combinations(range(3), 2):
        assert not graph.conflicts(i, j)
    offending = [list(map(int, c)) for c in graph.cliques
                 if len(set(int(i) for i in c)) >= 2]
    print("probe (b): build_conflicts cliques on the triple = %r"
          % [list(map(int, c)) for c in graph.cliques])
    assert offending == [], (
        "clique constraints %r contain >= 2 of the pairwise-feasible triple; "
        "sum x <= 1 over such a clique over-constrains §2" % offending)
    assert graph.is_feasible([0, 1, 2])

    # --- behavioural, on the builders' graph
    infl = _reference_influence(cands, pts, w, rx)
    obj = _try(plan_mod.Objective, influence=infl, conflicts=graph,
               rx_cgy=float(rx))
    ran = []
    for role in ("milp", "greedy", "sa"):
        fn = getattr(plan_mod, SOLVER_ROLES[role], None)
        if not callable(fn):
            continue
        try:
            res = fn(obj, 3, **({"seed": 0} if role == "sa" else {}))
        except NotImplementedError:
            continue
        sel = list(map(int, np.asarray(res.selection).reshape(-1)))
        ran.append(role)
        assert sorted(sel) == [0, 1, 2], (
            "%s selected %s instead of all three pairwise-feasible "
            "candidates (status %r, reason %r): a conflict/clique constraint "
            "forbids a §2-feasible selection" % (role, sel, res.status,
                                                   res.reason))
    print("probe (b): solvers that returned the full triple: %s" % ran)


def test_cliques_are_cliques_of_the_pairwise_graph(dense_grid):
    """Every clique emitted by ``build_conflicts`` must be a clique of the
    pairwise conflict graph, and the pairwise graph must equal
    ``find_overlapping_tiles`` (symmetric, no diagonal)."""
    cset = _candidate_set(dense_grid)
    graph = _try(_fn("build_conflicts"), cset)
    n = len(dense_grid)
    truth = set(find_overlapping_tiles(dense_grid))
    pairs = sp.csr_matrix(graph.pairs)
    got = {(int(i), int(j)) for i, j in zip(*pairs.nonzero()) if i < j}
    sym = {(int(i), int(j)) for i, j in zip(*pairs.nonzero()) if i > j}
    assert {(j, i) for i, j in sym} == got, "pairs matrix is not symmetric"
    assert pairs.diagonal().sum() == 0
    print("probe (b, grid): %d candidates, %d true conflicting pairs, "
          "builders %d pairs, %d cliques (sizes %s)"
          % (n, len(truth), len(got), len(graph.cliques),
             sorted(len(c) for c in graph.cliques)))
    assert got == truth, (
        "pairwise graph differs from find_overlapping_tiles: extra %r, "
        "missing %r" % (sorted(got - truth), sorted(truth - got)))
    bad = []
    for c in graph.cliques:
        ids = sorted(set(int(i) for i in np.asarray(c).reshape(-1)))
        for i, j in combinations(ids, 2):
            if (i, j) not in truth:
                bad.append((ids, (i, j)))
    assert bad == [], (
        "%d clique(s) contain a pair that does NOT conflict under "
        "find_overlapping_tiles, e.g. %r: sum x <= 1 over them forbids "
        "pairwise-feasible selections" % (len(bad), bad[:3]))


# ========================================================== (c) influence
@pytest.mark.parametrize("kernel", ["tabulated", "exact"])
def test_influence_rows_match_exact_engine(phantom_case, phantom_candidates,
                                           kernel):
    """Probe (c): each candidate's influence row vs dose_at_points(exact).

    Tolerance 1 % of rx per point for points >= 2.5 mm from every seed of
    the candidate; the maximum deviation is printed for the notes.
    ``m_opt`` is set to the full target size so no subsampling happens;
    ``target_index`` is honoured anyway.
    """
    build = _fn("build_influence")
    pts, w, rx = phantom_case["pts"], phantom_case["w"], RX_CGY
    cands = phantom_candidates[:6]
    cset = _candidate_set(cands)
    tset = _target_set(pts, w)
    infl = _try(build, cset, tset, rx_cgy=rx, m_opt=pts.shape[0],
                kernel=kernel)
    A = np.asarray(infl.dose, dtype=float)
    idx = np.asarray(infl.target_index, dtype=int).reshape(-1)
    assert A.shape == (len(cands), idx.shape[0]), A.shape
    used = pts[idx]
    if np.isfinite(infl.sk_per_seed_u):
        assert abs(float(infl.sk_per_seed_u) - 3.5) < 1e-9, (
            "influence built at S_K = %r U, engine default is 3.5 U"
            % infl.sk_per_seed_u)

    worst = 0.0
    worst_scale = (1.0, 1.0)
    for i, t in enumerate(cands):
        ref = tiles_dose(t.seed_centers, t.seed_axes, used)
        d_seed = np.min(np.linalg.norm(
            used[:, None, :] - t.seed_centers[None, :, :], axis=2), axis=1)
        far = d_seed >= 2.5
        dev = np.abs(A[i, far] - ref[far])
        worst = max(worst, float(dev.max()))
        denom = float(ref[far] @ ref[far])
        if denom > 0:                   # best-fit scale reveals unit errors
            s = float(A[i, far] @ ref[far]) / denom
            worst_scale = (min(worst_scale[0], s), max(worst_scale[1], s))
    print("probe (c, %s): max |influence - exact| over %d candidates x %d "
          "points (>= 2.5 mm from seeds) = %.2f cGy = %.3f %% of rx; best-fit "
          "scale range %.5f..%.5f" % (kernel, len(cands), idx.shape[0], worst,
                                      100 * worst / rx, *worst_scale))
    assert worst <= 0.01 * rx, (
        "influence row (%s kernel) deviates from the exact engine by %.1f cGy "
        "(> 1 %% of rx = %.1f cGy)" % (kernel, worst, 0.01 * rx))


# ============================================================ (d) SA seed
def test_sa_seed_reproducible_and_no_worse_than_greedy(phantom_case,
                                                        phantom_objective):
    pts, w, rx = phantom_case["pts"], phantom_case["w"], RX_CGY
    obj, cset = phantom_objective
    n = PHANTOM_N

    def key(tiles):
        return tuple(sorted(tuple(np.round(t.anchor_ras, 6)) for t in tiles))

    g_tiles, g_res = _solve("greedy", obj, cset, n)
    s0_tiles, s0_res = _solve("sa", obj, cset, n, seed=0, candidates=cset)
    s0b_tiles, _ = _solve("sa", obj, cset, n, seed=0, candidates=cset)
    s1_tiles, s1_res = _solve("sa", obj, cset, n, seed=1, candidates=cset)

    assert key(s0_tiles) == key(s0b_tiles), \
        "solve_sa is not reproducible from its seed"
    m_g = metrics_for_tiles(g_tiles, pts, w, rx)
    m_0 = metrics_for_tiles(s0_tiles, pts, w, rx)
    m_1 = metrics_for_tiles(s1_tiles, pts, w, rx)
    changed = key(s0_tiles) != key(s1_tiles)
    print("probe (d): greedy %s V100 %.4f obj %.4f | SA seed0 %s V100 %.4f "
          "obj %.4f | SA seed1 %s V100 %.4f obj %.4f | answer changes with "
          "seed: %s" % (list(g_res.selection), m_g["V100"], _p1_objective(m_g),
                        list(s0_res.selection), m_0["V100"], _p1_objective(m_0),
                        list(s1_res.selection), m_1["V100"], _p1_objective(m_1),
                        changed))
    for tiles, res in ((g_tiles, g_res), (s0_tiles, s0_res),
                       (s1_tiles, s1_res)):
        assert is_feasible(tiles)
        assert len(tiles) == n
        assert res.feasible
        assert res.status == "ok", (res.status, res.reason)
    # §2 P1 objective (no OARs) -- SA must not be worse than greedy
    for m in (m_0, m_1):
        assert _p1_objective(m) >= _p1_objective(m_g) - 1e-9, (
            "SA objective %.4f worse than greedy %.4f"
            % (_p1_objective(m), _p1_objective(m_g)))


# ======================================================== (e) re-derive
def rederive_metrics(tiles, mesh, rx_cgy=RX_CGY, offset_mm=5.0,
                     weights_from="shell"):
    """Recompute §2 metrics for stored tiles on the full +offset shell."""
    pts, w = shell_target(mesh, offset_mm, weights_from=weights_from)
    return metrics_for_tiles(list(tiles), pts, w, rx_cgy)


def test_objective_metrics_match_reference(phantom_case, phantom_objective):
    """Builders' ``Objective.metrics`` (weighted V100/V150/V200/D90) on the
    SAME influence rows vs the reference: isolates the metric / weighted
    quantile conventions from everything else."""
    pts, w, rx = phantom_case["pts"], phantom_case["w"], RX_CGY
    obj, cset = phantom_objective
    sel = np.array([0, 5, 10, 15])
    theirs = _try(obj.metrics, sel)
    mine = metrics_for_tiles(cset.tiles_of(sel), pts, w, rx)
    print("probe (e, metrics): builders %s | reference V100 %.4f V150 %.4f "
          "V200 %.4f D90 %.1f" % ({k: (round(float(v), 4) if isinstance(
              v, (int, float, np.floating)) else v) for k, v in theirs.items()},
                                  mine["V100"], mine["V150"], mine["V200"],
                                  mine["D90"]))
    for k in ("V100", "V150", "V200"):
        assert k in theirs, "metrics lack %s" % k
        assert abs(float(theirs[k]) - mine[k]) <= 0.005, (k, theirs[k], mine[k])
    assert "D90" in theirs
    assert abs(float(theirs["D90"]) - mine["D90"]) <= 0.01 * rx, (
        theirs["D90"], mine["D90"])


def test_greedy_result_metrics_rederived(phantom_case, phantom_objective):
    """Probe (e) on the greedy ``SolverResult.metrics``."""
    pts, w, rx = phantom_case["pts"], phantom_case["w"], RX_CGY
    obj, cset = phantom_objective
    tiles, res = _solve("greedy", obj, cset, PHANTOM_N)
    mine = metrics_for_tiles(tiles, pts, w, rx)
    print("probe (e, greedy): reported %s | re-derived V100 %.4f D90 %.1f "
          "V150 %.4f V200 %.4f" % (
              {k: round(float(v), 4) for k, v in res.metrics.items()
               if isinstance(v, (int, float, np.floating))},
              mine["V100"], mine["D90"], mine["V150"], mine["V200"]))
    assert "V100" in res.metrics and "D90" in res.metrics
    assert abs(float(res.metrics["V100"]) - mine["V100"]) <= 0.005
    assert abs(float(res.metrics["D90"]) - mine["D90"]) <= 0.01 * rx
    assert abs(float(res.objective) - _p1_objective(mine)) <= 0.005


def _shell_entry(metrics_grid, offset=5.0):
    for key in (offset, float(offset), str(offset), "5", "5.0", 5):
        if key in metrics_grid:
            return metrics_grid[key]
    for key, val in metrics_grid.items():
        try:
            if abs(float(key) - offset) < 1e-6:
                return val
        except (TypeError, ValueError):
            pass
    pytest.skip("metrics_grid has no +5 mm shell entry (keys %r)"
                % list(metrics_grid))


@pytest.fixture(scope="module")
def optimized(phantom_case):
    """``optimize`` on the phantom (greedy, N = 4, seed 0) -- shared by (e)/(f)."""
    import time
    optimize = _fn("optimize")
    t0 = time.time()
    out = _try(optimize, phantom_case["mesh"], PHANTOM_N, rx_cgy=RX_CGY,
               solver="greedy", seed=0)
    dt = time.time() - t0
    tiles, report = out
    print("optimize(greedy, N=%d, rx=%.0f): %.1f s wall, %d tiles"
          % (PHANTOM_N, RX_CGY, dt, len(tiles)))
    return tiles, report, dt


def test_reported_metrics_rederived_on_full_shell(phantom_case, optimized):
    """Probe (e): builders' reported V100/D90 vs my reference on the shell."""
    tiles, report, _ = optimized
    mesh = phantom_case["mesh"]
    mine = rederive_metrics(tiles, mesh)
    mine_wall = rederive_metrics(tiles, mesh, weights_from="wall")
    grid = _shell_entry(report.metrics_grid)
    v100_grid = float(grid["V100"])
    d90_grid = float(grid["D90"])
    infl = report.metrics_influence or {}
    print("probe (e): reported grid V100 %.4f D90 %.1f | influence-time %s | "
          "re-derived (shell weights) V100 %.4f D90 %.1f | (wall weights) "
          "V100 %.4f D90 %.1f" % (v100_grid, d90_grid,
                                  {k: round(float(v), 4) for k, v in infl.items()
                                   if isinstance(v, (int, float, np.floating))},
                                  mine["V100"], mine["D90"],
                                  mine_wall["V100"], mine_wall["D90"]))
    assert abs(v100_grid - mine["V100"]) <= 0.005, (
        "V100 reported %.4f vs re-derived %.4f" % (v100_grid, mine["V100"]))
    assert abs(d90_grid - mine["D90"]) <= 0.01 * RX_CGY, (
        "D90 reported %.1f vs re-derived %.1f" % (d90_grid, mine["D90"]))


# ============================================ (f) planner consistency
def test_optimizer_output_never_flagged_by_planner(optimized):
    tiles, report, _ = optimized
    assert len(tiles) == PHANTOM_N
    assert all(isinstance(t, PlacedTile) for t in tiles)
    assert find_overlapping_tiles(tiles) == []
    assert list(report.overlaps) == []


def test_greedy_output_never_flagged_by_planner(phantom_objective):
    obj, cset = phantom_objective
    tiles, res = _solve("greedy", obj, cset, PHANTOM_N)
    assert res.feasible
    assert find_overlapping_tiles(tiles) == []
