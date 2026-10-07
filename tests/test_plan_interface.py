"""Phase 0 tests for the frozen ``gtcore.plan`` interface and its stub fixtures."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import trimesh

import gtcore.plan as plan
import plan_fixtures as pf
from gtcore.interact import PlacedTile, SEED_WALL_OFFSET_MM, conform_tile, snap_to_wall
from gtcore.plan import (
    CandidateSet,
    ConflictGraph,
    InfluenceMatrix,
    Objective,
    OptimizeReport,
    SolverResult,
    TargetSet,
    TileCountRecommendation,
)

CONSTANTS = {
    "DEFAULT_H_MM": 2.5, "DEFAULT_N_SPINS_FULL": 6, "DEFAULT_N_SPINS_HALF": 12,
    "DETACHED_MM": 1.5, "DEFAULT_RX_CGY": 6000.0, "TARGET_SHELL_OFFSET_MM": 5.0,
    "M_OPT_MAX": 4000, "TAU_FRACTION": 0.05, "LAMBDA_HOT": 0.5, "V200_TOL": 0.10,
    "LAMBDA_OAR": 1e3, "LOCAL_RADIUS_MM": 10.0, "SA_ALPHA": 0.95,
    "SA_MOVES_PER_TILE_PER_SWEEP": 50, "SA_N_SWEEPS": 40, "SA_N_RESTARTS": 3,
    "MILP_TIME_LIMIT_S": 60.0, "TILE_AREA_CM2": 4.0, "CONFLICT_GAP_MM": 0.0,
}

STUBS = [
    "build_candidates", "visible_faces", "build_influence", "build_conflicts",
    "make_objective", "evaluate", "solve_greedy", "solve_local", "solve_sa",
    "solve_milp", "refine_continuous", "sweep_n", "final_report",
    "recommend_tile_count", "optimize", "suggest_next",
]


# ------------------------------------------------------------------ package
def test_package_imports_and_all():
    for name in plan.__all__:
        assert hasattr(plan, name), name
    for name in CONSTANTS:
        assert name in plan.__all__
    for name in STUBS:
        assert name in plan.__all__
    assert "Given" in plan.__doc__ and "P1 (fixed N)" in plan.__doc__


def test_constants_values():
    for name, value in CONSTANTS.items():
        assert getattr(plan, name) == value, name


def test_submodules_exist_but_are_not_imported_by_init():
    import importlib
    import subprocess
    import sys
    mods = ("candidates", "influence", "conflicts", "objective", "solvers",
            "milp", "sweep", "report")
    # importing the package in a fresh interpreter must not pull the branch
    # modules in (they are wired up by the plan/* branches)
    code = ("import sys, gtcore.plan; "
            "print(sorted(m for m in sys.modules if m.startswith('gtcore.plan.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", out
    for mod in mods:
        m = importlib.import_module("gtcore.plan." + mod)
        assert m.__doc__ and ("Owner" in m.__doc__)


# Functions already wired to their branch module (each branch adds its own).
IMPLEMENTED = {"build_influence", "make_objective", "evaluate",
               "solve_greedy", "solve_local", "solve_sa", "refine_continuous", "sweep_n"}


@pytest.mark.parametrize("name", STUBS)
def test_every_stub_raises_not_implemented(name):
    import inspect
    if name in IMPLEMENTED:
        pytest.skip("%s is implemented (see its own test module)" % name)
    fn = getattr(plan, name)
    n_required = sum(
        1 for prm in inspect.signature(fn).parameters.values()
        if prm.default is inspect.Parameter.empty
        and prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD))
    with pytest.raises(NotImplementedError, match=name + ": implemented on branch plan/"):
        fn(*([None] * n_required))


def test_objective_methods_are_wired():
    # implemented on plan/influence; behaviour is tested in test_plan_objective.py
    inst = pf.toy_instance(n_candidates=4, n_targets=10)
    obj = Objective(inst["influence"], inst["conflicts"], rx_cgy=inst["rx_cgy"])
    assert obj.tau_cgy == pytest.approx(plan.TAU_FRACTION * inst["rx_cgy"])
    assert obj.dose_of([0]).shape == (10,)
    assert set(obj.metrics([0])) >= {"V100", "V150", "V200", "D90", "Dmean"}
    assert np.isfinite(obj.hard([0])) and np.isfinite(obj.soft([0]))
    assert np.isfinite(obj.gain([0], 1))


# ---------------------------------------------------------------- TargetSet
def test_from_shell_weights_sum_to_shell_area():
    r = 25.0
    mesh = pf.sphere_cap_mesh(radius_mm=r, subdivisions=4)
    t = TargetSet.from_shell(mesh, offset_mm=5.0)
    assert len(t) == len(mesh.vertices)
    assert t.points.shape == (len(mesh.vertices), 3)
    radii = np.linalg.norm(t.points, axis=1)
    assert np.allclose(radii, r + 5.0, atol=1e-6)
    ref = trimesh.creation.icosphere(subdivisions=4, radius=r + 5.0).area
    assert abs(t.total_weight - ref) / ref < 0.02
    assert abs(t.total_weight - 4 * np.pi * (r + 5.0) ** 2) / ref < 0.02
    assert (t.weights > 0).all()
    assert t.name == "shell+5mm"


def test_from_points_equal_weights():
    pts = np.random.default_rng(0).normal(size=(7, 3))
    t = TargetSet.from_points(pts)
    assert np.all(t.weights == 1.0)
    assert t.total_weight == 7.0
    with pytest.raises(ValueError):
        TargetSet(points=pts, weights=np.ones(3))


def test_subsample_deterministic_and_weight_preserving():
    mesh = pf.sphere_cap_mesh(radius_mm=25.0, subdivisions=4)
    t = TargetSet.from_shell(mesh)
    m_full = len(t)
    sub_a, idx_a = t.subsample(300, rng_seed=3)
    sub_b, idx_b = t.subsample(300, rng_seed=3)
    sub_c, idx_c = t.subsample(300, rng_seed=4)
    assert len(sub_a) == 300 and idx_a.shape == (300,)
    assert np.array_equal(idx_a, idx_b)
    assert not np.array_equal(idx_a, idx_c)
    assert np.unique(idx_a).size == 300           # without replacement
    assert np.all(np.diff(idx_a) > 0)             # ascending index
    assert np.allclose(sub_a.points, t.points[idx_a])
    assert sub_a.total_weight == pytest.approx(t.total_weight, rel=1e-9)
    # a subsample covers the sphere evenly: hemisphere weight share ~ 1/2
    share = sub_a.weights[sub_a.points[:, 2] > 0].sum() / sub_a.total_weight
    assert abs(share - 0.5) < 0.1
    # no-op when already small enough
    same, idx = t.subsample(m_full)
    assert same is t and np.array_equal(idx, np.arange(m_full))


def test_subsample_heavy_points_taken_with_certainty():
    pts = np.zeros((50, 3))
    pts[:, 0] = np.arange(50)
    w = np.ones(50)
    w[7] = 100.0
    t = TargetSet.from_points(pts, w)
    sub, idx = t.subsample(10, rng_seed=1)
    assert 7 in idx
    assert sub.total_weight == pytest.approx(t.total_weight)
    assert sub.weights[np.flatnonzero(idx == 7)[0]] == 100.0


# ------------------------------------------------------------ ConflictGraph
def _hand_graph():
    # 0-1, 1-2, 0-2 (triangle), 3-4; 5 isolated
    import scipy.sparse as sp
    m = sp.lil_matrix((6, 6), dtype=bool)
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4)]:
        m[i, j] = True
        m[j, i] = True
    return ConflictGraph(n=6, pairs=m.tocsr(), cliques=[np.array([0, 1, 2]),
                                                        np.array([3, 4])])


def test_conflict_graph_helpers():
    g = _hand_graph()
    assert g.count_pairs() == 4
    assert g.conflicts(0, 1) and g.conflicts(1, 0) and not g.conflicts(0, 3)
    assert not g.conflicts(5, 5)
    assert np.array_equal(g.neighbors(0), [1, 2])
    assert np.array_equal(g.neighbors(5), [])
    assert g.is_feasible([])
    assert g.is_feasible([0])
    assert g.is_feasible([0, 3, 5])
    assert not g.is_feasible([0, 1])
    assert not g.is_feasible([5, 4, 3])
    mask = np.zeros(6, dtype=bool)
    mask[[0, 3]] = True
    assert g.is_feasible(mask)
    comp = g.compatible_mask([0, 3])
    assert comp.tolist() == [False, False, False, False, False, True]
    assert g.compatible_mask([]).all()
    assert g.compatible_mask([5]).tolist() == [True, True, True, True, True, False]
    assert len(g.cliques) == 2 and g.cliques[0].dtype.kind == "i"


# ------------------------------------------------------------- CandidateSet
def test_candidate_set_seeds_of_ordering_and_subset():
    inst = pf.toy_instance(n_candidates=9, n_targets=20)
    c = inst["candidates"]
    assert len(c) == 9
    centers, axes = c.seeds_of([5, 2])
    expect = np.vstack([c.tiles[2].seed_centers, c.tiles[5].seed_centers])
    assert centers.shape == (8, 3) and axes.shape == (8, 3)
    assert np.allclose(centers, expect)            # ascending id order
    mask = np.zeros(9, dtype=bool)
    mask[[2, 5]] = True
    assert np.allclose(c.seeds_of(mask)[0], expect)
    assert np.allclose(c.seeds_of([5, 2, 5])[0], expect)   # duplicates dropped
    assert c.seeds_of([])[0].shape == (0, 3)
    tiles = c.tiles_of([5, 2])
    assert tiles[0] is c.tiles[2] and tiles[1] is c.tiles[5]
    sub = c.subset([7, 1])
    assert len(sub) == 2 and sub.tiles[0] is c.tiles[1]
    assert np.array_equal(sub.anchor_ids, [1, 7])
    assert sub.seed_centers.shape == (2, 4, 3)
    assert np.allclose(sub.anchors[1], c.anchors[7])
    with pytest.raises(IndexError):
        c.seeds_of([9])


def test_candidate_set_half_tile_padding():
    mesh = pf.flat_wall_mesh()
    surf, n_in = snap_to_wall(mesh, [0.0, 0.0, 0.0])
    half = conform_tile(mesh, surf, n_in, [1.0, 0.0, 0.0], kind="half")
    full = conform_tile(mesh, surf, n_in, [1.0, 0.0, 0.0], kind="full")
    c = CandidateSet.from_tiles([half, full], spins_deg=[0.0, 0.0])
    assert c.n_seeds.tolist() == [2, 4]
    assert np.isnan(c.seed_centers[0, 2:]).all() and not np.isnan(c.seed_centers[1]).any()
    centers, _ = c.seeds_of([0, 1])
    assert centers.shape == (6, 3) and not np.isnan(centers).any()
    assert c.kinds.tolist() == ["half", "full"]


# ---------------------------------------------------------- InfluenceMatrix
def test_influence_dose_of_and_oar():
    inst = pf.toy_instance(n_candidates=6, n_targets=30)
    I = inst["influence"]
    d = I.dose_of([0, 3])
    assert d.dtype == np.float64 and d.shape == (30,)
    assert np.allclose(d, I.dose[0].astype(float) + I.dose[3].astype(float))
    assert np.all(I.dose_of([]) == 0.0)
    I.oar["brainstem"] = np.ones((6, 4), dtype=np.float32) * 10.0
    assert np.allclose(I.oar_dose_of("brainstem", [1, 2, 4]), 30.0)
    assert I.n_candidates == 6 and I.n_targets == 30


# ----------------------------------------------------------- OptimizeReport
def test_optimize_report_json_csv_round_trip(tmp_path):
    inst = pf.toy_instance(n_candidates=4, n_targets=10)
    tiles = inst["candidates"].tiles_of([0, 3])
    sr = SolverResult(selection=np.array([0, 3]), objective=0.75,
                      metrics={"V100": 0.75, "D90": np.float32(5900.0)},
                      history=[(0, 0.4), (1, 0.75)], runtime_s=0.01, solver="greedy",
                      seed=0, status="ok")
    rep = OptimizeReport(tiles=tiles, solver=sr, parameters={"h_mm": 2.5, "n_spins": 6},
                         candidate_stats={"n_candidates": 4},
                         metrics_grid={5.0: {"V100": 0.7, "D90": 5800.0, "V150": 0.2,
                                             "V200": np.float64(0.05)}},
                         overlaps=[], shadowing=[], runtime={"total": 1.5},
                         seed=0, wall_clock_s=1.5, notes=["toy"])
    assert rep.gtcore_version
    jp = tmp_path / "report.json"
    cp = tmp_path / "report.csv"
    rep.to_json(jp)
    rep.to_csv(cp)
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["n_tiles"] == 2 and len(data["tiles"]) == 2
    assert data["tiles"][0]["kind"] == "full"
    assert np.allclose(np.asarray(data["tiles"][1]["seed_centers"]), tiles[1].seed_centers)
    assert data["solver"]["selection"] == [0, 3]
    assert data["solver"]["metrics"]["D90"] == 5900.0
    assert data["metrics_grid"]["5.0"]["V200"] == 0.05
    assert data["parameters"]["h_mm"] == 2.5
    with open(cp, newline="", encoding="utf-8") as fh:
        rows = {r["key"]: r["value"] for r in csv.DictReader(fh)}
    assert rows["n_tiles"] == "2"
    assert rows["parameters.h_mm"] == "2.5"
    assert rows["solver.objective"] == "0.75"
    assert json.loads(rows["solver.selection"]) == [0, 3]
    assert rows["metrics_grid.5.0.D90"] == "5800.0"
    s = rep.summary()
    assert "2 tiles" in s and "greedy" in s and "V100 0.700" in s


def test_tile_count_recommendation_describe():
    rec = TileCountRecommendation(n_tiles=8, area_mm2=3100.0, treatable_area_mm2=3100.0,
                                  contraction_pct=0.0, untreated_pct=0.0,
                                  ellipsoid_area_mm2=3300.0, n_tiles_ellipsoid=9,
                                  diameters_mm=[40, 35, 30], volume_mm3=22000.0)
    s = rec.describe()
    assert "8" in s and "9" in s and "gammatile.com" in s and "22.0 cc" in s


# ---------------------------------------------------------------- fixtures
def test_flat_wall_fixture_and_conform_tile():
    mesh = pf.flat_wall_mesh(size_mm=60.0, step_mm=2.0, depth_mm=40.0)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert np.allclose(mesh.bounds, [[-30, -30, -40], [30, 30, 0]])
    top = pf.flat_wall_top_faces(mesh)
    assert top.sum() == 2 * 30 * 30
    assert np.allclose(mesh.face_normals[top], [0, 0, 1])      # outward +z
    surf, n_in = snap_to_wall(mesh, [3.0, -4.0, 2.0])
    assert np.allclose(surf, [3.0, -4.0, 0.0])
    assert np.allclose(n_in, [0, 0, -1])                        # inward = into box
    tile = conform_tile(mesh, surf, n_in, [1.0, 0.0, 0.0], kind="full")
    assert isinstance(tile, PlacedTile)
    assert tile.seed_centers.shape == (4, 3)
    # seeds sit 3 mm off the wall on the cavity side (z = -3)
    assert np.allclose(tile.seed_centers[:, 2], -SEED_WALL_OFFSET_MM)
    assert np.allclose(np.sort(tile.seed_centers[:, 0]), [-2, -2, 8, 8])
    assert np.allclose(np.sort(tile.seed_centers[:, 1]), [-9, -9, 1, 1])
    assert np.allclose(tile.seed_axes, [1, 0, 0])
    assert np.allclose(tile.corners_ras[:, 2], -SEED_WALL_OFFSET_MM)
    assert np.allclose(np.abs(tile.corners_ras[:, 0] - 3.0), 10.0)


def test_sphere_fixture():
    mesh = pf.sphere_cap_mesh(radius_mm=25.0, subdivisions=3)
    assert mesh.is_watertight
    assert abs(mesh.area - 4 * np.pi * 25 ** 2) / (4 * np.pi * 25 ** 2) < 0.02


def test_analytic_point_dose_calibration():
    rx = 6000.0
    d = pf.analytic_point_dose([[0, 0, 8.0], [0, 0, 16.0], [0, 0, 0.1]], [[0, 0, 0]], rx)
    assert d[0] == pytest.approx(rx)
    assert d[1] == pytest.approx(rx / 4)
    assert d[2] == pytest.approx(rx * 64)                       # clamped at 1 mm
    two = pf.analytic_point_dose([[0, 0, 8.0]], [[0, 0, 0], [0, 0, 0]], rx)
    assert two[0] == pytest.approx(2 * rx)


def test_toy_instance_self_consistent():
    inst = pf.toy_instance(n_candidates=30, n_targets=200, rng_seed=0)
    c, I, g, t = inst["candidates"], inst["influence"], inst["conflicts"], inst["target"]
    assert len(c) == 30 and len(c.tiles) == 30
    assert I.dose.shape == (30, 200) and I.dose.dtype == np.float32
    assert not np.isnan(I.dose).any() and (I.dose > 0).all()
    assert len(t) == 200 and np.allclose(t.points[:, 2], 5.0)
    assert I.target is t and np.array_equal(I.target_index, np.arange(200))
    assert not np.isnan(c.seed_centers).any() and c.n_seeds.tolist() == [4] * 30
    assert np.allclose(c.seed_centers[:, :, 2], -SEED_WALL_OFFSET_MM)
    assert np.allclose(c.anchors[:, :2], inst["anchor_grid"])
    # conflicts: symmetric, no self-loops, grid neighbours conflict, far ones not
    assert (g.pairs != g.pairs.T).nnz == 0
    assert g.pairs.diagonal().sum() == 0
    grid = inst["anchor_grid"]
    for i in range(30):
        for j in g.neighbors(i):
            assert np.max(np.abs(grid[i] - grid[j])) <= 20.0 + 1e-9
    far = [(i, j) for i in range(30) for j in range(i + 1, 30)
           if np.max(np.abs(grid[i] - grid[j])) >= 30.0]
    assert far and all(not g.conflicts(i, j) for i, j in far)
    near = [(i, j) for i in range(30) for j in range(i + 1, 30)
            if np.max(np.abs(grid[i] - grid[j])) <= 10.0 + 1e-9]
    assert near and all(g.conflicts(i, j) for i, j in near)
    # every clique is a true clique of the pairwise graph
    assert g.cliques
    for q in g.cliques:
        sub = g.pairs[q][:, q].toarray()
        assert np.all(sub[~np.eye(q.size, dtype=bool)])
    # a feasible 4-tile selection exists and is reported feasible
    sel = [i for i in range(30) if tuple(grid[i]) in {(-25., -25.), (5., -25.),
                                                        (-25., 5.), (5., 5.)}]
    assert len(sel) == 4 and g.is_feasible(sel)
    # a target point right above a seed gets rx from that tile alone
    j = 0
    above = c.tiles[j].seed_centers[0].copy()
    above[2] = 5.0
    d = pf.analytic_point_dose(above[None, :], c.tiles[j].seed_centers, inst["rx_cgy"])
    assert d[0] >= inst["rx_cgy"]
    # determinism
    inst2 = pf.toy_instance(n_candidates=30, n_targets=200, rng_seed=0)
    assert np.array_equal(inst2["influence"].dose, I.dose)
    assert np.allclose(inst2["target"].points, t.points)
