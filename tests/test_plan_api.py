"""Tests for ``gtcore.plan.api`` (A6): ``optimize`` / ``suggest_next`` /
``recommended_count`` wiring, the fixed-tile augmentation, the candidate
cache and the influence-style metrics.

Every ``gtcore.plan`` function the API depends on is monkeypatched with a
fake built on ``tests/plan_fixtures.py`` (the toy flat-wall instance), so
these tests are green before the A1-A5 branches land and stay green after
(the fakes only use the frozen dataclass contracts).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import gtcore.plan as plan
import plan_fixtures as pf
from gtcore.interact import conform_tile, find_overlapping_tiles, snap_to_wall
from gtcore.plan import (
    ConflictGraph,
    InfluenceMatrix,
    Objective,
    OptimizeReport,
    SolverResult,
    TargetSet,
    TileCountRecommendation,
)
from gtcore.plan import api

RX = plan.DEFAULT_RX_CGY


# ------------------------------------------------------------------- fakes
@pytest.fixture()
def toy():
    return pf.toy_instance(n_candidates=36, n_targets=150)


def _analytic_influence(cand, target, rx_cgy):
    dose = np.empty((len(cand), len(target)), dtype=np.float32)
    for c, t in enumerate(cand.tiles):
        dose[c] = pf.analytic_point_dose(target.points, t.seed_centers, rx_cgy)
    return InfluenceMatrix(dose=dose, target=target, target_index=np.arange(len(target)),
                           rx_cgy=rx_cgy, kernel="analytic")


def _planner_conflicts(cand, gap_mm=0.0):
    c = len(cand)
    pairs = sp.lil_matrix((c, c), dtype=bool)
    for i, j in find_overlapping_tiles(cand.tiles, threshold_mm=1.0):
        pairs[i, j] = pairs[j, i] = True
    return ConflictGraph(n=c, pairs=pairs.tocsr(), cliques=[], gap_mm=gap_mm)


def _fake_greedy(objective, n_tiles, fixed=(), kinds_required=None):
    """Independent greedy on the influence matrix (no A2 Objective methods)."""
    infl, conf = objective.influence, objective.conflicts
    sel = [int(i) for i in fixed]
    while len(sel) < int(n_tiles):
        base = infl.dose_of(sel)
        mask = conf.compatible_mask(sel)
        gains, _h0 = api._gains_over_base(infl.dose, base, infl.target.weights,
                                          objective.rx_cgy, mask)
        if not np.isfinite(gains).any():
            return SolverResult(selection=np.sort(sel), solver="greedy",
                                status="infeasible", reason="no compatible candidate",
                                feasible=False)
        sel.append(int(np.argmax(gains)))
    m = api.target_metrics(infl.dose_of(sel), infl.target.weights, objective.rx_cgy)
    return SolverResult(selection=np.sort(sel), objective=m["hard"], metrics=m,
                        solver="greedy", status="ok", feasible=True)


def _install_fakes(monkeypatch, toy, calls):
    def build_candidates(mesh, h_mm=plan.DEFAULT_H_MM, n_spins=None, kinds=("full",),
                         eligible_faces=None, rng_seed=0, **kw):
        calls["build_candidates"] += 1
        calls["last_kinds"] = tuple(kinds)
        return toy["candidates"]

    def build_influence(cand, target, rx_cgy=RX, **kw):
        calls["build_influence"] += 1
        return _analytic_influence(cand, target, rx_cgy)

    def build_conflicts(cand, gap_mm=0.0):
        calls["build_conflicts"] += 1
        return _planner_conflicts(cand, gap_mm)

    def make_objective(infl, conf, **w):
        return Objective(infl, conf, rx_cgy=w.get("rx_cgy", infl.rx_cgy))

    def solve_greedy(objective, n_tiles, fixed=(), kinds_required=None, **kw):
        calls["greedy_fixed"] = list(int(i) for i in fixed)
        calls["greedy_kinds"] = dict(kinds_required or {})
        calls["greedy_n"] = int(n_tiles)
        return _fake_greedy(objective, n_tiles, fixed, kinds_required)

    def solve_local(objective, n_tiles, start, radius_mm=10.0, candidates=None):
        res = _fake_greedy(objective, n_tiles, fixed=())
        res.selection = np.asarray(start, dtype=int)
        res.solver = "local"
        return res

    def solve_sa(objective, n_tiles, seed=0, n_sweeps=1, n_restarts=1, start=None,
                 candidates=None):
        calls["sa_seed"] = seed
        res = _fake_greedy(objective, n_tiles, fixed=())
        res.selection = np.asarray(start, dtype=int)
        res.solver, res.seed = "sa", seed
        return res

    def evaluate(objective, selection):
        infl = objective.influence
        m = api.target_metrics(infl.dose_of(selection), infl.target.weights, objective.rx_cgy)
        return {"hard": m["hard"], "soft": m["hard"], "metrics": m, "feasible": True}

    def final_report(mesh, tiles, rx_cgy=RX, target=None, solver_result=None,
                     parameters=None, **kw):
        calls["final_report"] += 1
        return OptimizeReport(
            tiles=list(tiles), solver=solver_result, parameters=dict(parameters or {}),
            metrics_grid={5.0: {"stats": {"V100": 0.91, "D90": 6100.0,
                                          "V150": 0.3, "V200": 0.08}}},
            overlaps=[(int(i), int(j)) for i, j in
                      find_overlapping_tiles(list(tiles), threshold_mm=1.0)])

    def recommend_tile_count(mesh, contraction_pct=0.0, untreated_pct=0.0,
                             eligible_faces=None):
        calls["recommend"] += 1
        return TileCountRecommendation(
            n_tiles=7, area_mm2=2800.0, treatable_area_mm2=2800.0, contraction_pct=0.0,
            untreated_pct=0.0, ellipsoid_area_mm2=3000.0, n_tiles_ellipsoid=8,
            diameters_mm=[40.0, 30.0, 25.0])

    for name, fn in (("build_candidates", build_candidates),
                     ("build_influence", build_influence),
                     ("build_conflicts", build_conflicts),
                     ("make_objective", make_objective),
                     ("solve_greedy", solve_greedy), ("solve_local", solve_local),
                     ("solve_sa", solve_sa), ("evaluate", evaluate),
                     ("final_report", final_report),
                     ("recommend_tile_count", recommend_tile_count)):
        monkeypatch.setattr(plan, name, fn)


@pytest.fixture()
def fakes(monkeypatch, toy):
    calls = {"build_candidates": 0, "build_influence": 0, "build_conflicts": 0,
             "final_report": 0, "recommend": 0}
    api.clear_cache()
    _install_fakes(monkeypatch, toy, calls)
    yield calls
    api.clear_cache()


# ----------------------------------------------------------------- metrics
def test_overlap_threshold_matches_planner():
    from gtcore.planner import OVERLAP_THRESHOLD_MM
    assert api.PLANNER_OVERLAP_THRESHOLD_MM == OVERLAP_THRESHOLD_MM


def test_weighted_quantile_and_target_metrics():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    assert api.weighted_quantile(d, w, 0.10) == 1.0
    assert api.weighted_quantile(d, w, 0.5) == 2.0
    assert api.weighted_quantile(d, [1.0, 1.0, 1.0, 100.0], 0.10) == 4.0
    m = api.target_metrics([0.5 * RX, RX, 1.6 * RX, 2.5 * RX], w, RX)
    assert m["V100"] == pytest.approx(0.75)
    assert m["V150"] == pytest.approx(0.5)
    assert m["V200"] == pytest.approx(0.25)
    assert m["hard"] == pytest.approx(0.75 - plan.LAMBDA_HOT * (0.25 - plan.V200_TOL))
    empty = api.target_metrics([], [], RX)
    assert empty["V100"] == 0.0 and empty["hard"] == 0.0


def test_evaluate_tiles_on_the_flat_wall(toy):
    mesh = toy["mesh"]
    tile = toy["candidates"].tiles[0]
    target = toy["target"]
    none = api.evaluate_tiles(mesh, [], rx_cgy=RX, target=target)
    assert none["V100"] == 0.0 and none["n_tiles"] == 0.0
    one = api.evaluate_tiles(mesh, [tile], rx_cgy=RX, target=target)
    assert 0.0 <= one["V100"] <= 1.0 and one["D90"] >= 0.0
    assert one["n_tiles"] == 1.0 and one["n_points"] == float(len(target))
    assert one["Dmean"] > none["Dmean"]
    # default target: the +5 mm shell of the mesh itself
    default = api.evaluate_tiles(mesh, [tile], rx_cgy=RX)
    assert default["target"].startswith("shell")


def test_default_target_masks_ineligible_wall(toy):
    """On the printed-phantom fallback only the eligible (inner) wall may
    carry target weight; the point set itself is unchanged."""
    mesh = toy["mesh"]
    full = api.default_target(mesh)
    assert full.total_weight == pytest.approx(TargetSet.from_shell(mesh).total_weight)
    top = pf.flat_wall_top_faces(mesh)
    masked = api.default_target(mesh, top)
    assert len(masked) == len(full) and np.array_equal(masked.points, full.points)
    assert 0 < masked.total_weight < full.total_weight
    faces = np.asarray(mesh.faces)
    off_wall = np.setdiff1d(np.arange(len(full)), np.unique(faces[top]))
    assert np.all(masked.weights[off_wall] == 0.0)
    assert "eligible" in masked.name
    with pytest.raises(ValueError, match="one entry per mesh face"):
        api.default_target(mesh, top[:-1])
    # and the before/after readout honours it
    m = api.evaluate_tiles(mesh, [toy["candidates"].tiles[0]], rx_cgy=RX, eligible_faces=top)
    assert "eligible" in m["target"]


def test_optimize_and_suggest_use_the_eligible_target(fakes, toy):
    mesh = toy["mesh"]
    top = pf.flat_wall_top_faces(mesh)
    tiles, rep = plan.optimize(mesh, 2, report=False, eligible_faces=top)
    assert "eligible" in rep.parameters["target"] and len(tiles) == 2
    _tile, info = plan.suggest_next(mesh, [], eligible_faces=top)
    assert "eligible" in info["target"]


def test_compatible_with_placed_agrees_with_the_toy_conflicts(toy):
    cand, conf = toy["candidates"], toy["conflicts"]
    mask = api.compatible_with_placed(cand, [cand.tiles[0]])
    expect = ~conf.pairs[0].toarray().ravel()
    expect[0] = False  # a tile on top of itself overlaps itself
    assert np.array_equal(mask, expect)
    assert api.compatible_with_placed(cand, []).all()


# ------------------------------------------------------------------- cache
def test_candidate_cache_reuses_until_inputs_change(fakes, toy):
    mesh = toy["mesh"]
    a = api.cached_candidates(mesh, h_mm=2.5, kinds=("full",))
    b = api.cached_candidates(mesh, h_mm=2.5, kinds=("full",))
    assert a is b and fakes["build_candidates"] == 1
    api.cached_candidates(mesh, h_mm=3.0, kinds=("full",))
    assert fakes["build_candidates"] == 2
    elig = np.ones(len(mesh.faces), dtype=bool)
    api.cached_candidates(mesh, h_mm=2.5, kinds=("full",), eligible_faces=elig)
    assert fakes["build_candidates"] == 3
    api.cached_candidates(mesh, h_mm=2.5, kinds=("full",), eligible_faces=elig.copy())
    assert fakes["build_candidates"] == 3, "an equal mask must hit the cache"
    other = pf.flat_wall_mesh(size_mm=80.0)
    api.cached_candidates(other, h_mm=2.5, kinds=("full",))
    assert fakes["build_candidates"] == 4
    api.clear_cache()
    api.cached_candidates(mesh, h_mm=2.5, kinds=("full",))
    assert fakes["build_candidates"] == 5


# ---------------------------------------------------------------- optimize
def test_optimize_greedy_end_to_end_with_fakes(fakes, toy):
    mesh = toy["mesh"]
    tiles, rep = plan.optimize(mesh, 3, solver="greedy", seed=4)
    assert len(tiles) == 3 and all(t.kind == "full" for t in tiles)
    assert find_overlapping_tiles(tiles, threshold_mm=1.0) == []
    assert rep.overlaps == []
    assert fakes["final_report"] == 1
    assert fakes["greedy_kinds"] == {"full": 3} and fakes["greedy_fixed"] == []
    assert rep.seed == 4 and rep.parameters["seed"] == 4
    for key in ("h_mm", "n_spins", "lambda_hot", "v200_tol", "lambda_oar", "tau_cgy",
                "detached_mm", "conflict_gap_mm", "m_opt_max", "solver", "n_full",
                "n_half", "target_shell_offset_mm", "sa_alpha", "milp_time_limit_s"):
        assert key in rep.parameters, key
    for stage in ("target", "candidates", "influence", "conflicts", "solver", "report"):
        assert stage in rep.runtime, stage
    assert rep.wall_clock_s > 0
    assert rep.candidate_stats["n_candidates"] == len(toy["candidates"])
    assert rep.metrics_influence["hard"] == pytest.approx(rep.solver.objective)
    assert 5.0 in rep.metrics_grid
    assert "3 tiles" in rep.summary()


def test_optimize_report_false_is_lightweight(fakes, toy):
    tiles, rep = plan.optimize(toy["mesh"], 2, report=False)
    assert len(tiles) == 2 and fakes["final_report"] == 0
    assert rep.metrics_grid == {} and rep.overlaps == []
    assert any("influence" in n for n in rep.notes)
    assert rep.metrics_influence["V100"] >= 0.0
    assert "report" not in rep.runtime


def test_optimize_reuses_cached_candidates_and_influence(fakes, toy):
    mesh = toy["mesh"]
    plan.optimize(mesh, 2, report=False)
    plan.optimize(mesh, 3, report=False)
    assert fakes["build_candidates"] == 1 and fakes["build_influence"] == 1
    assert fakes["build_conflicts"] == 1, "the conflict graph is cached per candidate set"
    # an explicit candidate set is used as given (no cache lookup)
    plan.optimize(mesh, 2, report=False, candidates=toy["candidates"])
    assert fakes["build_candidates"] == 1


def test_optimize_solver_variants_and_verbose(fakes, toy, capsys):
    tiles, rep = plan.optimize(toy["mesh"], 2, solver="sa", seed=11, report=False,
                               verbose=True)
    assert rep.solver.solver == "sa" and fakes["sa_seed"] == 11
    assert len(tiles) == 2
    out = capsys.readouterr().out
    assert "candidates:" in out and "solver sa" in out
    tiles, rep = plan.optimize(toy["mesh"], 2, solver="local", report=False)
    assert rep.solver.solver == "local" and len(tiles) == 2
    seen = []
    plan.optimize(toy["mesh"], 2, report=False, log=seen.append)
    assert any(s.startswith("done") for s in seen)


def test_optimize_validates_inputs(fakes, toy):
    mesh = toy["mesh"]
    with pytest.raises(ValueError, match="at least one tile"):
        plan.optimize(mesh, 0)
    with pytest.raises(ValueError, match="unknown solver"):
        plan.optimize(mesh, 2, solver="magic")
    with pytest.raises(ValueError, match="greedy solver only"):
        plan.optimize(mesh, 2, solver="sa", fixed_tiles=[toy["candidates"].tiles[0]])
    with pytest.raises(ValueError, match="integers"):
        plan.optimize(mesh, None)


def test_optimize_fails_loudly_on_short_or_infeasible_results(fakes, toy, monkeypatch):
    mesh = toy["mesh"]
    real = plan.solve_greedy

    def short(objective, n_tiles, fixed=(), kinds_required=None, **kw):
        res = real(objective, n_tiles, fixed, kinds_required, **kw)
        res.selection = res.selection[:-1]
        return res

    monkeypatch.setattr(plan, "solve_greedy", short)
    with pytest.raises(RuntimeError, match="returned 2 tiles, 3 requested"):
        plan.optimize(mesh, 3, report=False)

    def infeasible(objective, n_tiles, fixed=(), kinds_required=None, **kw):
        return SolverResult(selection=[], status="infeasible", reason="wall is full",
                            feasible=False, solver="greedy")

    monkeypatch.setattr(plan, "solve_greedy", infeasible)
    with pytest.raises(RuntimeError, match="wall is full"):
        plan.optimize(mesh, 3, report=False)

    def conflicting(objective, n_tiles, fixed=(), kinds_required=None, **kw):
        return SolverResult(selection=[0, 1], status="ok", feasible=True, solver="greedy")

    monkeypatch.setattr(plan, "solve_greedy", conflicting)
    with pytest.raises(RuntimeError, match="violates a conflict"):
        plan.optimize(mesh, 2, report=False)
    # more tiles than the toy wall holds: the fake greedy reports infeasible
    monkeypatch.setattr(plan, "solve_greedy", real)
    with pytest.raises(RuntimeError):
        plan.optimize(mesh, 36, report=False)


def test_optimize_with_fixed_tiles_keeps_them_as_obstacles(fakes, toy):
    mesh, cand = toy["mesh"], toy["candidates"]
    fixed = [cand.tiles[5]]
    tiles, rep = plan.optimize(mesh, 2, report=False, fixed_tiles=fixed)
    c = len(cand)
    assert fakes["greedy_fixed"] == [c] and fakes["greedy_n"] == 3
    assert fakes["greedy_kinds"] == {"full": 3}
    assert len(tiles) == 2
    assert all(not np.allclose(t.corners_ras, fixed[0].corners_ras) for t in tiles)
    assert find_overlapping_tiles(fixed + tiles, threshold_mm=1.0) == []
    assert len(rep.tiles) == 3 and rep.parameters["n_fixed"] == 1
    assert any("fixed" in n for n in rep.notes)


def test_augment_with_fixed_builds_a_consistent_instance(toy):
    cand, infl, conf = toy["candidates"], toy["influence"], toy["conflicts"]
    fixed = [cand.tiles[0], cand.tiles[35]]
    cand2, infl2, conf2, ids = api._augment_with_fixed(cand, infl, conf, fixed)
    c = len(cand)
    assert list(ids) == [c, c + 1]
    assert len(cand2) == c + 2 and infl2.n_candidates == c + 2 and conf2.n == c + 2
    assert conf2.is_feasible(ids), "fixed tiles never conflict with each other"
    # the fixed row conflicts exactly with the candidates overlapping it
    row = conf2.pairs[c].toarray().ravel()[:c]
    expect = conf.pairs[0].toarray().ravel().copy()
    expect[0] = True
    assert np.array_equal(row, expect)
    assert np.allclose(conf2.pairs.toarray(), conf2.pairs.toarray().T)
    assert infl2.dose[c].max() > 0 and infl2.dose.dtype == np.float32
    assert cand2.anchor_ids[c] > cand.anchor_ids.max()


def test_optimize_continuous_uses_greedy_start_and_coarse_grid(fakes, toy, monkeypatch):
    mesh, cand = toy["mesh"], toy["candidates"]
    seen = {}

    def solve_continuous(mesh_, candidates, target, rx_cgy, n_tiles, seed=0, n_starts=4,
                         n_passes=2, time_budget_s=60.0, start=None, kinds_required=None,
                         **kw):
        assert kw.get("objective") is not None and kw.get("conflicts") is not None
        seen.update(n_tiles=n_tiles, seed=seed, budget=time_budget_s,
                    start=list(np.asarray(start)), kinds=dict(kinds_required or {}))
        tiles = candidates.tiles_of(start)
        m = api.target_metrics(api.tiles_dose(tiles, target.points), target.weights, rx_cgy)
        return tiles, SolverResult(selection=np.asarray(start), objective=m["hard"],
                                   metrics=m, solver="continuous", seed=seed, status="ok")

    def build_candidates(mesh_, h_mm=plan.DEFAULT_H_MM, n_spins=None, kinds=("full",),
                         eligible_faces=None, rng_seed=0, **kw):
        seen.update(h_mm=h_mm, n_spins=n_spins)
        return cand

    monkeypatch.setattr(plan, "build_candidates", build_candidates)
    monkeypatch.setattr(plan, "solve_continuous", solve_continuous, raising=False)
    tiles, rep = plan.optimize(mesh, 2, solver="continuous", seed=3, report=False,
                               time_budget_s=7.5, refine=True)
    assert len(tiles) == 2 and rep.solver.solver == "continuous"
    assert seen["h_mm"] == api.CONTINUOUS_H_MM and seen["n_spins"] == api.CONTINUOUS_N_SPINS
    assert seen["budget"] == 7.5 and seen["seed"] == 3 and seen["n_tiles"] == 2
    assert len(seen["start"]) == 2 and seen["kinds"] == {"full": 2}
    assert rep.parameters["time_budget_s"] == 7.5 and rep.parameters["h_mm"] == 4.0
    assert rep.overlaps == [] and "refine" not in rep.runtime
    # explicit grid values are kept
    plan.optimize(mesh, 2, solver="continuous", report=False, h_mm=3.0, n_spins=2)
    assert seen["h_mm"] == 3.0 and seen["n_spins"] == 2
    with pytest.raises(ValueError, match="greedy solver only"):
        plan.optimize(mesh, 2, solver="continuous", fixed_tiles=[cand.tiles[0]])


def test_optimize_continuous_is_a_stub_until_a3_lands(fakes, toy, monkeypatch):
    import gtcore.plan.solvers as solvers_mod
    monkeypatch.delattr(plan, "solve_continuous", raising=False)
    monkeypatch.delattr(solvers_mod, "solve_continuous", raising=False)
    with pytest.raises(NotImplementedError, match="solve_continuous"):
        plan.optimize(toy["mesh"], 2, solver="continuous", report=False)


# ------------------------------------------------------------- suggest_next
def test_suggest_next_on_an_empty_board_picks_a_candidate(fakes, toy):
    mesh, cand = toy["mesh"], toy["candidates"]
    tile, info = plan.suggest_next(mesh, [])
    assert tile is cand.tiles[info["candidate_id"]]
    assert fakes["last_kinds"] == ("full",)
    for key in ("gain", "V100_before", "V100_after", "D90_before", "D90_after",
                "n_candidates", "n_compatible", "seconds", "hard_before", "hard_after"):
        assert key in info, key
    assert info["V100_before"] == 0.0 and info["D90_before"] == 0.0
    assert info["gain"] == pytest.approx(info["hard_after"] - info["hard_before"])
    assert info["n_compatible"] == len(cand)


def test_suggest_next_respects_the_board(fakes, toy):
    mesh, cand = toy["mesh"], toy["candidates"]
    first, info0 = plan.suggest_next(mesh, [])
    second, info = plan.suggest_next(mesh, [first])
    assert find_overlapping_tiles([first, second], threshold_mm=1.0) == []
    assert info["n_compatible"] < info["n_candidates"]
    assert info["n_placed"] == 1
    # the base dose comes from the real engine (the fake influence is the
    # analytic toy dose), so only its shape is checked here
    assert 0.0 <= info["V100_before"] <= 1.0 and info["D90_before"] >= 0.0
    assert info["hard_before"] == pytest.approx(
        info["V100_before"] - plan.LAMBDA_HOT * max(0.0, info["V200_before"] - plan.V200_TOL))
    assert fakes["build_candidates"] == 1 and fakes["build_influence"] == 1
    # a full wall has nothing compatible left
    with pytest.raises(ValueError, match="compatible"):
        plan.suggest_next(mesh, list(cand.tiles))
    with pytest.raises(ValueError, match="kind"):
        plan.suggest_next(mesh, [], kind="third")
    # an explicit candidate set is honoured
    tile, info = plan.suggest_next(mesh, [], candidates=cand.subset([3, 4]))
    assert info["n_candidates"] == 2 and tile in (cand.tiles[3], cand.tiles[4])


def test_suggest_next_does_not_use_ineligible_candidates(fakes, toy):
    cand = toy["candidates"]
    elig = np.ones(len(cand), dtype=bool)
    elig[:8] = False
    restricted = cand.subset(np.arange(len(cand)))
    restricted.eligible = elig
    tile, info = plan.suggest_next(toy["mesh"], [], candidates=restricted)
    assert info["candidate_id"] >= 8 and info["n_compatible"] == 28


# ----------------------------------------------------------- recommendation
def test_recommended_count_wraps_the_rule(fakes, toy):
    rec, line = api.recommended_count(toy["mesh"])
    assert isinstance(rec, TileCountRecommendation) and rec.n_tiles == 7
    assert line.startswith("recommended 7 tiles") and "ellipsoid estimate 8" in line
    assert fakes["recommend"] == 1


def test_not_implemented_from_a_branch_propagates(monkeypatch, toy):
    api.clear_cache()

    def stub(*a, **k):
        raise NotImplementedError("build_candidates: implemented on branch plan/candidates")

    monkeypatch.setattr(plan, "build_candidates", stub)
    with pytest.raises(NotImplementedError, match="plan/candidates"):
        plan.optimize(toy["mesh"], 2, report=False)
    with pytest.raises(NotImplementedError, match="plan/candidates"):
        plan.suggest_next(toy["mesh"], [])
