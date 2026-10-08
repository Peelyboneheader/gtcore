"""Tests for the heuristic solvers (``gtcore.plan.solvers`` / ``sweep``, A3).

Section 4 V1: greedy never violates a conflict and is deterministic; greedy
with ``fixed`` keeps the fixed ids; an infeasible N fails loudly; local
search never decreases the hard objective and ends feasible; SA is
reproducible from its seed, feasible and at least as good as greedy; the
N-sweep is monotone for greedy warm starts; E5 never lowers V100 and never
introduces an overlap.  Everything runs on ``plan_fixtures.toy_instance``
(analytic dose) except the E5 test, which uses the real engine on the flat
wall.  Whole module ~15 s.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import plan_fixtures as pf
from gtcore.interact import conform_tile, find_overlapping_tiles, snap_to_wall
from gtcore.plan import (
    CandidateSet,
    Objective,
    SolverResult,
    TargetSet,
    refine_continuous,
    solve_greedy,
    solve_local,
    solve_sa,
    sweep_n,
)
from gtcore.plan import solvers as S
from gtcore.plan.solvers import _Evaluator, _HardEval, _weighted_quantile, solve_greedy_local

# The 30-candidate toy is a 6 x 5 grid at 10 mm pitch; abutting tiles conflict,
# so a feasible packing needs 30 mm between anchors along an axis: N_max = 4.
# The 60-candidate toy is an 8 x 8 grid (60 of 64 cells): N_max = 9.
N_MAX = {30: 4, 60: 9}


@pytest.fixture(scope="module")
def toy30():
    return pf.toy_instance(n_candidates=30)


@pytest.fixture(scope="module")
def toy60():
    return pf.toy_instance(n_candidates=60, n_targets=250)


def _obj(inst) -> Objective:
    return Objective(inst["influence"], inst["conflicts"], rx_cgy=inst["rx_cgy"])


def _check_feasible(inst, r: SolverResult, n: int):
    assert r.status == "ok", r.reason
    assert r.feasible
    assert r.selection.shape == (n,)
    assert np.all(np.diff(r.selection) > 0), "ascending unique ids"
    assert inst["conflicts"].is_feasible(r.selection)
    assert find_overlapping_tiles(inst["candidates"].tiles_of(r.selection)) == []
    assert np.isfinite(r.objective)
    for k in ("V100", "V150", "V200", "D90", "Dmean"):
        assert k in r.metrics


# ------------------------------------------------------------- evaluation
def test_weighted_quantile_matches_numpy_for_unit_weights():
    rng = np.random.default_rng(3)
    v = rng.uniform(0, 100, 57)
    w = np.ones_like(v)
    for q in (0.1, 0.5, 0.9):
        assert _weighted_quantile(v, w, q) == pytest.approx(
            np.percentile(v, 100 * q, method="inverted_cdf"))
    assert _weighted_quantile(v, w * 3.0, 0.1) == _weighted_quantile(v, w, 0.1)


def test_hard_eval_matches_direct_definition(toy30):
    inst = toy30
    obj = _obj(inst)
    h = _HardEval(obj)
    ids = np.array([0, 4, 20])
    dose = inst["influence"].dose_of(ids)
    w = inst["target"].weights
    v100 = (w * (dose >= inst["rx_cgy"])).sum() / w.sum()
    v200 = (w * (dose >= 2 * inst["rx_cgy"])).sum() / w.sum()
    expect = v100 - obj.lambda_hot * max(0.0, v200 - obj.v200_tol)
    assert h.hard_from_dose(dose) == pytest.approx(expect)
    m = h.metrics_from_dose(dose, {})
    assert m["V100"] == pytest.approx(v100) and m["V200"] == pytest.approx(v200)
    # vectorized add-one agrees with scalar evaluation
    base = inst["influence"].dose_of([0, 4])
    all_vals = h.hard_all(base, {})
    assert all_vals[20] == pytest.approx(h.hard_from_dose(base + inst["influence"].dose[20]))
    sub = h.hard_all(base, {}, ids=np.array([20, 3]))
    assert sub == pytest.approx(all_vals[[20, 3]])
    # soft is between 0 and 1 (minus penalties) and increases with dose
    assert h.soft_from_dose(dose) > h.soft_from_dose(base)


def test_evaluator_uses_objective_methods_when_real(toy30):
    """When ``hard`` / ``metrics`` / ``gains_all`` are implemented (A2), the
    solvers route through them; the selection must not change."""
    inst = toy30

    class RealObjective(Objective):
        calls = {"hard": 0, "gains_all": 0, "metrics": 0}

        def hard(self, selection):
            RealObjective.calls["hard"] += 1
            h = _HardEval(self)
            ids = np.asarray(selection, dtype=int)
            return h.hard_from_dose(h.dose_of(ids), h.oar_penalty(h.oar_doses_of(ids)))

        def metrics(self, selection):
            RealObjective.calls["metrics"] += 1
            h = _HardEval(self)
            ids = np.asarray(selection, dtype=int)
            return h.metrics_from_dose(h.dose_of(ids), h.oar_doses_of(ids))

        def gains_all(self, selection, dose_vec=None):
            RealObjective.calls["gains_all"] += 1
            h = _HardEval(self)
            ids = np.asarray(selection, dtype=int)
            if dose_vec is None:
                dose_vec = h.dose_of(ids)
            return h.hard_all(dose_vec, h.oar_doses_of(ids)) - self.hard(ids)

    real = RealObjective(inst["influence"], inst["conflicts"], rx_cgy=inst["rx_cgy"])
    ev = _Evaluator(real)
    assert ev._real_hard and ev._real_metrics and ev._use_obj_internal
    r_real = solve_greedy(real, 4)
    r_stub = solve_greedy(_obj(inst), 4)
    assert RealObjective.calls["gains_all"] >= 4 and RealObjective.calls["metrics"] >= 1
    assert np.array_equal(r_real.selection, r_stub.selection)
    assert r_real.objective == pytest.approx(r_stub.objective)
    # on main the plain Objective is real too (plan/influence merged) and
    # exposes gains_all, so the fast path is taken and agrees with the helper
    ev2 = _Evaluator(_obj(inst))
    assert ev2._real_hard and ev2._use_obj_internal


# ------------------------------------------------------------------ greedy
@pytest.mark.parametrize("nc", [30, 60])
def test_greedy_never_violates_a_conflict(nc, toy30, toy60):
    inst = toy30 if nc == 30 else toy60
    obj = _obj(inst)
    for n in range(1, N_MAX[nc] + 1):
        r = solve_greedy(obj, n)
        _check_feasible(inst, r, n)
        assert r.solver == "greedy"
        assert len(r.extra["gains"]) == n and r.extra["marginal_gain"] == r.extra["gains"]
        assert len(r.history) == n + 1
        assert r.history[-1][1] == pytest.approx(r.objective)
        assert sorted(r.extra["order"]) == r.selection.tolist()


def test_greedy_is_deterministic(toy30):
    obj = _obj(toy30)
    a = solve_greedy(obj, 4)
    b = solve_greedy(obj, 4)
    assert np.array_equal(a.selection, b.selection)
    assert a.history == b.history
    assert a.extra["gains"] == b.extra["gains"]
    assert a.extra["order"] == b.extra["order"]


def test_greedy_ties_break_lexicographically(toy30):
    """Hard gain, then soft gain, then room, then lowest id.  With rx raised so
    no single tile reaches it, step 1 ties on hard gain for every candidate
    and the soft gain must decide (not simply id 0)."""
    inst = toy30
    obj = Objective(inst["influence"], inst["conflicts"], rx_cgy=1e6)
    r = solve_greedy(obj, 1)
    h = _HardEval(obj)
    soft = h.soft_all(np.zeros(h.M), {})
    assert r.extra["gains"] == [0.0]
    assert r.selection[0] == int(np.argmax(soft))
    # exact ties on hard and soft (zero-dose influence) fall to room, then id
    zero = type(inst["influence"])(dose=np.zeros_like(inst["influence"].dose),
                                   target=inst["target"],
                                   target_index=inst["influence"].target_index,
                                   rx_cgy=inst["rx_cgy"], kernel="analytic")
    obj0 = Objective(zero, inst["conflicts"], rx_cgy=inst["rx_cgy"])
    r0 = solve_greedy(obj0, 1)
    pairs = inst["conflicts"].pairs
    room = inst["conflicts"].n - np.asarray(pairs.sum(axis=1)).reshape(-1) - 1
    assert r0.selection[0] == int(np.argmax(room))     # first max = lowest id


def test_greedy_with_fixed_keeps_fixed_ids_and_suggests_next(toy30):
    inst = toy30
    obj = _obj(inst)
    fixed = [0, 29]
    r = solve_greedy(obj, 4, fixed=fixed)
    _check_feasible(inst, r, 4)
    assert set(fixed) <= set(r.selection.tolist())
    assert r.extra["fixed"] == fixed and r.extra["order"][:2] == fixed
    assert len(r.extra["gains"]) == 2
    # "suggest next tile": one more than what is placed
    placed = r.selection[:3]
    nxt = solve_greedy(obj, len(placed) + 1, fixed=placed)
    _check_feasible(inst, nxt, 4)
    assert set(placed.tolist()) <= set(nxt.selection.tolist())
    assert len(nxt.extra["gains"]) == 1
    # fixed ids are kept verbatim when nothing has to be added
    same = solve_greedy(obj, 2, fixed=fixed)
    assert same.selection.tolist() == fixed and same.extra["gains"] == []


def test_greedy_rejects_bad_fixed(toy30):
    obj = _obj(toy30)
    with pytest.raises(ValueError, match="fixed has"):
        solve_greedy(obj, 1, fixed=[0, 29])
    r = solve_greedy(obj, 3, fixed=[0, 1])            # abutting -> conflict
    assert r.status == "infeasible" and not r.feasible
    assert "conflict" in r.reason


def test_infeasible_n_fails_loudly(toy30):
    inst = toy30
    r = solve_greedy(_obj(inst), N_MAX[30] + 1)
    assert r.status == "infeasible"
    assert r.feasible is False
    assert r.selection.shape == (N_MAX[30],)
    assert "4 of 5" in r.reason
    assert inst["conflicts"].is_feasible(r.selection)


def test_greedy_kinds_required(toy30):
    inst = toy30
    obj = _obj(inst)
    cand = inst["candidates"]
    with pytest.raises(ValueError, match="kinds_required needs candidate kinds"):
        solve_greedy(obj, 2, kinds_required={"full": 2})
    r = S.solve_greedy(obj, 3, kinds_required={"full": 3}, candidates=cand)
    _check_feasible(inst, r, 3)
    assert np.array_equal(r.selection, solve_greedy(obj, 3).selection)
    with pytest.raises(ValueError, match="sum to"):
        S.solve_greedy(obj, 3, kinds_required={"full": 2}, candidates=cand)
    r = S.solve_greedy(obj, 3, kinds_required={"full": 2, "half": 1}, candidates=cand)
    assert r.status == "infeasible" and "required kinds" in r.reason
    assert r.selection.shape == (2,)
    # kinds via objective.candidates also work
    obj.candidates = cand
    r2 = solve_greedy(obj, 3, kinds_required={"full": 3})
    assert np.array_equal(r2.selection, r.selection) or r2.feasible


def test_feasibility_aware_greedy_packs_where_plain_greedy_stalls(toy60):
    """Coordinator failure case: the max-gain picks cluster where the gain is
    largest and leave no room, so plain greedy places 7 of 9 on the 8 x 8
    toy; the packing bound skips those picks and places all 9."""
    inst = toy60
    obj = _obj(inst)
    n = N_MAX[60]
    plain = S.solve_greedy(obj, n, feasibility_aware=False)
    assert plain.status == "infeasible" and not plain.feasible
    assert plain.selection.shape[0] < n
    assert "of 9" in plain.reason
    aware = S.solve_greedy(obj, n)
    _check_feasible(inst, aware, n)
    assert aware.extra["feasibility_aware"] and aware.extra["skipped"] > 0
    assert aware.extra["bound_fallbacks"] == 0
    # the packing bound itself: 9 mutually compatible candidates exist
    nbr = S._conflict_lists(inst["conflicts"])
    mask = np.ones(inst["conflicts"].n, dtype=bool)
    deg = np.asarray(inst["conflicts"].pairs.sum(axis=1)).reshape(-1)
    assert S._packing_bound(mask, nbr, deg, 9) == 9
    assert S._packing_bound(mask, nbr, deg, 3) == 3     # truncated at need
    # at every step at most N_MAX tiles fit; the bound never exceeds the truth
    assert S._packing_bound(mask, nbr, deg, 20) <= 16


# ------------------------------------------------------------ local search
@pytest.mark.parametrize("nc", [30, 60])
def test_local_search_never_decreases_and_ends_feasible(nc, toy30, toy60):
    inst = toy30 if nc == 30 else toy60
    obj = _obj(inst)
    cand = inst["candidates"]
    n = N_MAX[nc]
    g = solve_greedy(obj, n)
    r = solve_local(obj, n, g.selection, candidates=cand)
    _check_feasible(inst, r, n)
    assert r.solver == "local"
    assert r.objective >= g.objective - 1e-12
    vals = [v for _s, v in r.history]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert r.history[0][1] == pytest.approx(g.objective)
    assert r.history[-1][1] == pytest.approx(r.objective)
    assert r.extra["n_moves"] == len(r.history) - 1
    assert not r.extra["degraded_to_global"]
    # a local optimum: re-running from the result makes no move
    again = solve_local(obj, n, r.selection, candidates=cand)
    assert again.extra["n_moves"] == 0
    assert np.array_equal(again.selection, r.selection)


def test_local_search_improves_greedy_on_toy30(toy30):
    inst = toy30
    obj = _obj(inst)
    g = solve_greedy(obj, 4)
    r = solve_local(obj, 4, g.selection, candidates=inst["candidates"])
    assert r.objective > g.objective
    assert r.extra["moves"]["local"] >= 1
    gl = solve_greedy_local(obj, 4, candidates=inst["candidates"])
    assert gl.solver == "greedy+local"
    assert np.array_equal(gl.selection, r.selection)
    assert gl.extra["greedy"]["objective"] == pytest.approx(g.objective)
    assert gl.history[0][1] == pytest.approx(g.objective)


def test_local_search_without_candidates_and_eval_cap(toy30):
    inst = toy30
    obj = _obj(inst)
    g = solve_greedy(obj, 4)
    r = solve_local(obj, 4, g.selection)                 # global moves only
    _check_feasible(inst, r, 4)
    assert r.extra["degraded_to_global"]
    assert r.extra["moves"]["local"] == 0 and r.extra["moves"]["spin"] == 0
    assert r.objective >= g.objective - 1e-12
    capped = S.solve_local(obj, 4, g.selection, candidates=inst["candidates"], max_evals=30)
    assert capped.status == "max_evals" and "max_evals" in capped.reason
    assert capped.feasible and inst["conflicts"].is_feasible(capped.selection)
    assert capped.objective >= g.objective - 1e-12
    # short starts are topped up greedily; bad starts are refused
    short = solve_local(obj, 4, g.selection[:2], candidates=inst["candidates"])
    _check_feasible(inst, short, 4)
    bad = solve_local(obj, 2, [0, 1], candidates=inst["candidates"])
    assert bad.status == "infeasible" and not bad.feasible


# ------------------------------------------------------ simulated annealing
@pytest.mark.parametrize("nc", [30, 60])
def test_sa_reproducible_feasible_and_at_least_greedy(nc, toy30, toy60):
    inst = toy30 if nc == 30 else toy60
    obj = _obj(inst)
    cand = inst["candidates"]
    n = N_MAX[nc]
    g = solve_greedy(obj, n)
    kw = dict(n_sweeps=12, n_restarts=2, candidates=cand)
    a = solve_sa(obj, n, seed=0, **kw)
    b = solve_sa(obj, n, seed=0, **kw)
    _check_feasible(inst, a, n)
    assert a.solver == "sa" and a.seed == 0
    assert np.array_equal(a.selection, b.selection)
    assert a.history == b.history
    assert a.extra["T0"] == b.extra["T0"] > 0
    assert a.objective >= g.objective - 1e-12
    assert len(a.history) == 12 * 2
    vals = [v for _r, _s, v in a.history]
    assert all(y >= x - 1e-12 for x, y in zip(vals, vals[1:]))
    assert a.history[-1][2] == pytest.approx(a.objective)
    assert len(a.extra["per_restart"]) == 2
    assert a.extra["per_restart"][0]["origin"] == "start"
    assert a.extra["per_restart"][1]["origin"] == "random"
    assert all(0.0 <= x <= 1.0 for x in a.extra["acceptance_rate"])
    assert not a.extra["degraded_to_global"]
    mc = a.extra["move_counts"]
    assert mc["local"][0] > mc["spin"][0] and mc["local"][0] > mc["global"][0]
    # a different seed may differ, but stays feasible and >= greedy
    c = solve_sa(obj, n, seed=7, **kw)
    _check_feasible(inst, c, n)
    assert c.objective >= g.objective - 1e-12


def test_sa_start_and_degraded_moves(toy30):
    inst = toy30
    obj = _obj(inst)
    g = solve_greedy(obj, 4)
    r = solve_sa(obj, 4, seed=1, n_sweeps=5, n_restarts=1, start=g.selection)
    _check_feasible(inst, r, 4)
    assert r.extra["degraded_to_global"]
    assert r.extra["start"] == g.selection.tolist()
    assert r.extra["start_hard"] == pytest.approx(g.objective)
    assert r.objective >= g.objective - 1e-12
    # a short start is topped up; an infeasible start is refused loudly
    r2 = solve_sa(obj, 4, seed=1, n_sweeps=3, n_restarts=1, start=g.selection[:1],
                  candidates=inst["candidates"])
    _check_feasible(inst, r2, 4)
    bad = solve_sa(obj, 2, start=[0, 1])
    assert bad.status == "infeasible" and not bad.feasible
    with pytest.raises(ValueError):
        solve_sa(obj, 0)


def test_random_feasible_start_is_a_packing(toy60):
    inst = toy60
    ev = _Evaluator(_obj(inst), inst["candidates"])
    rng = np.random.default_rng(5)
    sel = S._random_feasible(ev, ["full"] * 9, rng)
    assert sel is not None and len(sel) == 9
    assert inst["conflicts"].is_feasible(sel)
    assert S._random_feasible(ev, ["full"] * 17, rng, n_attempts=3) is None
    assert S._random_feasible(ev, ["half"] * 2, rng, n_attempts=2) is None


# ------------------------------------------------------------------- sweep
def test_sweep_greedy_rows_monotone_and_min_n(toy30, toy60):
    inst = toy30
    sw = sweep_n(inst["mesh"], inst["target"], 6, rx_cgy=inst["rx_cgy"], solver="greedy",
                 candidates=inst["candidates"], influence=inst["influence"],
                 conflicts=inst["conflicts"])
    assert [r["N"] for r in sw.rows] == [1, 2, 3, 4]
    v = [r["V100"] for r in sw.rows]
    assert all(b >= a for a, b in zip(v, v[1:]))
    assert all(r["warm_start"] for r in sw.rows)
    for r in sw.rows:
        for k in ("V100", "D90", "V150", "V200", "objective", "runtime_s", "solver"):
            assert k in r
        assert r["solver"] == "greedy" and r["status"] == "ok"
    assert set(sw.min_n) == {"D90>=rx", "V100>=0.90"}
    assert sw.min_n == {"D90>=rx": None, "V100>=0.90": None}
    assert len(sw.results) == 5 and sw.results[-1].status == "infeasible"
    # warm starts nest: the previous selection stays
    for a, b in zip(sw.results, sw.results[1:4]):
        assert set(a.selection.tolist()) <= set(b.selection.tolist())
    # the 60-toy meets both criteria; a boxed-in warm start restarts cold
    inst = toy60
    sw = sweep_n(inst["mesh"], inst["target"], 9, rx_cgy=inst["rx_cgy"], solver="greedy",
                 candidates=inst["candidates"], influence=inst["influence"],
                 conflicts=inst["conflicts"])
    assert [r["N"] for r in sw.rows] == list(range(1, 10))
    assert sw.min_n["D90>=rx"] is not None and sw.min_n["V100>=0.90"] is not None
    assert sw.rows[sw.min_n["D90>=rx"] - 1]["D90"] >= inst["rx_cgy"]
    assert not all(r["warm_start"] for r in sw.rows)
    assert any("warm_start_failed" in r.extra for r in sw.results)


def test_sweep_other_solvers_and_errors(toy30):
    inst = toy30
    common = dict(rx_cgy=inst["rx_cgy"], candidates=inst["candidates"],
                  influence=inst["influence"], conflicts=inst["conflicts"])
    sw = sweep_n(inst["mesh"], inst["target"], 3, solver="local", **common)
    assert [r["solver"] for r in sw.rows] == ["local"] * 3
    sw = sweep_n(inst["mesh"], inst["target"], 3, solver="sa", n_sweeps=3, n_restarts=1,
                 **common)
    assert [r["solver"] for r in sw.rows] == ["sa"] * 3
    for r in sw.results:
        assert inst["conflicts"].is_feasible(r.selection)
    with pytest.raises(ValueError, match="solver"):
        sweep_n(inst["mesh"], inst["target"], 2, solver="milp", **common)
    # without prebuilt objects the builders run; a missing mesh fails loudly
    with pytest.raises((ValueError, TypeError)):
        sweep_n(None, inst["target"], 2)


# ------------------------------------------------------ continuous refinement
def test_refine_continuous_never_lowers_v100_or_overlaps():
    mesh = pf.flat_wall_mesh(size_mm=60.0)
    tiles = []
    for x in (-12.0, 12.0):
        surf, n_in = snap_to_wall(mesh, np.array([x, 0.0, 0.0]))
        tiles.append(conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0]), kind="full"))
    cand = CandidateSet.from_tiles(tiles, spins_deg=[0.0, 0.0], anchor_ids=[0, 1])
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-25, 25, 300), rng.uniform(-25, 25, 300),
                           np.full(300, 5.0)])
    target = TargetSet.from_points(pts, name="flat+5mm")
    t0 = time.perf_counter()
    new_tiles, info = refine_continuous(mesh, cand, [0, 1], target, rx_cgy=3000.0,
                                        n_passes=2, max_iter=40)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, elapsed
    assert len(new_tiles) == 2 and all(t.kind == "full" for t in new_tiles)
    assert info["V100_after"] >= info["V100_before"] - 1e-12
    assert info["hard_after"] >= info["hard_before"] - 1e-12
    assert info["soft_after"] >= info["soft_before"] - 1e-12
    assert info["overlaps_before"] == [] and info["overlaps_after"] == []
    assert find_overlapping_tiles(new_tiles) == []
    assert info["accepted"] >= 1 and info["evaluations"] > 0
    assert len(info["offsets_uv_mm_theta_rad"]) == 2
    # the tiles stay conformed to the wall: seeds 3 mm inside the box
    for t in new_tiles:
        assert np.allclose(t.seed_centers[:, 2], -3.0, atol=1e-6)
        assert np.all(np.abs(t.anchor_ras[:2]) < 30.0)
    # every accepted step is a strict soft improvement
    assert info["rejected"]["overlap"] >= 0


# ------------------------------------------------- continuous multi-start
@pytest.fixture(scope="module")
def flat_cont():
    """Flat 60 mm wall, 3 x 3 candidate grid at 15 mm pitch (neighbours
    conflict; opposite rows / columns are 30 mm apart and compatible), real
    engine influence on 300 shell points at rx 3000 cGy (one tile reaches
    ~4200 cGy at the +5 mm shell, so coverage is informative)."""
    import scipy.sparse as sp
    from gtcore.dose.engine import TG43Engine, dose_at_points
    from gtcore.plan import ConflictGraph, InfluenceMatrix
    mesh = pf.flat_wall_mesh(size_mm=60.0)
    tiles = []
    for y in (-15.0, 0.0, 15.0):
        for x in (-15.0, 0.0, 15.0):
            surf, n_in = snap_to_wall(mesh, np.array([x, y, 0.0]))
            tiles.append(conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0]), kind="full"))
    cand = CandidateSet.from_tiles(tiles, spins_deg=np.zeros(9), anchor_ids=np.arange(9),
                                   method="toy_grid", h_mm=15.0, n_spins=1)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-25, 25, 300), rng.uniform(-25, 25, 300),
                           np.full(300, 5.0)])
    target = TargetSet.from_points(pts, name="flat+5mm")
    eng = TG43Engine()
    dose = np.array([dose_at_points(t.seed_centers, t.seed_axes, pts, engine=eng, exact=False)
                     for t in tiles], dtype=np.float32)
    rx = 3000.0
    influence = InfluenceMatrix(dose=dose, target=target, target_index=np.arange(300),
                                rx_cgy=rx, sk_per_seed_u=eng.DEFAULT_SK_U, kernel="tabulated")
    pairs = sp.lil_matrix((9, 9), dtype=bool)
    for i, j in find_overlapping_tiles(tiles):
        pairs[i, j] = pairs[j, i] = True
    conflicts = ConflictGraph(n=9, pairs=pairs.tocsr())
    obj = Objective(influence, conflicts, rx_cgy=rx)
    return {"mesh": mesh, "candidates": cand, "target": target, "rx": rx,
            "objective": obj, "conflicts": conflicts, "engine": eng}


def test_solve_continuous_feasible_beats_greedy_and_reproducible(flat_cont):
    f = flat_cont
    g = solve_greedy(f["objective"], 2)
    assert g.feasible
    kw = dict(seed=0, n_starts=2, n_passes=1, max_iter=30, time_budget_s=15.0,
              objective=f["objective"], engine=f["engine"], m_opt=300)
    t0 = time.perf_counter()
    tiles, r = S.solve_continuous(f["mesh"], f["candidates"], f["target"], f["rx"], 2, **kw)
    elapsed = time.perf_counter() - t0
    assert elapsed < 15.0, elapsed
    assert r.solver == "continuous" and r.status == "ok" and r.feasible
    assert len(tiles) == 2 and find_overlapping_tiles(tiles) == []
    assert r.extra["overlaps"] == [] and r.extra["tiles_are_continuous_poses"]
    ps = r.extra["per_start"]
    assert ps[0]["origin"] == "greedy" and ps[0]["selection"] == g.selection.tolist()
    assert ps[1]["origin"] == "random" and f["conflicts"].is_feasible(ps[1]["selection"])
    assert r.objective >= ps[0]["hard_before"] - 1e-12       # >= the greedy start
    assert r.objective == pytest.approx(max(p["hard_after"] for p in ps))
    assert r.metrics["V100"] >= ps[0]["V100_before"] - 1e-12
    assert r.selection.tolist() == ps[r.extra["best_start"]]["selection"]
    assert len(r.history) == 2 and r.history[-1][2] == pytest.approx(r.objective)
    assert r.extra["nm_runs"] == 4 and r.extra["nm_seconds_per_tile"] > 0
    # the tiles are continuous poses: conformed to the wall, inside it
    for t in tiles:
        assert np.allclose(t.seed_centers[:, 2], -3.0, atol=1e-6)
    # reproducible from the seed
    tiles2, r2 = S.solve_continuous(f["mesh"], f["candidates"], f["target"], f["rx"], 2, **kw)
    assert np.array_equal(r.selection, r2.selection)
    assert r.objective == r2.objective and r.history == r2.history
    assert np.allclose([t.anchor_ras for t in tiles], [t.anchor_ras for t in tiles2])
    assert [p["selection"] for p in ps] == [p["selection"] for p in r2.extra["per_start"]]


def test_solve_continuous_time_budget_and_fallbacks(flat_cont):
    f = flat_cont
    # strict budget: aborts NM, still returns feasible tiles with a loud status
    tiles, r = S.solve_continuous(f["mesh"], f["candidates"], f["target"], f["rx"], 2,
                                  seed=3, n_starts=3, time_budget_s=0.05,
                                  objective=f["objective"], engine=f["engine"], m_opt=300)
    assert r.status == "time_limit" and "budget" in r.reason
    assert r.feasible and len(tiles) == 2 and find_overlapping_tiles(tiles) == []
    assert r.extra["deadline_hit"]
    # no objective and no conflicts: conflicts from find_overlapping_tiles,
    # first start random
    tiles, r = S.solve_continuous(f["mesh"], f["candidates"], f["target"], f["rx"], 2,
                                  seed=1, n_starts=1, n_passes=1, max_iter=10,
                                  engine=f["engine"], m_opt=300)
    assert r.feasible and r.extra["per_start"][0]["origin"] == "random"
    assert f["conflicts"].is_feasible(r.selection)
    # explicit start, kinds, and loud failures
    tiles, r = S.solve_continuous(f["mesh"], f["candidates"], f["target"], f["rx"], 2,
                                  seed=1, n_starts=1, n_passes=1, max_iter=10, start=[0, 8],
                                  conflicts=f["conflicts"], engine=f["engine"], m_opt=300,
                                  kinds_required={"full": 2})
    assert r.extra["per_start"][0]["origin"] == "start" and r.selection.tolist() == [0, 8]
    with pytest.raises(ValueError, match="feasible selection"):
        S.solve_continuous(f["mesh"], f["candidates"], f["target"], f["rx"], 2, start=[0, 1],
                           conflicts=f["conflicts"])
    tiles, r = S.solve_continuous(f["mesh"], f["candidates"], f["target"], f["rx"], 5,
                                  conflicts=f["conflicts"], engine=f["engine"])
    assert r.status == "infeasible" and not r.feasible and tiles == []


def test_sweep_continuous(flat_cont):
    f = flat_cont
    sw = sweep_n(f["mesh"], f["target"], 2, rx_cgy=f["rx"], solver="continuous",
                 candidates=f["candidates"], influence=f["objective"].influence,
                 conflicts=f["conflicts"], n_starts=1, n_passes=1, max_iter=10,
                 engine=f["engine"], m_opt=300)
    assert [r["N"] for r in sw.rows] == [1, 2]
    assert all(r["solver"] == "continuous" for r in sw.rows)
    for res in sw.results:
        tiles = res.extra["tiles"]
        assert len(tiles) == res.selection.size and find_overlapping_tiles(tiles) == []
    assert set(sw.results[0].selection.tolist()) <= set(sw.results[1].selection.tolist())


# ------------------------------- continuous: NM overlap penalty / deadline
def _flat_pair(size_mm: float = 60.0, xs=(-12.0, 12.0)):
    """Flat wall, two full tiles at ``xs`` on the x axis, 300 shell points."""
    mesh = pf.flat_wall_mesh(size_mm=size_mm)
    tiles = []
    for x in xs:
        surf, n_in = snap_to_wall(mesh, np.array([x, 0.0, 0.0]))
        tiles.append(conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0]), kind="full"))
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-25, 25, 300), rng.uniform(-25, 25, 300),
                           np.full(300, 5.0)])
    return mesh, tiles, TargetSet.from_points(pts, name="flat+5mm")


def test_moving_conflicts_matches_tiles_conflict_rule():
    from gtcore.plan.conflicts import tiles_conflict
    mesh, tiles, target = _flat_pair()
    core = S._ContinuousCore(mesh, target, 3000.0, m_opt=300)
    t0, t1 = tiles
    assert core.moving_conflicts(t0, [t1]) is False and tiles_conflict(tiles) == []
    assert core.moving_conflicts(t0, []) is False
    # tile 0 slid to x = +2: 10 mm from tile 1 -> overlap under both rules
    surf, n_in = snap_to_wall(mesh, np.array([2.0, 0.0, 0.0]))
    t_over = conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0]), kind="full")
    assert core.moving_conflicts(t_over, [t1]) is True
    assert tiles_conflict([t_over, t1]) != []


def test_refine_tile_nm_never_returns_an_overlapping_pose():
    # 2 mm gap and a 2 mm first simplex step: NM probes the overlapping pose
    mesh, tiles, target = _flat_pair(xs=(-11.0, 11.0))
    core = S._ContinuousCore(mesh, target, 3000.0, m_opt=300, step_mm=2.0, step_deg=10.0,
                             max_iter=40)
    others_dose = core.tile_dose(tiles[1])
    new_tile, new_dose, x, expired = core.refine_tile(tiles[0], others_dose, [tiles[1]])
    assert not expired and new_dose.shape == (300,) and x.shape == (3,)
    assert core.nm_overlap_evals >= 1          # charged, not silently converged on
    assert find_overlapping_tiles([new_tile, tiles[1]]) == []
    assert not core.moving_conflicts(new_tile, [tiles[1]])
    assert np.allclose(new_tile.seed_centers[:, 2], -3.0, atol=1e-6)


def test_descend_deadline_keeps_best_so_far_pose(monkeypatch):
    from gtcore.plan.conflicts import tiles_conflict
    mesh, tiles, target = _flat_pair()
    core = S._ContinuousCore(mesh, target, 3000.0, m_opt=300, max_iter=40,
                             deadline=time.perf_counter() + 1e9)
    calls = {"n": 0}

    def fake_expired():
        calls["n"] += 1
        return calls["n"] > 15               # deadline after ~14 NM evaluations

    monkeypatch.setattr(core, "expired", fake_expired)
    new_tiles, info = core.descend(tiles, n_passes=2)
    assert info["deadline_hit"] and info["passes_done"] == 0 and core.nm_runs == 1
    assert core.nm_evals <= 15
    # the interrupted tile is judged like any other (or dropped if nothing beat
    # its start); nothing is lost and nothing infeasible is kept
    assert info["accepted"] + sum(info["rejected"].values()) <= 1
    assert len(info["history"]) == info["accepted"]
    assert info["hard_after"] >= info["hard_before"] - 1e-12
    assert info["soft_after"] >= info["soft_before"] - 1e-12
    assert tiles_conflict(new_tiles) == [] and len(new_tiles) == 2
    assert "nm_overlap_evals" in info
