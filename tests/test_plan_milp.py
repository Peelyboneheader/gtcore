"""Tests for ``gtcore.plan.milp`` (section 4 V1: MILP equals brute force on the
toy instance; clique rows never cut a pairwise-feasible selection; bound and
status semantics).  Everything runs on ``plan_fixtures.toy_instance``.

The default toy (pitch 10 mm) has a maximum independent set of 2, so N = 3
is infeasible there; the pitch-15 toy is feasible up to N = 4.  Both are
used so the brute-force agreement covers feasible and infeasible N.
"""
from __future__ import annotations

import dataclasses
import itertools
import time
import warnings

import numpy as np
import pytest

import gtcore.plan as plan
import plan_fixtures as pf
from gtcore.plan import ConflictGraph, InfluenceMatrix, Objective, SolverResult
from gtcore.plan.milp import (
    PRUNE_FRACTION,
    brute_force,
    build_formulation,
    evaluate_selection,
    lp_bound,
    solve_milp,
    validate_cliques,
)


# ------------------------------------------------------------------ helpers
def _objective(inst, **kw) -> Objective:
    return Objective(inst["influence"], inst["conflicts"], rx_cgy=inst["rx_cgy"], **kw)


@pytest.fixture(scope="module")
def toy10():
    return pf.toy_instance(n_candidates=12, n_targets=150, pitch_mm=10.0)


@pytest.fixture(scope="module")
def toy15():
    return pf.toy_instance(n_candidates=12, n_targets=150, pitch_mm=15.0)


def _v100(obj: Objective, ids) -> float:
    dose = obj.influence.dose_of(ids)
    w = obj.influence.target.weights
    return float(w[dose >= obj.rx_cgy].sum() / w.sum())


# ------------------------------------------------- MILP vs brute force (V1)
@pytest.mark.parametrize("pitch,n_tiles", [(10.0, 1), (10.0, 2), (10.0, 3),
                                           (15.0, 1), (15.0, 2), (15.0, 3)])
def test_milp_equals_brute_force(toy10, toy15, pitch, n_tiles):
    obj = _objective(toy10 if pitch == 10.0 else toy15)
    bf = brute_force(obj, n_tiles)
    r = solve_milp(obj, n_tiles)
    assert r.solver == "milp"
    assert r.status == bf.status
    if bf.status == "infeasible":
        assert r.selection.size == 0 and not r.feasible and r.reason
        return
    assert r.status == "optimal" and r.feasible
    assert r.selection.size == n_tiles
    assert obj.conflicts.is_feasible(r.selection)
    # the MILP's own coverage objective is the brute-force optimum
    assert r.extra["milp_objective"] >= bf.extra["milp_objective"] - 1e-6
    assert r.extra["milp_objective"] == pytest.approx(bf.extra["milp_objective"], abs=1e-6)
    # and the returned selection really achieves it (recomputed V100)
    assert _v100(obj, r.selection) == pytest.approx(bf.extra["milp_objective"], abs=1e-9)
    assert r.metrics["V100"] == pytest.approx(_v100(obj, r.selection))
    # the P1 hard value is the same function of the selection for both
    assert r.objective == pytest.approx(evaluate_selection(obj, r.selection)[0])
    assert bf.objective == pytest.approx(evaluate_selection(obj, bf.selection)[0])
    assert r.bound >= r.extra["milp_objective"] - 1e-9
    assert r.mip_gap is not None and r.mip_gap <= 1e-4 + 1e-12
    assert r.history == [(0, r.objective)]


def test_default_toy_max_independent_set_is_two(toy10):
    """Documents why N = 3 is infeasible on the default toy."""
    P = toy10["conflicts"].pairs.toarray()
    feas3 = [s for s in itertools.combinations(range(12), 3) if not P[np.ix_(s, s)].any()]
    feas2 = [s for s in itertools.combinations(range(12), 2) if not P[np.ix_(s, s)].any()]
    assert feas3 == [] and len(feas2) > 0


# ---------------------------------------------------- clique vs pairwise rows
def test_pairwise_only_and_clique_formulations_agree(toy15):
    obj = _objective(toy15)
    assert len(obj.conflicts.cliques) > 0
    for n in (2, 3):
        a = solve_milp(obj, n, use_cliques=True)
        b = solve_milp(obj, n, use_cliques=False)
        assert a.status == b.status == "optimal"
        assert a.extra["milp_objective"] == pytest.approx(b.extra["milp_objective"], abs=1e-6)
        assert a.extra["formulation"]["n_cliques_used"] > 0
        assert b.extra["formulation"]["n_cliques_used"] == 0
        # pairwise-only has one row per conflicting pair
        assert b.extra["formulation"]["n_rows_pair"] == obj.conflicts.count_pairs()
        # with cliques, every pair is covered by a clique row or a pairwise row
        assert a.extra["formulation"]["n_rows_pair"] < obj.conflicts.count_pairs()


def test_conflict_rows_equal_pairwise_graph_exactly(toy15):
    """Every subset is cut by the clique+pair rows iff it is pairwise infeasible."""
    obj = _objective(toy15)
    form = build_formulation(obj, 3)
    kinds = np.asarray(form["row_kind"])
    conf = np.isin(kinds, ["clique", "pair"])
    A = form["A"][conf][:, :form["n_x"]].toarray()
    ub = form["ub"][conf]
    n_cut = 0
    for k in (2, 3):
        for s in itertools.combinations(range(12), k):
            x = np.zeros(12)
            x[list(s)] = 1.0
            cut = bool((A @ x > ub + 1e-9).any())
            assert cut == (not obj.conflicts.is_feasible(list(s))), s
            n_cut += cut
    assert n_cut > 0


def test_bogus_clique_is_dropped_with_warning(toy15):
    """A 'clique' over mutually non-conflicting candidates must not cut them."""
    obj = _objective(toy15)
    cg = obj.conflicts
    trio = [0, 2, 8]
    assert cg.is_feasible(trio)                      # pairwise feasible
    bogus = ConflictGraph(n=cg.n, pairs=cg.pairs,
                          cliques=list(cg.cliques) + [np.asarray(trio)], gap_mm=cg.gap_mm)
    obj_bogus = dataclasses.replace(obj, conflicts=bogus)

    with pytest.warns(UserWarning, match="dropped 1 of"):
        valid, dropped = validate_cliques(bogus)
    assert len(valid) == len(cg.cliques)
    assert [d[0] for d in dropped] == [len(cg.cliques)]
    assert "non-conflicting" in dropped[0][1]

    with pytest.warns(UserWarning):
        r = solve_milp(obj_bogus, 3)
    clean = solve_milp(obj, 3)
    assert r.status == "optimal"
    assert r.extra["formulation"]["n_cliques_dropped"] == 1
    assert r.extra["milp_objective"] == pytest.approx(clean.extra["milp_objective"], abs=1e-6)

    # and the trio itself is not cut by the rows actually used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        form = build_formulation(obj_bogus, 3)
    x = np.zeros(form["n_x"] + form["n_y"])
    x[trio] = 1.0
    conf = np.isin(np.asarray(form["row_kind"]), ["clique", "pair"])
    assert not ((form["A"][conf] @ x) > form["ub"][conf] + 1e-9).any()


def test_validate_cliques_rejects_malformed(toy15):
    cg = toy15["conflicts"]
    bad = ConflictGraph(n=cg.n, pairs=cg.pairs, gap_mm=0.0, cliques=[
        np.asarray([0, 1]),          # a real conflicting pair -> valid
        np.asarray([0, 1]),          # duplicate
        np.asarray([5]),             # too small
        np.asarray([0, 0, 1]),       # repeated id
        np.asarray([0, 99]),         # out of range
    ])
    with pytest.warns(UserWarning):
        valid, dropped = validate_cliques(bad)
    assert len(valid) == 1 and list(valid[0]) == [0, 1]
    assert sorted(d[0] for d in dropped) == [1, 2, 3, 4]
    # no warning when asked not to
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_cliques(bad, warn=False)


# ------------------------------------------------------------- exact_n flag
def test_exact_n_false_never_worse(toy10, toy15):
    for inst in (toy10, toy15):
        obj = _objective(inst)
        for n in (1, 2, 3):
            le = solve_milp(obj, n, exact_n=False)
            eq = solve_milp(obj, n, exact_n=True)
            assert le.status == "optimal" and le.feasible
            assert le.selection.size <= n
            if eq.status == "optimal":
                assert le.extra["milp_objective"] >= eq.extra["milp_objective"] - 1e-9
            else:
                # exact N infeasible, but <= N is not
                assert eq.status == "infeasible"
                bf = brute_force(obj, n, exact_n=False)
                assert le.extra["milp_objective"] == pytest.approx(
                    bf.extra["milp_objective"], abs=1e-6)


# ------------------------------------------------------------- pruning
def test_pruning_threshold_does_not_change_optimum(toy15):
    obj = _objective(toy15)
    full = solve_milp(obj, 3, prune_frac=0.0)
    default = solve_milp(obj, 3)
    coarse = solve_milp(obj, 3, prune_frac=0.05)
    assert default.extra["prune_frac"] == PRUNE_FRACTION
    assert full.extra["milp_objective"] == pytest.approx(default.extra["milp_objective"], abs=1e-9)
    # the toy's inverse-square dose never falls below 1e-3 rx within the wall
    assert default.extra["formulation"]["n_pruned"] == 0
    # a coarse threshold really drops entries and can only be conservative
    assert coarse.extra["formulation"]["n_pruned"] > 0
    assert coarse.extra["formulation"]["nnz"] < full.extra["formulation"]["nnz"]
    assert coarse.extra["milp_objective"] <= full.extra["milp_objective"] + 1e-9
    # whatever the pruned model claims, the true V100 of its selection is at
    # least what it claims (never counts an uncovered point)
    assert _v100(obj, coarse.selection) >= coarse.extra["milp_objective"] - 1e-9


# ------------------------------------------------------------- cover cuts
def test_cover_cuts_do_not_change_optimum(toy15, toy10):
    """Pigeonhole rows y_m <= sum_{c: D[c,m] >= rx/N} x_c are valid for both
    count forms: the optimum equals the plain formulation and brute force."""
    from gtcore.plan.milp import COVER_CUTS, LP_COVER_CUTS
    assert COVER_CUTS is False and LP_COVER_CUTS is True   # measured, V3 notes
    obj = _objective(toy15)
    for n in (1, 2, 3, 4):
        for exact in (True, False):
            plain = solve_milp(obj, n, exact_n=exact, cover_cuts=False)
            cut = solve_milp(obj, n, exact_n=exact, cover_cuts=True)
            bf = brute_force(obj, n, exact_n=exact)
            assert plain.status == cut.status == "optimal"
            assert cut.extra["cover_cuts"] is True and plain.extra["cover_cuts"] is False
            assert cut.extra["formulation"]["n_rows_cover_cut"] > 0
            assert plain.extra["formulation"]["n_rows_cover_cut"] == 0
            assert cut.extra["milp_objective"] == pytest.approx(plain.extra["milp_objective"], abs=1e-6)
            assert cut.extra["milp_objective"] == pytest.approx(bf.extra["milp_objective"], abs=1e-6)
            assert _v100(obj, cut.selection) == pytest.approx(bf.extra["milp_objective"], abs=1e-9)
            # the LP bound with cuts is at least as tight and still a bound
            lp_plain = lp_bound(obj, n, exact_n=exact, cover_cuts=False)
            lp_cut = lp_bound(obj, n, exact_n=exact, cover_cuts=True)
            assert lp_cut <= lp_plain + 1e-9
            assert lp_cut >= cut.extra["milp_objective"] - 1e-9
    # the fixed-to-zero points really are uncoverable: at N = 1 a point needs
    # a single candidate at >= rx, so the fixed count is the number of points
    # with max_c D[c, m] < rx
    r1 = solve_milp(obj, 1, cover_cuts=True)
    D = np.asarray(obj.influence.dose, dtype=float)
    assert r1.extra["formulation"]["n_fixed_zero"] == int((D.max(axis=0) < obj.rx_cgy).sum())
    assert r1.extra["formulation"]["n_rows_coverage"] == D.shape[1] - r1.extra["formulation"]["n_fixed_zero"]
    # infeasible N stays infeasible with cuts on
    assert solve_milp(_objective(toy10), 3, cover_cuts=True).status == "infeasible"


# ------------------------------------------------------------- time limit
def test_time_limit_on_larger_toy():
    inst = pf.toy_instance(n_candidates=80, n_targets=300)
    obj = _objective(inst)
    t0 = time.perf_counter()
    r = solve_milp(obj, 6, time_limit_s=0.5)
    elapsed = time.perf_counter() - t0
    assert r.status in ("time_limit", "optimal"), r.reason
    assert elapsed < 15.0
    assert r.bound is not None and np.isfinite(r.bound)
    assert r.mip_gap is not None
    if r.status == "time_limit":
        assert "bound" in r.reason and "reference" in r.reason
    if r.selection.size:
        assert r.feasible and r.selection.size == 6
        assert obj.conflicts.is_feasible(r.selection)
        assert r.bound >= r.extra["milp_objective"] - 1e-9
        assert r.bound >= r.objective - 1e-9
        # an incumbent's y need not be maximal for its x at a time limit, so
        # the solver's value may sit below the true V100 of the selection,
        # never above it
        v100 = _v100(obj, r.selection)
        assert r.extra["milp_objective"] <= v100 + 1e-6
        if r.status == "optimal":
            assert r.extra["milp_objective"] == pytest.approx(v100, abs=1e-6)
        assert r.extra["n_rows"] == r.extra["formulation"]["n_rows"]
        assert r.extra["nnz"] > 0 and r.extra["build_s"] >= 0.0
    else:
        assert not r.feasible and r.reason
    assert r.bound <= 1.0 + 1e-6
    # the LP relaxation is a (weaker) upper bound too
    lp = lp_bound(obj, 6, time_limit_s=10.0)
    assert np.isfinite(lp) and lp >= r.bound - 1e-6


# ------------------------------------------------------------------ LP bound
def test_lp_bound_dominates_milp(toy15):
    obj = _objective(toy15)
    for n in (1, 2, 3):
        r = solve_milp(obj, n)
        lp = lp_bound(obj, n)
        assert np.isfinite(lp)
        assert lp >= r.extra["milp_objective"] - 1e-9
        assert lp <= 1.0 + 1e-9
    assert np.isnan(lp_bound(obj, 9))          # LP infeasible -> no bound
    assert lp_bound(obj, 0) == 0.0


# ------------------------------------------------------------------ OAR rows
def test_oar_limit_respected(toy10):
    obj0 = _objective(toy10)
    free = solve_milp(obj0, 2)
    assert free.status == "optimal"
    hot = int(free.selection[-1])
    # OAR samples 1 mm above the seeds of the candidate the free optimum uses
    inf0 = obj0.influence
    tile = toy10["candidates"].tiles[hot]
    oar_pts = tile.seed_centers + np.array([0.0, 0.0, 1.0])
    oar = np.empty((inf0.n_candidates, oar_pts.shape[0]), dtype=np.float32)
    for c, t in enumerate(toy10["candidates"].tiles):
        oar[c] = pf.analytic_point_dose(oar_pts, t.seed_centers, inf0.rx_cgy)
    limit = float(oar[hot].max()) * 0.5     # the free optimum violates it
    inf = InfluenceMatrix(dose=inf0.dose, target=inf0.target, target_index=inf0.target_index,
                          rx_cgy=inf0.rx_cgy, oar={"oar": oar}, oar_limits={"oar": limit},
                          kernel="analytic")
    obj = Objective(inf, obj0.conflicts, rx_cgy=obj0.rx_cgy)

    r = solve_milp(obj, 2)
    assert r.status == "optimal" and r.feasible
    assert r.extra["formulation"]["n_rows_oar"] == oar_pts.shape[0]
    assert hot not in r.selection
    dmax = inf.oar_dose_of("oar", r.selection).max()
    assert dmax <= limit * (1 + 1e-6)
    assert r.metrics["oar_dmax_oar"] == pytest.approx(dmax)
    assert r.extra["milp_objective"] <= free.extra["milp_objective"] + 1e-9
    # brute force applies the same hard limit and agrees
    bf = brute_force(obj, 2)
    assert bf.extra["milp_objective"] == pytest.approx(r.extra["milp_objective"], abs=1e-6)
    # no OAR penalty in the P1 value when the limit is met
    assert r.objective == pytest.approx(r.metrics["V100"]
                                        - obj.lambda_hot * max(0.0, r.metrics["V200"] - obj.v200_tol))
    # an infeasible OAR limit -> infeasible, not an exception
    inf_bad = dataclasses.replace(inf, oar_limits={"oar": 0.0})
    r_bad = solve_milp(Objective(inf_bad, obj0.conflicts, rx_cgy=obj0.rx_cgy), 2)
    assert r_bad.status == "infeasible" and r_bad.selection.size == 0 and r_bad.reason


# --------------------------------------------------------------- infeasible
def test_infeasible_n_reports_status(toy10, toy15):
    for inst, n in ((toy10, 3), (toy15, 5)):
        obj = _objective(inst)
        bf = brute_force(obj, n)
        assert bf.status == "infeasible"
        r = solve_milp(obj, n)
        assert isinstance(r, SolverResult)
        assert r.status == "infeasible"
        assert r.selection.size == 0 and not r.feasible
        assert r.reason and "no %d-candidate selection" % n in r.reason
        assert np.isnan(r.objective) and np.isnan(r.bound) and np.isnan(r.mip_gap)
    obj = _objective(toy10)
    r = solve_milp(obj, 13)
    assert r.status == "infeasible" and "exceeds" in r.reason
    with pytest.raises(ValueError):
        solve_milp(obj, -1)


def test_zero_tiles_is_trivial(toy10):
    r = solve_milp(_objective(toy10), 0)
    assert r.status == "optimal" and r.selection.size == 0 and r.feasible
    assert r.objective == 0.0 and r.bound == 0.0 and r.metrics["V100"] == 0.0


# --------------------------------------------------------- result contents
def test_p1_hard_value_definition(toy15):
    obj = _objective(toy15, lambda_hot=0.5, v200_tol=0.0)
    r = solve_milp(obj, 3)
    m = r.metrics
    assert set(m) >= {"V100", "V150", "V200", "D90", "Dmean"}
    assert m["V200"] > 0.0                       # inverse-square toy has hot spots
    assert r.objective == pytest.approx(m["V100"] - 0.5 * m["V200"])
    assert r.objective < r.extra["milp_objective"]  # penalty is outside the MILP
    assert 0.0 <= m["V200"] <= m["V150"] <= m["V100"] <= 1.0
    dose = obj.influence.dose_of(r.selection)
    # D90: at least 90 % of the (equal) weights are at or above it
    assert np.mean(dose >= m["D90"]) >= 0.9 - 1e-9
    assert np.mean(dose < m["D90"]) <= 0.1 + 1e-9
    for key in ("milp_objective", "formulation", "highs_status", "node_count",
                "exact_n", "time_limit_s", "mip_rel_gap", "prune_frac", "n_tiles", "v100"):
        assert key in r.extra, key
    assert r.seed is None and r.runtime_s > 0.0


def test_package_wrapper_delegates(toy15):
    obj = _objective(toy15)
    a = plan.solve_milp(obj, 2, time_limit_s=10.0, exact_n=True, mip_rel_gap=1e-4)
    b = solve_milp(obj, 2)
    assert a.status == b.status == "optimal"
    assert a.extra["milp_objective"] == pytest.approx(b.extra["milp_objective"])
    assert list(a.selection) == list(b.selection)


def test_brute_force_guard(toy15):
    obj = _objective(toy15)
    with pytest.raises(ValueError, match="max_subsets"):
        brute_force(obj, 3, max_subsets=10)
    r = brute_force(obj, 2)
    assert r.solver == "brute_force" and r.status == "optimal"
    assert r.extra["n_enumerated"] == 66 and 0 < r.extra["n_feasible"] < 66
    assert r.bound == r.extra["milp_objective"] and r.mip_gap == 0.0
