"""Tests for ``gtcore.plan.objective`` (A2 Dose, branch plan/influence)."""
from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

import plan_fixtures as pf
from gtcore.plan import (
    LAMBDA_HOT,
    LAMBDA_OAR,
    TAU_FRACTION,
    V200_TOL,
    ConflictGraph,
    InfluenceMatrix,
    Objective,
    TargetSet,
    evaluate,
    make_objective,
)
from gtcore.plan import objective as ob

RX = 6000.0


# ----------------------------------------------------------------- helpers
def synthetic(c_n, m, rng_seed=0, dose_scale=3000.0, weights="random", oars=None,
              oar_limits=None):
    """Random InfluenceMatrix (float32 doses in [0, dose_scale]) + no conflicts."""
    rng = np.random.default_rng(rng_seed)
    dose = rng.uniform(0.0, dose_scale, size=(c_n, m)).astype(np.float32)
    if weights == "random":
        w = rng.uniform(0.5, 1.5, size=m)
    else:
        w = np.ones(m)
    target = TargetSet.from_points(rng.normal(size=(m, 3)), w)
    oar = {}
    if oars:
        for name, mo in oars.items():
            oar[name] = rng.uniform(0.0, dose_scale / 4, size=(c_n, mo)).astype(np.float32)
    inf = InfluenceMatrix(dose=dose, target=target, target_index=np.arange(m),
                          rx_cgy=RX, oar=oar, oar_limits=dict(oar_limits or {}),
                          kernel="synthetic")
    conflicts = ConflictGraph(n=c_n, pairs=sp.csr_matrix((c_n, c_n), dtype=bool))
    return inf, conflicts


def brute_metrics(dose, weights, rx):
    """Independent re-derivation: expand weights by repetition where integer."""
    d = np.asarray(dose, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    out = {
        "V100": float(w[d >= rx].sum() / total),
        "V150": float(w[d >= 1.5 * rx].sum() / total),
        "V200": float(w[d >= 2.0 * rx].sum() / total),
        "Dmean": float((w * d).sum() / total),
    }
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(w[order])
    i = int(np.argmax(cum >= 0.10 * total - 1e-9))
    out["D90"] = float(d[order][i])
    return out


# --------------------------------------------------------- weighted_quantile
@pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 25, 64, 200, 1001])
def test_weighted_quantile_unit_weights_matches_numpy_inverted_cdf(n):
    rng = np.random.default_rng(n)
    v = rng.normal(size=n)
    srt = np.sort(v)
    for q in np.concatenate([np.linspace(0.0, 1.0, 41), rng.uniform(0, 1, 40)]):
        got = ob.weighted_quantile(v, np.ones(n), q)
        ref = float(np.percentile(v, 100.0 * q, method="inverted_cdf"))
        nq = n * q
        if abs(nq - round(nq)) < 1e-9 and nq > round(nq):
            # n*q overshoots an integer k by floating-point rounding: numpy
            # steps to element k+1, the inverted-CDF definition says element k
            assert got == srt[int(round(nq)) - 1]
        else:
            assert got == ref, (n, q)


def test_weighted_quantile_weights_equal_repetition():
    rng = np.random.default_rng(1)
    v = rng.normal(size=30)
    reps = rng.integers(1, 6, size=30)
    expanded = np.repeat(v, reps)
    for q in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        got = ob.weighted_quantile(v, reps.astype(float), q)
        ref = ob.weighted_quantile(expanded, np.ones(expanded.size), q)
        assert got == ref
        assert got in v
    # scale invariance of the weights
    assert ob.weighted_quantile(v, 0.37 * reps, 0.3) == ob.weighted_quantile(v, reps, 0.3)


def test_weighted_quantile_edge_cases():
    v = np.array([5.0, 1.0, 3.0])
    assert ob.weighted_quantile(v, [1, 1, 1], 0.0) == 1.0
    assert ob.weighted_quantile(v, [1, 1, 1], 1.0) == 5.0
    # zero-weight entries are ignored
    assert ob.weighted_quantile(v, [0, 1, 1], 0.0) == 1.0
    assert ob.weighted_quantile(v, [1, 0, 1], 0.0) == 3.0
    assert ob.weighted_quantile(v, [1, 0, 0], 0.5) == 5.0
    with pytest.raises(ValueError):
        ob.weighted_quantile(v, [0, 0, 0], 0.5)
    with pytest.raises(ValueError):
        ob.weighted_quantile([], [], 0.5)
    with pytest.raises(ValueError):
        ob.weighted_quantile(v, [1, 1, 1], 1.5)
    with pytest.raises(ValueError):
        ob.weighted_quantile(v, [1, -1, 1], 0.5)


# ---------------------------------------------------------------- metrics
def test_metrics_toy_instance_against_brute_force():
    inst = pf.toy_instance(n_candidates=16, n_targets=300, rng_seed=2)
    obj = make_objective(inst["influence"], inst["conflicts"])
    assert obj.rx_cgy == inst["rx_cgy"]
    rng = np.random.default_rng(0)
    w = inst["target"].weights
    for _ in range(6):
        sel = rng.choice(16, size=int(rng.integers(1, 7)), replace=False)
        d = inst["influence"].dose[np.sort(sel)].astype(np.float64).sum(0)
        np.testing.assert_allclose(obj.dose_of(sel), d)
        m = obj.metrics(sel)
        ref = brute_metrics(d, w, obj.rx_cgy)
        for k, val in ref.items():
            assert m[k] == pytest.approx(val, rel=1e-12), k
        assert set(m) == {"V100", "V150", "V200", "D90", "Dmean"}
    # empty selection
    m0 = obj.metrics([])
    assert m0["V100"] == 0.0 and m0["D90"] == 0.0 and m0["Dmean"] == 0.0
    # boolean mask selection equals the id selection
    mask = np.zeros(16, dtype=bool)
    mask[[1, 5, 9]] = True
    assert obj.hard(mask) == obj.hard([9, 1, 5])


def test_metrics_weighted_and_oar_keys():
    inf, cg = synthetic(5, 40, weights="random", oars={"stem": 6, "eye": 3},
                        oar_limits={"stem": 500.0})
    obj = make_objective(inf, cg)
    m = obj.metrics([0, 3])
    ref = brute_metrics(inf.dose_of([0, 3]), inf.target.weights, RX)
    for k, val in ref.items():
        assert m[k] == pytest.approx(val, rel=1e-12)
    assert m["oar_dmax_stem"] == pytest.approx(inf.oar_dose_of("stem", [0, 3]).max())
    assert m["oar_dmax_eye"] == pytest.approx(inf.oar_dose_of("eye", [0, 3]).max())
    e = evaluate(obj, [0, 3])
    assert set(e) == {"hard", "soft", "metrics", "feasible"}
    assert e["metrics"] == m


# ------------------------------------------------------------- hard / soft
def test_hard_and_soft_monotone_in_dose():
    """Scaling every dose up never lowers coverage (penalties off)."""
    rng = np.random.default_rng(3)
    base = rng.uniform(0.5 * RX, 2.0 * RX, size=(1, 500)).astype(np.float32)
    target = TargetSet.from_points(rng.normal(size=(500, 3)), rng.uniform(0.5, 2.0, 500))
    cg = ConflictGraph(n=1, pairs=sp.csr_matrix((1, 1), dtype=bool))
    prev_h, prev_s = -1.0, -1.0
    for s in np.linspace(0.2, 3.0, 29):
        inf = InfluenceMatrix(dose=s * base, target=target, target_index=np.arange(500), rx_cgy=RX)
        obj = make_objective(inf, cg, lambda_hot=0.0)
        h, so = obj.hard([0]), obj.soft([0])
        assert h >= prev_h - 1e-15 and so >= prev_s - 1e-15
        assert 0.0 <= h <= 1.0 and 0.0 <= so <= 1.0
        prev_h, prev_s = h, so
    assert prev_h == 1.0 and prev_s > 0.99
    # the soft coverage is a smooth version of the hard one around rx
    inf = InfluenceMatrix(dose=base, target=target, target_index=np.arange(500), rx_cgy=RX)
    obj = make_objective(inf, cg, lambda_hot=0.0)
    assert abs(obj.soft([0]) - obj.hard([0])) < 0.05
    assert obj.tau_cgy == pytest.approx(TAU_FRACTION * RX)
    # a narrower tau brings soft closer to hard
    obj2 = make_objective(inf, cg, lambda_hot=0.0, tau_cgy=1.0)
    assert abs(obj2.soft([0]) - obj2.hard([0])) < 1e-3


def test_hot_penalty_kicks_in_exactly_at_tolerance():
    m = 20
    target = TargetSet.from_points(np.zeros((m, 3)))           # unit weights
    cg = ConflictGraph(n=1, pairs=sp.csr_matrix((1, 1), dtype=bool))
    for k in range(0, 8):                                       # k points at >= 2 rx
        dose = np.full((1, m), 1.2 * RX, dtype=np.float32)
        dose[0, :k] = 2.0 * RX
        inf = InfluenceMatrix(dose=dose, target=target, target_index=np.arange(m), rx_cgy=RX)
        obj = make_objective(inf, cg)
        v200 = k / m
        expected = 1.0 - LAMBDA_HOT * max(0.0, v200 - V200_TOL)
        assert obj.hard([0]) == pytest.approx(expected, abs=1e-12)
        assert obj.metrics([0])["V200"] == pytest.approx(v200)
        # at exactly the tolerance (k = 2 -> V200 = 0.10) there is no penalty
        if k == 2:
            assert obj.hard([0]) == pytest.approx(1.0, abs=1e-12)
        # soft carries the same penalty
        expected_soft = ob.soft_coverage(obj, inf.dose_of([0])) - LAMBDA_HOT * max(0.0, v200 - V200_TOL)
        assert obj.soft([0]) == pytest.approx(expected_soft, abs=1e-12)
        # custom weights
        obj3 = make_objective(inf, cg, lambda_hot=2.0, v200_tol=0.0)
        assert obj3.hard([0]) == pytest.approx(1.0 - 2.0 * v200, abs=1e-12)


def test_oar_penalty_kicks_in_exactly_at_limit():
    m = 10
    target = TargetSet.from_points(np.zeros((m, 3)))
    cg = ConflictGraph(n=2, pairs=sp.csr_matrix((2, 2), dtype=bool))
    dose = np.full((2, m), 0.6 * RX, dtype=np.float32)           # two tiles -> rx everywhere
    limit = 800.0
    for dmax in (0.0, 400.0, 799.0, 800.0, 801.0, 1300.0):
        oar = {"stem": np.array([[dmax / 2, 0.0, 0.0], [dmax / 2, 0.0, 0.0]], dtype=np.float32),
               "free": np.array([[5000.0], [5000.0]], dtype=np.float32)}    # no limit
        inf = InfluenceMatrix(dose=dose, target=target, target_index=np.arange(m),
                              rx_cgy=RX, oar=oar, oar_limits={"stem": limit})
        obj = make_objective(inf, cg)
        expected = 1.0 - LAMBDA_OAR * max(0.0, dmax - limit)
        assert obj.hard([0, 1]) == pytest.approx(expected, rel=1e-12, abs=1e-9)
        assert obj.metrics([0, 1])["oar_dmax_stem"] == pytest.approx(dmax)
        assert obj.metrics([0, 1])["oar_dmax_free"] == pytest.approx(10000.0)
        s_cov = ob.soft_coverage(obj, inf.dose_of([0, 1]))
        assert obj.soft([0, 1]) == pytest.approx(s_cov - LAMBDA_OAR * max(0.0, dmax - limit),
                                                  rel=1e-12, abs=1e-9)
        # a single tile stays below the limit -> its gain carries the full penalty jump
        g = obj.gain([0], 1)
        assert g == pytest.approx(obj.hard([0, 1]) - obj.hard([0]), rel=1e-9, abs=1e-9)
        ga = ob.gains_all(obj, [0])
        assert ga[1] == pytest.approx(g, abs=1e-12) and ga[0] == 0.0


# ------------------------------------------------------------------ gains
@pytest.mark.parametrize("oars", [None, {"stem": 9}])
def test_gains_all_equals_gain_loop_and_hard_difference(oars):
    limits = {"stem": 300.0} if oars else None
    inf, cg = synthetic(60, 150, rng_seed=5, dose_scale=0.8 * RX, oars=oars, oar_limits=limits)
    obj = make_objective(inf, cg)
    rng = np.random.default_rng(1)
    for n_sel in (0, 1, 3, 8):
        sel = np.sort(rng.choice(60, size=n_sel, replace=False))
        ga = ob.gains_all(obj, sel)
        assert ga.shape == (60,)
        loop = np.array([obj.gain(sel, c) for c in range(60)])
        np.testing.assert_allclose(ga, loop, rtol=0, atol=1e-12)
        h0 = obj.hard(sel)
        full = np.array([obj.hard(np.append(sel, c)) - h0 for c in range(60)])
        np.testing.assert_allclose(ga, full, rtol=0, atol=1e-9)
        assert np.all(ga[sel] == 0.0)
        # cached dose vector path and boolean-mask selection give the same answer
        mask = np.zeros(60, dtype=bool)
        mask[sel] = True
        np.testing.assert_array_equal(ob.gains_all(obj, mask, dose_vec=obj.dose_of(sel)), ga)
        # soft gains
        sg = ob.soft_gains_all(obj, sel)
        s0 = obj.soft(sel)
        sfull = np.array([obj.soft(np.append(sel, c)) - s0 for c in range(60)])
        np.testing.assert_allclose(sg, sfull, rtol=0, atol=1e-9)
        assert np.all(sg[sel] == 0.0)
    with pytest.raises(IndexError):
        obj.gain([0], 60)
    with pytest.raises(ValueError):
        ob.gains_all(obj, [0], dose_vec=np.zeros(3))


def test_gain_is_additive_coverage_on_toy_instance():
    inst = pf.toy_instance(n_candidates=9, n_targets=200)
    obj = make_objective(inst["influence"], inst["conflicts"], lambda_hot=0.0)
    # the gain of the first tile equals its own V100
    for c in range(9):
        assert obj.gain([], c) == pytest.approx(obj.metrics([c])["V100"])
    ga = ob.gains_all(obj, [])
    assert int(np.argmax(ga)) == int(np.argmax([obj.metrics([c])["V100"] for c in range(9)]))


# ------------------------------------------------------- constructors / eval
def test_make_objective_weights_and_errors():
    inst = pf.toy_instance(n_candidates=4, n_targets=20)
    inf, cg = inst["influence"], inst["conflicts"]
    obj = make_objective(inf, cg)
    assert isinstance(obj, Objective)
    assert obj.rx_cgy == inf.rx_cgy and obj.lambda_hot == LAMBDA_HOT
    assert obj.v200_tol == V200_TOL and obj.lambda_oar == LAMBDA_OAR
    assert obj.tau_cgy == pytest.approx(TAU_FRACTION * inf.rx_cgy)
    obj2 = make_objective(inf, cg, lambda_hot=0.1, v200_tol=0.2, lambda_oar=5.0,
                          tau_cgy=10.0, rx_cgy=5000.0)
    assert (obj2.lambda_hot, obj2.v200_tol, obj2.lambda_oar, obj2.tau_cgy, obj2.rx_cgy) \
        == (0.1, 0.2, 5.0, 10.0, 5000.0)
    with pytest.raises(TypeError):
        make_objective(inf, cg, lambda_cold=1.0)
    bad = ConflictGraph(n=3, pairs=sp.csr_matrix((3, 3), dtype=bool))
    with pytest.raises(ValueError):
        make_objective(inf, bad)
    # bound methods route to the module functions
    assert obj.hard([0, 2]) == ob.hard(obj, [0, 2])
    assert obj.soft([0, 2]) == ob.soft(obj, [0, 2])
    assert obj.gain([0], 2) == ob.gain(obj, [0], 2)
    assert obj.metrics([0]) == ob.metrics(obj, [0])
    np.testing.assert_array_equal(obj.dose_of([0]), ob.dose_of(obj, [0]))


def test_evaluate_feasibility_flag():
    inst = pf.toy_instance(n_candidates=16, n_targets=100)
    obj = make_objective(inst["influence"], inst["conflicts"])
    cg = inst["conflicts"]
    # on the 10 mm grid, anchors 0 and 1 abut (conflict); 0 and 3 are 30 mm apart
    assert cg.conflicts(0, 1) and not cg.conflicts(0, 3)
    e_bad = evaluate(obj, [0, 1])
    e_ok = evaluate(obj, [0, 3])
    assert e_bad["feasible"] is False and e_ok["feasible"] is True
    assert e_ok["hard"] == pytest.approx(obj.hard([0, 3]))
    assert e_ok["soft"] == pytest.approx(obj.soft([0, 3]))
    assert e_ok["metrics"] == obj.metrics([0, 3])
    assert evaluate(obj, [])["feasible"] is True
    assert evaluate(obj, [])["hard"] == 0.0


# ------------------------------------------------------------- performance
def test_hard_timing_m4000_n8():
    inf, cg = synthetic(64, 4000, rng_seed=0)
    obj = make_objective(inf, cg)
    sel = np.arange(0, 64, 8)
    obj.hard(sel)
    t0 = time.perf_counter()
    for _ in range(100):
        obj.hard(sel)
    per = (time.perf_counter() - t0) / 100.0
    print("\nhard(): %.3f ms per call at M = 4000, N = 8" % (1e3 * per))
    assert per <= 0.020  # budget 2 ms; 10x slack so a loaded machine does not fail the suite (the printed number is what gets reported)


def test_gains_all_timing_c2000_m4000():
    inf, cg = synthetic(2000, 4000, rng_seed=0)
    obj = make_objective(inf, cg)
    sel = np.arange(0, 2000, 250)
    ob.gains_all(obj, sel)
    t0 = time.perf_counter()
    for _ in range(5):
        ob.gains_all(obj, sel)
    per = (time.perf_counter() - t0) / 5.0
    print("\ngains_all(): %.1f ms at C = 2000, M = 4000" % (1e3 * per))
    assert per <= 1.000  # budget 100 ms; 10x slack for loaded machines (the printed number is what gets reported)
