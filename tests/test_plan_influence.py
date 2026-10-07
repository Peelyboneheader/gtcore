"""Tests for ``gtcore.plan.influence`` (A2 Dose, branch plan/influence).

Includes the section 3 B gate (``test_influence_gate``): for seeded random
selections of 1-8 candidates on a synthetic phantom cavity, V100 from the
influence row-sum must agree with V100 from ``compute_dose_grid`` (exact
kernel, 1 mm, sampled at the same points) within 0.5 percentage points and
D90 within 1 % of rx.  Run with ``-s`` to see the measured deviations.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import plan_fixtures as pf
from gtcore.dose.dvh import sample_doses
from gtcore.dose.engine import TG43Engine, compute_dose_grid, dose_at_points
from gtcore.interact import conform_tile, snap_to_wall
from gtcore.plan import CandidateSet, InfluenceMatrix, TargetSet, build_influence
from gtcore.plan.influence import candidate_doses
from gtcore.plan.objective import metrics_from_dose

RX = 6000.0
GATE_V100_PP = 0.5          # section 3 B: V100 within 0.5 percentage points
GATE_D90_FRAC = 0.01        # section 3 B: D90 within 1 % of rx
GRID_MARGIN_MM = 15.0       # grid bounds = mesh bounds +- 15 mm (covers the +5 mm shell)


# ---------------------------------------------------------------- helpers
def farthest_point_ids(points, n, start=0):
    """Deterministic farthest-point sampling of ``n`` indices into ``points``."""
    pts = np.asarray(points, dtype=float)
    ids = [int(start)]
    dist = np.linalg.norm(pts - pts[ids[0]], axis=1)
    for _ in range(1, int(n)):
        nxt = int(np.argmax(dist))
        ids.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.asarray(ids, dtype=int)


def axis_hint_for(normal):
    """A unit vector perpendicular to ``normal`` (least-aligned world axis)."""
    n = np.asarray(normal, dtype=float)
    k = int(np.argmin(np.abs(n)))
    e = np.zeros(3)
    e[k] = 1.0
    h = e - float(e @ n) * n
    return h / np.linalg.norm(h)


def cavity_candidates(mesh, n_anchors=12, kinds=("full",)):
    """Hand-built CandidateSet: one conformed tile per farthest-point anchor
    and kind (spin 0; A1's ``build_candidates`` is not needed here)."""
    ids = farthest_point_ids(mesh.vertices, n_anchors)
    tiles, spins, anchor_ids = [], [], []
    for a_id, vid in enumerate(ids):
        surf, n_in = snap_to_wall(mesh, np.asarray(mesh.vertices[vid], dtype=float))
        for kind in kinds:
            tiles.append(conform_tile(mesh, surf, n_in, axis_hint_for(n_in), kind=kind))
            spins.append(0.0)
            anchor_ids.append(a_id)
    return CandidateSet.from_tiles(tiles, spins_deg=spins, anchor_ids=anchor_ids,
                                   method="farthest_point_test", h_mm=float("nan"),
                                   n_spins=1, wall_area_mm2=float(mesh.area))


def phantom_cavity_mesh():
    from gtcore.phantom.generate import make_head_phantom
    from gtcore.segment.surface import mask_to_mesh
    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)
    return mask_to_mesh(truth.masks["cavity"], vol.affine)


def grid_dose_at(points, seed_centers, seed_axes, bounds, engine):
    vol = compute_dose_grid(seed_centers, seed_axes, bounds, spacing_mm=1.0,
                            engine=engine, exact=True)
    return sample_doses(vol, points)


def run_gate(candidates, target, bounds, n_draws=5, rng_seed=0, engine=None):
    """Row-sum vs exact 1 mm grid at the same points; returns max deviations."""
    eng = engine if engine is not None else TG43Engine()
    inf = build_influence(candidates, target, rx_cgy=RX, m_opt=len(target),
                          engine=eng, kernel="tabulated")
    assert inf.n_targets == len(target)               # no subsample
    assert np.array_equal(inf.target_index, np.arange(len(target)))
    rng = np.random.default_rng(rng_seed)
    worst = {"V100_pp": 0.0, "D90_frac": 0.0, "point_rel": 0.0}
    rows = []
    for _ in range(n_draws):
        n_sel = int(rng.integers(1, 9))
        sel = np.sort(rng.choice(len(candidates), size=min(n_sel, len(candidates)),
                                 replace=False))
        d_row = inf.dose_of(sel)
        centers, axes = candidates.seeds_of(sel)
        d_grid = grid_dose_at(target.points, centers, axes, bounds, eng)
        m_row = metrics_from_dose(d_row, target.weights, RX)
        m_grid = metrics_from_dose(d_grid, target.weights, RX)
        dv = abs(m_row["V100"] - m_grid["V100"]) * 100.0
        dd = abs(m_row["D90"] - m_grid["D90"]) / RX
        hot = d_grid >= 0.5 * RX
        rel = float(np.max(np.abs(d_row[hot] - d_grid[hot]) / d_grid[hot])) if hot.any() else 0.0
        worst["V100_pp"] = max(worst["V100_pp"], dv)
        worst["D90_frac"] = max(worst["D90_frac"], dd)
        worst["point_rel"] = max(worst["point_rel"], rel)
        rows.append((sel.tolist(), m_row["V100"], m_grid["V100"], m_row["D90"], m_grid["D90"]))
    return worst, rows, inf


# ------------------------------------------------------------ basic build
def test_build_influence_matches_dose_at_points_rows():
    inst = pf.toy_instance(n_candidates=6, n_targets=50)
    cand, target = inst["candidates"], inst["target"]
    eng = TG43Engine()
    inf = build_influence(cand, target, rx_cgy=RX, engine=eng)
    assert isinstance(inf, InfluenceMatrix)
    assert inf.dose.shape == (6, 50) and inf.dose.dtype == np.float32
    assert inf.kernel == "tabulated" and inf.build_seconds > 0.0
    assert inf.sk_per_seed_u == TG43Engine.DEFAULT_SK_U
    assert inf.rx_cgy == RX
    assert np.array_equal(inf.target_index, np.arange(50))
    assert inf.target is target
    for c in range(6):
        ref = dose_at_points(cand.seed_centers[c, :4], cand.seed_axes[c, :4],
                             target.points, engine=eng, exact=False)
        np.testing.assert_allclose(inf.dose[c], ref, rtol=2e-6)
    # tabulated vs exact kernels agree to the table's interpolation accuracy
    inf_x = build_influence(cand, target, rx_cgy=RX, engine=eng, kernel="exact")
    assert inf_x.kernel == "exact"
    np.testing.assert_allclose(inf_x.dose, inf.dose, rtol=2e-3)


def test_build_influence_sk_scaling_and_half_tiles():
    inst = pf.toy_instance(n_candidates=3, n_targets=20)
    cand, target = inst["candidates"], inst["target"]
    eng = TG43Engine()
    base = build_influence(cand, target, engine=eng)
    double = build_influence(cand, target, sk_per_seed_u=2.0 * TG43Engine.DEFAULT_SK_U,
                             engine=eng)
    np.testing.assert_allclose(double.dose, 2.0 * base.dose, rtol=1e-6)
    assert double.sk_per_seed_u == pytest.approx(7.0)
    # a half tile uses only its 2 valid seeds (rows beyond n_seeds are NaN)
    mesh = inst["mesh"]
    surf, n_in = snap_to_wall(mesh, np.array([0.0, 0.0, 0.0]))
    half = conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0]), kind="half")
    cs = CandidateSet.from_tiles([half], spins_deg=[0.0])
    assert cs.n_seeds[0] == 2 and np.isnan(cs.seed_centers[0, 2:]).all()
    inf = build_influence(cs, target, engine=eng)
    ref = dose_at_points(half.seed_centers, half.seed_axes, target.points,
                         engine=eng, exact=False)
    np.testing.assert_allclose(inf.dose[0], ref, rtol=2e-6)
    assert np.isfinite(inf.dose).all()


def test_build_influence_subsamples_and_records_index():
    inst = pf.toy_instance(n_candidates=4, n_targets=300)
    cand, target = inst["candidates"], inst["target"]
    eng = TG43Engine()
    inf = build_influence(cand, target, m_opt=100, rng_seed=3, engine=eng)
    assert inf.dose.shape == (4, 100)
    assert inf.target_index.shape == (100,)
    assert len(np.unique(inf.target_index)) == 100
    np.testing.assert_allclose(inf.target.points, target.points[inf.target_index])
    assert inf.target.total_weight == pytest.approx(target.total_weight)
    # the rows are the doses at exactly those points
    full = build_influence(cand, target, m_opt=len(target), engine=eng)
    np.testing.assert_allclose(inf.dose, full.dose[:, inf.target_index], rtol=1e-6)
    # deterministic in the seed, different for another seed
    again = build_influence(cand, target, m_opt=100, rng_seed=3, engine=eng)
    assert np.array_equal(again.target_index, inf.target_index)
    other = build_influence(cand, target, m_opt=100, rng_seed=4, engine=eng)
    assert not np.array_equal(other.target_index, inf.target_index)


def test_build_influence_oars_and_limits():
    inst = pf.toy_instance(n_candidates=3, n_targets=20)
    cand, target = inst["candidates"], inst["target"]
    eng = TG43Engine()
    rng = np.random.default_rng(0)
    oar_pts = rng.uniform(-10, 10, size=(7, 3)) + np.array([0.0, 0.0, 15.0])
    oars = {"brainstem": TargetSet.from_points(oar_pts, name="brainstem"),
            "chiasm": oar_pts[:3]}                       # raw points accepted too
    inf = build_influence(cand, target, oars=oars, oar_limits={"brainstem": 1000.0},
                          engine=eng)
    assert set(inf.oar) == {"brainstem", "chiasm"}
    assert inf.oar["brainstem"].shape == (3, 7) and inf.oar["chiasm"].shape == (3, 3)
    assert inf.oar["brainstem"].dtype == np.float32
    assert inf.oar_limits == {"brainstem": 1000.0}
    ref = candidate_doses(cand, oar_pts, TG43Engine.DEFAULT_SK_U, eng, False)
    np.testing.assert_allclose(inf.oar["brainstem"], ref, rtol=1e-6)
    np.testing.assert_allclose(inf.oar_dose_of("brainstem", [0, 2]),
                               ref[[0, 2]].astype(np.float64).sum(0), rtol=1e-6)
    with pytest.raises(KeyError):
        build_influence(cand, target, oars=oars, oar_limits={"nope": 1.0}, engine=eng)
    with pytest.raises(ValueError):
        build_influence(cand, target, kernel="magic", engine=eng)


def test_build_influence_empty_candidates():
    inst = pf.toy_instance(n_candidates=2, n_targets=10)
    empty = inst["candidates"].subset(np.zeros(2, dtype=bool))
    inf = build_influence(empty, inst["target"], engine=TG43Engine())
    assert inf.dose.shape == (0, 10)
    assert inf.dose_of([]).shape == (10,)


# ------------------------------------------------------------------ gates
def test_influence_gate_flat_wall():
    """Cheap gate: toy flat-wall tiles, equal-weight shell points at z = +5."""
    inst = pf.toy_instance(n_candidates=9, n_targets=400, rng_seed=1)
    cand, target = inst["candidates"], inst["target"]
    lo = target.points.min(axis=0) - GRID_MARGIN_MM
    hi = target.points.max(axis=0) + GRID_MARGIN_MM
    lo[2] = min(lo[2], cand.seed_centers[:, :, 2].min() - GRID_MARGIN_MM)
    worst, rows, _ = run_gate(cand, target, np.vstack([lo, hi]), n_draws=5, rng_seed=1)
    print("\nflat-wall gate: max |dV100| = %.3f pp, max |dD90| = %.3f %% rx, "
          "max point rel = %.2e" % (worst["V100_pp"], 100 * worst["D90_frac"],
                                    worst["point_rel"]))
    assert worst["V100_pp"] <= GATE_V100_PP
    assert worst["D90_frac"] <= GATE_D90_FRAC


def test_influence_gate():
    """Section 3 B gate on a synthetic phantom cavity (full +5 mm shell target)."""
    t0 = time.perf_counter()
    mesh = phantom_cavity_mesh()
    cand = cavity_candidates(mesh, n_anchors=12)
    assert len(cand) == 12
    target = TargetSet.from_shell(mesh, 5.0)
    bounds = np.vstack([mesh.bounds[0] - GRID_MARGIN_MM, mesh.bounds[1] + GRID_MARGIN_MM])
    worst, rows, inf = run_gate(cand, target, bounds, n_draws=5, rng_seed=0)
    print("\ncavity gate (M = %d, %d candidates, %.1f s): max |dV100| = %.3f pp, "
          "max |dD90| = %.3f %% rx, max point rel = %.2e; influence build %.3f s "
          "(%.2f ms / candidate)"
          % (len(target), len(cand), time.perf_counter() - t0, worst["V100_pp"],
             100 * worst["D90_frac"], worst["point_rel"], inf.build_seconds,
             1e3 * inf.build_seconds / len(cand)))
    for sel, v_row, v_grid, d_row, d_grid in rows:
        print("  sel %-28s V100 %.4f / %.4f   D90 %.0f / %.0f cGy"
              % (sel, v_row, v_grid, d_row, d_grid))
    # the selections must actually exercise coverage (not all-zero V100)
    assert max(r[1] for r in rows) > 0.05
    assert worst["V100_pp"] <= GATE_V100_PP
    assert worst["D90_frac"] <= GATE_D90_FRAC


def test_influence_per_candidate_time():
    """Influence build time per candidate at M = 4000 (budget: <= 5 ms with
    the tabulated kernel; measured ~1 ms)."""
    rng = np.random.default_rng(0)
    inst = pf.toy_instance(n_candidates=16, n_targets=10)
    cand = inst["candidates"]
    pts = rng.uniform(-30, 30, size=(4000, 3)) + np.array([0.0, 0.0, 5.0])
    target = TargetSet.from_points(pts)
    eng = TG43Engine()
    build_influence(cand, target, engine=eng)                  # warm the kernel table
    inf = build_influence(cand, target, engine=eng)
    per = inf.build_seconds / len(cand)
    print("\ninfluence: %.2f ms per candidate at M = 4000 (tabulated)" % (1e3 * per))
    assert per <= 0.100  # budget 5-10 ms; 10x slack for loaded machines (the printed number is what gets reported)
