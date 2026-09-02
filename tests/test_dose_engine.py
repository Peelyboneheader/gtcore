"""Verification campaign for the corrected, vectorized TG-43U1 engine (v2).

Layers of evidence, roughly independent of one another:

- reference identities at (r0, theta0);
- an *independent numerical integration* of the line-source geometry
  function (1/d^2 integrated along the active line) against G_L;
- v1-vs-v2 regression where v1 is valid (transverse / mid-angle points);
- regressions on each of the five documented v1 defects;
- exact superposition of per-seed grids;
- grid-vs-point cross-check through trilinear sampling;
- physical sanity against the GammaTile prescription (60 Gy at 5 mm);
- nested isodose surfaces;
- a wall-clock performance budget.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from gtcore.dose import DoseInterpolator, TG43Engine, compute_dose_grid, \
    isodose_surfaces
from gtcore.phantom import make_head_phantom


@pytest.fixture(scope="module")
def eng():
    return TG43Engine()


@pytest.fixture(scope="module")
def v1():
    return DoseInterpolator()


# ----------------------------------------------------------------- identities
def test_reference_rate_is_lambda(eng):
    """Rate at (theta0=90, r0=1 cm) equals Lambda within 2%."""
    rate = eng.dose_rate(90.0, 1.0)
    assert rate == pytest.approx(TG43Engine._LAMBDA, rel=0.02)


def test_reference_factors_are_unity(eng):
    assert eng._radial_dose(1.0) == pytest.approx(1.0, rel=0.02)
    assert float(eng._anisotropy(1.0, 90.0)) == pytest.approx(1.0, abs=1e-12)
    g = eng._geometry_factor(np.array(1.0), np.array(90.0))
    assert float(g) / eng._GL_ref == pytest.approx(1.0, abs=1e-12)


def test_constants_match_v1_exactly(eng, v1):
    """Seed data must be copied byte-for-byte from the v1 port."""
    assert TG43Engine._LAMBDA == DoseInterpolator._LAMBDA
    assert TG43Engine._L == DoseInterpolator._L
    assert np.array_equal(TG43Engine._GR_COEFFS, DoseInterpolator._GR_COEFFS)
    assert np.array_equal(TG43Engine._F_THETAS, DoseInterpolator._F_THETAS)
    assert np.array_equal(TG43Engine._F_RADII, DoseInterpolator._F_RADII)
    assert np.array_equal(TG43Engine._F_TABLE, DoseInterpolator._F_TABLE,
                          equal_nan=True)
    assert eng._GL_ref == pytest.approx(v1._GL_ref, rel=1e-12)


# --------------------------------------- independent geometry-function check
def test_geometry_factor_against_numerical_integration(eng):
    """G_L(r, theta) == (1/L) * integral of 1/d^2 along the active line.

    The TG-43U1 line-source geometry function is, by definition, the
    unnormalized integral over the active length of the inverse-square
    kernel: G_L = (1/L) * int_{-L/2}^{+L/2} dl / d(P, l)^2, which reduces
    analytically to beta / (L * r * sin(theta)). Integrate it numerically
    at random (r, theta) with no shared code and demand 1% agreement.
    """
    rng = np.random.default_rng(42)
    L = TG43Engine._L
    zline = np.linspace(-L / 2.0, L / 2.0, 20001)
    for _ in range(10):
        r = float(rng.uniform(0.3, 8.0))
        theta = float(rng.uniform(5.0, 90.0))
        y = r * np.sin(np.radians(theta))
        z = r * np.cos(np.radians(theta))
        integrand = 1.0 / (y * y + (z - zline) ** 2)
        g_num = np.trapezoid(integrand, zline) / L
        g_eng = float(eng._geometry_factor(np.array(r), np.array(theta)))
        assert g_eng == pytest.approx(g_num, rel=0.01), (r, theta)


def test_geometry_on_axis_limit_matches_formula(eng):
    """Exactly on-axis, r > L/2: G = 1/(r^2 - L^2/4)."""
    L = TG43Engine._L
    for r in (0.25, 0.5, 1.0, 3.0):
        g = float(eng._geometry_factor(np.array(r), np.array(0.0)))
        assert g == pytest.approx(1.0 / (r * r - L * L / 4.0), rel=1e-9)


# ------------------------------------------------------------ v1 regression
V1_V2_POINTS = ([(90.0, r) for r in (0.5, 1.0, 2.0, 3.0, 5.0)]
                + [(th, r) for th in (30.0, 45.0, 60.0) for r in (1.0, 2.0)])


@pytest.mark.parametrize("theta,r", V1_V2_POINTS)
def test_v2_matches_v1_where_v1_is_valid(eng, v1, theta, r):
    """Total-decay dose agrees with the v1 reference within 1%.

    The residual ~0.07% is the documented tau difference (335.48 h derived
    from T_half vs v1's opaque 335.23 h).
    """
    v2_dose = eng.total_dose_at_point(theta, r, sk_per_seed_u=3.5)
    v1_dose = v1.dose_at_point(theta, r)
    assert v2_dose == pytest.approx(v1_dose, rel=0.01), (
        "theta=%g r=%g v1=%.4f v2=%.4f" % (theta, r, v1_dose, v2_dose))


# ---------------------------------------------------------- defect regressions
def test_on_axis_dose_positive_finite_decreasing(eng, v1):
    """Defect 1: v1 reads exactly 0 on the long axis out to ~2 mm; v2 must
    be positive, finite and monotone decreasing there."""
    d15 = eng.total_dose_at_point(0.0, 0.15)
    d19 = eng.total_dose_at_point(0.0, 0.19)
    assert np.isfinite(d15) and np.isfinite(d19)
    assert d15 > 0.0 and d19 > 0.0
    assert d15 > d19
    # And the v1 defect really was there (guards against a silently
    # "fixed" reference making this test vacuous).
    assert v1.dose_at_point(0.0, 0.15) == 0.0
    assert v1.dose_at_point(0.0, 0.19) == 0.0


def test_on_axis_dose_positive_finite_everywhere(eng):
    """Dose along the axis is positive/finite everywhere and monotone
    decreasing on each side of the active-line tip.

    Across the tip itself (r = L/2 = 0.2 cm) the exact line-source G
    genuinely diverges, so dose rises as the tip is approached from
    outside — that is real TG-43 physics, not a defect; monotonicity is
    asserted separately inside the capsule clamp and beyond the tip.
    """
    inside = np.array([0.05, 0.10, 0.15, 0.19, 0.20])   # capsule clamp zone
    beyond = np.array([0.21, 0.25, 0.5, 1.0, 2.0, 5.0])  # analytic limit zone
    d_in = eng.total_dose_at_point(np.zeros_like(inside), inside)
    d_out = eng.total_dose_at_point(np.zeros_like(beyond), beyond)
    for d in (d_in, d_out):
        assert np.all(np.isfinite(d)) and np.all(d > 0.0)
        assert np.all(np.diff(d) < 0.0), d


def test_core_rate_folds_theta(eng):
    """Defect 2: the CORE function folds theta, exactly."""
    assert eng.dose_rate(30.0, 1.0) == eng.dose_rate(150.0, 1.0)
    assert eng.dose_rate(45.0, 2.0) == eng.dose_rate(135.0, 2.0)
    assert eng.dose_rate(90.0, 1.0) == eng.dose_rate(270.0, 1.0)
    assert eng.dose_rate(10.0, 1.0) == eng.dose_rate(-10.0, 1.0)


def test_anisotropy_nan_fill_precomputed(eng):
    """Defect 3: no NaN survives init; holes take the nearest valid value
    along r at the same theta (== first tabulated radius with data)."""
    assert not np.isnan(eng._F_FILLED).any()
    # Row theta=0: holes at r=0.10, 0.15 take the r=0.25 value 0.617.
    assert eng._F_FILLED[0, 0] == 0.617
    assert eng._F_FILLED[0, 1] == 0.617
    # Row theta=20: hole at r=0.10 takes the r=0.15 value 1.141.
    it20 = int(np.where(TG43Engine._F_THETAS == 20)[0][0])
    assert eng._F_FILLED[it20, 0] == 1.141
    # Valid entries are untouched.
    valid = ~np.isnan(TG43Engine._F_TABLE)
    assert np.array_equal(eng._F_FILLED[valid], TG43Engine._F_TABLE[valid])


def test_single_radial_clip(eng):
    """Defect 5: one clip, [0.05, 10.0] cm, applied identically everywhere.

    v1 froze g_L at r=0.05 but let geometry keep varying down to r=0.041;
    in v2 everything below the floor collapses to the floor value.
    """
    assert TG43Engine.R_CLIP_CM == (0.05, 10.0)
    floor = eng.dose_rate(90.0, 0.05)
    for r in (0.0, 0.01, 0.041, 0.049):
        assert eng.dose_rate(90.0, r) == floor
    ceil = eng.dose_rate(90.0, 10.0)
    for r in (10.1, 50.0, 1e6):
        assert eng.dose_rate(90.0, r) == ceil


def test_dose_conversion_is_explicit(eng):
    """Defect 4 (conversion provenance): rate * S_K * tau, tau = T_half/ln2."""
    assert eng.T_HALF_HOURS == pytest.approx(232.536)
    assert eng.TAU_HOURS == pytest.approx(232.536 / np.log(2.0), rel=1e-12)
    assert eng.TAU_HOURS == pytest.approx(335.48, abs=0.01)
    assert float(eng.dose_to_total_decay(2.0, 3.5)) == pytest.approx(
        2.0 * 3.5 * eng.TAU_HOURS)
    # S_K scales linearly and defaults to the historical 3.5 U.
    assert float(eng.dose_to_total_decay(1.0, 7.0)) == pytest.approx(
        2.0 * float(eng.dose_to_total_decay(1.0, 3.5)))
    # v1's opaque factor was tau_v1 * S_K with tau_v1 = 335.23 h, S_K = 3.5 U:
    # within 0.1% of the derived tau * 3.5.
    assert DoseInterpolator.DOSE_CONVERSION_FACTOR == pytest.approx(
        eng.TAU_HOURS * 3.5, rel=1e-3)


# ------------------------------------------------------------------ the grid
BOUNDS = np.array([[-30.0, -30.0, -30.0], [30.0, 30.0, 30.0]])


def test_grid_superposition_is_exact(eng):
    c1 = np.array([[-6.0, 0.0, 0.0]])
    c2 = np.array([[6.0, 3.0, -2.0]])
    a1 = np.array([[0.0, 0.0, 1.0]])
    a2 = np.array([[1.0, 1.0, 0.0]])
    both = compute_dose_grid(np.vstack([c1, c2]), np.vstack([a1, a2]),
                             BOUNDS, 2.0, engine=eng)
    solo1 = compute_dose_grid(c1, a1, BOUNDS, 2.0, engine=eng)
    solo2 = compute_dose_grid(c2, a2, BOUNDS, 2.0, engine=eng)
    assert np.allclose(both.array, solo1.array + solo2.array,
                       rtol=1e-12, atol=0.0)
    assert np.array_equal(both.affine, solo1.affine)


def test_grid_geometry_and_metadata(eng):
    vol = compute_dose_grid(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]),
                            BOUNDS, 2.0, engine=eng)
    assert vol.array.shape == (31, 31, 31)          # [k, j, i]
    assert np.allclose(vol.spacing, 2.0)
    assert np.allclose(vol.origin_ras, BOUNDS[0])
    assert np.allclose(vol.index_to_ras([30, 30, 30]), BOUNDS[1])
    assert vol.meta["units"] == "cGy"
    assert np.all(np.isfinite(vol.array)) and np.all(vol.array > 0.0)


def test_grid_matches_point_evaluation(eng):
    """Trilinear samples of the grid agree with direct point sums to 3%
    at random points more than 5 mm from every seed."""
    rng = np.random.default_rng(7)
    centers = rng.uniform(-8.0, 8.0, (4, 3))
    axes = rng.standard_normal((4, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    vol = compute_dose_grid(centers, axes, BOUNDS, 2.0, engine=eng)

    pts = []
    while len(pts) < 20:
        p = rng.uniform(-24.0, 24.0, 3)     # stay off the grid boundary
        if np.min(np.linalg.norm(centers - p, axis=1)) > 5.0:
            pts.append(p)
    pts = np.asarray(pts)

    sampled = vol.sample_ras(pts)
    for p, s in zip(pts, sampled):
        direct = 0.0
        for c, a in zip(centers, axes):
            d = p - c
            r_mm = np.linalg.norm(d)
            theta = np.degrees(np.arccos(np.clip(d @ a / r_mm, -1, 1)))
            direct += eng.total_dose_at_point(theta, r_mm / 10.0)
        assert s == pytest.approx(direct, rel=0.03), (p, s, direct)


# ------------------------------------------------------------ physical sanity
def test_prescription_scale_dose_at_5mm(eng):
    """12 seeds in the phantom's tile layout: dose 5 mm outside a tile
    centre should be prescription-scale (GammaTile: 60 Gy at 5 mm)."""
    _, truth = make_head_phantom(spacing=2.0, n_tiles=3, noise_hu=0.0,
                                 rng_seed=0)
    centers = np.array([s.center_ras for s in truth.seeds])
    axes = np.array([s.axis_ras for s in truth.seeds])
    assert centers.shape == (12, 3)

    lo = centers.min(axis=0) - 25.0
    hi = centers.max(axis=0) + 25.0
    vol = compute_dose_grid(centers, axes, np.vstack([lo, hi]), 2.0,
                            engine=eng)

    tile = truth.tiles[0]
    probe = tile.center_ras + 5.0 * tile.normal_ras     # 5 mm into the wall
    dose_cgy = float(vol.sample_ras(probe))
    print("\n[report] dose 5 mm outside tile 0 centre: %.1f cGy (%.1f Gy); "
          "target 60 Gy, accepted band 20-180 Gy" % (dose_cgy, dose_cgy / 100))
    assert 60.0e2 / 3.0 < dose_cgy < 60.0e2 * 3.0


# ---------------------------------------------------------------------- isodose
def test_isodose_surfaces_nested_and_nonempty(eng):
    # A 2x2 "tile" of parallel seeds -> smooth nested isodose shells.
    centers = np.array([[-5.0, -5.0, 0.0], [-5.0, 5.0, 0.0],
                        [5.0, -5.0, 0.0], [5.0, 5.0, 0.0]])
    axes = np.tile([0.0, 1.0, 0.0], (4, 1))
    bounds = np.array([[-40.0, -40.0, -40.0], [40.0, 40.0, 40.0]])
    vol = compute_dose_grid(centers, axes, bounds, 2.0, engine=eng)

    levels = [1000.0, 3000.0, 6000.0, 12000.0]
    surf = isodose_surfaces(vol, levels)
    assert set(surf.keys()) == set(levels)

    mesh_vols = []
    for level in levels:
        mesh = surf[level]
        assert len(mesh.faces) > 0, "empty mesh at %g cGy" % level
        mesh_vols.append(abs(mesh.volume))
        # The mask itself must also shrink monotonically.
    mask_vols = [(vol.array >= lv).sum() for lv in levels]
    assert all(a > b for a, b in zip(mask_vols, mask_vols[1:])), mask_vols
    assert all(a > b for a, b in zip(mesh_vols, mesh_vols[1:])), mesh_vols


def test_isodose_disconnected_components_survive(eng):
    """Two far-apart seeds: a high isodose is legitimately two shells."""
    centers = np.array([[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    axes = np.tile([0.0, 0.0, 1.0], (2, 1))
    bounds = np.array([[-40.0, -20.0, -20.0], [40.0, 20.0, 20.0]])
    vol = compute_dose_grid(centers, axes, bounds, 2.0, engine=eng)
    level = float(eng.total_dose_at_point(90.0, 0.5))   # ~dose 5 mm out
    mesh = isodose_surfaces(vol, [level])[level]
    parts = mesh.split(only_watertight=False)
    assert len(parts) == 2, "expected two disconnected isodose shells"


# ------------------------------------------------------------------ performance
def test_grid_performance_budget(eng):
    """12 seeds on a 2 mm grid spanning a 100 mm cube in under 5 s."""
    rng = np.random.default_rng(3)
    centers = rng.uniform(-15.0, 15.0, (12, 3))
    axes = rng.standard_normal((12, 3))
    bounds = np.array([[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0]])
    t0 = time.perf_counter()
    vol = compute_dose_grid(centers, axes, bounds, 2.0, engine=eng)
    elapsed = time.perf_counter() - t0
    print("\n[report] 12-seed 51^3 grid: %.2f s" % elapsed)
    assert vol.array.shape == (51, 51, 51)
    assert elapsed < 5.0, "grid took %.2f s (budget 5 s)" % elapsed
