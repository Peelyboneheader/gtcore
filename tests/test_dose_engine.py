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
    dose_at_points, isodose_surfaces
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
    """Dose along the axis is positive/finite everywhere, strictly
    decreasing outside the titanium can, and never larger inside the can
    than the largest value on the can's surface.

    Inside the capsule (|z| <= 0.225 cm) there is no tissue; points there
    take the dose of the nearest capsule-surface point, so along the axis
    the value follows the capsule side (0 < z < 0.185) and then the end
    face (a plateau to z = 0.225). Monotonicity is only a physical claim
    outside the can and is asserted only there.
    """
    inside = np.array([0.05, 0.10, 0.15, 0.19, 0.20, 0.225])
    beyond = np.array([0.22501, 0.23, 0.25, 0.5, 1.0, 2.0, 5.0])
    d_in = eng.total_dose_at_point(np.zeros_like(inside), inside)
    d_out = eng.total_dose_at_point(np.zeros_like(beyond), beyond)
    for d in (d_in, d_out):
        assert np.all(np.isfinite(d)) and np.all(d > 0.0)
    assert np.all(np.diff(d_out) < 0.0), d_out
    # Surface bound: nothing inside exceeds the hottest point ON the can
    # (densely sampled side + end face).
    rho, cap = TG43Engine._RHO_SURFACE, TG43Engine._CAP_HALF_CM
    ys = np.concatenate([np.full(2000, rho), np.linspace(0.0, rho, 2000)])
    zs = np.concatenate([np.linspace(0.0, cap, 2000), np.full(2000, cap)])
    surf = eng.total_dose_at_point(np.degrees(np.arctan2(ys, zs)),
                                   np.hypot(ys, zs))
    assert np.all(d_in <= surf.max() * (1 + 1e-9))
    # End-face plateau: interior points nearer the end face than the side
    # all read the end-face value, continuous with the first point outside.
    assert d_in[-1] == pytest.approx(d_in[-2], rel=1e-12)
    assert d_out[0] == pytest.approx(d_in[-1], rel=2e-3)


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


def test_single_radial_floor(eng):
    """Defect 5: one floor, 0.05 cm, applied identically everywhere.

    v1 froze g_L at r=0.05 but let geometry keep varying down to r=0.041;
    in v2 everything below the floor collapses to the floor value.
    """
    assert TG43Engine.R_FLOOR_CM == 0.05
    assert TG43Engine.R_CLIP_CM == (0.05, 10.0)      # legacy alias
    floor = eng.dose_rate(90.0, 0.05)
    for r in (0.0, 0.01, 0.041, 0.049):
        assert eng.dose_rate(90.0, r) == floor


def test_far_field_falls_off_beyond_data_domain(eng):
    """Beyond R_DATA_MAX_CM (10 cm) g_L and F are held at their 10 cm
    values but the geometry factor keeps falling, so the rate is strictly
    decreasing (the earlier v2 plateaued at the 10 cm value)."""
    assert TG43Engine.R_DATA_MAX_CM == 10.0
    radii = np.array([5.0, 10.0, 10.1, 15.0, 20.0, 40.0])
    rate = eng.dose_rate(np.full_like(radii, 90.0), radii)
    assert np.all(np.diff(rate) < 0.0), rate
    # Exactly the geometry-factor ratio, since the data terms are frozen.
    g10 = float(eng._geometry_factor(np.array(10.0), np.array(90.0)))
    for r in (15.0, 20.0, 40.0):
        g = float(eng._geometry_factor(np.array(r), np.array(90.0)))
        assert eng.dose_rate(90.0, r) / eng.dose_rate(90.0, 10.0) \
            == pytest.approx(g / g10, rel=1e-12)
    # The same holds off the transverse plane.
    assert eng.dose_rate(30.0, 20.0) < eng.dose_rate(30.0, 10.0)


def test_anisotropy_holes_are_inside_the_capsule(eng):
    """Every NaN in the anisotropy table sits inside the titanium can
    (perpendicular distance < 0.04 cm AND |axial| <= 0.225 cm), so the
    borrowed fill values are never read for a field point in tissue."""
    it, ir = np.where(np.isnan(TG43Engine._F_TABLE))
    assert it.size == 19
    r = TG43Engine._F_RADII[ir]
    th = np.radians(TG43Engine._F_THETAS[it])
    y = r * np.sin(th)
    z = r * np.cos(th)
    assert np.all(y < TG43Engine._RHO_SURFACE)
    assert np.all(z <= TG43Engine._CAP_HALF_CM)


def test_capsule_boundary_is_continuous(eng):
    """Dose rate is continuous across the whole capsule surface (side and
    end face): the interior projects to the nearest surface point."""
    rng = np.random.default_rng(11)
    n = 5000
    rho, cap = TG43Engine._RHO_SURFACE, TG43Engine._CAP_HALF_CM
    ys = np.concatenate([np.full(n, rho), rng.uniform(0.0, rho, n)])
    zs = np.concatenate([rng.uniform(0.0, cap, n), np.full(n, cap)])
    ny = np.concatenate([np.ones(n), np.zeros(n)])
    nz = np.concatenate([np.zeros(n), np.ones(n)])
    eps = 1.0e-5

    def rate(y, z):
        return eng.dose_rate(np.degrees(np.arctan2(y, z)), np.hypot(y, z))

    inner = rate(ys - eps * ny, zs - eps * nz)
    outer = rate(ys + eps * ny, zs + eps * nz)
    assert np.all(np.isfinite(inner)) and np.all(inner > 0.0)
    assert np.allclose(inner, outer, rtol=1e-3)


def test_tabulated_kernel_matches_exact(eng):
    """Bilinear (ln r, theta) kernel vs the analytic rate.

    < 0.1 % beyond r = 0.25 cm (any angle), < 2 % for r < 0.25 cm outside
    the capsule; inside the capsule the projection is piecewise and the
    lattice is not required to resolve it (no tissue there).
    """
    rng = np.random.default_rng(5)
    n = 100000
    th = rng.uniform(-360.0, 720.0, n)
    r = np.exp(rng.uniform(np.log(0.25), np.log(40.0), n))
    rel = np.abs(eng.dose_rate_tabulated(th, r) / eng.dose_rate(th, r) - 1.0)
    assert rel.max() < 1.0e-3, rel.max()

    r = np.exp(rng.uniform(np.log(0.05), np.log(0.25), n))
    th = rng.uniform(0.0, 90.0, n)
    y = r * np.sin(np.radians(th))
    z = r * np.cos(np.radians(th))
    outside = (y >= TG43Engine._RHO_SURFACE) | (z > TG43Engine._CAP_HALF_CM)
    rel = np.abs(eng.dose_rate_tabulated(th, r) / eng.dose_rate(th, r) - 1.0)
    assert rel[outside].max() < 2.0e-2, rel[outside].max()
    assert np.all(np.isfinite(rel))
    # Scalars in, scalars out; folding identical to the exact path.
    assert isinstance(eng.dose_rate_tabulated(30.0, 1.0), float)
    assert eng.dose_rate_tabulated(30.0, 1.0) == eng.dose_rate_tabulated(150.0, 1.0)


def test_decay_helpers(eng):
    """S_K decay to implant day and the delivered-dose fraction."""
    assert float(eng.decay_factor(0.0)) == 1.0
    assert float(eng.decay_factor(eng.T_HALF_HOURS)) == pytest.approx(0.5)
    assert float(eng.sk_decayed(3.5, eng.T_HALF_HOURS)) == pytest.approx(1.75)
    assert eng.delivered_fraction(None) == 1.0
    assert float(eng.delivered_fraction(0.0)) == 0.0
    assert float(eng.delivered_fraction(eng.T_HALF_HOURS)) == pytest.approx(0.5)
    assert float(eng.delivered_fraction(eng.TAU_HOURS)) == pytest.approx(
        1.0 - np.exp(-1.0))
    assert float(eng.delivered_fraction(np.inf)) == 1.0
    with pytest.raises(ValueError):
        eng.delivered_fraction(-1.0)


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
    assert vol.meta["kind"] == "tg43_total_decay_dose"
    assert vol.meta["kernel"] == "table"
    assert vol.meta["delivered_fraction"] == 1.0
    assert vol.meta["elapsed_hours"] is None
    assert np.all(np.isfinite(vol.array)) and np.all(vol.array > 0.0)


def test_grid_tabulated_matches_exact_grid(eng):
    rng = np.random.default_rng(9)
    centers = rng.uniform(-8.0, 8.0, (5, 3))
    axes = rng.standard_normal((5, 3))
    fast = compute_dose_grid(centers, axes, BOUNDS, 2.0, engine=eng)
    slow = compute_dose_grid(centers, axes, BOUNDS, 2.0, engine=eng,
                             exact=True)
    assert slow.meta["kernel"] == "exact"
    assert np.allclose(fast.array, slow.array, rtol=5e-4, atol=0.0)


def test_grid_elapsed_hours_scales_by_delivered_fraction(eng):
    c = np.zeros((1, 3))
    a = np.array([[0.0, 0.0, 1.0]])
    full = compute_dose_grid(c, a, BOUNDS, 4.0, engine=eng)
    part = compute_dose_grid(c, a, BOUNDS, 4.0, engine=eng,
                             elapsed_hours=eng.T_HALF_HOURS)
    assert part.meta["kind"] == "tg43_dose_at_time"
    assert part.meta["delivered_fraction"] == pytest.approx(0.5)
    assert np.allclose(part.array, 0.5 * full.array, rtol=1e-12)
    zero = compute_dose_grid(c, a, BOUNDS, 4.0, engine=eng, elapsed_hours=0.0)
    assert np.all(zero.array == 0.0)


def test_grid_per_seed_sk_and_zero_sk(eng):
    c = np.array([[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    a = np.tile([0.0, 0.0, 1.0], (2, 1))
    both = compute_dose_grid(c, a, BOUNDS, 4.0, engine=eng,
                             sk_per_seed_u=[3.5, 0.0])
    solo = compute_dose_grid(c[:1], a[:1], BOUNDS, 4.0, engine=eng)
    assert np.allclose(both.array, solo.array, rtol=1e-12)
    with pytest.raises(ValueError):
        compute_dose_grid(c, a, BOUNDS, 4.0, engine=eng,
                          sk_per_seed_u=[3.5, -1.0])


def test_dose_at_points_matches_engine_and_grid(eng):
    rng = np.random.default_rng(13)
    centers = rng.uniform(-8.0, 8.0, (4, 3))
    axes = rng.standard_normal((4, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    pts = rng.uniform(-20.0, 20.0, (30, 3))
    got = dose_at_points(centers, axes, pts, engine=eng)
    assert got.shape == (30,)
    for p, g in zip(pts, got):
        direct = 0.0
        for c, a in zip(centers, axes):
            d = p - c
            r_mm = np.linalg.norm(d)
            theta = np.degrees(np.arccos(np.clip(d @ a / r_mm, -1, 1)))
            direct += eng.total_dose_at_point(theta, r_mm / 10.0)
        assert g == pytest.approx(direct, rel=1e-12)
    # Single point in -> float out; tabulated path within 0.1 %.
    single = dose_at_points(centers, axes, pts[0], engine=eng)
    assert isinstance(single, float) and single == pytest.approx(got[0])
    fast = dose_at_points(centers, axes, pts, engine=eng, exact=False)
    assert np.allclose(fast, got, rtol=1e-3)
    # Chunking does not change the answer.
    chunked = dose_at_points(centers, axes, pts, engine=eng,
                             max_chunk_points=7)
    assert np.array_equal(chunked, got)


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


def test_isodose_position_is_subvoxel_accurate(eng):
    """Log-space marching cubes places a single-seed isodose within 0.1 mm
    rms (0.3 mm max) of the analytic isodose radius on a 2 mm grid."""
    c = np.zeros((1, 3))
    a = np.array([[0.0, 0.0, 1.0]])
    bounds = np.array([[-30.0, -30.0, -30.0], [30.0, 30.0, 30.0]])
    vol = compute_dose_grid(c, a, bounds, 2.0, engine=eng)
    level = float(eng.total_dose_at_point(90.0, 0.5))    # ~5 mm out
    mesh = isodose_surfaces(vol, [level])[level]
    assert mesh.is_watertight
    verts = np.asarray(mesh.vertices)
    dist = np.linalg.norm(verts, axis=1)
    unit = verts / dist[:, None]
    theta = np.degrees(np.arccos(np.abs(unit[:, 2])))
    lo = np.full(len(verts), 1.0)
    hi = np.full(len(verts), 30.0)
    for _ in range(60):                                  # bisection, mm
        mid = 0.5 * (lo + hi)
        above = eng.total_dose_at_point(theta, mid / 10.0) >= level
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    err = dist - 0.5 * (lo + hi)
    rms = float(np.sqrt(np.mean(err ** 2)))
    print("\n[report] isodose radial error: rms %.3f mm, max %.3f mm"
          % (rms, np.abs(err).max()))
    assert rms < 0.1 and np.abs(err).max() < 0.3
    # Normals point outward (away from the seed).
    assert np.mean(np.sum(mesh.vertex_normals * unit, axis=1)) > 0.9


def test_isodose_levels_outside_grid_range_are_empty(eng):
    c = np.zeros((1, 3))
    a = np.array([[0.0, 0.0, 1.0]])
    vol = compute_dose_grid(c, a, BOUNDS, 4.0, engine=eng)
    too_high = float(vol.array.max()) * 10.0
    too_low = float(vol.array.min()) * 0.1
    surf = isodose_surfaces(vol, [too_high, too_low])
    assert len(surf[too_high].faces) == 0
    assert len(surf[too_low].faces) == 0
    with pytest.raises(ValueError):
        isodose_surfaces(vol, [0.0])


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
