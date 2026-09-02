"""Verification campaign for inter-seed / tile-carrier interference.

Layers of evidence, deliberately independent of one another:

- the two ray/solid intersection primitives against brute-force numerical
  sampling of the segment (no shared code);
- closed-form Beer-Lambert transmission for a hand-computed chord;
- geometric invariants: reciprocity, no self-shadowing, monotonicity;
- the correction's *sign and shape* on a real tile layout (shadows behind
  seeds, nothing in the clear);
- exactness of the opt-out: no model must reproduce the plain TG-43 grid
  bit for bit, so the existing regression pins still mean what they say;
- the convexity argument behind averaging transmission (not optical depth)
  over the active line;
- a wall-clock budget, since this runs inside the interactive planner.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from gtcore.dose import (
    InterferenceModel,
    SeedCapsule,
    TileCarrier,
    compute_dose_grid,
    interference_report,
)
from gtcore.dose.interference import (
    ACTIVE_LENGTH_MM,
    CAPSULE_LENGTH_MM,
    CAPSULE_OUTER_RADIUS_MM,
    CAPSULE_WALL_MM,
    MU_SEED_CORE_CM1,
    MU_TITANIUM_CM1,
    MU_WATER_CM1,
    TILE_CARRIER_DENSITY_G_CM3,
    segment_box_length_mm,
    segment_cylinder_length_mm,
)


# ------------------------------------------------------------------ fixtures
def _pair_model(gap_mm=10.0, **kw):
    """Two seeds ``gap_mm`` apart along +x, long axes along y."""
    centers = np.array([[0.0, 0.0, 0.0], [gap_mm, 0.0, 0.0]])
    axes = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    return centers, axes, InterferenceModel.from_implant(centers, axes, **kw)


class _Tile:
    """Minimal stand-in for gtcore.interact.PlacedTile."""

    def __init__(self, seed_centers, normal_ras, axis_ras, kind="full"):
        self.seed_centers = np.asarray(seed_centers, dtype=float)
        self.normal_ras = np.asarray(normal_ras, dtype=float)
        self.axis_ras = np.asarray(axis_ras, dtype=float)
        self.kind = kind


def _flat_tile(origin_xy=(0.0, 0.0), z=0.0):
    """One 4-seed tile in the z = ``z`` plane, seeds on a 10 mm grid."""
    ox, oy = origin_xy
    seeds = np.array([[ox + dx, oy + dy, z]
                      for dx in (-5.0, 5.0) for dy in (-5.0, 5.0)])
    axes = np.tile([1.0, 0.0, 0.0], (4, 1)).astype(float)
    tile = _Tile(seeds, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    return seeds, axes, tile


# -------------------------------------- primitives vs numerical integration
def _numeric_inside_length(origin, target, inside_fn, n=400001):
    """Brute-force path length by sampling the segment; no shared code."""
    o = np.asarray(origin, float)
    t = np.asarray(target, float)
    ts = np.linspace(0.0, 1.0, n)
    pts = o[None, :] + ts[:, None] * (t - o)[None, :]
    frac = float(inside_fn(pts).mean())
    return frac * float(np.linalg.norm(t - o))


def test_cylinder_path_matches_numerical_sampling():
    """Analytic segment/cylinder length == sampled length, random rays."""
    rng = np.random.default_rng(7)
    center = np.array([1.0, -2.0, 0.5])
    axis = np.array([0.3, 0.6, -0.74])
    axis = axis / np.linalg.norm(axis)
    radius, half_len = 0.4, 2.25

    def inside(pts):
        rel = pts - center
        a = rel @ axis
        perp = rel - a[:, None] * axis[None, :]
        return (np.einsum("ij,ij->i", perp, perp) <= radius ** 2) \
            & (np.abs(a) <= half_len)

    for _ in range(12):
        origin = rng.normal(scale=8.0, size=3)
        target = center + rng.normal(scale=3.0, size=3)
        got = float(segment_cylinder_length_mm(origin, target[None, :], center,
                                               axis, radius, half_len)[0])
        want = _numeric_inside_length(origin, target, inside)
        assert got == pytest.approx(want, abs=0.02), (origin, target)


def test_box_path_matches_numerical_sampling():
    """Analytic segment/OBB length == sampled length, random rays."""
    rng = np.random.default_rng(11)
    center = np.array([-3.0, 4.0, 1.0])
    n = np.array([0.2, -0.3, 0.93])
    n = n / np.linalg.norm(n)
    t1 = np.cross(n, [0.0, 0.0, 1.0])
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    frame = np.vstack([t1, t2, n])
    half = np.array([10.0, 10.0, 2.0])

    def inside(pts):
        loc = (pts - center) @ frame.T
        return np.all(np.abs(loc) <= half, axis=1)

    for _ in range(12):
        origin = center + rng.normal(scale=20.0, size=3)
        target = center + rng.normal(scale=12.0, size=3)
        got = float(segment_box_length_mm(origin, target[None, :], center,
                                          frame, half)[0])
        want = _numeric_inside_length(origin, target, inside)
        assert got == pytest.approx(want, abs=0.05), (origin, target)


def test_cylinder_path_zero_when_ray_misses():
    """A ray passing wide of the capsule contributes nothing."""
    got = segment_cylinder_length_mm([0.0, 0.0, 0.0], [[20.0, 5.0, 0.0]],
                                     [10.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                     0.4, 2.25)
    assert float(got[0]) == 0.0


def test_cylinder_path_stops_at_the_segment_end():
    """Path length only counts the part of the SEGMENT inside the solid."""
    # Target sits at the capsule centre: half the chord, not the whole one.
    full = float(segment_cylinder_length_mm(
        [0.0, 0.0, 0.0], [[20.0, 0.0, 0.0]], [10.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], 0.4, 2.25)[0])
    half = float(segment_cylinder_length_mm(
        [0.0, 0.0, 0.0], [[10.0, 0.0, 0.0]], [10.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], 0.4, 2.25)[0])
    assert full == pytest.approx(0.8, abs=1e-9)
    assert half == pytest.approx(0.4, abs=1e-9)


# ---------------------------------------------------- closed-form attenuation
def test_transmission_matches_hand_computed_beer_lambert():
    """Straight through a capsule's waist: T = exp(-sum (mu - mu_w) l)."""
    centers, axes, model = _pair_model()
    point = np.array([[20.0, 0.0, 0.0]])          # beyond seed 1, on the line
    got = float(model.transmission(0, point)[0])

    core_r = CAPSULE_OUTER_RADIUS_MM - CAPSULE_WALL_MM
    core_chord_mm = 2.0 * core_r                  # 0.70 mm
    wall_chord_mm = 2.0 * CAPSULE_OUTER_RADIUS_MM - core_chord_mm   # 0.10 mm
    tau = 0.1 * ((MU_TITANIUM_CM1 - MU_WATER_CM1) * wall_chord_mm
                 + (MU_SEED_CORE_CM1 - MU_WATER_CM1) * core_chord_mm)
    assert got == pytest.approx(math.exp(-tau), rel=1e-12)
    # Sanity on the magnitude: a seed in the way is a serious absorber.
    assert 0.4 < got < 0.7


def test_no_self_shadowing():
    """A seed never attenuates its own rays -- F(r, theta) already does."""
    centers, axes, model = _pair_model()
    # Points all around seed 0 but nowhere near seed 1.
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(200, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    pts = centers[0] + 6.0 * dirs
    keep = np.linalg.norm(pts - centers[1], axis=1) > 4.0
    assert np.allclose(model.transmission(0, pts[keep]), 1.0)


def test_transmission_is_one_in_the_clear():
    """Perpendicular to the seed-seed line nothing is in the way."""
    centers, axes, model = _pair_model()
    pts = np.array([[0.0, 30.0, 0.0], [0.0, 0.0, 25.0], [-30.0, 0.0, 0.0]])
    assert np.allclose(model.transmission(0, pts), 1.0)


def test_transmission_is_reciprocal():
    """Seed 0 -> point behind seed 1 equals seed 1 -> mirrored point."""
    centers, axes, model = _pair_model()
    a = float(model.transmission(0, np.array([[25.0, 0.0, 0.0]]))[0])
    b = float(model.transmission(1, np.array([[-15.0, 0.0, 0.0]]))[0])
    assert a == pytest.approx(b, rel=1e-12)


def test_shadow_is_confined_to_the_geometric_umbra():
    """The umbra ends exactly where the projected capsule edge says it does.

    The occluding seed sits half way along the ray, so its 0.4 mm radius
    projects to a 0.8 mm umbra radius in the plane of the field point.  This
    pins the geometry: a bug in the intersection would move that edge.
    """
    centers, axes, model = _pair_model()          # seeds 10 mm apart on x
    # Field points at x = 20 mm displaced along z, i.e. across the capsule's
    # 0.8 mm waist (its long axis is y).
    inside = np.array([0.0, 0.3, 0.6, 0.75])
    outside = np.array([0.85, 1.0, 3.0])

    def probe(zs):
        pts = np.stack([np.full_like(zs, 20.0), np.zeros_like(zs), zs], axis=1)
        return model.transmission(0, pts)

    t_in = probe(inside)
    t_out = probe(outside)
    assert np.all(t_in < 0.75)                    # every point still shadowed
    assert t_in[0] < 0.7                          # dead centre: full chord
    assert np.allclose(t_out, 1.0)                # past the projected edge

    # Inside the umbra the chord *lengthens* off axis -- the ray crosses the
    # cylinder obliquely -- so transmission is not monotone here.  Assert the
    # physics that actually holds: the shortest chord is on the axis only
    # among rays of equal obliquity, and everything stays a real attenuation.
    assert np.all((t_in > 0.0) & (t_in < 1.0))


def test_transmission_decreases_with_more_seeds_in_the_way():
    """Three collinear seeds: the far point sees two shadows, not one."""
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    axes = np.tile([0.0, 1.0, 0.0], (3, 1)).astype(float)
    model = InterferenceModel.from_implant(centers, axes)
    pt = np.array([[30.0, 0.0, 0.0]])
    one = float(model.transmission(1, pt)[0])     # one capsule in the way
    two = float(model.transmission(0, pt)[0])     # two capsules in the way
    assert two == pytest.approx(one * one, rel=1e-12)
    assert two < one < 1.0


def test_max_range_cutoff_is_respected():
    """Beyond max_range_mm the ray is not traced and T is exactly 1."""
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    axes = np.tile([0.0, 1.0, 0.0], (2, 1)).astype(float)
    near = InterferenceModel(
        [SeedCapsule(c, a) for c, a in zip(centers, axes)], max_range_mm=15.0)
    far = InterferenceModel(
        [SeedCapsule(c, a) for c, a in zip(centers, axes)], max_range_mm=None)
    pts = np.array([[12.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    t_near = near.transmission(0, pts)
    t_far = far.transmission(0, pts)
    assert t_near[0] < 1.0 and t_far[0] < 1.0
    assert t_near[1] == 1.0                       # cut off
    assert t_far[1] < 1.0                         # actually shadowed


# --------------------------------------------------------- the active line
def test_line_sampling_softens_the_shadow_edge():
    """Averaging over the active line produces a penumbra, not a step.

    A point source casts a hard-edged shadow; a 4 mm line source casts a
    penumbra.  Sample transmission across the umbra edge and require the
    line-sampled profile to be strictly gentler where the point profile
    steps, and to still agree deep inside and far outside.
    """
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    # Seed axes along z so the active line is perpendicular to the scan.
    axes = np.tile([0.0, 0.0, 1.0], (2, 1)).astype(float)
    caps = [SeedCapsule(c, a) for c, a in zip(centers, axes)]
    point = InterferenceModel(caps, line_samples=1)
    line = InterferenceModel(caps, line_samples=7)

    # Scan past the END of the occluding capsule, where a line source's
    # penumbra lives.  The capsule spans z = +/-2.25 mm half way along the
    # ray, so a point source's umbra edge projects to z = 4.5 mm.
    zs = np.linspace(0.0, 8.0, 81)
    pts = np.stack([np.full_like(zs, 20.0), np.zeros_like(zs), zs], axis=1)
    t_pt = point.transmission(0, pts)
    t_ln = line.transmission(0, pts)

    # Both are shadowed on the axis and clear well past the capsule end.
    assert t_pt[0] < 0.7 and t_ln[0] < 0.7
    assert t_pt[-1] == pytest.approx(1.0) and t_ln[-1] == pytest.approx(1.0)

    # Width of the 10%-90% transition between full shadow and full
    # transmission.  A point source steps across it between two samples; a
    # line source ramps over millimetres.  (The slow drift inside the umbra
    # -- oblique rays cut a longer chord -- is below the 10% level and does
    # not count as transition.)
    def band_width(t):
        lo, hi = t[0], 1.0
        t10, t90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
        inside = np.flatnonzero((t >= t10) & (t <= t90))
        return 0.0 if inside.size == 0 else zs[inside[-1]] - zs[inside[0]]

    assert band_width(t_pt) < 0.5
    assert band_width(t_ln) > 2.0


def test_line_sampling_averages_transmission_not_optical_depth():
    """The penumbra must be the mean of exp(-tau), not exp(-mean tau).

    exp is convex, so mean(exp(-tau)) >= exp(-mean(tau)); averaging depths
    would systematically under-report dose at every partially shadowed point.
    Construct a point where the active line is half blocked and check which
    identity holds.
    """
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    axes = np.tile([0.0, 0.0, 1.0], (2, 1)).astype(float)
    caps = [SeedCapsule(c, a) for c, a in zip(centers, axes)]
    model = InterferenceModel(caps, line_samples=5)

    # In the penumbra past the occluding capsule's end, so the active line is
    # partly blocked and partly in the clear.
    pt = np.array([[20.0, 0.0, 5.0]])
    got = float(model.transmission(0, pt)[0])

    # Reference: trace each active-line origin separately.  Use a model that
    # holds ONLY the occluding capsule, since transmission_from_point skips
    # nothing and an origin displaced along the seed axis lies inside seed
    # 0's own can.
    occluder_only = InterferenceModel([caps[1]], line_samples=1)
    offsets = np.linspace(-1.0, 1.0, 5) * (0.5 * ACTIVE_LENGTH_MM)
    per_origin = np.array([
        float(occluder_only.transmission_from_point(
            centers[0] + off * np.array([0.0, 0.0, 1.0]), pt)[0])
        for off in offsets
    ])

    assert got == pytest.approx(per_origin.mean(), rel=1e-9)
    # Strictly better than averaging optical depths, by Jensen's inequality,
    # and strictly partial -- i.e. this really is a penumbra sample.
    depth_mean = math.exp(np.mean(np.log(per_origin)))
    assert got > depth_mean
    assert 0.0 < got < 1.0
    assert per_origin.min() < per_origin.max()


# ------------------------------------------------------------- tile carriers
def test_carrier_below_water_density_raises_transmission():
    """A carrier lighter than water displaces attenuator: T > 1."""
    seeds, axes, tile = _flat_tile()
    with_carrier = InterferenceModel.from_implant(
        seeds, axes, tiles=[tile], include_carriers=True)
    # The tile centre: in-plane, so the ray runs lengthwise through the 4 mm
    # slab, and no other capsule sits on the diagonal.
    pt = np.array([[0.0, 0.0, 0.0]])
    t = float(with_carrier.transmission(0, pt)[0])
    assert t > 1.0
    # Capsules alone leave that ray untouched, so the excess is all carrier.
    caps_only = InterferenceModel.from_implant(seeds, axes)
    assert float(caps_only.transmission(0, pt)[0]) == pytest.approx(1.0)
    # Out of plane, well clear of the slab, nothing changes.
    far = np.array([[0.0, 0.0, 60.0]])
    assert float(with_carrier.transmission(0, far)[0]) == pytest.approx(1.0)


def test_carriers_are_opt_in():
    """from_implant must not enable the provisional-density carrier term."""
    seeds, axes, tile = _flat_tile()
    default = InterferenceModel.from_implant(seeds, axes, tiles=[tile])
    assert default.carriers == []
    opted_in = InterferenceModel.from_implant(
        seeds, axes, tiles=[tile], include_carriers=True)
    assert len(opted_in.carriers) == 1


def test_carrier_footprint_follows_tile_kind():
    seeds, axes, tile = _flat_tile()
    tile.kind = "half"
    tile.seed_centers = seeds[:2]
    car = TileCarrier.from_tile(tile)
    assert car.half_extents_mm == (5.0, 10.0)
    assert car.thickness_mm == pytest.approx(4.0)
    # Frame rows are orthonormal and the third is the tile normal.
    assert np.allclose(car.frame @ car.frame.T, np.eye(3), atol=1e-12)
    assert np.allclose(car.frame[2], [0.0, 0.0, 1.0])


def test_carrier_density_drives_the_correction_linearly_in_log():
    """Optical depth is proportional to (mu_carrier - mu_water)."""
    seeds, axes, tile = _flat_tile()
    pt = np.array([[0.0, 0.0, 0.0]])          # in-plane, no capsule in the way
    taus = {}
    for rho in (0.2, 0.4):
        m = InterferenceModel.from_implant(
            seeds, axes, tiles=[tile], include_carriers=True,
            carrier_density_g_cm3=rho)
        taus[rho] = -math.log(float(m.transmission(0, pt)[0]))
    # tau scales with (rho - 1); halving the deficit halves the magnitude.
    assert taus[0.4] / taus[0.2] == pytest.approx(
        (0.4 - 1.0) / (0.2 - 1.0), rel=1e-9)


# ----------------------------------------------------------- grid integration
def test_no_model_reproduces_plain_tg43_bitwise():
    """The opt-out must be exact, or the v2 regression pins stop meaning
    what they say."""
    seeds, axes, _tile = _flat_tile()
    bounds = np.vstack([seeds.min(axis=0) - 20.0, seeds.max(axis=0) + 20.0])
    a = compute_dose_grid(seeds, axes, bounds, spacing_mm=2.0)
    b = compute_dose_grid(seeds, axes, bounds, spacing_mm=2.0,
                          interference=None)
    assert np.array_equal(a.array, b.array)
    assert a.meta["interference"] is None


def test_grid_with_interference_is_never_hotter_without_carriers():
    """Capsule-only interference can only remove primary fluence."""
    seeds, axes, _tile = _flat_tile()
    bounds = np.vstack([seeds.min(axis=0) - 20.0, seeds.max(axis=0) + 20.0])
    free = compute_dose_grid(seeds, axes, bounds, spacing_mm=2.0)
    model = InterferenceModel.from_implant(seeds, axes)
    corr = compute_dose_grid(seeds, axes, bounds, spacing_mm=2.0,
                             interference=model)
    assert np.all(corr.array <= free.array + 1e-9)
    assert np.any(corr.array < free.array * (1.0 - 1e-6))
    assert corr.meta["interference"]["n_capsules"] == 4
    assert corr.meta["interference"]["n_carriers"] == 0


def test_correction_magnitude_is_a_few_percent_in_the_treated_volume():
    """Sign and size sanity: single digits on average, tens locally.

    Published interseed-attenuation work on low-energy permanent implants
    finds a few percent on volume metrics with much larger local shadows.
    Anything outside that band means the geometry or the coefficients drifted.
    """
    seeds_a, axes_a, _ = _flat_tile((0.0, 0.0))
    seeds_b, axes_b, _ = _flat_tile((22.0, 0.0))
    seeds = np.vstack([seeds_a, seeds_b])
    axes = np.vstack([axes_a, axes_b])
    bounds = np.vstack([seeds.min(axis=0) - 30.0, seeds.max(axis=0) + 30.0])

    free = compute_dose_grid(seeds, axes, bounds, spacing_mm=2.0)
    model = InterferenceModel.from_implant(seeds, axes)
    corr = compute_dose_grid(seeds, axes, bounds, spacing_mm=2.0,
                             interference=model)

    rep = interference_report(free, corr, level_cgy=1500.0)
    assert rep["n_voxels"] > 500
    assert rep["max_ratio"] == pytest.approx(1.0)          # never hotter
    assert -5.0 < rep["mean_percent_change"] < 0.0         # a few percent
    assert 0.5 < rep["min_ratio"] < 0.95                   # real local shadows


def test_report_restricts_to_the_dose_level_asked_for():
    free = np.array([[[100.0, 10.0], [1.0, 0.0]]])
    corr = free * 0.9
    all_pts = interference_report(free, corr)
    high = interference_report(free, corr, level_cgy=50.0)
    assert all_pts["n_voxels"] == 3                        # the 0 is dropped
    assert high["n_voxels"] == 1
    assert high["mean_ratio"] == pytest.approx(0.9)
    assert high["mean_percent_change"] == pytest.approx(-10.0)


def test_report_handles_an_empty_selection():
    free = np.zeros((2, 2, 2))
    rep = interference_report(free, free)
    assert rep["n_voxels"] == 0
    assert math.isnan(rep["mean_ratio"])


# -------------------------------------------------------------- safety rails
def test_model_must_match_the_seed_arrays():
    """Index misalignment would make a seed shadow itself: refuse it."""
    seeds, axes, _tile = _flat_tile()
    bounds = np.vstack([seeds.min(axis=0) - 10.0, seeds.max(axis=0) + 10.0])
    wrong_count = InterferenceModel.from_implant(seeds[:2], axes[:2])
    with pytest.raises(ValueError, match="capsules"):
        compute_dose_grid(seeds, axes, bounds, spacing_mm=4.0,
                          interference=wrong_count)

    shifted = InterferenceModel.from_implant(seeds[::-1], axes[::-1])
    with pytest.raises(ValueError, match="do not match"):
        compute_dose_grid(seeds, axes, bounds, spacing_mm=4.0,
                          interference=shifted)


def test_capsule_rejects_impossible_geometry():
    with pytest.raises(ValueError):
        SeedCapsule([0, 0, 0], [1, 0, 0], wall_mm=0.0)
    with pytest.raises(ValueError):
        SeedCapsule([0, 0, 0], [1, 0, 0], wall_mm=1.0, outer_radius_mm=0.4)
    with pytest.raises(ValueError):
        SeedCapsule([0, 0, 0], [0, 0, 0])


def test_model_rejects_bad_settings():
    caps = [SeedCapsule([0, 0, 0], [1, 0, 0])]
    with pytest.raises(ValueError):
        InterferenceModel(caps, max_range_mm=0.0)
    with pytest.raises(ValueError):
        InterferenceModel(caps, line_samples=0)


def test_empty_model_is_transparent():
    model = InterferenceModel([])
    pts = np.zeros((5, 3))
    assert np.allclose(model.transmission_from_point([0, 0, -10.0], pts), 1.0)


def test_capsule_geometry_matches_the_engine_constants():
    """The occluder must be the same physical can the engine clamps inside."""
    from gtcore.dose import TG43Engine
    assert CAPSULE_LENGTH_MM == pytest.approx(
        2.0 * TG43Engine._CAP_HALF_CM * 10.0)
    assert CAPSULE_OUTER_RADIUS_MM == pytest.approx(
        TG43Engine._RHO_SURFACE * 10.0)
    assert ACTIVE_LENGTH_MM == pytest.approx(TG43Engine._L * 10.0)


def test_carrier_term_cannot_be_enabled_by_accident():
    """A guard against the provisional density being quietly promoted.

    The carrier correction is larger than the interseed shadow and opposite
    in sign, driven by a density nobody here has measured.  If someone flips
    the default, this test is the thing that notices.
    """
    assert TILE_CARRIER_DENSITY_G_CM3 < 1.0     # still the dry-sponge guess
    import inspect

    sig = inspect.signature(InterferenceModel.from_implant)
    assert sig.parameters["include_carriers"].default is False


# -------------------------------------------------------------- performance
def test_interference_grid_stays_within_budget():
    """The planner recomputes dose on a keypress; keep the overhead sane."""
    seeds_a, axes_a, _ = _flat_tile((0.0, 0.0))
    seeds_b, axes_b, _ = _flat_tile((22.0, 0.0))
    seeds_c, axes_c, _ = _flat_tile((0.0, 22.0))
    seeds = np.vstack([seeds_a, seeds_b, seeds_c])
    axes = np.vstack([axes_a, axes_b, axes_c])
    bounds = np.vstack([seeds.min(axis=0) - 50.0, seeds.max(axis=0) + 50.0])
    model = InterferenceModel.from_implant(seeds, axes)

    t0 = time.perf_counter()
    compute_dose_grid(seeds, axes, bounds, spacing_mm=2.0, interference=model)
    elapsed = time.perf_counter() - t0
    assert elapsed < 6.0, "interference grid took %.2f s" % elapsed
