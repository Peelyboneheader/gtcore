"""Deformable (developable) tile fit without a surface (gtcore.tiles.deform)
and its use in count-free auto mode.

Acceptance (plan Step 3): the phantom's wall-conformed tiles fit with
rms < 0.5 mm; the 8-tile scan's crumpled tile fits WITHOUT the count
fallback (auto mode finds it as a degraded full tile); random non-tile
quads are rejected (residual / bending energy / axis error explode).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from gtcore import geometry
from gtcore.phantom import make_head_phantom
from gtcore.tiles import (
    DeformParams,
    TilePose6,
    deformable_score,
    deformed_footprint,
    deformed_points,
    deformed_seed_points,
    deformed_surface_grid,
    fit_deformable,
    fit_rigid,
    fit_tiles,
)
from gtcore.tiles.auto import LAMBDA_FULL, LOOSE_RMS_MAX_MM
from gtcore.tiles.deform import deformed_seed_axes

# The crumpled tile of the physical 8-tile printed phantom (Philips
# 0.59 x 0.59 x 1.0 mm, O-MAR): four detected seed centres / PCA axes, RAS
# mm.  Two sides are squeezed to ~5-6 mm -- outside every chord gate -- and
# the count-constrained completion was the only way to recover it before.
CRUMPLED_PTS = np.array([
    [-10.809, -206.023, -617.337],
    [-10.025, -200.913, -614.464],
    [-19.178, -206.496, -614.795],
    [-17.625, -202.414, -612.795],
])
CRUMPLED_AXES = np.array([
    [-0.871, -0.029, 0.491],
    [-0.879, -0.193, 0.436],
    [-0.998, -0.067, 0.0],
    [-0.978, -0.209, 0.0],
])
# a regular tile from the same scan, for contrast
REGULAR_PTS = np.array([
    [-2.78, -177.442, -607.108],
    [-7.942, -184.116, -617.427],
    [0.736, -181.329, -614.109],
    [-11.287, -178.795, -610.687],
])
REGULAR_AXES = np.array([
    [-0.882, -0.122, -0.455],
    [-0.863, -0.293, -0.412],
    [-0.882, -0.213, -0.42],
    [-0.921, -0.275, -0.276],
])


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def _random_pose(rng, kind="full"):
    R = Rotation.random(random_state=int(rng.integers(0, 2 ** 31))).as_matrix()
    return TilePose6(R, rng.uniform(-50.0, 50.0, 3), kind)


# ------------------------------------------------------------------- model
def test_flat_model_is_the_rigid_tile():
    rng = np.random.default_rng(0)
    pose = _random_pose(rng)
    flat = DeformParams()
    assert flat.bending_energy == 0.0 and flat.fold_deg == 0.0
    assert np.allclose(deformed_seed_points(pose, flat), pose.seed_points())
    assert np.allclose(deformed_footprint(pose, flat, offset_mm=0.0),
                       pose.footprint())
    ax = deformed_seed_axes(pose, flat)
    assert np.allclose(np.abs(ax @ pose.t1), 1.0)
    # the tissue face lies SEED_PLANE_OFFSET_MM behind the seed sheet
    fp = deformed_footprint(pose, flat)
    assert np.allclose((fp - pose.footprint()) @ pose.normal,
                       -geometry.SEED_PLANE_OFFSET_MM)


def test_curvature_contracts_chords_like_a_conformed_tile():
    """Seed sheet on a sphere of radius 14 mm (17 mm cavity wall minus the
    3 mm seed offset): the 10 mm tissue-face pitch becomes 10 * 14 / 17 on
    the sheet, i.e. the chords the phantom generator produces."""
    pose = TilePose6(np.eye(3), np.zeros(3))
    k = 1.0 / 14.0
    pts = deformed_seed_points(pose, DeformParams(k, k, 0.0))
    D = cdist(pts, pts)
    sides = sorted(D[np.triu_indices(4, 1)])[:4]
    pitch = 10.0 * 14.0 / 17.0                     # 8.24 mm geodesic
    chord = 2.0 * 14.0 * np.sin(pitch / 2.0 / 14.0)
    assert np.allclose(sides, chord, atol=0.02)
    # bowl toward +n: seeds rise above the tile centre plane
    assert np.all(pts[:, 2] > 0.0)
    # the sheet normal at the centre is n, and tilts away from it at seeds
    _, nrm = deformed_points(pose, DeformParams(k, k, 0.0), [[0.0, 0.0]])
    assert np.allclose(nrm[0], [0.0, 0.0, 1.0])
    assert DeformParams(k, k, 0.0).fold_deg == pytest.approx(
        np.degrees(20.0 / 14.0))


def test_hinge_limit_squeezes_sides_like_a_fold():
    """A sharp fold about the tile midline (kappa1 large, kappa2 = 0) pulls
    the cross-fold sides far below 10 mm while the along-fold sides keep
    the (flat) 10 mm pitch."""
    pose = TilePose6(np.eye(3), np.zeros(3))
    pts = deformed_seed_points(pose, DeformParams(0.3, 0.0, 0.0))
    D = cdist(pts, pts)
    sides = sorted(D[np.triu_indices(4, 1)])[:4]
    assert sides[0] < 6.0 and sides[1] < 6.0
    assert sides[2] == pytest.approx(10.0) and sides[3] == pytest.approx(10.0)


def test_surface_grid_shape_and_footprint_extent():
    pose = TilePose6(np.eye(3), np.zeros(3))
    grid = deformed_surface_grid(pose, DeformParams(0.05, 0.02, 0.3), n=5)
    assert grid.shape == (25, 3)
    fp = deformed_footprint(pose, DeformParams(), offset_mm=0.0)
    assert np.allclose(np.abs(fp[:, :2]), geometry.TILE_HALF_SIZE_MM)
    half = deformed_footprint(TilePose6(np.eye(3), np.zeros(3), "half"),
                              DeformParams(), offset_mm=0.0)
    assert np.allclose(np.abs(half[:, 0]), geometry.TILE_HALF_SIZE_MM / 2.0)


# ---------------------------------------------------------- exact recovery
@pytest.mark.parametrize("seed", range(12))
def test_recovers_synthetic_bent_tile(seed):
    rng = np.random.default_rng(200 + seed)
    pose = _random_pose(rng)
    truth = DeformParams(kappa1=rng.uniform(0.0, 0.2),
                         kappa2=rng.uniform(0.0, 0.1),
                         psi=rng.uniform(-1.2, 1.2))
    pts = deformed_seed_points(pose, truth)
    axes = deformed_seed_axes(pose, truth)
    axes = axes * rng.choice([-1.0, 1.0], size=(4, 1))
    order = rng.permutation(4)
    fit = fit_deformable(pts[order], axes[order])
    assert fit.rms_mm < 0.05
    assert fit.axis_err_deg < 1.0
    assert np.linalg.norm(fit.pose.center - pose.center) < 0.2
    assert _angle_deg(fit.pose.normal, pose.normal) < 1.5
    assert fit.bending_energy == pytest.approx(truth.bending_energy, abs=0.004)
    D = cdist(fit.seed_points(), pts)
    assert D.min(axis=1).max() < 0.05
    # the bowl side is resolved by the axes: n points the same way
    assert float(fit.pose.normal @ pose.normal) > 0.0


def test_half_tile_falls_back_to_rigid():
    rng = np.random.default_rng(5)
    pose = _random_pose(rng, "half")
    pts = pose.seed_points()
    axes = np.repeat(pose.t1[None, :], 2, axis=0)
    fit = fit_deformable(pts, axes)
    assert fit.pose.kind == "half"
    assert fit.bending_energy == 0.0
    assert fit.rms_mm < 1e-6


# -------------------------------------------------------------- phantom
@pytest.mark.parametrize("rng_seed", range(4))
def test_phantom_conformed_tiles_fit_below_half_mm(rng_seed):
    _, truth = make_head_phantom(spacing=2.0, n_tiles=3, noise_hu=0.0,
                                 rng_seed=rng_seed, fov_mm=160.0)
    for t in truth.tiles:
        pts = np.array([truth.seeds[i].center_ras for i in t.seed_ids])
        axes = np.array([truth.seeds[i].axis_ras for i in t.seed_ids])
        rigid = fit_rigid(pts, axes)
        fit = fit_deformable(pts, axes)
        assert fit.rms_mm < 0.5, fit.rms_mm
        assert fit.rms_mm < rigid.rms_mm / 2.0
        # a ~17 mm cavity: seed-sheet curvature ~1/14 in both directions
        assert 0.04 < fit.params.kappa1 < 0.12
        assert 0.04 < fit.params.kappa2 < 0.12
        assert fit.bending_energy < 0.03
        assert fit.axis_err_deg < 8.0
        # the fitted normal bows toward the cavity centre
        to_cav = truth.cavity_center_ras - fit.pose.center
        assert float(fit.pose.normal @ to_cav) > 0.0
        assert deformable_score(fit) > LAMBDA_FULL + 2.0


# ---------------------------------------------------------- real crumple
def test_crumpled_printed_phantom_tile_is_explained_by_a_fold():
    fit = fit_deformable(CRUMPLED_PTS, CRUMPLED_AXES)
    assert fit.rms_mm < LOOSE_RMS_MAX_MM
    assert fit.axis_err_deg < 15.0
    assert fit.params.fold_deg > 150.0            # a real fold, not a bend
    assert 0.03 < fit.bending_energy < 0.10
    assert deformable_score(fit) > LAMBDA_FULL
    # the regular tile from the same scan bends only mildly
    reg = fit_deformable(REGULAR_PTS, REGULAR_AXES)
    assert reg.rms_mm < 0.6
    assert reg.bending_energy < 0.01
    assert deformable_score(reg) > deformable_score(fit)


def test_crumpled_tile_recovered_in_auto_mode_without_a_count():
    """The count-fallback case of tests/test_tiles_degraded.py: two regular
    tiles plus the crumpled one, no count given -> three full tiles, the
    crumpled one flagged degraded and carrying its fold."""
    from tests.test_tiles_degraded import _build_case

    centers, axes = _build_case()
    result = fit_tiles(centers, axes, "auto")
    assert result.n_selected == 3
    assert sorted(sorted(p.seed_indices) for p in result.tiles) == \
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    crumpled = [p for p in result.tiles if sorted(p.seed_indices) == [8, 9, 10, 11]][0]
    assert crumpled.degraded
    assert crumpled.deform is not None
    assert crumpled.deform.params.fold_deg > 100.0
    assert all(not p.degraded for p in result.tiles if p is not crumpled)
    assert all(p.deform is not None for p in result.tiles)


def test_auto_mode_still_refuses_clip_lines_and_scatter():
    from tests.test_tiles_degraded import _build_case

    centers, axes = _build_case()
    t1 = np.array([1.0, 0.0, 0.0])
    clip_line = [np.array([90.0, 0.0, 0.0]) + i * np.array([0.0, 4.0, 0.0])
                 for i in range(4)]
    centers = np.vstack([centers, clip_line])
    axes = np.vstack([axes, [t1] * 4])
    result = fit_tiles(centers, axes, "auto")
    assert result.n_selected == 3
    assert sorted(result.rejected_indices) == [12, 13, 14, 15]

    rng = np.random.default_rng(0)
    centers2, axes2 = _build_case()
    centers2[8:] = np.array([[60, 0, 0], [95, 40, 0], [60, -60, 30],
                             [130, 0, -40]], dtype=float)
    axes2[8:] = rng.normal(size=(4, 3))
    result = fit_tiles(centers2, axes2, "auto")
    assert result.n_selected == 2
    assert sorted(result.rejected_indices) == [8, 9, 10, 11]


# ------------------------------------------------------------ non-tiles
def test_random_quads_are_rejected():
    """Loose-window random 4-point groups with random axes: the bent-tile
    model must not explain them (residual, energy or axis error blows up)."""
    rng = np.random.default_rng(42)
    n_ok = 0
    n = 0
    while n < 60:
        pts = rng.uniform(-8.0, 8.0, (4, 3))
        D = cdist(pts, pts)
        chords = D[np.triu_indices(4, 1)]
        if chords.min() < 3.5 or chords.max() > 16.5:
            continue
        n += 1
        axes = rng.normal(size=(4, 3))
        fit = fit_deformable(pts, axes)
        ok = (fit.rms_mm <= LOOSE_RMS_MAX_MM and fit.bending_energy <= 0.10
              and fit.axis_err_deg <= 25.0)
        n_ok += ok
    assert n_ok <= 2, n_ok
