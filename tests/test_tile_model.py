"""Rigid nominal tile model + Kabsch pose fit (gtcore.tiles.model).

Acceptance (plan Step 1): on synthetic rigid layouts the pose is recovered to
< 0.1 mm / < 1 deg, for full and half tiles, from arbitrarily ordered seeds,
under every symmetry of the square, with the seed axes resolving the
90-degree ambiguity.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from gtcore import geometry
from gtcore.tiles import RigidTile, TilePose6, fit_rigid
from gtcore.tiles.model import kabsch


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def _random_pose(rng, kind="full"):
    R = Rotation.random(random_state=int(rng.integers(0, 2 ** 31))).as_matrix()
    t = rng.uniform(-60.0, 60.0, 3)
    return TilePose6(R, t, kind)


def _observe(pose, rng, noise_mm=0.0, shuffle=True):
    pts = pose.seed_points()
    axes = np.repeat(pose.t1[None, :], pts.shape[0], axis=0)
    # arbitrary per-seed sign of the axis, as PCA returns it
    axes = axes * rng.choice([-1.0, 1.0], size=(pts.shape[0], 1))
    if noise_mm > 0.0:
        pts = pts + rng.normal(0.0, noise_mm, pts.shape)
    order = np.arange(pts.shape[0])
    if shuffle:
        rng.shuffle(order)
    return pts[order], axes[order]


def _assert_pose_matches(fit, pose, tol_mm=0.1, tol_deg=1.0):
    got = fit.pose
    assert np.linalg.norm(got.center - pose.center) < tol_mm
    assert _angle_deg(got.normal, pose.normal) < tol_deg          # up to sign
    assert _angle_deg(got.t1, pose.t1) < tol_deg                  # up to sign
    # seed sets coincide (bipartite nearest-neighbour)
    D = cdist(got.seed_points(), pose.seed_points())
    assert D.min(axis=1).max() < tol_mm
    assert sorted(D.argmin(axis=1).tolist()) == list(range(D.shape[0]))


# --------------------------------------------------------------------- model
def test_canonical_geometry_matches_manufactured_tile():
    full = RigidTile("full")
    half = RigidTile("half")
    assert full.n_seeds == 4 and half.n_seeds == 2
    D = cdist(full.seed_xyz, full.seed_xyz)
    sides = sorted(D[np.triu_indices(4, 1)])
    assert np.allclose(sides[:4], geometry.SEED_PITCH_MM)
    assert np.allclose(sides[4:], geometry.SEED_PITCH_MM * np.sqrt(2.0))
    fp = full.footprint_uv
    assert np.allclose(np.abs(fp), geometry.TILE_HALF_SIZE_MM)
    # seeds sit SEED_EDGE_MARGIN_MM from the two nearest edges
    assert np.allclose(geometry.TILE_HALF_SIZE_MM - np.abs(full.seed_uv),
                       geometry.SEED_EDGE_MARGIN_MM)
    assert np.linalg.norm(half.seed_xyz[0] - half.seed_xyz[1]) == \
        pytest.approx(geometry.SEED_PITCH_MM)
    hf = half.footprint_uv
    assert np.allclose(np.abs(hf[:, 0]), geometry.TILE_HALF_SIZE_MM / 2.0)
    assert np.allclose(np.abs(hf[:, 1]), geometry.TILE_HALF_SIZE_MM)
    with pytest.raises(ValueError):
        RigidTile("third")


def test_pose_apply_and_flip():
    rng = np.random.default_rng(0)
    pose = _random_pose(rng)
    assert np.allclose(pose.R @ pose.R.T, np.eye(3))
    pts = pose.seed_points()
    assert pts.shape == (4, 3)
    assert np.allclose(pts.mean(axis=0), pose.center)
    flipped = pose.flipped_normal()
    assert np.allclose(flipped.normal, -pose.normal)
    assert np.allclose(flipped.t1, pose.t1)
    assert np.linalg.det(flipped.R) == pytest.approx(1.0)
    D = cdist(flipped.seed_points(), pts)
    assert D.min(axis=1).max() < 1e-9


def test_kabsch_exact():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(6, 3))
    R = Rotation.random(random_state=3).as_matrix()
    t = np.array([1.0, -2.0, 3.0])
    P = Q @ R.T + t
    Rh, th, s = kabsch(P, Q)
    assert np.allclose(Rh, R, atol=1e-9) and np.allclose(th, t, atol=1e-9)
    assert s == 1.0
    Rh, th, s = kabsch(0.8 * P, Q, allow_scale=True)
    assert s == pytest.approx(0.8)
    # reflection-safe: a mirrored cloud still yields a proper rotation
    Rh, _, _ = kabsch(P * np.array([1.0, 1.0, -1.0]), Q)
    assert np.linalg.det(Rh) == pytest.approx(1.0)


# ---------------------------------------------------------------- full tiles
@pytest.mark.parametrize("seed", range(40))
def test_full_tile_pose_recovery(seed):
    rng = np.random.default_rng(seed)
    pose = _random_pose(rng, "full")
    pts, axes = _observe(pose, rng, noise_mm=0.03)
    fit = fit_rigid(pts, axes)
    assert fit.pose.kind == "full"
    assert fit.rms_mm < 0.1
    assert fit.axis_err_deg < 1.0
    _assert_pose_matches(fit, pose)
    # assignment maps every observed seed onto its canonical partner
    canon = RigidTile("full").seed_xyz[list(fit.assignment)]
    assert np.linalg.norm(fit.pose.apply(canon) - pts, axis=1).max() < 0.15


def test_full_tile_exact_is_exact():
    rng = np.random.default_rng(99)
    pose = _random_pose(rng)
    pts, axes = _observe(pose, rng)
    fit = fit_rigid(pts, axes)
    assert fit.rms_mm < 1e-9
    assert fit.axis_err_deg < 1e-4     # arccos near 1 is sqrt-precision
    _assert_pose_matches(fit, pose, tol_mm=1e-6, tol_deg=1e-5)


def test_axes_resolve_the_90_degree_symmetry():
    """Without axes the fitted t1 may land on t2 (a rotated labelling of the
    same square); with axes it must land on the true seed direction."""
    rng = np.random.default_rng(5)
    hits_without = 0
    for _ in range(20):
        pose = _random_pose(rng)
        pts, axes = _observe(pose, rng)
        f_no = fit_rigid(pts)
        f_ax = fit_rigid(pts, axes)
        assert f_no.rms_mm < 1e-9 and f_ax.rms_mm < 1e-9
        assert _angle_deg(f_no.pose.normal, pose.normal) < 1e-5
        assert _angle_deg(f_ax.pose.t1, pose.t1) < 1e-5
        # the axis-free fit is right only up to the square's symmetries
        a = _angle_deg(f_no.pose.t1, pose.t1)
        assert a < 1e-5 or abs(a - 90.0) < 1e-5
        hits_without += a < 1e-5
    assert hits_without < 20     # the symmetry is real: it was not always right


def test_flipped_tile_is_recovered():
    """A tile laid face-down is the mirror labelling; the fit still returns a
    proper rotation whose seed set and axis match."""
    rng = np.random.default_rng(11)
    pose = _random_pose(rng)
    pts, axes = _observe(pose.flipped_normal(), rng)
    fit = fit_rigid(pts, axes)
    assert fit.rms_mm < 1e-9
    assert np.linalg.det(fit.pose.R) == pytest.approx(1.0)
    _assert_pose_matches(fit, pose, tol_mm=1e-6, tol_deg=1e-5)


def test_non_square_quad_has_large_residual():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-8.0, 8.0, (4, 3))
    fit = fit_rigid(pts, rng.normal(size=(4, 3)))
    assert fit.rms_mm > 1.0


def test_similarity_scale_absorbs_chord_contraction():
    """Wall conforming contracts a tile's chords ~18 % (seed plane 3 mm
    inside a ~17 mm cavity); the similarity fit reads that scale off and
    leaves a near-zero residual, while the rigid fit sees ~1 mm."""
    rng = np.random.default_rng(3)
    pose = _random_pose(rng)
    pose.scale = 0.82
    pts, axes = _observe(pose, rng)
    rigid = fit_rigid(pts, axes)
    assert rigid.rms_mm > 0.8
    sim = fit_rigid(pts, axes, allow_scale=True)
    assert sim.pose.scale == pytest.approx(0.82, abs=1e-6)
    assert sim.rms_mm < 1e-6
    assert np.linalg.norm(sim.pose.center - pose.center) < 1e-6
    # the scale window is enforced
    clipped = fit_rigid(pts * 0.5, axes, allow_scale=True, scale_range=(0.7, 1.1))
    assert clipped.pose.scale == pytest.approx(0.7)


# ---------------------------------------------------------------- half tiles
@pytest.mark.parametrize("seed", range(20))
def test_half_tile_pose_recovery(seed):
    rng = np.random.default_rng(100 + seed)
    pose = _random_pose(rng, "half")
    pts, axes = _observe(pose, rng, noise_mm=0.03)
    fit = fit_rigid(pts, axes)
    assert fit.pose.kind == "half"
    assert fit.rms_mm < 0.1
    _assert_pose_matches(fit, pose)


def test_half_tile_without_axes_is_underdetermined_but_sane():
    rng = np.random.default_rng(7)
    pose = _random_pose(rng, "half")
    pts, _ = _observe(pose, rng)
    fit = fit_rigid(pts)
    assert fit.rms_mm < 1e-9
    assert np.linalg.norm(fit.pose.center - pose.center) < 1e-9
    assert _angle_deg(fit.pose.t2, pose.t2) < 1e-5     # the pair direction
    assert abs(float(fit.pose.normal @ pose.t2)) < 1e-6


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        fit_rigid(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        fit_rigid(np.zeros((4, 3)), kind="half")
    with pytest.raises(ValueError):
        fit_rigid(np.zeros((4, 3)), np.zeros((2, 3)))
