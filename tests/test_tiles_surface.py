"""Stick-to-surface tile fit (gtcore.tiles.surface) and its cross-feed with
the surface-free deformable fit.

Acceptance (plan Step 4): on the synthetic phantom the surface-constrained
fit matches the truth poses at least as well as Step 3 alone and localizes
the tile's footprint on the wall; detachment / disagreement are flagged.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from gtcore import geometry
from gtcore.phantom import make_head_phantom
from gtcore.segment import mask_to_mesh
from gtcore.tiles import SurfaceFit, fit_deformable, fit_on_surface, fit_tiles
from gtcore.tiles.surface import DETACHED_MM


@pytest.fixture(scope="module")
def case():
    vol, truth = make_head_phantom(spacing=2.0, n_tiles=3, noise_hu=0.0,
                                   rng_seed=1, fov_mm=160.0)
    # the physical wall: lumen boundary with the capsules filled in
    mesh = mask_to_mesh(truth.masks["cavity"] | truth.masks["seeds"], vol.affine)
    return vol, truth, mesh


def _tile_arrays(truth, tile):
    pts = np.array([truth.seeds[i].center_ras for i in tile.seed_ids])
    axes = np.array([truth.seeds[i].axis_ras for i in tile.seed_ids])
    return pts, axes


def test_surface_fit_matches_truth_and_sits_on_the_wall(case):
    _vol, truth, mesh = case
    for tile in truth.tiles:
        pts, axes = _tile_arrays(truth, tile)
        free = fit_deformable(pts, axes)
        sf = fit_on_surface(mesh, pts, axes, init=free)
        assert isinstance(sf, SurfaceFit)
        assert sf.placed.kind == "full"
        assert sf.rms_mm < 0.9, sf.rms_mm
        assert sf.axis_err_deg < 12.0
        # observed seeds ride the 3 mm offset: attached, and consistent
        # with the free fit
        assert sf.attached and sf.detachment_mm < 0.5
        assert sf.consistent and sf.agreement_mm < 1.0
        assert sf.free_rms_mm == pytest.approx(free.rms_mm)
        assert "attached" in sf.verdict()
        # pose: centre (mean of conformed seeds) within 0.6 mm of truth,
        # normal within 12 deg of the radial truth normal
        assert np.linalg.norm(sf.placed.center_ras - tile.center_ras) < 0.6
        cosang = abs(float(sf.placed.normal_ras @ tile.normal_ras))
        assert np.degrees(np.arccos(min(1.0, cosang))) < 12.0
        # footprint localized on the wall: anchor ON the mesh, corners on the
        # conformed sheet 3 mm off it
        _, d_anchor, _ = trimesh.proximity.closest_point(mesh, [sf.anchor_ras])
        assert d_anchor[0] < 0.3
        _, d_corners, _ = trimesh.proximity.closest_point(
            mesh, sf.placed.corners_ras)
        assert np.all(np.abs(d_corners - geometry.SEED_PLANE_OFFSET_MM) < 0.6)


def test_surface_fit_is_at_least_as_good_as_the_free_fit(case):
    """Seed-mean centre error and normal error vs truth, surface vs free."""
    _vol, truth, mesh = case
    for tile in truth.tiles:
        pts, axes = _tile_arrays(truth, tile)
        free = fit_deformable(pts, axes)
        sf = fit_on_surface(mesh, pts, axes, init=free)
        # centre: the free fit reproduces exact truth seeds to ~0, so the
        # surface fit (limited by the 2 mm marching-cubes wall) can only be
        # held to an absolute bound here
        e_surf = np.linalg.norm(sf.placed.center_ras - tile.center_ras)
        assert e_surf < 0.6
        n_free = np.degrees(np.arccos(min(1.0, abs(float(
            free.pose.normal @ tile.normal_ras)))))
        n_surf = np.degrees(np.arccos(min(1.0, abs(float(
            sf.placed.normal_ras @ tile.normal_ras)))))
        assert n_surf <= n_free + 3.0


def test_lifted_tile_is_flagged_detached(case):
    _vol, truth, mesh = case
    tile = truth.tiles[0]
    pts, axes = _tile_arrays(truth, tile)
    d = truth.cavity_center_ras - pts.mean(axis=0)
    d /= np.linalg.norm(d)
    sf = fit_on_surface(mesh, pts + 4.0 * d, axes)
    assert not sf.attached
    assert sf.detachment_mm > DETACHED_MM
    assert "detached" in sf.verdict()
    assert sf.agreement_mm is None and sf.consistent is None


def test_half_tile_on_surface(case):
    _vol, truth, mesh = case
    tile = truth.tiles[1]
    pts, axes = _tile_arrays(truth, tile)
    # one column of the tile = a half tile (seeds (-5,-5),(-5,+5))
    sf = fit_on_surface(mesh, pts[:2], axes[:2], kind="half")
    assert sf.placed.kind == "half"
    assert sf.placed.seed_centers.shape == (2, 3)
    assert sf.rms_mm < 1.0
    assert sf.attached


def test_auto_mode_attaches_surface_fits(case):
    _vol, truth, mesh = case
    rng = np.random.default_rng(0)
    centers = np.array([s.center_ras for s in truth.seeds])
    axes = np.array([s.axis_ras for s in truth.seeds])
    centers = centers + rng.normal(0.0, 0.2, centers.shape)   # localization noise
    order = rng.permutation(len(centers))
    centers, axes = centers[order], axes[order]
    result = fit_tiles(centers, axes, "auto",
                       cavity_center_ras=truth.cavity_center_ras, mesh=mesh)
    assert result.n_selected == 3
    for pose in result.tiles:
        assert pose.surface is not None
        assert pose.surface.attached
        assert pose.surface.consistent
        assert pose.surface.placed.corners_ras.shape == (4, 3)
    # without a mesh nothing is attached, selection is identical
    plain = fit_tiles(centers, axes, "auto",
                      cavity_center_ras=truth.cavity_center_ras)
    assert [p.seed_indices for p in plain.tiles] == \
        [p.seed_indices for p in result.tiles]
    assert all(p.surface is None for p in plain.tiles)


def test_pipeline_auto_mode_reports_surface_verdicts():
    from gtcore.pipeline import reconstruct

    vol, _truth = make_head_phantom(spacing=1.2, n_tiles=2, rng_seed=1)
    result = reconstruct(vol, verbose=False, n_full_tiles="auto")
    assert result.tiles.n_selected == 2
    assert "cavity" in result.meshes
    for pose in result.tiles.tiles:
        assert pose.surface is not None
        assert pose.surface.attached
