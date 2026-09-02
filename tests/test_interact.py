"""Verification campaign for the tile drag-and-drop geometry (gtcore.interact).

The phantom generator is the ground truth: it conforms tiles to the analytic
lumpy cavity wall (per-seed re-projection + 3 mm inset), so reproducing its
seed positions from the *meshed* cavity mask proves the interactive conform
implements the same physics.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.optimize import linear_sum_assignment

from gtcore.interact import (
    SEED_WALL_OFFSET_MM,
    PlacedTile,
    conform_tile,
    rotate_on_wall,
    snap_to_wall,
    tiles_to_seed_arrays,
    translate_on_wall,
)
from gtcore.phantom import make_head_phantom
from gtcore.phantom.generate import ENTRY_DIR_U0
from gtcore.segment import mask_to_mesh

FULL_ADJACENT_PAIRS = ((0, 1), (0, 2), (1, 3), (2, 3))  # 10 mm grid neighbours


@pytest.fixture(scope="module")
def case():
    vol, truth = make_head_phantom(spacing=0.8)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    assert len(mesh.vertices) > 1000
    return vol, truth, mesh


def _wall_point_along(mesh, origin, direction):
    """Ray-cast from inside the cavity to the wall (outermost hit)."""
    locs, _, _ = mesh.ray.intersects_location(
        ray_origins=[origin], ray_directions=[direction]
    )
    locs = np.atleast_2d(np.asarray(locs, dtype=float))
    if locs.size == 0:
        surf, _ = snap_to_wall(mesh, origin + 25.0 * direction)
        return surf
    return locs[np.argmax(np.linalg.norm(locs - origin, axis=1))]


def _seatable(mesh, tile, max_sag_mm=3.0):
    """A tile can seat only where the wall stays close to its seating plane.

    Rejects placements straddling a sharp fold / the tract mouth, where a
    rigid 20 mm tile could not physically make contact.
    """
    for s in tile.seed_centers:
        _, n = snap_to_wall(mesh, s)
        wall = s - SEED_WALL_OFFSET_MM * n
        if abs(float(np.dot(wall - tile.anchor_ras, tile.normal_ras))) > max_sag_mm:
            return False
    return True


# ------------------------------------------------- ground-truth reproduction
def test_conform_reproduces_phantom_truth(case):
    """conform_tile must reproduce the phantom's own tile physics.

    For each truth tile, seat a tile at the truth wall point with the truth
    normal and in-plane axis: the conformed seed centres must match the truth
    seed centres within 1.5 mm each (matched by optimal assignment -- the
    grid traversal order is not part of the contract).
    """
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    all_errs = []
    for tile in truth.tiles:
        d = tile.normal_ras / np.linalg.norm(tile.normal_ras)  # outward radial
        w = _wall_point_along(mesh, cav_c, d)                  # truth centre -> wall
        t1 = np.cross(d, ENTRY_DIR_U0)
        t1 /= np.linalg.norm(t1)                               # truth in-plane axis

        placed = conform_tile(mesh, w, -d, t1, kind="full")
        assert placed.seed_centers.shape == (4, 3)

        truth_centers = np.array(
            [truth.seeds[i].center_ras for i in tile.seed_ids]
        )
        D = np.linalg.norm(
            placed.seed_centers[:, None, :] - truth_centers[None, :, :], axis=2
        )
        rows, cols = linear_sum_assignment(D)
        errs = D[rows, cols]
        all_errs.extend(errs)
        assert errs.max() < 1.5, (
            "tile %d: conform vs truth seed errors %s mm"
            % (tile.tile_id, np.round(errs, 2))
        )
    # the whole campaign should be sub-millimetre, not just sub-1.5
    assert float(np.mean(all_errs)) < 1.0


# ------------------------------------------------------------------- snapping
def test_snap_to_wall_from_offset_points(case):
    _vol, _truth, mesh = case
    rng = np.random.default_rng(1)
    centroid = mesh.centroid
    for _ in range(15):
        v = mesh.vertices[rng.integers(0, len(mesh.vertices))]
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        probe = v + rng.uniform(5.0, 15.0) * direction

        surf, n_in = snap_to_wall(mesh, probe)
        _, dist, _ = trimesh.proximity.closest_point(mesh, [surf])
        assert dist[0] < 0.5, "snapped point %.3f mm off the mesh" % dist[0]
        assert abs(np.linalg.norm(n_in) - 1.0) < 1e-6
        assert float(np.dot(n_in, centroid - surf)) > 0.0, (
            "inward normal does not point toward the cavity interior"
        )


def test_snap_normal_robust_to_mesh_winding(case):
    """Either mesh orientation must give the same inward normal."""
    _vol, _truth, mesh = case
    flipped = mesh.copy()
    flipped.invert()
    p = mesh.vertices[123] + 4.0
    s1, n1 = snap_to_wall(mesh, p)
    s2, n2 = snap_to_wall(flipped, p)
    assert np.allclose(s1, s2, atol=1e-6)
    assert float(np.dot(n1, n2)) > 0.99


# ------------------------------------------------------ placement invariants
def test_placement_invariants_random(case):
    _vol, _truth, mesh = case
    rng = np.random.default_rng(0)
    centroid = mesh.centroid

    accepted = 0
    attempts = 0
    while accepted < 20:
        attempts += 1
        assert attempts < 200, "could not find 20 seatable placements"
        vi = rng.integers(0, len(mesh.vertices))
        sp, n_in = snap_to_wall(mesh, mesh.vertices[vi])
        tile = conform_tile(mesh, sp, n_in, rng.standard_normal(3), kind="full")
        if not _seatable(mesh, tile):
            continue
        accepted += 1

        # every seed 2.25-3.75 mm off the mesh (hydrated seed-plane spread),
        # plus mesh-projection slack, on the cavity side
        surf, dist, _ = trimesh.proximity.closest_point(mesh, tile.seed_centers)
        for s, q, dd in zip(tile.seed_centers, surf, dist):
            assert 2.0 <= dd <= 4.0, "seed %.2f mm from wall" % dd
            _, nq = snap_to_wall(mesh, q)
            assert float(np.dot(s - q, nq)) > 0.0, "seed on tissue side of wall"

        # 10 mm grid pitch, chord-shortened by curvature
        for a, b in FULL_ADJACENT_PAIRS:
            d = np.linalg.norm(tile.seed_centers[a] - tile.seed_centers[b])
            assert 7.5 <= d <= 10.5, "adjacent seed spacing %.2f mm" % d

        # seed axes unit and tangent to the local wall
        for s, ax in zip(tile.seed_centers, tile.seed_axes):
            assert abs(np.linalg.norm(ax) - 1.0) < 1e-6
            _, nl = snap_to_wall(mesh, s)
            assert abs(float(np.dot(ax, nl))) < 0.3

        # translate then inverse-translate returns near the start
        delta = np.cross(tile.normal_ras, rng.standard_normal(3))
        delta = 2.0 * delta / np.linalg.norm(delta)
        there = translate_on_wall(mesh, tile, delta)
        back = translate_on_wall(mesh, there, -delta)
        assert np.linalg.norm(back.center_ras - tile.center_ras) < 1.0

        # a full turn about the local normal is the identity (axis-wise)
        turned = rotate_on_wall(mesh, tile, 2.0 * np.pi)
        cosang = abs(float(np.dot(turned.axis_ras, tile.axis_ras)))
        assert np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) < 5.0

        # bookkeeping fields
        assert tile.kind == "full"
        assert np.allclose(tile.center_ras, tile.seed_centers.mean(axis=0))
        assert tile.corners_ras.shape == (4, 3)
        assert float(np.dot(tile.normal_ras, centroid - tile.anchor_ras)) > 0.0


# ------------------------------------------------------------------ half tiles
def test_half_tiles(case):
    """Half tiles: 2 seeds ~10 mm apart, seated at the truth tile sites."""
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    for tile in truth.tiles:
        d = tile.normal_ras / np.linalg.norm(tile.normal_ras)
        w = _wall_point_along(mesh, cav_c, d)
        t1 = np.cross(d, ENTRY_DIR_U0)
        t1 /= np.linalg.norm(t1)
        for angle in (0.5, 1.5):
            half = conform_tile(mesh, w, -d, t1, kind="half")
            half = rotate_on_wall(mesh, half, angle)
            assert half.kind == "half"
            assert half.seed_centers.shape == (2, 3)
            assert half.seed_axes.shape == (2, 3)

            spacing = np.linalg.norm(half.seed_centers[0] - half.seed_centers[1])
            # 10 mm pitch, chord-shortened by the concave wall at the 3 mm inset
            assert 7.5 <= spacing <= 10.5, "half-tile spacing %.2f mm" % spacing

            _, dist, _ = trimesh.proximity.closest_point(mesh, half.seed_centers)
            assert np.all(dist >= 2.0) and np.all(dist <= 4.0)
            for s, ax in zip(half.seed_centers, half.seed_axes):
                assert abs(np.linalg.norm(ax) - 1.0) < 1e-6
                _, nl = snap_to_wall(mesh, s)
                assert abs(float(np.dot(ax, nl))) < 0.3


def test_conform_rejects_unknown_kind(case):
    _vol, _truth, mesh = case
    sp, n_in = snap_to_wall(mesh, mesh.vertices[0])
    with pytest.raises(ValueError):
        conform_tile(mesh, sp, n_in, [1.0, 0.0, 0.0], kind="quarter")


# ---------------------------------------------------------------- seed arrays
def test_tiles_to_seed_arrays(case):
    _vol, _truth, mesh = case
    sp, n_in = snap_to_wall(mesh, mesh.vertices[42])
    full = conform_tile(mesh, sp, n_in, [1.0, 0.0, 0.0], kind="full")
    half = conform_tile(mesh, sp, n_in, [0.0, 1.0, 0.0], kind="half")

    centers, axes = tiles_to_seed_arrays([full, half])
    assert centers.shape == (6, 3)
    assert axes.shape == (6, 3)
    assert np.allclose(centers[:4], full.seed_centers)
    assert np.allclose(centers[4:], half.seed_centers)
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)

    empty_c, empty_a = tiles_to_seed_arrays([])
    assert empty_c.shape == (0, 3) and empty_a.shape == (0, 3)
