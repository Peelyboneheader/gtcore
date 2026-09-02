"""Tile-overlap detection (gtcore.interact.find_overlapping_tiles).

All placements are deterministic: the phantom (rng_seed=0, spacing=0.8) and
its meshed cavity are bit-stable, and tiles are seated by ray-casting fixed
directions from the cavity centre.  Wall directions are parametrized as
(polar, azimuth) about the entry direction, matching the phantom's own tile
layout band (polar 100-150 deg keeps clear of the tract mouth).

Geometry notes that shaped the expected values below:

- The cavity is only ~30-40 mm across, so a 20 mm tile is strongly curved
  (corners sag 5-10 mm below the seating tangent plane) and legal
  edge-to-edge abutment happens at ~21 mm anchor separation with a real 3D
  gap of ~2.5 mm -- comfortably outside the default 1 mm threshold but well
  inside a generous 4 mm one.
- Truly antipodal tiles on the phantom cavity are >26 mm apart
  centre-to-centre, so the opposite-wall case (centres closer than 20 mm)
  uses a purpose-built thin lens-shaped cavity mesh where the two walls are
  8 mm apart; a same-wall control shows the pair is only cleared by the
  normal check, not by distance.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import trimesh

from gtcore.interact import (
    SEED_WALL_OFFSET_MM,
    conform_tile,
    find_overlapping_tiles,
    snap_to_wall,
    _footprint_surface,
)
from gtcore.phantom import make_head_phantom
from gtcore.phantom.generate import ENTRY_DIR_U0
from gtcore.segment import mask_to_mesh


@pytest.fixture(scope="module")
def case():
    vol, truth = make_head_phantom(spacing=0.8)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    assert len(mesh.vertices) > 1000
    return vol, truth, mesh


_U0 = ENTRY_DIR_U0
_E1 = np.cross(_U0, [0.0, 0.0, 1.0])
_E1 /= np.linalg.norm(_E1)
_E2 = np.cross(_U0, _E1)


def _wall_dir(polar_deg, azimuth_deg):
    """Unit direction at (polar, azimuth) about the entry direction."""
    th = np.deg2rad(polar_deg)
    ph = np.deg2rad(azimuth_deg)
    return (np.cos(th) * _U0
            + np.sin(th) * (np.cos(ph) * _E1 + np.sin(ph) * _E2))


def _rotated(base, alpha):
    """``base`` rotated by ``alpha`` rad toward its cross-product tangent."""
    perp = np.cross(base, _U0)
    perp /= np.linalg.norm(perp)
    d = np.cos(alpha) * base + np.sin(alpha) * perp
    return d / np.linalg.norm(d)


def _place(mesh, cav_c, direction, kind="full", axis=None):
    """Conform a tile at the wall point hit by a ray from the cavity centre."""
    locs, _, _ = mesh.ray.intersects_location(
        ray_origins=[cav_c], ray_directions=[direction]
    )
    locs = np.atleast_2d(np.asarray(locs, dtype=float))
    assert locs.size, "wall ray missed the cavity mesh"
    w = locs[np.argmax(np.linalg.norm(locs - cav_c, axis=1))]
    surf, n_in = snap_to_wall(mesh, w)
    if axis is None:
        axis = np.cross(n_in, _U0)
    return conform_tile(mesh, surf, n_in, axis, kind=kind)


def _sep(ta, tb):
    return float(np.linalg.norm(ta.anchor_ras - tb.anchor_ras))


# ------------------------------------------------------------------ trivial
def test_empty_and_single(case):
    _vol, truth, mesh = case
    assert find_overlapping_tiles([]) == []
    tile = _place(mesh, truth.cavity_center_ras, _wall_dir(140.0, 180.0))
    assert find_overlapping_tiles([tile]) == []
    assert find_overlapping_tiles((tile,), threshold_mm=25.0) == []


# ------------------------------------------------------------ far / near
def test_far_apart_not_flagged(case):
    """Wall anchors ~30 mm apart: footprints are >8 mm clear -> no overlap."""
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    base = _wall_dir(140.0, 180.0)
    a = _place(mesh, cav_c, base)
    b = _place(mesh, cav_c, _rotated(base, 1.9), axis=a.axis_ras)
    assert 28.0 <= _sep(a, b) <= 35.0
    assert find_overlapping_tiles([a, b]) == []
    assert find_overlapping_tiles([b, a]) == []


def test_intersecting_footprints_flagged(case):
    """Anchors ~8 mm apart: two 20 mm footprints must interpenetrate."""
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    base = _wall_dir(140.0, 180.0)
    a = _place(mesh, cav_c, base)
    b = _place(mesh, cav_c, _rotated(base, 0.45), axis=a.axis_ras)
    assert 6.0 <= _sep(a, b) <= 10.0
    assert find_overlapping_tiles([a, b], threshold_mm=1.0) == [(0, 1)]


def test_edge_to_edge_abutment_is_legal(case):
    """Edge-to-edge tiles (~21.5 mm anchors, ~2.5 mm real gap on this curved
    wall): legal at the default 1 mm threshold, flagged at a generous 4 mm."""
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    base = _wall_dir(140.0, 180.0)
    a = _place(mesh, cav_c, base)
    b = _place(mesh, cav_c, _rotated(base, 1.25), axis=a.axis_ras)
    assert 20.0 <= _sep(a, b) <= 23.0
    assert find_overlapping_tiles([a, b], threshold_mm=1.0) == []
    assert find_overlapping_tiles([a, b], threshold_mm=4.0) == [(0, 1)]


# -------------------------------------------------------------- normal gate
def test_opposite_walls_not_flagged():
    """Tiles on the two walls of a thin (8 mm) cavity sit ~4 mm apart
    centre-to-centre yet must NOT be flagged: their inward normals are
    anti-parallel (opposite walls).  A same-wall pair at comparable distance
    IS flagged, proving distance alone would have fired."""
    lens = trimesh.creation.icosphere(subdivisions=4)
    lens.apply_scale([30.0, 30.0, 4.0])
    top = conform_tile(lens, [0.0, 0.0, 4.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
    bot = conform_tile(lens, [0.0, 0.0, -4.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])

    assert float(np.linalg.norm(top.center_ras - bot.center_ras)) < 20.0
    assert float(np.dot(top.normal_ras, bot.normal_ras)) < -0.9
    assert find_overlapping_tiles([top, bot], threshold_mm=6.0) == []
    assert find_overlapping_tiles([top, bot], threshold_mm=1.0) == []

    # control: same wall, similar 3D proximity -> the 6 mm threshold fires
    top2 = conform_tile(lens, [6.0, 0.0, 3.75], [0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0])
    assert find_overlapping_tiles([top, top2], threshold_mm=6.0) == [(0, 1)]


# -------------------------------------------------------------- half tiles
def test_half_vs_full_adjacency(case):
    """At ~18.7 mm anchors with the neighbour's axis facing the gap, a FULL
    neighbour (reaches 10 mm) still collides while a HALF neighbour at the
    same seat (reaches only 5 mm across its 10 mm side) leaves ~2 mm of
    clearance -- legal at 1 mm, caught at 4 mm."""
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    base = _wall_dir(135.0, 60.0)
    a = _place(mesh, cav_c, base)
    perp = np.cross(base, _U0)
    perp /= np.linalg.norm(perp)
    d2 = _rotated(base, 1.10)
    b_full = _place(mesh, cav_c, d2, kind="full", axis=perp)
    b_half = _place(mesh, cav_c, d2, kind="half", axis=perp)

    assert 17.0 <= _sep(a, b_full) <= 20.5
    assert np.allclose(b_full.anchor_ras, b_half.anchor_ras)
    assert find_overlapping_tiles([a, b_full], threshold_mm=1.0) == [(0, 1)]
    assert find_overlapping_tiles([a, b_half], threshold_mm=1.0) == []
    assert find_overlapping_tiles([a, b_half], threshold_mm=4.0) == [(0, 1)]


def test_mixed_list_ordering(case):
    """Mixed list: only the touching pair is reported, as (i, j) with i<j."""
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    base = _wall_dir(140.0, 180.0)
    a = _place(mesh, cav_c, base)
    c_far = _place(mesh, cav_c, _wall_dir(105.0, 0.0))
    b_near = _place(mesh, cav_c, _rotated(base, 0.45), axis=a.axis_ras)
    assert _sep(a, c_far) > 26.0 and _sep(b_near, c_far) > 26.0

    result = find_overlapping_tiles([a, c_far, b_near], threshold_mm=1.0)
    assert result == [(0, 2)]
    for i, j in result:
        assert i < j


# ------------------------------------------------------- footprint fidelity
def test_footprint_tracks_cavity_wall(case):
    """The sampled footprint must ride the mesh at its 2 mm collagen inset:
    every sample within ~1 mm of that offset surface, for full and half
    tiles, with unit inward-pointing sample normals."""
    _vol, truth, mesh = case
    cav_c = truth.cavity_center_ras
    for kind in ("full", "half"):
        tile = _place(mesh, cav_c, _wall_dir(135.0, 60.0), kind=kind)
        pts, nrm = _footprint_surface(tile)
        assert pts.shape == nrm.shape and pts.shape[1] == 3

        _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
        err = np.abs(dist - SEED_WALL_OFFSET_MM)
        assert float(err.max()) <= 1.1, (
            "%s footprint strays %.2f mm from the 2 mm-offset wall"
            % (kind, err.max())
        )
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-9)
        # normals point into the cavity, not out through the wall
        inward = ((cav_c[None, :] - pts) * nrm).sum(axis=1)
        assert np.all(inward > 0.0)


# ------------------------------------------------------------- performance
def test_ten_tiles_under_50ms(case):
    """The check fires on every drag step: 10 tiles must clear 50 ms."""
    _vol, _truth, mesh = case
    rng = np.random.default_rng(3)
    tiles = []
    while len(tiles) < 10:
        v = mesh.vertices[rng.integers(0, len(mesh.vertices))]
        surf, n_in = snap_to_wall(mesh, v)
        tiles.append(
            conform_tile(mesh, surf, n_in, rng.standard_normal(3),
                         kind="half" if len(tiles) % 3 == 2 else "full")
        )

    result = find_overlapping_tiles(tiles, threshold_mm=1.0)  # warm-up
    assert all(i < j for i, j in result)
    assert all(0 <= i < 10 and 0 <= j < 10 for i, j in result)

    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        again = find_overlapping_tiles(tiles, threshold_mm=1.0)
        best = min(best, time.perf_counter() - t0)
    assert again == result, "overlap result must be deterministic"
    assert best < 0.050, "10-tile overlap check took %.1f ms" % (best * 1e3)
