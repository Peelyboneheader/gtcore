"""Tests for ``gtcore.plan.candidates`` (A1 Geometry, branch plan/candidates).

Covers section 3 A of docs/plan-tile-optimize.md: anchor sampling, spin
sets, the conformer's hanging / detached rejections (checked against a
direct replica of ``interact._project_to_wall``'s ray cast), determinism,
``visible_faces`` on a hollow shell, ``recommend_tile_count`` (section 10)
and the section 4 V8 failure modes.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import trimesh

import gtcore.plan as plan
import plan_fixtures as pf
from gtcore.interact import SEED_WALL_OFFSET_MM, _grid_offsets, _rodrigues, snap_to_wall
from gtcore.plan import DETACHED_MM, TILE_AREA_CM2, CandidateSet, TileCountRecommendation
from gtcore.plan import candidates as C


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def sphere():
    return pf.sphere_cap_mesh(25.0)


@pytest.fixture(scope="module")
def sphere_cands(sphere):
    return C.build_candidates(sphere, h_mm=6.0, n_spins=2)


@pytest.fixture(scope="module")
def flat():
    mesh = pf.flat_wall_mesh()
    return mesh, pf.flat_wall_top_faces(mesh)


@pytest.fixture(scope="module")
def flat_cands(flat):
    mesh, top = flat
    return C.build_candidates(mesh, h_mm=5.0, n_spins=3, eligible_faces=top)


@pytest.fixture(scope="module")
def cavity():
    from gtcore.phantom import make_head_phantom
    from gtcore.segment import mask_to_mesh
    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)
    return mask_to_mesh(truth.masks["cavity"], vol.affine)


@pytest.fixture(scope="module")
def hollow():
    inner = trimesh.creation.icosphere(subdivisions=3, radius=25.0)
    inner.invert()                                   # inner wall faces the cavity
    outer = trimesh.creation.icosphere(subdivisions=3, radius=35.0)
    return trimesh.util.concatenate([inner, outer])


def _flat_grid(tile):
    """Flat tangent-plane grid points of a conformed tile and its normal,
    exactly as ``_project_to_wall`` lays them out."""
    n, t1, t2 = C._conform_frame(tile.normal_ras, tile.axis_ras)
    uv = np.vstack(_grid_offsets(tile.kind))
    return tile.anchor_ras[None, :] + uv @ np.vstack([t1, t2]), np.tile(n, (uv.shape[0], 1))


def _true_fallbacks(mesh, tile):
    flat, nrm = _flat_grid(tile)
    return int(C.count_ray_fallbacks_trimesh(mesh, flat, nrm).sum())


# ------------------------------------------------------------------ basics
def test_sphere_candidate_set_fields(sphere, sphere_cands):
    cs = sphere_cands
    c = len(cs)
    assert isinstance(cs, CandidateSet) and c > 100
    assert cs.anchors.shape == (c, 3) and cs.corners.shape == (c, 4, 3)
    assert set(cs.kinds.tolist()) == {"full"}
    assert np.all(cs.n_seeds == 4)
    assert np.isfinite(cs.seed_centers).all() and np.isfinite(cs.seed_axes).all()
    assert cs.eligible.all()
    assert cs.method.startswith("farthest_point")
    assert cs.h_mm == 6.0 and cs.n_spins == 2
    assert cs.wall_area_mm2 == pytest.approx(sphere.area)
    assert set(cs.n_rejected) >= {"hanging", "detached", "ineligible"}
    assert sorted(set(cs.spins_deg.tolist())) == [0.0, 45.0]
    # anchors lie on the mesh
    _p, dist, _t = trimesh.proximity.closest_point(sphere, cs.anchors)
    assert np.max(dist) < 1e-6
    # anchor-major, spin-minor ordering; every anchor carries both spins
    assert np.all(np.diff(cs.anchor_ids) >= 0)
    for a in np.unique(cs.anchor_ids):
        ids = np.flatnonzero(cs.anchor_ids == a)
        assert cs.spins_deg[ids].tolist() == [0.0, 45.0]
        assert np.allclose(cs.anchors[ids] - cs.anchors[ids[0]], 0.0)


def test_anchor_spacing_matches_h(sphere_cands):
    anchors = np.unique(sphere_cands.anchors, axis=0)
    d = np.linalg.norm(anchors[:, None] - anchors[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    assert nn.min() >= 6.0 - 1e-9          # farthest-point guarantee
    assert nn.max() <= 2.0 * 6.0           # and no large gaps
    # ~area / (h^2 * packing) anchors
    assert 90 <= anchors.shape[0] <= 220


def test_seeds_sit_three_mm_off_the_wall(sphere, sphere_cands):
    seeds = sphere_cands.seed_centers.reshape(-1, 3)
    _p, dist, _t = trimesh.proximity.closest_point(sphere, seeds)
    assert np.all(np.abs(dist - SEED_WALL_OFFSET_MM) <= DETACHED_MM)
    assert abs(np.median(dist) - SEED_WALL_OFFSET_MM) < 0.3
    # seeds are inside the sphere (offset INTO the cavity)
    assert np.all(np.linalg.norm(seeds, axis=1) < 25.0)


def test_deterministic_and_seed_dependent(sphere, sphere_cands):
    again = C.build_candidates(sphere, h_mm=6.0, n_spins=2)
    assert np.array_equal(again.anchors, sphere_cands.anchors)
    assert np.array_equal(again.seed_centers, sphere_cands.seed_centers)
    assert np.array_equal(again.spins_deg, sphere_cands.spins_deg)
    assert again.n_rejected == sphere_cands.n_rejected
    other = C.build_candidates(sphere, h_mm=6.0, n_spins=2, rng_seed=7)
    assert not np.array_equal(other.anchors[:10], sphere_cands.anchors[:10])


def test_spin_sets_and_rotated_axis(sphere_cands):
    assert C.spin_set("full", None).tolist() == [0, 15, 30, 45, 60, 75]
    assert C.spin_set("half", None).tolist() == list(range(0, 180, 15))
    assert C.spin_set("full", 4).tolist() == [0.0, 22.5, 45.0, 67.5]
    assert C.spin_set("half", 3).tolist() == [0.0, 60.0, 120.0]
    with pytest.raises(ValueError):
        C.spin_set("quarter", 2)
    # the spin-45 tile's axis is the spin-0 axis rotated 45 deg about the normal
    cs = sphere_cands
    a = np.unique(cs.anchor_ids)[3]
    t0, t45 = cs.tiles_of(np.flatnonzero(cs.anchor_ids == a))
    assert np.allclose(t0.normal_ras, t45.normal_ras)
    expect = _rodrigues(t0.axis_ras, t0.normal_ras, np.radians(45.0))
    assert np.allclose(t45.axis_ras, expect, atol=1e-9)


def test_half_and_mixed_kinds(flat):
    mesh, top = flat
    half = C.build_candidates(mesh, h_mm=12.0, n_spins=2, kinds=("half",), eligible_faces=top)
    assert set(half.kinds.tolist()) == {"half"} and np.all(half.n_seeds == 2)
    assert np.isnan(half.seed_centers[:, 2:]).all() and np.isfinite(half.seed_centers[:, :2]).all()
    assert sorted(set(half.spins_deg.tolist())) == [0.0, 90.0]
    both = C.build_candidates(mesh, h_mm=12.0, n_spins=2, kinds=("full", "half"), eligible_faces=top)
    for a in np.unique(both.anchor_ids):
        ids = np.flatnonzero(both.anchor_ids == a)
        rows = list(zip(both.kinds[ids].tolist(), both.spins_deg[ids].tolist()))
        # anchor-major, kinds in the given order, spins ascending
        full_rows = [r for r in rows if r[0] == "full"]
        half_rows = [r for r in rows if r[0] == "half"]
        assert rows == full_rows + half_rows
        assert [s for _, s in full_rows] == sorted(s for _, s in full_rows)
    assert both.n_spins == 2


# ----------------------------------------------------- hanging / detached
def test_hanging_tiles_rejected_on_flat_wall(flat, flat_cands):
    mesh, top = flat
    cs = flat_cands
    assert cs.n_rejected["hanging"] > 0
    assert cs.wall_area_mm2 == pytest.approx(3600.0)
    assert np.allclose(cs.anchors[:, 2], 0.0, atol=1e-9)
    # accepted tiles have at most one fallback point in the conformer's own cast
    assert max(_true_fallbacks(mesh, t) for t in cs.tiles) <= 1
    # the permissive build keeps every hanging tile; counting the truth there
    # reproduces the strict build's rejection count exactly
    loose = C.build_candidates(mesh, h_mm=5.0, n_spins=3, eligible_faces=top,
                               min_fraction_on_wall=0.0)
    assert len(loose) == len(cs) + cs.n_rejected["hanging"]
    n_truth = sum(_true_fallbacks(mesh, t) > 1 for t in loose.tiles)
    assert n_truth == cs.n_rejected["hanging"]
    # min_fraction_on_wall = 1.0 demands every grid point on the wall
    strict = C.build_candidates(mesh, h_mm=5.0, n_spins=3, eligible_faces=top,
                                min_fraction_on_wall=1.0)
    assert len(strict) <= len(cs)
    assert max(_true_fallbacks(mesh, t) for t in strict.tiles) == 0


def test_fast_fallback_cast_matches_trimesh_reference(flat, cavity):
    mesh, top = flat
    rng = np.random.default_rng(0)
    for m, elig in ((mesh, top), (cavity, None)):
        eligible = C._eligible_mask(m, elig)
        anchors, _fid, _meth = C.sample_anchors(m, 9.0, eligible, 3)
        flats, nrms = [], []
        for a in anchors:
            surf, n_in = snap_to_wall(m, a)
            hint = _rodrigues(C._axis_hint(n_in), n_in, rng.uniform(0, np.pi / 2))
            n, t1, t2 = C._conform_frame(n_in, hint)
            uv = np.vstack(_grid_offsets("full"))
            flats.append(surf[None, :] + uv @ np.vstack([t1, t2]))
            nrms.append(np.tile(n, (uv.shape[0], 1)))
        flats = np.vstack(flats)
        nrms = np.vstack(nrms)
        fast = C.count_ray_fallbacks(m, flats, nrms)
        ref = C.count_ray_fallbacks_trimesh(m, flats, nrms)
        assert np.array_equal(fast, ref)
    # a tiny sphere: every ray misses (all fallbacks) in both
    tiny = trimesh.creation.icosphere(2, 2.0)
    pts = np.array([[5.0, 0, 0], [0, 6.0, 0]])
    nn = np.array([[0, 0, 1.0], [0, 0, 1.0]])
    assert C.count_ray_fallbacks(tiny, pts, nn).all()
    assert C.count_ray_fallbacks_trimesh(tiny, pts, nn).all()


def test_detached_rejection_counts(cavity):
    base = C.build_candidates(cavity, h_mm=8.0, n_spins=2)
    assert base.n_rejected["detached"] == 0
    # an impossibly tight tolerance rejects every candidate loudly
    with pytest.raises(ValueError, match="every candidate was rejected"):
        C.build_candidates(cavity, h_mm=8.0, n_spins=2, detached_mm=0.0)
    seeds = base.seed_centers.reshape(-1, 3)
    _p, dist, _t = trimesh.proximity.closest_point(cavity, seeds)
    assert np.all(np.abs(dist - SEED_WALL_OFFSET_MM) <= DETACHED_MM)


def test_grid_fallback_flags_diagnostic(flat, flat_cands):
    mesh, _top = flat
    # an interior tile on a flat wall: recovered wall points sit on the rays
    inner = [t for t in flat_cands.tiles if np.abs(t.anchor_ras[:2]).max() < 12.0][0]
    pts = np.vstack([inner.seed_centers, inner.corners_ras])
    wall, _d, _t = trimesh.proximity.ProximityQuery(mesh).on_surface(pts)
    assert not C.grid_fallback_flags(inner, wall).any()
    with pytest.raises(ValueError):
        C.grid_fallback_flags(inner, wall[:3])


# ---------------------------------------------------------- eligibility
def test_eligible_faces_mask_and_index_forms(flat):
    mesh, top = flat
    by_mask = C.build_candidates(mesh, h_mm=10.0, n_spins=1, eligible_faces=top)
    by_index = C.build_candidates(mesh, h_mm=10.0, n_spins=1,
                                  eligible_faces=np.flatnonzero(top))
    assert np.array_equal(by_mask.anchors, by_index.anchors)
    assert by_mask.wall_area_mm2 == pytest.approx(3600.0)
    with pytest.raises(ValueError, match="one entry per face"):
        C.build_candidates(mesh, eligible_faces=top[:-1])


def test_tile_diagonal():
    assert C.tile_diagonal_mm("full") == pytest.approx(28.284271)
    assert C.tile_diagonal_mm("half") == pytest.approx(22.360680)
    with pytest.raises(ValueError):
        C.tile_diagonal_mm("third")


# ----------------------------------------------------------- failure modes
def test_failure_modes_fail_loudly(sphere):
    with pytest.raises(ValueError, match="degenerate mesh"):
        C.build_candidates(trimesh.Trimesh())
    tri = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    with pytest.raises(ValueError, match="degenerate mesh"):
        C.build_candidates(tri)
    with pytest.raises(ValueError, match="eligible region is empty"):
        C.build_candidates(sphere, eligible_faces=np.zeros(len(sphere.faces), dtype=bool))
    with pytest.raises(ValueError, match="every candidate was rejected"):
        C.build_candidates(trimesh.creation.icosphere(2, 2.0), h_mm=1.0, n_spins=1)
    with pytest.raises(ValueError, match="kind"):
        C.build_candidates(sphere, kinds=("third",))
    with pytest.raises(ValueError):
        C.build_candidates(sphere, h_mm=0.0)
    with pytest.raises(ValueError):
        C.build_candidates(None)
    with pytest.raises(ValueError, match="degenerate mesh"):
        C.visible_faces(trimesh.Trimesh(), [0, 0, 0])
    with pytest.raises(ValueError, match="degenerate mesh"):
        C.recommend_tile_count(trimesh.Trimesh())


def test_package_entry_points_are_wired(sphere):
    cs = plan.build_candidates(sphere, h_mm=12.0, n_spins=1)
    assert isinstance(cs, CandidateSet) and len(cs) > 10
    rec = plan.recommend_tile_count(sphere)
    assert isinstance(rec, TileCountRecommendation)
    vis = plan.visible_faces(sphere, [0.0, 0.0, 0.0])
    assert vis.shape == (len(sphere.faces),) and vis.all()


# ----------------------------------------------------------- visible faces
def test_visible_faces_hollow_sphere_isolates_inner_wall(hollow):
    assert hollow.is_watertight
    vis = C.visible_faces(hollow, [0.0, 0.0, 0.0])
    r = np.linalg.norm(hollow.triangles_center, axis=1)
    inner, outer = r < 30.0, r > 30.0
    assert vis[inner].all()
    assert not vis[outer].any()
    # from outside the shell only the near outer faces are first hits
    vis_out = C.visible_faces(hollow, [100.0, 0.0, 0.0])
    assert not vis_out[inner].any()
    assert 0 < vis_out[outer].sum() < outer.sum()


def test_candidates_on_inner_wall_of_hollow_shell(hollow):
    vis = C.visible_faces(hollow, [0.0, 0.0, 0.0])
    cs = C.build_candidates(hollow, h_mm=8.0, n_spins=2, eligible_faces=vis)
    assert len(cs) > 50
    assert cs.wall_area_mm2 == pytest.approx(hollow.area_faces[vis].sum())
    assert np.all(np.linalg.norm(cs.anchors, axis=1) < 26.0)
    radii = np.linalg.norm(cs.seed_centers.reshape(-1, 3), axis=1)
    assert np.all(np.abs(radii - (25.0 - SEED_WALL_OFFSET_MM)) < 0.6)


# ----------------------------------------------------------- tile count
def test_recommend_tile_count_sphere(sphere):
    rec = C.recommend_tile_count(sphere)
    area = 4.0 * np.pi * 25.0 ** 2
    assert rec.area_mm2 == pytest.approx(sphere.area)
    assert abs(rec.area_mm2 - area) / area < 0.01
    assert rec.n_tiles == 20
    assert rec.treatable_area_mm2 == pytest.approx(rec.area_mm2)
    assert abs(rec.ellipsoid_area_mm2 - area) / area < 0.01
    assert rec.n_tiles_ellipsoid == 20
    assert np.allclose(rec.diameters_mm, 50.0, atol=0.5)
    assert np.all(np.diff(rec.diameters_mm) <= 1e-9)
    assert rec.volume_mm3 == pytest.approx(4.0 / 3.0 * np.pi * 25.0 ** 3, rel=0.02)
    assert "gammatile.com" in rec.source and "4 cm^2" in rec.source
    assert "Recommended tiles: 20" in rec.describe()
    # deductions
    rec2 = C.recommend_tile_count(sphere, contraction_pct=10.0, untreated_pct=20.0)
    assert rec2.treatable_area_mm2 == pytest.approx(rec.area_mm2 * 0.9 * 0.8)
    assert rec2.n_tiles == int(np.ceil(rec2.treatable_area_mm2 / (TILE_AREA_CM2 * 100.0)))
    assert rec2.n_tiles == 15 and rec2.n_tiles_ellipsoid == 15
    with pytest.raises(ValueError):
        C.recommend_tile_count(sphere, contraction_pct=100.0)


def test_recommend_tile_count_eligible_half(sphere):
    upper = sphere.triangles_center[:, 2] > 0.0
    rec = C.recommend_tile_count(sphere, eligible_faces=upper)
    assert rec.area_mm2 == pytest.approx(sphere.area_faces[upper].sum())
    assert abs(rec.area_mm2 - sphere.area / 2.0) / sphere.area < 0.02
    assert rec.n_tiles == 10
    # the hemisphere's principal extents: 50 x 50 x 25
    assert np.allclose(np.sort(rec.diameters_mm)[::-1], [50.0, 50.0, 25.0], atol=0.6)
    with pytest.raises(ValueError, match="eligible region is empty"):
        C.recommend_tile_count(sphere, eligible_faces=np.zeros(len(sphere.faces), bool))


def test_ellipsoid_area_formula():
    assert C.ellipsoid_area_mm2([50.0, 50.0, 50.0]) == pytest.approx(4 * np.pi * 625.0, rel=1e-9)
    # Knud Thomsen vs the exact prolate spheroid a=b=10, c=20
    a, c = 10.0, 20.0
    e = np.sqrt(1 - a * a / (c * c))
    exact = 2 * np.pi * a * a * (1 + c / (a * e) * np.arcsin(e))
    assert abs(C.ellipsoid_area_mm2([40.0, 20.0, 20.0]) - exact) / exact < 0.011


# ------------------------------------------------------------ performance
def test_build_rate_on_30mm_cavity():
    mesh = pf.sphere_cap_mesh(15.0, subdivisions=3)      # 30 mm cavity
    C.build_candidates(mesh, h_mm=12.0, n_spins=1)       # warm caches
    t0 = time.perf_counter()
    cs = C.build_candidates(mesh, h_mm=4.0, n_spins=2)
    dt = time.perf_counter() - t0
    n = len(cs) + sum(cs.n_rejected.values())
    assert n >= 100
    # target: <= 2 s per 100 candidates (section 3 A / A1 brief)
    assert 100.0 * dt / n <= 2.0, "%.2f s per 100 candidates" % (100.0 * dt / n)
