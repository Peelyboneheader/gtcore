"""Clinical dose metrics: DVH order statistics, cavity rind, wall coverage."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from gtcore.dose import (TG43Engine, compute_dose_grid, dose_at_points,
                         dose_metrics, dvh, resample_mask_to, rind_mask,
                         surface_coverage, wall_dose)
from gtcore.phantom import make_head_phantom
from gtcore.segment import mask_to_mesh
from gtcore.volume import Volume


@pytest.fixture(scope="module")
def eng():
    return TG43Engine()


@pytest.fixture(scope="module")
def phantom():
    vol, truth = make_head_phantom(spacing=2.0, n_tiles=3, noise_hu=0.0,
                                   rng_seed=0)
    return vol, truth


@pytest.fixture(scope="module")
def phantom_dose(phantom, eng):
    vol, truth = phantom
    centers = np.array([s.center_ras for s in truth.seeds])
    axes = np.array([s.axis_ras for s in truth.seeds])
    cav = truth.masks["cavity"]
    kji = np.argwhere(cav)
    lo = vol.index_to_ras(kji.min(axis=0)[::-1]) - 15.0
    hi = vol.index_to_ras(kji.max(axis=0)[::-1]) + 15.0
    return compute_dose_grid(centers, axes, np.vstack([lo, hi]), 2.0,
                             engine=eng)


# -------------------------------------------------------------------- DVH
def _ramp_volume():
    arr = np.arange(1000, dtype=float).reshape(10, 10, 10) + 1.0  # 1..1000
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    return Volume(arr, aff)


def test_dvh_order_statistics_on_a_ramp():
    h = dvh(_ramp_volume())
    assert h.n_voxels == 1000
    assert h.volume_cc == pytest.approx(1000 * 8.0 / 1000.0)
    assert h.D(100.0) == 1.0                     # minimum
    assert h.D(0.0) == 1000.0                    # maximum
    assert h.D(90.0) == pytest.approx(101.0, abs=1.0)
    assert h.D(50.0) == pytest.approx(501.0, abs=1.0)
    assert h.V(500.5) == pytest.approx(0.5)      # 501..1000 -> 500 voxels
    assert h.V(1.0) == 1.0 and h.V(1000.5) == 0.0
    assert h.V(500.5, relative=False) == pytest.approx(500 * 8.0 / 1000.0)
    axis, frac = h.curve(n_bins=11)
    assert axis[0] == 0.0 and axis[-1] == 1000.0
    assert np.all(np.diff(frac) <= 0.0) and frac[0] == 1.0
    with pytest.raises(ValueError):
        h.D(120.0)


def test_dvh_with_mask_and_metrics():
    v = _ramp_volume()
    mask = v.array > 900.0                       # 901..1000, 100 voxels
    h = dvh(v, mask)
    assert h.n_voxels == 100
    m = dose_metrics(v, mask, rx_cgy=950.0)
    assert m["volume_cc"] == pytest.approx(0.8)
    assert m["D100"] == 901.0 and m["Dmax"] == 1000.0
    assert m["Dmean"] == pytest.approx(950.5)
    assert m["V100"] == pytest.approx(0.51)      # 950..1000
    assert m["V150"] == 0.0 and m["V200"] == 0.0
    assert m["D90"] == pytest.approx(911.0, abs=1.0)
    empty = dose_metrics(v, np.zeros_like(mask), rx_cgy=950.0)
    assert empty["volume_cc"] == 0.0 and np.isnan(empty["D90"])
    with pytest.raises(ValueError):
        dvh(v, np.zeros((2, 2, 2), bool))


# ------------------------------------------------------------- structures
def test_resample_mask_round_trips_on_same_grid():
    v = _ramp_volume()
    mask = v.array > 300.0
    back = resample_mask_to(v, mask, v.affine)
    assert np.array_equal(back, mask)


def test_rind_is_a_shell_of_the_requested_depth():
    # A ball of radius 10 mm on a 1 mm grid; a 5 mm rind is the shell
    # 10 < r <= 15 mm (up to voxelization).
    n = 41
    aff = np.eye(4)
    aff[:3, 3] = -20.0
    k, j, i = np.mgrid[:n, :n, :n]
    r = np.sqrt((i - 20.0) ** 2 + (j - 20.0) ** 2 + (k - 20.0) ** 2)
    ball = r <= 10.0
    dose = Volume(np.ones((n, n, n)), aff)
    rind = rind_mask(dose, ball, aff, depth_mm=5.0)
    assert rind.any()
    assert not (rind & ball).any()
    assert np.all(r[rind] > 10.0) and np.all(r[rind] <= 15.0 + 1.0)
    assert np.all(rind[(r > 10.5) & (r < 14.5)])
    # Exclusion mask is honoured.
    half = np.zeros_like(ball)
    half[:, :, :20] = True
    rind2 = rind_mask(dose, ball, aff, depth_mm=5.0, exclude_mask=half)
    assert not (rind2 & half).any() and rind2.sum() < rind.sum()
    assert not rind_mask(dose, np.zeros_like(ball), aff, 5.0).any()
    with pytest.raises(ValueError):
        rind_mask(dose, ball, aff, depth_mm=0.0)


def test_rind_handles_different_grids_and_anisotropic_spacing():
    # Structure on a 1 mm grid, dose on a 2 x 2 x 3 mm grid offset by 0.5 mm.
    n = 41
    aff_s = np.eye(4)
    aff_s[:3, 3] = -20.0
    k, j, i = np.mgrid[:n, :n, :n]
    ball = np.sqrt((i - 20.0) ** 2 + (j - 20.0) ** 2 + (k - 20.0) ** 2) <= 8.0
    aff_d = np.diag([2.0, 2.0, 3.0, 1.0])
    aff_d[:3, 3] = [-19.5, -19.5, -19.5]
    dose = Volume(np.ones((14, 21, 21)), aff_d)
    rind = rind_mask(dose, ball, aff_s, depth_mm=4.0)
    assert rind.any()
    kk, jj, ii = np.nonzero(rind)
    pts = dose.index_to_ras(np.stack([ii, jj, kk], axis=1).astype(float))
    rr = np.linalg.norm(pts, axis=1)
    assert np.all(rr > 8.0 - 1.0) and np.all(rr <= 12.0 + 3.0)


def test_phantom_rind_dvh_is_prescription_scale(phantom, phantom_dose):
    vol, truth = phantom
    rind = rind_mask(phantom_dose, truth.masks["cavity"], vol.affine, 5.0)
    assert rind.sum() > 100
    m = dose_metrics(phantom_dose, rind, rx_cgy=6000.0)
    print("\n[report] phantom 5 mm rind (3 tiles): %.1f cc, D90 %.0f cGy, "
          "V100 %.2f, V150 %.2f, Dmax %.0f cGy"
          % (m["volume_cc"], m["D90"], m["V100"], m["V150"], m["Dmax"]))
    # Three tiles cover only part of the wall: some rind is at rx, most
    # is not, and the hottest voxels sit right against the seeds.
    assert 0.0 < m["V100"] < 1.0
    assert m["Dmax"] > 6000.0 > m["D90"]
    assert m["D100"] > 0.0


# -------------------------------------------------------------- wall dose
def _square_mesh(z=0.0, size=10.0):
    """Two triangles in the z-plane, normals +z, area size^2."""
    v = np.array([[0, 0, z], [size, 0, z], [size, size, z], [0, size, z]],
                 dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(v, f, process=False)


def test_wall_dose_offsets_along_normals(eng):
    mesh = _square_mesh()
    assert np.allclose(mesh.vertex_normals, [0.0, 0.0, 1.0])
    seed_c = np.array([[5.0, 5.0, 0.0]])
    seed_a = np.array([[1.0, 0.0, 0.0]])
    d = wall_dose(mesh, 5.0, seed_c, seed_a, engine=eng)
    pts = mesh.vertices + [0.0, 0.0, 5.0]
    expect = dose_at_points(seed_c, seed_a, pts, engine=eng)
    assert np.allclose(d, expect)
    # Via a dose grid instead of the seeds.
    bounds = np.array([[-20.0, -20.0, -20.0], [30.0, 30.0, 30.0]])
    vol = compute_dose_grid(seed_c, seed_a, bounds, 1.0, engine=eng)
    d2 = wall_dose(mesh, 5.0, dose_volume=vol)
    assert np.allclose(d2, expect, rtol=0.05)
    with pytest.raises(ValueError):
        wall_dose(mesh, 5.0)
    assert wall_dose(trimesh.Trimesh(), 5.0, seed_c, seed_a).size == 0


def test_surface_coverage_is_area_weighted():
    mesh = _square_mesh()
    # Face 0 = (0,1,2), face 1 = (0,2,3); equal areas.
    assert surface_coverage(mesh, [1.0, 1.0, 1.0, 1.0], 1.0) == 1.0
    assert surface_coverage(mesh, [0.0, 0.0, 0.0, 0.0], 1.0) == 0.0
    # Vertex 1 hot only: face 0 mean = 3/3 = 1 >= 1, face 1 mean = 0.
    assert surface_coverage(mesh, [0.0, 3.0, 0.0, 0.0], 1.0) == 0.5
    # Unequal areas: stretch face 1 by moving vertex 3.
    big = trimesh.Trimesh(np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0],
                                    [0, 30, 0]], float),
                          mesh.faces.copy(), process=False)
    a0, a1 = big.area_faces
    assert surface_coverage(big, [0.0, 3.0, 0.0, 0.0], 1.0) \
        == pytest.approx(a0 / (a0 + a1))
    assert np.isnan(surface_coverage(trimesh.Trimesh(), [], 1.0))
    with pytest.raises(ValueError):
        surface_coverage(mesh, [1.0, 2.0], 1.0)


def test_phantom_wall_coverage_at_5mm(phantom, eng):
    """Wall-at-depth coverage on the phantom: partial at the prescription,
    total at zero, none at an absurd level; tile centres are covered."""
    vol, truth = phantom
    # The physical wall is the lumen boundary WITH the tiles inside it: the
    # truth lumen mask has the seed capsules carved out, and at the 3 mm
    # seed-plane inset those notches face the seeds, so their vertex normals
    # would aim the 5 mm depth point straight at a capsule.
    cav_mesh = mask_to_mesh(truth.masks["cavity"] | truth.masks["seeds"],
                            vol.affine)
    centers = np.array([s.center_ras for s in truth.seeds])
    axes = np.array([s.axis_ras for s in truth.seeds])
    d = wall_dose(cav_mesh, 5.0, centers, axes, engine=eng)
    assert d.shape == (len(cav_mesh.vertices),) and np.all(d > 0.0)
    cov = surface_coverage(cav_mesh, d, 6000.0)
    print("\n[report] phantom wall coverage at 5 mm depth, 60 Gy: %.1f%%"
          % (100 * cov))
    assert 0.0 < cov < 1.0
    assert surface_coverage(cav_mesh, d, 0.0) == 1.0
    assert surface_coverage(cav_mesh, d, 1e12) == 0.0
    # The wall vertices nearest each tile centre see prescription-scale
    # dose (same 20-180 Gy band as the engine's physical-sanity test).
    for tile in truth.tiles:
        near = np.argmin(np.linalg.norm(cav_mesh.vertices - tile.center_ras,
                                        axis=1))
        assert 2000.0 < d[near] < 18000.0, d[near]
