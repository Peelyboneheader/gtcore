"""gtcore.dose.hrctv: the 5 mm tissue rind as a voxel volume.

- the rind is a band of the requested depth on the tissue side of the wall:
  every voxel centre is within depth of the mesh and outside the cavity,
  and the far surface sits ~depth away;
- its volume is the shell volume a sphere predicts (phantom cavity);
- the display mesh is the far surface only (no second copy of the wall);
- ``side=-1`` puts the rind on the other side of the wall;
- statistics are exact order statistics with voxel-volume weighting.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from gtcore.dose.hrctv import HRCTV_DEPTH_MM, build_hrctv, hrctv_stats
from gtcore.volume import Volume


@pytest.fixture(scope="module")
def sphere():
    return trimesh.creation.icosphere(subdivisions=4, radius=15.0)


def test_rind_is_a_band_of_the_requested_depth_outside_a_sphere(sphere):
    h = build_hrctv(sphere, depth_mm=5.0, spacing_mm=1.0)
    pts = h.voxel_centers_ras()
    r = np.linalg.norm(pts, axis=1)
    # on the tissue side (outside), within the depth (+ half a voxel)
    assert r.min() > 15.0 - 0.75
    assert r.max() < 20.0 + 0.75
    # dense: nothing missing in the band
    expect = 4.0 / 3.0 * np.pi * (20.0 ** 3 - 15.0 ** 3) / 1000.0
    assert abs(h.volume_cc - expect) / expect < 0.08
    assert h.describe().startswith("HR-CTV: 5 mm rind")
    # display mesh = the far surface only
    assert h.mesh is not None and len(h.mesh.faces)
    rv = np.linalg.norm(np.asarray(h.mesh.vertices), axis=1)
    assert np.percentile(rv, 5) > 18.0 and np.percentile(rv, 95) < 21.5


def test_side_flag_puts_the_rind_inside_a_shell(sphere):
    h = build_hrctv(sphere, depth_mm=5.0, spacing_mm=1.0, side=-1,
                    source="phantom shell")
    r = np.linalg.norm(h.voxel_centers_ras(), axis=1)
    assert r.max() < 15.0 + 0.75 and r.min() > 10.0 - 0.75
    assert "phantom shell" in h.describe()


def test_inside_and_keep_masks_trim_the_rind(sphere):
    # a cavity mask slightly larger than the mesh removes the straddling
    # wall voxels; a keep mask of the +x half keeps only that half
    sp = 1.0
    lo = np.full(3, -25.0)
    n = 51
    aff = np.eye(4) * sp
    aff[3, 3] = 1.0
    aff[:3, 3] = lo
    kk, jj, ii = np.indices((n, n, n))
    xyz = np.stack([ii, jj, kk], axis=-1) * sp + lo
    inside = np.linalg.norm(xyz, axis=-1) <= 15.5
    keep = xyz[..., 0] > 0.0
    h = build_hrctv(sphere, depth_mm=5.0, spacing_mm=sp,
                    inside_mask=inside, inside_affine=aff,
                    keep_mask=keep, keep_affine=aff, with_mesh=False)
    pts = h.voxel_centers_ras()
    r = np.linalg.norm(pts, axis=1)
    assert r.min() > 15.5 - 1e-9
    assert np.all(pts[:, 0] > -0.5 - 1e-9)       # wall voxels straddle
    assert h.mesh is None


def test_hrctv_stats_are_voxel_weighted_order_statistics():
    d = np.arange(1, 101, dtype=float) * 100.0    # 100..10000 cGy
    s = hrctv_stats(d, rx_cgy=6000.0, voxel_volume_mm3=8.0)
    assert s["Dmin"] == 100.0 and s["Dmax"] == 10000.0
    assert s["D50"] == pytest.approx(5000.0, abs=100.0)
    assert s["D90"] == pytest.approx(1000.0, abs=100.0)
    assert s["V100"] == pytest.approx(41 / 100.0)    # 6000..10000 inclusive
    assert s["V150"] == pytest.approx(11 / 100.0)
    assert s["V200"] == 0.0
    assert s["volume_cc"] == pytest.approx(100 * 8.0 / 1000.0)
    empty = hrctv_stats(np.zeros(0), 6000.0)
    assert np.isnan(empty["D90"]) and empty["volume_cc"] == 0.0


def test_doses_exact_and_from_grid_agree_far_from_seeds(sphere):
    from gtcore.dose import compute_dose_grid
    h = build_hrctv(sphere, depth_mm=5.0, spacing_mm=2.0, with_mesh=False)
    c = np.array([[0.0, 0.0, 0.0]])
    a = np.array([[1.0, 0.0, 0.0]])
    exact = h.doses_exact(c, a)
    grid = compute_dose_grid(c, a, [[-30.0] * 3, [30.0] * 3], spacing_mm=1.0)
    sampled = h.doses_from_grid(grid)
    assert exact.shape == sampled.shape == (h.n_voxels,)
    # 15-20 mm from a single seed, a 1 mm grid interpolates to ~1 %
    assert np.median(np.abs(sampled - exact) / exact) < 0.01
    assert h.dvh(exact).n_voxels == h.n_voxels


def test_build_rejects_bad_inputs(sphere):
    with pytest.raises(ValueError):
        build_hrctv(sphere, depth_mm=0.0)
    with pytest.raises(ValueError):
        build_hrctv(sphere, side=0)
    with pytest.raises(ValueError):
        build_hrctv(trimesh.Trimesh())
    assert HRCTV_DEPTH_MM == 5.0
