"""resample_iso must preserve physical space, not just array shape."""
from __future__ import annotations

import numpy as np

from gtcore import Volume
from gtcore.preprocess import resample_iso


def _ramp(x, y, z):
    """Analytic field the resampler must reproduce exactly (it is linear)."""
    return x + 2.0 * y + 3.0 * z


def _ramp_volume():
    spacing = np.array([0.5, 0.5, 1.25])
    shape_ijk = (40, 36, 20)  # ni, nj, nk
    affine = np.eye(4)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = [-11.0, 5.5, -3.25]

    ni, nj, nk = shape_ijk
    ii, jj, kk = np.meshgrid(
        np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij"
    )
    ijk = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(float)
    ras = ijk @ affine[:3, :3].T + affine[:3, 3]
    vals = _ramp(ras[:, 0], ras[:, 1], ras[:, 2]).reshape(ni, nj, nk)

    # array is [k, j, i]
    array = np.transpose(vals, (2, 1, 0)).astype(np.float32)
    return Volume(array, affine, {})


def test_resample_iso_is_isotropic_and_tagged():
    vol = _ramp_volume()
    out = resample_iso(vol, spacing=0.8)

    np.testing.assert_allclose(out.spacing, [0.8, 0.8, 0.8], atol=1e-9)
    np.testing.assert_allclose(out.direction, vol.direction, atol=1e-9)
    assert out.meta["resampled_iso_mm"] == 0.8
    assert out.array.dtype == np.float32


def test_resample_iso_preserves_physical_extent():
    vol = _ramp_volume()
    out = resample_iso(vol, spacing=0.8)

    corner_in = vol.origin_ras - vol.direction @ (vol.spacing / 2.0)
    corner_out = out.origin_ras - out.direction @ (out.spacing / 2.0)
    np.testing.assert_allclose(corner_out, corner_in, atol=1e-9)

    extent_in = np.asarray(vol.shape_ijk) * vol.spacing
    extent_out = np.asarray(out.shape_ijk) * out.spacing
    # Shapes are rounded to whole voxels, so allow up to half a new voxel.
    assert np.all(np.abs(extent_out - extent_in) <= 0.8 / 2 + 1e-6)


def test_resample_iso_reproduces_analytic_ramp():
    vol = _ramp_volume()
    out = resample_iso(vol, spacing=0.8)

    # Sample well inside the field of view so no test point falls on the
    # extrapolated border of either grid.
    lo, hi = vol.bounds_ras()
    margin = 2.0
    rng = np.random.default_rng(7)
    pts = rng.uniform(lo + margin, hi - margin, size=(10, 3))

    got = out.sample_ras(pts, order=1)
    want = _ramp(pts[:, 0], pts[:, 1], pts[:, 2])
    np.testing.assert_allclose(got, want, atol=0.1)


def test_resample_iso_matches_source_sampling():
    """Resampled values agree with sampling the original volume directly."""
    vol = _ramp_volume()
    out = resample_iso(vol, spacing=0.8)

    lo, hi = vol.bounds_ras()
    rng = np.random.default_rng(11)
    pts = rng.uniform(lo + 2.0, hi - 2.0, size=(25, 3))

    np.testing.assert_allclose(
        out.sample_ras(pts), vol.sample_ras(pts), atol=0.1
    )
