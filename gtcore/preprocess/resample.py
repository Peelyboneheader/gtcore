"""Isotropic resampling.

Clinical head CT is routinely anisotropic (e.g. 0.5 x 0.5 x 1.25 mm). Every
downstream step -- morphology, marching cubes, seed-axis PCA, dose grids --
assumes a metric that is the same in all three directions, so we normalise
once here and carry an isotropic grid from then on.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..volume import Volume

__all__ = ["resample_iso"]


def resample_iso(vol, spacing=1.0, order=1, fill=-1000.0):
    """Resample ``vol`` onto an isotropic grid of ``spacing`` mm.

    The output keeps the input's direction cosines and covers the same
    physical box. ``origin_ras`` is the *centre* of voxel (0, 0, 0), so the
    outer corner of the field of view is ``origin - R @ (spacing_old / 2)``
    and the new origin is that corner plus half a new voxel.

    Parameters
    ----------
    vol : Volume
    spacing : float or sequence of 3 floats
        Target spacing in mm (a scalar gives a truly isotropic grid).
    order : int
        Spline order for :func:`scipy.ndimage.map_coordinates` (1 = trilinear).
    fill : float
        Value for samples outside the input volume. The default is air HU.

    Returns
    -------
    Volume
        ``float32`` array, with ``meta["resampled_iso_mm"]`` recorded.
    """
    new_spacing = np.asarray(spacing, dtype=float)
    if new_spacing.ndim == 0:
        new_spacing = np.repeat(new_spacing, 3)
    if new_spacing.shape != (3,):
        raise ValueError("spacing must be a scalar or 3 values, got %r" % (spacing,))
    if np.any(new_spacing <= 0):
        raise ValueError("spacing must be positive, got %r" % (new_spacing,))

    old_spacing = vol.spacing
    direction = vol.direction  # columns = unit i/j/k axes in RAS
    old_shape_ijk = np.asarray(vol.shape_ijk, dtype=float)  # (ni, nj, nk)

    # Physical extent measured corner-to-corner, not centre-to-centre.
    extent = old_shape_ijk * old_spacing
    new_shape_ijk = np.maximum(1, np.round(extent / new_spacing).astype(int))

    corner = vol.origin_ras - direction @ (old_spacing / 2.0)
    new_origin = corner + direction @ (new_spacing / 2.0)

    new_affine = np.eye(4)
    new_affine[:3, :3] = direction @ np.diag(new_spacing)
    new_affine[:3, 3] = new_origin

    # new ijk -> RAS -> old ijk
    M = np.linalg.inv(vol.affine) @ new_affine
    A, b = M[:3, :3], M[:3, 3]

    ni, nj, nk = (int(n) for n in new_shape_ijk)
    src = vol.array.astype(np.float32, copy=False)
    out = np.empty((nk, nj, ni), dtype=np.float32)

    # Output i/j grid, reused for every slice.
    jj, ii = np.meshgrid(
        np.arange(nj, dtype=float), np.arange(ni, dtype=float), indexing="ij"
    )
    ij_flat = np.stack([ii.ravel(), jj.ravel()], axis=0)  # (2, nj*ni)
    # Contribution of the (constant-per-slice) i and j indices.
    base = A[:, 0:2] @ ij_flat + b[:, None]  # (3, nj*ni) in old ijk

    for k in range(nk):
        old_ijk = base + A[:, 2:3] * float(k)
        # map_coordinates wants array index order [k, j, i]
        coords_kji = old_ijk[::-1, :]
        vals = ndimage.map_coordinates(
            src, coords_kji, order=order, mode="constant", cval=fill
        )
        out[k] = vals.reshape(nj, ni)

    meta = dict(vol.meta)
    meta["resampled_iso_mm"] = (
        float(new_spacing[0])
        if np.allclose(new_spacing, new_spacing[0])
        else tuple(float(s) for s in new_spacing)
    )
    return Volume(out, new_affine, meta)
