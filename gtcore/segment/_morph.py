"""Small binary-morphology helpers shared by the segmentation modules.

Everything here works on boolean arrays indexed ``[k, j, i]`` (see
``gtcore.volume``) and takes physical spacing in ``(si, sj, sk)`` millimetres,
so callers can express structuring-element sizes in millimetres instead of
voxels.  Head CT is routinely anisotropic (0.4 mm in-plane, 1--3 mm slice), so
a voxel-radius ball would be a very different physical shape depending on the
scan; the helpers below fix that by resampling to an isotropic working grid.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def largest_cc(mask):
    """Return the largest 6-connected component of ``mask`` as a bool array.

    An empty (or all-False) mask returns an all-False array of the same shape,
    which keeps downstream boolean algebra total -- no special-casing at the
    call sites.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    labels, n = ndimage.label(mask)
    if n <= 1:
        return labels > 0
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background
    return labels == int(counts.argmax())


def match_shape(mask, shape):
    """Crop and/or zero-pad a bool array so it has exactly ``shape``.

    Used to undo the rounding that ``ndimage.zoom`` introduces when a mask is
    taken down to a coarse working grid and back again.
    """
    mask = np.asarray(mask, dtype=bool)
    shape = tuple(int(s) for s in shape)
    if mask.ndim != len(shape):
        raise ValueError("match_shape: rank mismatch %s vs %s" % (mask.shape, shape))

    # crop first
    slicer = tuple(slice(0, min(m, s)) for m, s in zip(mask.shape, shape))
    cropped = mask[slicer]
    if cropped.shape == shape:
        return np.ascontiguousarray(cropped)

    out = np.zeros(shape, dtype=bool)
    dst = tuple(slice(0, c) for c in cropped.shape)
    out[dst] = cropped
    return out


def close_mm(mask, spacing, radius_mm, work_mm=2.0):
    """Morphological closing with a *physical* radius, in millimetres.

    The mask is resampled (nearest neighbour) onto an isotropic ``work_mm``
    grid, closed there with a Euclidean ball of ``radius_mm``, then resampled
    back.  Doing the work coarsely matters: sealing a 14 mm craniotomy on a
    512x512x300 CT with a true voxel ball is minutes of work, and at 2 mm
    isotropic it is a fraction of a second, with an error bounded by the
    working voxel size -- far below the anatomical tolerance we need.

    The closing is implemented with distance transforms rather than
    ``binary_closing``: dilation is ``edt(~m) <= r`` and the following erosion
    is ``edt(dilated) > r - work_mm/2``, which gives an exact Euclidean ball
    instead of an iterated 3x3x3 approximation.

    That half-voxel on the erosion is a deliberate sampling correction, and it
    is the difference between sealing a craniotomy and not.  Both transforms
    measure centre-to-centre distances on the working lattice, so the sampled
    background sits up to half a voxel further out than the continuous
    background it stands for, and the erosion consequently eats half a voxel
    too much.  The bridge a closing lays over a hole in a *thin* shell is only
    a few millimetres thick to begin with, so an uncorrected 14 mm closing at
    a 2 mm working resolution loses the bridge entirely and the "sealed" vault
    still leaks.  Measured on a 30 mm shell with a conical defect, the
    correction moves the largest sealable defect from under 8 degrees of
    half-angle to about 16.

    ``spacing`` is ``(si, sj, sk)`` as returned by ``Volume.spacing``; the
    array axes are ``[k, j, i]``, hence the reversed zoom factors.

    The result is OR-ed with the input, because closing is an extensive
    operator by definition and resampling must not be allowed to erode away
    thin structures that were there to begin with.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or radius_mm <= 0:
        return mask.copy()

    spacing = np.asarray(spacing, dtype=float)
    full_shape = mask.shape
    # array is [k, j, i]; spacing is (si, sj, sk)
    factors = np.array([spacing[2], spacing[1], spacing[0]], dtype=float) / float(work_mm)

    if np.allclose(factors, 1.0):
        small = mask
    else:
        small = ndimage.zoom(mask.astype(np.uint8), factors, order=0, grid_mode=False)
        small = small.astype(bool)
    if not small.any():
        return mask.copy()

    pad = int(math.ceil(float(radius_mm) / float(work_mm))) + 2
    padded = np.pad(small, pad, mode="constant", constant_values=False)

    r = float(radius_mm)
    r_erode = max(r - float(work_mm) / 2.0, 0.0)  # sampling correction, see docstring
    dilated = ndimage.distance_transform_edt(~padded, sampling=work_mm) <= r
    closed = ndimage.distance_transform_edt(dilated, sampling=work_mm) > r_erode

    closed = closed[tuple(slice(pad, s - pad) for s in closed.shape)]

    if closed.shape != full_shape:
        inv = 1.0 / factors
        big = ndimage.zoom(closed.astype(np.uint8), inv, order=0, grid_mode=False)
        big = match_shape(big.astype(bool), full_shape)
    else:
        big = closed

    return big | mask
