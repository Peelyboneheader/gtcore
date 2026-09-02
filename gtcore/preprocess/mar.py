"""Metal-artifact reduction (image domain, v1).

Cs-131 seeds and the titanium markers around them read at +3000 HU and bloom
into a halo of streaks. Left alone, that halo drags every intensity-driven
step after it: thresholds, region growing, marching cubes and the brain/skull
segmentation all get pulled toward the metal.

:func:`inpaint_metal` is a deliberately simple **image-domain** fix -- grow the
metal mask a little, fill it by nearest-neighbour extrapolation from
surrounding tissue, and feather the seam. It removes the bloom so the
segmentation is not dragged by +3000 HU voxels; it does **not** undo the
underlying projection inconsistency.

Sinogram-domain MAR (linear interpolation of the metal trace, or NMAR with a
prior image and reprojection) is future work and needs raw projections, which
a post-implant clinical export rarely carries.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..volume import Volume

__all__ = ["inpaint_metal"]


def inpaint_metal(vol, metal_mask, dilate_mm=1.5, smooth_mm=1.0):
    """Replace metal voxels with plausible surrounding tissue values.

    Parameters
    ----------
    vol : Volume
    metal_mask : array_like of bool
        Same shape as ``vol.array`` ([k, j, i]); True where metal is.
    dilate_mm : float
        Isotropic (mm-metric, spacing aware) growth of the mask, to catch the
        partial-volume rim around each seed.
    smooth_mm : float
        Gaussian sigma used to feather the filled region so the inpaint does
        not leave a hard nearest-neighbour mosaic edge. Feathering is applied
        **only inside** the grown region; real tissue is untouched.

    Returns
    -------
    Volume
        Same geometry, ``float32`` array, ``meta`` records the MAR parameters
        and the grown voxel count.
    """
    mask = np.asarray(metal_mask, dtype=bool)
    if mask.shape != vol.array.shape:
        raise ValueError(
            "metal_mask shape %s does not match volume %s"
            % (mask.shape, vol.array.shape)
        )

    data = vol.array.astype(np.float32, copy=True)
    if not mask.any():
        out = vol.copy_with(array=data)
        out.meta["mar"] = {
            "dilate_mm": float(dilate_mm),
            "smooth_mm": float(smooth_mm),
            "n_voxels": 0,
        }
        return out

    # vol.spacing is (si, sj, sk); the array is indexed [k, j, i].
    sampling = tuple(float(s) for s in vol.spacing[::-1])

    # --- restrict all EDT work to the metal bounding box (+margin) --------
    # Two full-volume EDTs (one with return_indices) peak at gigabytes on a
    # thin-cut 512^3 scan, to inpaint a few cm^3 of seeds. The margin covers
    # the dilation radius plus the feather support, so results inside the box
    # match the full-volume computation.
    nz = np.argwhere(mask)
    margin_mm = float(dilate_mm or 0) + 4.0 * float(smooth_mm or 0) + 2.0
    pad_vox = np.ceil(margin_mm / np.asarray(sampling)).astype(int)
    lo = np.maximum(nz.min(axis=0) - pad_vox, 0)
    hi = np.minimum(nz.max(axis=0) + pad_vox + 1, np.asarray(mask.shape))
    sb = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    m_sub = mask[sb]

    # --- grow the mask in millimetres, not voxels -------------------------
    if dilate_mm and dilate_mm > 0:
        dist_to_metal = ndimage.distance_transform_edt(~m_sub, sampling=sampling)
        grow_sub = dist_to_metal <= float(dilate_mm)
    else:
        grow_sub = m_sub.copy()

    if grow_sub.all():
        raise ValueError("metal mask covers the entire volume; nothing to inpaint from")

    # --- nearest out-of-mask value ---------------------------------------
    # EDT measures distance to the nearest zero of its input, so feeding it
    # ``grow_sub`` gives, for every masked voxel, the index of the closest
    # unmasked (tissue) voxel.
    _, indices = ndimage.distance_transform_edt(
        grow_sub, sampling=sampling, return_distances=True, return_indices=True
    )
    filled = data.copy()
    f_sub = filled[sb]  # view: edits land in ``filled``
    f_sub[grow_sub] = f_sub[indices[0][grow_sub], indices[1][grow_sub],
                            indices[2][grow_sub]]

    # --- feather the seam, inside the grown region only -------------------
    if smooth_mm and smooth_mm > 0:
        sigma = [float(smooth_mm) / s for s in sampling]
        if any(s > 0 for s in sigma):
            smoothed = ndimage.gaussian_filter(f_sub, sigma=sigma, mode="nearest")
            f_sub[grow_sub] = smoothed[grow_sub]
    grow = grow_sub  # for the meta voxel count below

    out = vol.copy_with(array=filled.astype(np.float32, copy=False))
    out.meta["mar"] = {
        "dilate_mm": float(dilate_mm),
        "smooth_mm": float(smooth_mm),
        "n_voxels": int(grow.sum()),
        "method": "image-domain nearest-neighbour inpaint v1",
    }
    return out
