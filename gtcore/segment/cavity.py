"""Resection-cavity segmentation, optionally guided by the detected seed cloud.

The seed-cloud prior
--------------------
GammaTiles are collagen tiles carrying Cs-131 seeds that the surgeon lays
against the *wall of the resection cavity* -- that is what the device is for.
So once ``gtcore.seeds.detect`` has found the seeds, their centres are not
merely an output, they are a strong spatial prior on where the cavity is: the
cavity is the low-density pocket that the seed cloud surrounds.

That prior does real work.  A post-op brain contains several dark pockets --
ventricles, sulcal CSF, pneumocephalus far from the surgical bed, a
contralateral cyst -- and intensity alone cannot tell you which one the
surgeon operated in.  Selecting the candidate components that intersect a
dilated seed cloud picks the right one directly, and degrades gracefully:
with no seeds supplied we fall back to the largest candidate component, which
is the usual heuristic and usually right but not reliably so.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._morph import close_mm, largest_cc

AIR_HU = -250.0          # pneumocephalus / trapped air
FLUID_HI_HU = 26.0       # CSF and serosanguinous fluid sit around 0-20 HU
BRAIN_ENVELOPE_MM = 10.0  # closing radius used to build the "inside brain" hull
CAVITY_CLOSE_MM = 5.0     # tidies the final mask across thin septa / clot flecks


def segment_cavity(vol, interior_mask, brain_mask, seed_centers_ras=None,
                   seed_radius_mm=14.0):
    """Segment the resection cavity inside an already-segmented brain.

    Parameters
    ----------
    vol : gtcore.volume.Volume
        The CT, HU in ``array`` indexed ``[k, j, i]``.
    interior_mask : ndarray of bool
        ``cranial_interior`` from :func:`gtcore.segment.head.segment_head`.
    brain_mask : ndarray of bool
        ``brain`` from the same call.
    seed_centers_ras : array_like, shape (N, 3), optional
        Seed centres in RAS mm from ``detect_seed_candidates``.  See module
        docstring.
    seed_radius_mm : float
        How far from the seed cloud a cavity component may be and still be
        accepted.  14 mm is roughly a tile diagonal plus the wall thickness a
        tile is pressed into.

    Returns
    -------
    ndarray of bool, shaped like ``vol.array``.
    """
    arr = np.asarray(vol.array, dtype=np.float32)
    spacing = vol.spacing
    interior = np.asarray(interior_mask, dtype=bool)
    brain = np.asarray(brain_mask, dtype=bool)

    # ------------------------------------------------ intensity candidates
    air = (arr < AIR_HU) & interior
    fluid = (arr < FLUID_HI_HU) & interior & ~brain & ~air
    cand = ndimage.binary_opening(air | fluid, iterations=1)

    # Restrict to holes *within* the brain envelope.  Closing the brain mask
    # by 10 mm produces a solid hull whose surface follows the cortex; this
    # both excludes the subarachnoid CSF rim between brain and skull (which is
    # fluid-density and touches everything) and re-admits the cavity itself,
    # which was punched out of the brain mask by the intensity threshold.
    envelope = close_mm(brain, spacing, BRAIN_ENVELOPE_MM)
    envelope = ndimage.binary_fill_holes(envelope)
    cand = cand & envelope

    if not cand.any():
        return np.zeros(arr.shape, dtype=bool)

    # --------------------------------------------------- seed-guided select
    selected = None
    if seed_centers_ras is not None:
        centers = np.atleast_2d(np.asarray(seed_centers_ras, dtype=float))
        if centers.size:
            seed_vol = np.zeros(arr.shape, dtype=bool)
            ijk = np.rint(vol.ras_to_index(centers)).astype(int)
            nk, nj, ni = arr.shape
            # ras_to_index gives (i, j, k); the array is [k, j, i] -- flip.
            kji = ijk[:, ::-1]
            ok = np.all((kji >= 0) & (kji < np.array([nk, nj, ni])), axis=1)
            kji = kji[ok]
            if len(kji):
                seed_vol[kji[:, 0], kji[:, 1], kji[:, 2]] = True
                cloud = ndimage.distance_transform_edt(
                    ~seed_vol, sampling=(spacing[2], spacing[1], spacing[0])
                ) <= float(seed_radius_mm)

                labels, n = ndimage.label(cand)
                if n:
                    hit = np.unique(labels[cloud & cand])
                    hit = hit[hit > 0]
                    if hit.size:
                        selected = np.isin(labels, hit)

    if selected is None:
        selected = largest_cc(cand)

    # --------------------------------------------------------------- tidy
    out = close_mm(selected, spacing, CAVITY_CLOSE_MM) & interior
    out = ndimage.binary_fill_holes(out)
    return out
