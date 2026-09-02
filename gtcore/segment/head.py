"""Head / skull / cranial-interior / brain segmentation from a post-implant CT.

Pipeline order matters
----------------------
Seed detection (``gtcore.seeds.detect``) runs **first** in the overall
pipeline and its metal mask is passed in here as ``metal_mask``.  The reason is
physical: a Cs-131 seed is titanium-encapsulated and blooms on CT to a
2--3 mm blob well above 2000 HU.  Left in place that bloom is
indistinguishable from cortical bone by threshold alone, so it would be
welded into the skull mask, and -- worse -- the 14 mm closing used to seal the
craniotomy would happily bridge from a cluster of seeds sitting on the
resection-cavity wall out to the vault, carving a false wall through the
middle of the brain.  Removing a dilated metal mask from the bone threshold
first keeps the vault an anatomical structure.

Everything is threshold + morphology; no atlas, no registration.  The intent
is an intraoperative-speed approximation good enough to seat GammaTiles
against a cavity wall, not a research-grade brain extraction.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._morph import close_mm, largest_cc

# Hounsfield windows.  Wide-ish on purpose: intraoperative CT is noisy and
# often has contrast on board.
AIR_BODY_HU = -300.0     # anything above this is "not room air"
BONE_HU = 300.0          # cortical + trabecular bone
BRAIN_LO_HU = 24.0       # grey/white matter window, excludes CSF (~0-15 HU)
BRAIN_HI_HU = 90.0       # excludes bone, metal, acute blood clot cores
VAULT_CLOSE_MM = 14.0    # must exceed the craniotomy defect half-width
# Fallbacks for when it does not -- see _sealed_interior().  Escalation is
# NON-monotonic in outcome: once the ball outgrows the cranial interior's
# inscribed radius (~65-70 mm in an adult), closing fills the head solid and
# the interior vanishes entirely, so rungs past ~64 mm are not merely useless
# but destructive, and the viable window for a large (~45 mm) craniotomy is
# narrow -- hence the fine steps at the top of the ladder.
VAULT_CLOSE_ESCALATION_MM = (20.0, 28.0, 40.0, 48.0, 56.0, 64.0)
MIN_INTERIOR_FRACTION = 0.15  # of the body mask
METAL_DILATE_VOXELS = 2  # bloom halo around each detected seed


def _sealed_interior(bone, body, spacing):
    """Close the bone mask hard enough that it actually encloses something.

    A closing bridges a defect only if its ball cannot roll through, so the
    radius has to exceed roughly the defect's half-width.  14 mm covers a
    burr hole or a small craniotomy, but a decompressive craniectomy is
    50 mm and more across and no fixed radius covers both.  Worse, the
    failure is silent and total: if the vault is still open, ``fill_holes``
    finds no hole at all and the interior -- and therefore the brain, and
    therefore every downstream mask -- comes back **empty**, with nothing in
    the output to say why.

    So the radius escalates until the vault encloses a plausible fraction of
    the body, and the best attempt is kept if none does.  The common case
    still costs exactly one closing; only a genuinely large defect pays for
    more.  The alternative -- defaulting to a radius big enough for the worst
    case -- would bridge real anatomy (the temporal fossa, the skull base)
    on every ordinary scan, which is a much worse trade.
    """
    body_count = float(body.sum())
    best = None
    best_count = -1
    for radius in (VAULT_CLOSE_MM,) + VAULT_CLOSE_ESCALATION_MM:
        vault = close_mm(bone, spacing, radius_mm=radius)
        interior = largest_cc(ndimage.binary_fill_holes(vault) & ~vault & body)
        count = int(interior.sum())
        if count > best_count:
            best, best_count = interior, count
        if body_count and count >= MIN_INTERIOR_FRACTION * body_count:
            return interior
    return best


def segment_head(vol, metal_mask=None):
    """Segment the head from a CT ``Volume``.

    Parameters
    ----------
    vol : gtcore.volume.Volume
        Post-implant head CT, ``array`` in HU indexed ``[k, j, i]``.
    metal_mask : ndarray of bool, optional
        Seed/metal voxels from ``detect_seed_candidates``, same shape as
        ``vol.array``.  See the module docstring for why this is not optional
        in practice.

    Returns
    -------
    dict with keys ``body``, ``skull``, ``cranial_interior``, ``brain``, each a
    bool array shaped like ``vol.array``.

    Notes
    -----
    ``cranial_interior`` is the key intermediate: the skull is *not* a closed
    shell after a craniotomy, so filling holes in the raw bone mask leaks
    straight out through the bone flap defect and floods the scalp.  We first
    close the bone mask with a 14 mm physical ball, which bridges the defect
    without meaningfully thickening the vault, and only then fill.  The
    interior is intersected with ``body`` and reduced to its largest component
    as a second line of defence against any residual leak.
    """
    arr = np.asarray(vol.array, dtype=np.float32)
    spacing = vol.spacing

    # ---------------------------------------------------------------- body
    body = largest_cc(arr > AIR_BODY_HU)
    body = ndimage.binary_fill_holes(body)

    # ---------------------------------------------------------------- bone
    bone = (arr > BONE_HU) & body
    if metal_mask is not None:
        metal = np.asarray(metal_mask, dtype=bool)
        if metal.shape != arr.shape:
            raise ValueError(
                "metal_mask shape %s does not match volume %s"
                % (metal.shape, arr.shape)
            )
        if metal.any():
            metal = ndimage.binary_dilation(metal, iterations=METAL_DILATE_VOXELS)
            bone = bone & ~metal

    skull = largest_cc(bone)

    # ------------------------------------------------------ cranial interior
    # Seal the craniotomy, then take what the sealed vault encloses.
    interior = _sealed_interior(bone, body, spacing)

    # --------------------------------------------------------------- brain
    brain = interior & (arr > BRAIN_LO_HU) & (arr < BRAIN_HI_HU)
    brain = ndimage.binary_opening(brain, iterations=1)
    brain = largest_cc(brain)

    return {
        "body": body,
        "skull": skull,
        "cranial_interior": interior,
        "brain": brain,
    }
