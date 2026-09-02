"""One-call reconstruction pipeline shared by the CLI, scripts, and viewers.

CT Volume -> seed candidates -> seed-shape filtering -> metal inpainting
-> skull/brain segmentation -> cavity segmentation -> surface meshes.

On a pre-implant scan (no seeds on board) the shape filter empties the seed
list and the pipeline degrades gracefully: no inpainting, cavity segmentation
falls back to its no-prior mode (which, on an intact brain, tends to find the
ventricles -- interpret accordingly).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
from scipy import ndimage

from .preprocess import inpaint_metal
from .seeds import SeedCandidates, detect_seed_candidates
from .segment import mask_to_mesh, segment_cavity, segment_head
from .tiles import TileFitResult, fit_tiles
from .volume import Volume

# a real Cs-131 seed blooms to roughly this window on CT
SEED_MIN_MM3 = 1.0
SEED_MAX_MM3 = 15.0
SEED_MIN_ELONG = 1.5
SEED_MAX_ELONG = 10.0


@dataclass
class PipelineResult:
    volume: Volume
    seeds: SeedCandidates          # shape-filtered, plausible seeds only
    seeds_raw: SeedCandidates      # every supra-threshold blob (incl. dense bone)
    masks: Dict[str, np.ndarray]   # body / skull / cranial_interior / brain
    cavity_mask: np.ndarray
    meshes: Dict[str, object]      # trimesh.Trimesh per structure
    timings: Dict[str, float] = field(default_factory=dict)
    tiles: Optional[TileFitResult] = None  # set when n_full_tiles was given


def filter_seed_shaped(cands: SeedCandidates,
                       min_mm3=SEED_MIN_MM3, max_mm3=SEED_MAX_MM3,
                       min_elong=SEED_MIN_ELONG, max_elong=SEED_MAX_ELONG):
    """Keep only capsule-plausible candidates.

    Dense cortical/petrous bone crosses a 2000 HU threshold on sharp-kernel
    head CT, but it comes back as large and/or sheet-like blobs; a bloomed
    seed is a small (few mm^3) moderately elongated blob.
    """
    keep = (
        (cands.volumes_mm3 >= min_mm3) & (cands.volumes_mm3 <= max_mm3)
        & (cands.elongations >= min_elong) & (cands.elongations <= max_elong)
    )
    return SeedCandidates(
        mask=cands.mask,
        centers_ras=cands.centers_ras[keep],
        axes_ras=cands.axes_ras[keep],
        volumes_mm3=cands.volumes_mm3[keep],
        elongations=cands.elongations[keep],
    )


def _seed_scale_metal_mask(mask, spacing, max_mm3=SEED_MAX_MM3 + 5.0):
    """Metal components small enough to be seeds (or seed pairs).

    Only these get inpainted / excluded from bone: stripping *large*
    supra-threshold components would carve real dense bone out of the skull.
    """
    if not mask.any():
        return np.zeros_like(mask)
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    voxel_mm3 = float(np.prod(spacing))
    sizes = np.bincount(lab.ravel()) * voxel_mm3
    small = np.zeros(n + 1, dtype=bool)
    small[1:] = sizes[1:] <= max_mm3 * 2.5  # allow merged pairs pre-split
    return small[lab]


def seed_detection_params(spacing):
    """Spacing-aware seed-detection parameters.

    Partial-volume averaging scales a seed's peak HU by roughly (capsule
    diameter / slice thickness): with the fixed thin-cut defaults the phantom
    study measured recall 0.58 at 1.4 mm slices and 0.00 at >= 2.1 mm, and the
    real 2 mm post-op export showed seeds peaking at only 1500-1950 HU.
    These values restore recall 1.00 out to 2.8 mm on the phantom
    (scripts/validation_spacing.py). Elongation is likewise meaningless when
    the capsule spans fewer than ~3 slices, so its lower bound is dropped on
    coarse scans -- the vault filter and tile fitting carry the rejection load
    there.
    """
    dz = float(np.max(spacing))
    if dz <= 1.2:
        return dict(hu_threshold=2000.0, min_mm3=SEED_MIN_MM3,
                    max_mm3=SEED_MAX_MM3, min_elong=SEED_MIN_ELONG,
                    max_elong=SEED_MAX_ELONG)
    if dz <= 1.8:
        thr = 1500.0
    elif dz <= 2.4:
        thr = 1200.0
    else:
        thr = 1000.0
    # No elongation bounds at all on coarse scans: a seed spanning a single
    # slice is a pancake whose PCA minor axis is ~zero, so its "elongation" is
    # arbitrarily large -- the measure is degenerate for true seeds and false
    # positives alike (measured: recall 1.00 unbounded vs 0.40-0.55 with any
    # ceiling). Precision is recovered downstream by the cranial-vault filter
    # and tile-fit rejection.
    return dict(hu_threshold=thr, min_mm3=0.5, max_mm3=60.0,
                min_elong=0.0, max_elong=float("inf"))


def reconstruct(vol: Volume, verbose: bool = True,
                n_full_tiles: Optional[Union[int, str]] = None,
                n_half_tiles: int = 0,
                complete_degraded: bool = True) -> PipelineResult:
    """Run the full reconstruction pipeline on one CT volume.

    When ``n_full_tiles`` is given (the OR team's implant count; ``None``
    skips tile fitting entirely), the shape-filtered seed candidates are
    additionally grouped into that many full tiles plus ``n_half_tiles`` half
    tiles, and the :class:`TileFitResult` lands on ``PipelineResult.tiles``.
    ``n_full_tiles="auto"`` needs no count: the configuration is inferred
    from the seed cloud by model selection (``n_half_tiles`` non-zero then
    merely allows half tiles to be selected) and ``PipelineResult.tiles`` is
    an :class:`~gtcore.tiles.auto.AutoFitResult` with the score curve.
    """
    timings = {}

    def stage(name, fn):
        t0 = time.perf_counter()
        out = fn()
        timings[name] = time.perf_counter() - t0
        if verbose:
            print("%-24s %6.2f s" % (name, timings[name]))
        return out

    params = seed_detection_params(vol.spacing)
    seeds_raw = stage("seed detection", lambda: detect_seed_candidates(
        vol, hu_threshold=params["hu_threshold"],
        min_mm3=params["min_mm3"], max_mm3=params["max_mm3"]))
    seeds = filter_seed_shaped(
        seeds_raw, min_mm3=params["min_mm3"], max_mm3=params["max_mm3"],
        min_elong=params["min_elong"], max_elong=params["max_elong"])
    if verbose:
        print("  %d supra-threshold blobs (thr %.0f HU) -> %d plausible seeds"
              % (len(seeds_raw), params["hu_threshold"], len(seeds)))

    metal = _seed_scale_metal_mask(seeds_raw.mask, vol.spacing,
                                   max_mm3=params["max_mm3"])
    if metal.any():
        clean = stage("metal inpainting", lambda: inpaint_metal(vol, metal))
    else:
        clean = vol

    masks = stage("head segmentation",
                  lambda: segment_head(clean, metal_mask=metal if metal.any() else None))

    # anatomical filter: a GammaTile seed lies inside the cranial vault, so
    # drop candidates outside it (dental work and jaw streaks live there).
    # Fail OPEN when no credible vault exists -- a 3D-printed phantom has no
    # skull (plastic ~300 HU), so its "cranial interior" is empty and the
    # filter would silently discard every real seed. 300 mL is well under any
    # adult cranial volume (~1300-1500 mL) and well over segmentation noise.
    interior_ml = float(masks["cranial_interior"].sum()) * float(np.prod(vol.spacing)) / 1000.0
    vault_info = {"applied": False, "interior_ml": round(interior_ml, 1),
                  "n_dropped": 0}
    if len(seeds) and interior_ml < 300.0:
        import warnings as _warnings

        _warnings.warn(
            "vault filter skipped: cranial interior %.0f mL is not credible "
            "(phantom or failed skull segmentation) -- extracranial false "
            "positives may pass through to tile fitting" % interior_ml)
        if verbose:
            print("  vault filter SKIPPED: cranial interior %.0f mL is not credible"
                  " (phantom / failed skull segmentation)" % interior_ml)
    elif len(seeds):
        interior = ndimage.binary_dilation(masks["cranial_interior"], iterations=2)
        ijk = np.atleast_2d(vol.ras_to_index(seeds.centers_ras))
        kji = np.clip(np.round(ijk[:, ::-1]).astype(int), 0,
                      np.array(interior.shape) - 1)
        inside = interior[kji[:, 0], kji[:, 1], kji[:, 2]]
        vault_info["applied"] = True
        vault_info["n_dropped"] = int(len(seeds) - int(inside.sum()))
        seeds = SeedCandidates(
            mask=seeds.mask,
            centers_ras=seeds.centers_ras[inside],
            axes_ras=seeds.axes_ras[inside],
            volumes_mm3=seeds.volumes_mm3[inside],
            elongations=seeds.elongations[inside],
        )
        if verbose:
            print("  vault filter: %d seeds inside the cranial interior" % len(seeds))
    vol.meta["vault_filter"] = vault_info

    cavity = stage("cavity segmentation", lambda: segment_cavity(
        clean, masks["cranial_interior"], masks["brain"],
        seeds.centers_ras if len(seeds) else None,
    ))

    def _meshes():
        big = vol.array.size > 3e7  # coarsen marching cubes on full-res clinical CT
        step = 2 if big else 1
        out = {}
        for name, m in (("brain", masks["brain"]), ("skull", masks["skull"]),
                        ("cavity", cavity)):
            if np.asarray(m).any():
                out[name] = mask_to_mesh(m, vol.affine, step_size=step)
        if "brain" not in out and masks["body"].any():
            # phantom / non-head scan: no brain-window tissue exists, so show
            # the scanned object's surface as the anatomical context (a
            # 3D-printed shell at ~150-330 HU produces at most a few skull
            # specks, which alone render as near-nothing)
            out["body"] = mask_to_mesh(masks["body"], vol.affine, step_size=step)
        return out

    meshes = stage("surface meshes", _meshes)

    tiles = None
    if n_full_tiles is not None:
        cavity_center = None
        if np.asarray(cavity).any():
            kji = np.argwhere(cavity).mean(axis=0)          # (k, j, i)
            cavity_center = vol.index_to_ras(kji[::-1])     # wants (i, j, k)

        auto = isinstance(n_full_tiles, str)

        def _fit():
            if auto:
                return fit_tiles(seeds.centers_ras, seeds.axes_ras,
                                 n_full_tiles, int(n_half_tiles),
                                 cavity_center_ras=cavity_center,
                                 mesh=meshes.get("cavity"))
            return fit_tiles(
                seeds.centers_ras, seeds.axes_ras,
                int(n_full_tiles), int(n_half_tiles),
                cavity_center_ras=cavity_center,
                complete_degraded=complete_degraded,
            )

        tiles = stage("tile fitting", _fit)
        if verbose and auto:
            print("  auto: %d tiles, %d candidates rejected; %s"
                  % (len(tiles.tiles), len(tiles.rejected_indices),
                     tiles.summary()))
            for pose in tiles.tiles:
                if pose.surface is not None:
                    print("    tile %d: %s" % (pose.tile_id,
                                               pose.surface.verdict()))
        elif verbose:
            print("  %d/%d tiles recovered, %d candidates rejected"
                  % (len(tiles.tiles), tiles.n_expected,
                     len(tiles.rejected_indices)))

    return PipelineResult(
        volume=vol, seeds=seeds, seeds_raw=seeds_raw, masks=masks,
        cavity_mask=cavity, meshes=meshes, timings=timings, tiles=tiles,
    )
