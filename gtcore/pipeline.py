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
from dataclasses import dataclass, field, replace
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
    implant: Optional[Dict] = None         # assess_implant() verdict
    # free-form extras; "seed_posterior" (fuse_tiles=True, plan stage 5)
    # mirrors vol.meta["seed_posterior"]: raw AND posterior seed positions
    meta: Dict = field(default_factory=dict)


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
    return cands.subset(keep)


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


def assess_implant(centers_ras, axes_ras=None):
    """Decide whether the scan actually contains an implant.

    Evidence is MANUFACTURED GEOMETRY, not proximity: a genuine implant
    contains at least one 4-seed group that passes the tile fitter's quad
    gates (10 mm grid chords, quad topology, coplanarity, axis parallelism).
    Mere spatial clustering is NOT enough -- a pre-implant head CT's ~30
    dense-bone candidates chain into one big 15 mm-linkage cluster along the
    skull base, but form zero gate-passing quads (measured on the DOE
    negative-control scan), while every real implant scan yields its full
    tile count.

    Returns dict(present, n_candidates, n_tile_evidence, reason).
    """
    from .tiles import fit_tiles

    centers = np.asarray(centers_ras, dtype=float).reshape(-1, 3)
    n = len(centers)
    out = {"present": False, "n_candidates": int(n), "n_tile_evidence": 0}
    if n < 4:
        out["reason"] = "only %d candidates (a tile needs 4 seeds)" % n
        return out
    if axes_ras is None:
        axes = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    else:
        axes = np.asarray(axes_ras, dtype=float).reshape(-1, 3)
    fit = fit_tiles(centers, axes, n_full=max(1, n // 4))
    n_tiles = len(fit.tiles)
    out["n_tile_evidence"] = int(n_tiles)

    # An implant is ONE cavity's lining: its tiles group within a cavity-
    # sized region. Chance quads from physiologic calcifications (pineal,
    # choroid plexus, falx -- measured on the pre-implant negative control:
    # 6 quads with genuinely tile-like geometry) scatter across the head.
    grouped = 0
    if n_tiles:
        from scipy.spatial.distance import cdist

        tc = np.array([t.center_ras for t in fit.tiles])
        grouped = int((cdist(tc, tc) < 40.0).sum(axis=1).max())  # incl. self
    if n_tiles >= 2 and grouped >= 2:
        verdict = "confirmed"
        reason = ("%d gate-passing tile quads, %d grouped within one "
                  "cavity-sized region" % (n_tiles, grouped))
    elif n_tiles >= 1:
        verdict = "uncertain"
        reason = ("%d tile-like quad(s) among %d candidates but not grouped "
                  "as one implant -- could be calcifications; needs review"
                  % (n_tiles, n))
    else:
        verdict = "absent"
        reason = ("%d candidates but none form a manufactured 4-seed tile "
                  "geometry (likely pre-implant scan or scattered false "
                  "positives)" % n)
    out["verdict"] = verdict
    out["grouped_tiles"] = grouped
    out["present"] = verdict == "confirmed"
    out["reason"] = reason
    return out


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


# coarse-scan threshold search when the implanted seed count is known:
# each step re-detects at a lower HU floor and is kept only while it moves
# the in-implant count toward the expected one without flooding
SEED_SEARCH_THRESHOLDS_HU = (1000.0, 900.0, 800.0)
SEED_SEARCH_MAX_CANDIDATES = 60     # C(N,4) quad enumeration stays tractable
SEED_SEARCH_RADIUS_MM = 30.0        # "near the implant": around the first cluster


def _count_near(centers, ref, radius_mm):
    if ref is None or not len(centers):
        return 0
    d = np.linalg.norm(np.asarray(centers, float) - ref[None, :], axis=1)
    return int((d <= radius_mm).sum())


# fragment merge default: slabs at most this fraction of the slice spacing
# leave an unimaged gap between neighbouring slices (PostOp: 1.0 / 2.0 mm)
MERGE_SLAB_GAP_RATIO = 0.75


def merge_fragments_default(vol: Volume):
    """Whether ``reconstruct`` merges seed fragments when not told.

    Returns ``(enabled, reason)``.  On only for coarse scans (slices >
    1.2 mm) whose reconstructed slabs are THINNER than their spacing
    (DICOM SliceThickness <= ``MERGE_SLAB_GAP_RATIO`` x spacing): a capsule
    crossing the unimaged gap between two slabs leaves two non-touching
    traces, which is how both duplicate pairs on the PostOp export (1 mm
    slices every 2 mm) arose.  Measured in docs/localization-notes.md
    (stage 3): on that geometry ~15 % of random seeds fragment and the merge
    rejoins all of them; on contiguous 2 mm slabs (with or without
    interpolated gap slices) fragments are >= 4.2 mm apart, the merge
    rejoins none of them and only adds false merges between close distinct
    seeds; thin cuts have no fragments.  Unknown slice thickness -> off.
    """
    dz = float(np.max(vol.spacing))
    from .seeds.detect import MERGE_COARSE_SLICE_MM

    if dz <= MERGE_COARSE_SLICE_MM:
        return False, "thin slices (%.2f mm): capsules stay connected" % dz
    try:
        thick = float(vol.meta.get("slice_thickness"))
    except (TypeError, ValueError):
        return False, "slice thickness unknown"
    if not np.isfinite(thick) or thick <= 0:
        return False, "slice thickness unknown"
    if thick <= MERGE_SLAB_GAP_RATIO * dz:
        return True, ("%.2f mm slices every %.2f mm leave unimaged gaps"
                      % (thick, dz))
    return False, ("contiguous %.2f mm slabs at %.2f mm spacing" % (thick, dz))


def reconstruct(vol: Volume, verbose: bool = True,
                n_full_tiles: Optional[Union[int, str]] = None,
                n_half_tiles: int = 0,
                complete_degraded: bool = True,
                n_seeds_expected: Optional[int] = None,
                refine_seeds: Optional[str] = None,
                fuse_tiles: bool = False,
                merge_fragments: Optional[bool] = None) -> PipelineResult:
    """Run the full reconstruction pipeline on one CT volume.

    ``n_seeds_expected`` (the implanted seed count, when the OR team knows
    it) is a consistency check on detection: on coarse scans (> 1.2 mm
    slices) where partial volume hides faint seeds, detection is repeated
    at lower HU floors (``SEED_SEARCH_THRESHOLDS_HU``) while that raises the
    number of in-vault candidates near the implant toward the expected
    count without flooding (``SEED_SEARCH_MAX_CANDIDATES``).  Each step is
    logged in ``vol.meta["seed_search"]``.

    When ``n_full_tiles`` is given (the OR team's implant count; ``None``
    skips tile fitting entirely), the shape-filtered seed candidates are
    additionally grouped into that many full tiles plus ``n_half_tiles`` half
    tiles, and the :class:`TileFitResult` lands on ``PipelineResult.tiles``.
    ``n_full_tiles="auto"`` needs no count: the configuration is inferred
    from the seed cloud by model selection (``n_half_tiles`` non-zero then
    merely allows half tiles to be selected) and ``PipelineResult.tiles`` is
    an :class:`~gtcore.tiles.auto.AutoFitResult` with the score curve.

    ``merge_fragments`` rejoins candidate pairs that are fragments of one
    seed (``gtcore.seeds.detect._merge_fragments``); ``None`` uses
    :func:`merge_fragments_default` (on for coarse scans whose slices are
    thinner than their spacing).  The decision and every merge are logged in
    ``vol.meta["seed_merge"]``.

    ``refine_seeds`` (``None`` = off, the default; ``"centroid"``; ``"model"``
    = the experimental stage-7 line-source fit on top of it) re-measures
    every surviving candidate after the vault filter and the threshold
    search with :func:`gtcore.seeds.refine.refine_seed_candidates` -- the
    threshold-free background-subtracted centroid with a per-seed analytic
    covariance (``PipelineResult.seeds.cov_ras``).  It runs on the RAW
    volume, never the metal-inpainted one (inpainting erases the very seed
    signal being measured), and before the implant assessment, so every
    consumer downstream sees the refined centres.  Per-seed status
    ("ok" / "fallback:<reason>") is logged in ``vol.meta["seed_refine"]``.

    ``fuse_tiles=True`` (plan stage 5, opt-in) needs fitted tiles AND a
    per-seed covariance (``seeds.cov_ras``, from the seed refinement of
    stage 2): the tiles are fitted with the hierarchical weighting
    (``seed_cov``; counted mode goes through
    :func:`~gtcore.tiles.auto.fit_tiles_prior` without its cover pass so
    every tile carries the bent-tile fit) and the posterior seed positions
    of :func:`gtcore.tiles.fuse.posterior_seed_positions` are computed.
    Raw and posterior positions are kept in ``vol.meta["seed_posterior"]``
    (and ``PipelineResult.meta``).  THE SWITCH: only with the flag on does
    ``PipelineResult.seeds`` -- the feed of the dose engine, the planner and
    ``to_placed_tiles`` -- carry the posterior centres and covariances; the
    raw detections stay in ``PipelineResult.meta["seeds_unfused"]``.
    Without ``cov_ras`` nothing is fused and the reason is recorded.
    """
    if refine_seeds not in (None, "centroid", "model"):
        raise ValueError("refine_seeds must be None, 'centroid' or 'model', "
                         "got %r" % (refine_seeds,))
    timings = {}
    meta = {}
    fusion = None           # vol.meta["seed_posterior"] when fuse_tiles
    if merge_fragments is None:
        merge_fragments, merge_reason = merge_fragments_default(vol)
    else:
        merge_reason = "set by caller"
    merge_fragments = bool(merge_fragments)

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
        min_mm3=params["min_mm3"], max_mm3=params["max_mm3"],
        merge_fragments=merge_fragments))
    merge_log = list((seeds_raw.info or {}).get("merge_log", ()))
    vol.meta["seed_merge"] = dict(enabled=merge_fragments, reason=merge_reason,
                                  n_merged=len(merge_log), merges=merge_log)
    if verbose and merge_fragments:
        print("  fragment merge: %d pair(s) rejoined (%s)"
              % (len(merge_log), merge_reason))
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
        seeds = seeds.subset(inside)
        if verbose:
            print("  vault filter: %d seeds inside the cranial interior" % len(seeds))
    vol.meta["vault_filter"] = vault_info

    # count-driven threshold search (coarse scans only; see docstring)
    search_log = []
    dz = float(np.max(vol.spacing))
    if (n_seeds_expected is not None and int(n_seeds_expected) > 0
            and vault_info["applied"] and dz > 1.2):
        expected = int(n_seeds_expected)
        ref = np.median(seeds.centers_ras, axis=0) if len(seeds) else None
        have = _count_near(seeds.centers_ras, ref, SEED_SEARCH_RADIUS_MM)
        search_log.append(dict(hu=params["hu_threshold"], near=have,
                               total=int(len(seeds)), kept=True))
        interior = ndimage.binary_dilation(masks["cranial_interior"], iterations=2)
        for thr in SEED_SEARCH_THRESHOLDS_HU:
            if have >= expected or thr >= params["hu_threshold"]:
                break
            raw2 = detect_seed_candidates(vol, hu_threshold=thr,
                                          min_mm3=params["min_mm3"],
                                          max_mm3=params["max_mm3"],
                                          merge_fragments=merge_fragments)
            s2 = filter_seed_shaped(raw2, min_mm3=params["min_mm3"],
                                    max_mm3=params["max_mm3"],
                                    min_elong=params["min_elong"],
                                    max_elong=params["max_elong"])
            if len(s2):
                ijk = np.atleast_2d(vol.ras_to_index(s2.centers_ras))
                kji = np.clip(np.round(ijk[:, ::-1]).astype(int), 0,
                              np.array(interior.shape) - 1)
                inside = interior[kji[:, 0], kji[:, 1], kji[:, 2]]
                s2 = s2.subset(inside)
            near2 = _count_near(s2.centers_ras, ref, SEED_SEARCH_RADIUS_MM)
            keep = (near2 > have and near2 <= max(expected + 8, have)
                    and near2 <= SEED_SEARCH_MAX_CANDIDATES)
            search_log.append(dict(hu=thr, near=near2, total=int(len(s2)),
                                   kept=bool(keep)))
            if verbose:
                print("  seed search @ %.0f HU: %d near the implant (want %d)"
                      " -> %s" % (thr, near2, expected,
                                  "kept" if keep else "rejected"))
            if not keep:
                break
            seeds, have = s2, near2
        if verbose:
            print("  seed count check: %d near the implant vs %d implanted"
                  " (recall %.2f)" % (have, expected,
                                      min(1.0, have / float(expected))))
    vol.meta["seed_search"] = dict(expected=n_seeds_expected, steps=search_log)

    if refine_seeds is not None:
        from .seeds.refine import refine_seed_candidates

        unrefined = seeds
        seeds = stage("seed refinement", lambda: refine_seed_candidates(
            vol, unrefined, method=refine_seeds))
        status = [str(s_) for s_ in seeds.info["refine_status"]]
        n_ok = sum(1 for s_ in status if s_ == "ok")
        vol.meta["seed_refine"] = dict(
            method=refine_seeds, n=len(status), n_ok=n_ok, status=status,
            shift_mm=[round(float(x), 4) for x in seeds.info["refine_shift_mm"]],
            saturation_hu=seeds.info["saturation_hu"])
        if verbose:
            print("  seed refinement (%s): %d/%d refined, %d kept the detected"
                  " centre" % (refine_seeds, n_ok, len(status),
                               len(status) - n_ok))

    # Implant assessment uses only candidates AWAY from bone: dense inner-
    # table spots pass every filter and even form chance quads with tile-like
    # geometry (measured on the pre-implant negative control: 6 quads,
    # residuals 0.2-0.7 mm, axis coherence 0.94-0.98 -- indistinguishable
    # from real tiles), but they sit ON the skull, while implanted seeds sit
    # in tissue lining the cavity.
    # (Checking against masks["skull"] does NOT work: segment_head carves the
    # metal mask out of the bone mask, so candidate locations are holes in
    # it. Instead, probe the INPAINTED volume on a 4 mm shell around each
    # candidate: a bone spot is embedded in bone, a seed floats in tissue.)
    if len(seeds):
        rng_dirs = np.random.default_rng(0).normal(size=(32, 3))
        rng_dirs /= np.linalg.norm(rng_dirs, axis=1, keepdims=True)
        shell = seeds.centers_ras[:, None, :] + 4.0 * rng_dirs[None, :, :]
        hu = clean.sample_ras(shell.reshape(-1, 3)).reshape(len(seeds), -1)
        in_bone = np.median(hu, axis=1) > 200.0
        implant = assess_implant(seeds.centers_ras[~in_bone],
                                 seeds.axes_ras[~in_bone])
        implant["n_candidates"] = int(len(seeds))
        implant["n_in_bone"] = int(in_bone.sum())
    else:
        implant = assess_implant(seeds.centers_ras, seeds.axes_ras)
    vol.meta["implant"] = implant
    if verbose:
        print("  implant assessment: %s -- %s"
              % ("PRESENT" if implant["present"] else "NOT PRESENT",
                 implant["reason"]))

    cavity = stage("cavity segmentation", lambda: segment_cavity(
        clean, masks["cranial_interior"], masks["brain"],
        seeds.centers_ras if (len(seeds) and implant["present"]) else None,
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
                                 mesh=meshes.get("cavity"),
                                 spacing_mm=vol.spacing)
            return fit_tiles(
                seeds.centers_ras, seeds.axes_ras,
                int(n_full_tiles), int(n_half_tiles),
                cavity_center_ras=cavity_center,
                complete_degraded=complete_degraded,
            )

        fuse_cov = None
        if fuse_tiles:
            fuse_cov = getattr(seeds, "cov_ras", None)
            if fuse_cov is None:
                fusion = dict(
                    applied=False,
                    reason="seeds.cov_ras is None: no per-seed covariance "
                           "(needs the stage-2 seed refinement)")
            elif len(seeds):
                _plain_fit = _fit

                def _fit():
                    from .tiles import ImplantPrior, fit_tiles_auto, fit_tiles_prior

                    if auto:
                        return fit_tiles_auto(
                            seeds.centers_ras, seeds.axes_ras,
                            cavity_center_ras=cavity_center,
                            allow_half=bool(n_half_tiles),
                            mesh=meshes.get("cavity"),
                            spacing_mm=vol.spacing, seed_cov=fuse_cov)
                    if int(n_full_tiles) <= 0 and int(n_half_tiles) <= 0:
                        return _plain_fit()
                    return fit_tiles_prior(
                        seeds.centers_ras, seeds.axes_ras,
                        ImplantPrior(n_full=int(n_full_tiles),
                                     n_half=int(n_half_tiles)),
                        cavity_center_ras=cavity_center,
                        spacing_mm=vol.spacing, cover=False,
                        complete_degraded=complete_degraded,
                        seed_cov=fuse_cov)

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

        if fuse_cov is not None and len(seeds):
            from .tiles.fuse import posterior_seed_positions

            def _fuse():
                out = posterior_seed_positions(tiles, seeds.centers_ras,
                                               fuse_cov)
                # same mean; covariance that also carries the fitted-pose
                # uncertainty (fuse docstring) -- kept for the NEES check
                pev = posterior_seed_positions(tiles, seeds.centers_ras,
                                               fuse_cov, cov_mode="pev")[1]
                return out + (pev,)

            post, post_cov, finfo, post_cov_pev = stage("tile fusion", _fuse)
            fusion = dict(
                applied=True,
                raw_centers_ras=np.array(seeds.centers_ras, dtype=float),
                raw_cov_ras=np.array(fuse_cov, dtype=float),
                centers_ras=post, cov_ras=post_cov,
                cov_ras_pev=post_cov_pev, info=finfo)
            meta["seeds_unfused"] = seeds
            info = dict(seeds.info or {})
            info["fused"] = True
            seeds = replace(seeds, centers_ras=post, cov_ras=post_cov,
                            info=info)
            if verbose:
                print("  tile fusion: %d seeds moved toward their tile "
                      "(mean %.2f mm, max %.2f mm), %d passed through"
                      % (finfo["n_fused"],
                         float(finfo["shift_mm"][finfo["tile_of"] >= 0].mean())
                         if finfo["n_fused"] else 0.0,
                         float(finfo["shift_mm"].max()) if len(post) else 0.0,
                         finfo["n_passthrough"]))
    if fuse_tiles:
        if fusion is None:
            fusion = dict(applied=False,
                          reason="no tiles fitted (n_full_tiles is None) "
                                 "or no seeds")
        vol.meta["seed_posterior"] = fusion
        meta["seed_posterior"] = fusion

    return PipelineResult(
        volume=vol, seeds=seeds, seeds_raw=seeds_raw, masks=masks,
        cavity_mask=cavity, meshes=meshes, timings=timings, tiles=tiles,
        implant=implant, meta=meta,
    )
