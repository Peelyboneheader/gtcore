"""Cs-131 seed candidate detection in a post-implant CT.

A GammaTile seed is a titanium capsule 4.5 mm long and 0.8 mm across.  On CT
the titanium is far denser than anything biological, so a plain high-HU
threshold finds it reliably; the difficulty is entirely in *sub-voxel
localisation*, because the capsule is thinner than a typical slice and the
reconstruction blooms it into a 2--3 mm blob whose apparent size depends on
the window, not on the seed.

Two consequences shape this module:

* Centres are **intensity weighted**, not simple centroids.  The bloom is
  roughly symmetric about the true capsule, so weighting by HU above the
  threshold recovers the centre to well under a voxel, whereas an unweighted
  centroid of a thresholded blob quantises to the voxel grid.
* A **long axis** is extracted per blob by weighted PCA.  Even blurred, a
  4.5:0.8 capsule leaves a clear principal direction, and that direction is
  what later lets seeds be grouped into the rigid tile geometry they were
  manufactured in.  The eigenvector's sign is arbitrary (a seed has no head or
  tail), so consumers must compare axes with ``abs(dot(...))``.

This stage is deliberately *candidates*, not *seeds*: it over-detects (surgical
clips, staples, dental work, a stereotactic frame all pass) and leaves
rejection to the tile-fitting stage, which has the geometric context to do it.
Two seeds whose blooms touch come back from labelling as one elongated blob --
seen in practice when adjacent tiles meet edge-to-edge on the cavity wall, so
inter-tile seed gaps get smaller than the 8 mm intra-tile spacing.  Those are
recognisable without any tile knowledge: seeds are identical capsules, so
lone-seed blooms have near-identical thresholded volumes, and a blob at ~2x
the population median is two seeds.  ``split_merged`` breaks them apart with
k-means on the blob's voxels; tile-aware refinement of genuinely ambiguous
cases still belongs downstream, where seed spacing is known.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..volume import apply_affine

MIN_PCA_VOXELS = 4  # below this a covariance is meaningless
# 26-connectivity, not the scipy default of 6.  A 0.8 mm capsule lying oblique
# to the voxel grid is one voxel thick along most of its length, so its
# thresholded trace steps diagonally and a 6-connected labelling chops a single
# seed into two or three "candidates".  On the reference phantom this alone
# accounts for every spurious detection.
SEED_CONNECTIVITY = 3
# a blob this many times the median blob volume is treated as merged seeds
SPLIT_FACTOR = 1.6


def _measure(pts, w, quant_cov=None):
    """Weighted centre, PCA long axis, and elongation for one blob's voxels.

    ``quant_cov`` is the voxel-quantization covariance (affine[:3,:3] @
    diag(1/12) @ affine[:3,:3].T): each sample really represents a uniform
    distribution over one voxel, whose per-axis variance is spacing^2/12.
    Without it, a seed lying flat within a single slice has an exactly-zero
    minor eigenvalue and its "elongation" explodes to ~1e6 (observed on the
    8-tile printed phantom at 1 mm slices, and on every coarse scan), which
    then trips any elongation ceiling. With it, the same seed reports the
    physically sensible ratio of rod length to voxel size (~4).
    """
    wsum = w.sum()
    center = (pts * w[:, None]).sum(axis=0) / wsum
    if len(pts) < MIN_PCA_VOXELS:
        return center, np.array([0.0, 0.0, 1.0]), 1.0
    d = pts - center
    cov_data = (d * w[:, None]).T @ d / wsum
    cov = cov_data
    if quant_cov is not None:
        cov = cov_data + quant_cov
    evals, evecs = np.linalg.eigh(cov)
    if quant_cov is not None:
        # AXIS from the data-only covariance whenever the data really has a
        # principal direction: the regularized covariance is dominated by the
        # quantization term for marginal blobs, which tilts the axis toward
        # the thickest voxel dimension (a 2 mm-slice in-plane trace reported
        # axis [0,0,1]). Elongation still uses the regularized eigenvalues.
        evals_d, evecs_d = np.linalg.eigh(cov_data)
        if float(evals_d[-1]) > float(np.linalg.eigvalsh(quant_cov)[-1]):
            evecs = evecs.copy()
            evecs[:, -1] = evecs_d[:, -1]
    evals = np.clip(evals, 0.0, None)
    axis = evecs[:, -1]
    nrm = np.linalg.norm(axis)
    axis = axis / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])
    major = float(evals[-1])
    floor = max(major * 1e-12, 1e-12)
    elong = float(np.sqrt(major / max(float(evals[0]), floor)))
    return center, axis, elong


def _split_merged_blobs(blobs, voxel_mm3):
    """Split blobs whose volume is a clean multiple of the population median.

    Needs no tile geometry: identical capsules bloom to near-identical
    thresholded volumes, so the median over all blobs estimates the lone-seed
    volume, and k-means on an oversized blob's voxel positions separates the
    constituent seeds (observed when two tiles meet edge-to-edge and their
    seeds sit closer than the intra-tile spacing).
    """
    from scipy.cluster.vq import kmeans2

    med = float(np.median([b[2] for b in blobs]))
    if med <= 0:
        return blobs
    out = []
    for pts, w, vol_mm3 in blobs:
        k = int(round(vol_mm3 / med))
        # k > 4 is not a run of touching seeds; it's large foreign metal
        # (plate, frame) that the tile-fitting stage rejects by geometry.
        if vol_mm3 <= SPLIT_FACTOR * med or k < 2 or k > 4:
            out.append((pts, w, vol_mm3))
            continue
        try:
            _, lab = kmeans2(pts, k, minit="++", seed=0)
        except Exception:
            out.append((pts, w, vol_mm3))
            continue
        parts = [(pts[lab == c], w[lab == c]) for c in range(k)]
        if any(len(p) < MIN_PCA_VOXELS for p, _ in parts):
            out.append((pts, w, vol_mm3))  # degenerate split: keep the blob
            continue
        for p, pw in parts:
            out.append((p, pw, len(p) * voxel_mm3))
    return out


@dataclass
class SeedCandidates:
    """Detected high-density blobs and their per-blob geometry.

    Attributes
    ----------
    mask : ndarray of bool, ``[k, j, i]``
        Union of all accepted blobs.  This is what gets handed to
        ``segment_head`` as ``metal_mask`` so bloom is not read as bone.
    centers_ras : ndarray (N, 3)
        Intensity-weighted centres in RAS mm.
    axes_ras : ndarray (N, 3)
        Unit long axes in RAS.  Sign is arbitrary.
    volumes_mm3 : ndarray (N,)
        Thresholded (bloomed) volume, not the true capsule volume.
    elongations : ndarray (N,)
        ``sqrt(major / minor)`` PCA eigenvalue ratio; ~1 for a blob, large for
        a capsule.  Degenerate blobs report 1.0.
    """

    mask: np.ndarray
    centers_ras: np.ndarray
    axes_ras: np.ndarray
    volumes_mm3: np.ndarray
    elongations: np.ndarray

    def __len__(self):
        return int(self.centers_ras.shape[0])


def detect_seed_candidates(vol, hu_threshold=2000.0, min_mm3=0.2, max_mm3=120.0,
                           split_merged=True):
    """Find high-density seed candidates in a CT ``Volume``.

    Parameters
    ----------
    vol : gtcore.volume.Volume
        HU array indexed ``[k, j, i]`` with a RAS affine.
    hu_threshold : float
        Metal threshold.  2000 HU sits above every bone and above iodinated
        contrast, and below the plateau the titanium capsule reaches.
    min_mm3, max_mm3 : float
        Physical volume window on the thresholded blob.  The lower bound drops
        single-voxel noise spikes; the upper bound drops large metal such as a
        cranial plate or a stereotactic frame, which no amount of PCA would
        turn into a seed.
    split_merged : bool
        Split blobs whose volume is a multiple of the population median into
        that many candidates by k-means (touching seeds, e.g. from adjacent
        tiles).  Needs at least 4 blobs to estimate the median.

    Returns
    -------
    SeedCandidates, ordered by descending blob volume.
    """
    arr = np.asarray(vol.array, dtype=np.float32)
    affine = np.asarray(vol.affine, dtype=float)
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

    hot = arr > float(hu_threshold)
    empty = SeedCandidates(
        mask=np.zeros(arr.shape, dtype=bool),
        centers_ras=np.zeros((0, 3), dtype=float),
        axes_ras=np.zeros((0, 3), dtype=float),
        volumes_mm3=np.zeros((0,), dtype=float),
        elongations=np.zeros((0,), dtype=float),
    )
    if not hot.any():
        return empty

    labels, n = ndimage.label(
        hot, structure=ndimage.generate_binary_structure(3, SEED_CONNECTIVITY)
    )
    if n == 0:
        return empty

    # find_objects gives a bounding box per label, so all the per-blob work
    # below touches only a handful of voxels instead of the whole volume.
    boxes = ndimage.find_objects(labels)

    mask = np.zeros(arr.shape, dtype=bool)
    blobs = []  # (pts_ras, weights, volume_mm3) per accepted blob

    for idx, box in enumerate(boxes):
        if box is None:
            continue
        lab = idx + 1
        sub_lab = labels[box] == lab
        nvox = int(sub_lab.sum())
        vol_mm3 = nvox * voxel_mm3
        # admit up to a 4-seed merged run here; the per-seed window is
        # enforced AFTER splitting (gating first made the k=3/4 split
        # branches unreachable and silently discarded whole seed runs)
        if vol_mm3 < float(min_mm3) or vol_mm3 > 4.0 * float(max_mm3):
            continue

        mask[box] |= sub_lab

        kk, jj, ii = np.nonzero(sub_lab)
        kk = kk + box[0].start
        jj = jj + box[1].start
        ii = ii + box[2].start

        vals = arr[kk, jj, ii].astype(float)
        w = np.clip(vals - float(hu_threshold), 1.0, None)

        ijk = np.stack([ii, jj, kk], axis=1).astype(float)
        pts = apply_affine(affine, ijk)
        blobs.append((pts, w, vol_mm3))

    if not blobs:
        return empty

    if split_merged and len(blobs) >= 4:
        blobs = _split_merged_blobs(blobs, voxel_mm3)

    # per-seed volume window, applied post-split so merged runs are first
    # separated into their constituents rather than discarded whole
    blobs = [b for b in blobs if float(min_mm3) <= b[2] <= float(max_mm3)]
    if not blobs:
        return empty

    quant_cov = affine[:3, :3] @ np.diag([1.0 / 12.0] * 3) @ affine[:3, :3].T
    centers, axes, volumes, elongs = [], [], [], []
    for pts, w, vol_mm3 in blobs:
        center, axis, elong = _measure(pts, w, quant_cov)
        centers.append(center)
        axes.append(axis)
        volumes.append(vol_mm3)
        elongs.append(elong)

    centers = np.asarray(centers, dtype=float)
    axes = np.asarray(axes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    elongs = np.asarray(elongs, dtype=float)

    order = np.argsort(-volumes)
    return SeedCandidates(
        mask=mask,
        centers_ras=centers[order],
        axes_ras=axes[order],
        volumes_mm3=volumes[order],
        elongations=elongs[order],
    )
