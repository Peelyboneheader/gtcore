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

The opposite failure -- one seed coming back as two candidates -- happens on
coarse (2 mm) exports: a capsule lying near the slice plane but tilted across
a slab boundary leaves its brightest part in one slice and its tip (or the
rest of its body) in the next, offset in-plane by a few voxels, and the two
thresholded traces are not 26-connected.  ``merge_fragments`` (see
:func:`_merge_fragments`) rejoins such pairs with a disclosed rule-based test
whose distance threshold is tied to the 4.5 mm capsule length.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

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
# the reference (lone-seed) volume needs at least this many blobs inside the
# per-seed volume window; with fewer, the median falls back to all blobs
SPLIT_MIN_REFERENCE_BLOBS = 4
# a blob is a ROD (one capsule) when sqrt(major / middle) weighted-PCA
# eigenvalue ratio reaches this: a bloomed capsule gives ~(L + 2r) / 2r ~ 3-4
# (lone seeds measured >= 2.3 on the 0.59 mm grid, 2.4-3.2 on the DOE 1 mm
# thin-cut); two seeds side by side give a planar blob, ~1.3-1.4 at 2.5 mm
SPLIT_ROD_LINEARITY = 2.0

# ---- fragment merge (rule-based; every threshold is disclosed and swept) ----
# Physical capsule length.  Two DISTINCT capsules lying along a common axis
# cannot have centres closer than this without overlapping, so a collinear
# candidate pair closer than ~L can only be one capsule seen twice.
SEED_LENGTH_MM = 4.5
# Merge distance: L minus a 0.5 mm margin for centroid error.  Sensitivity
# (3.0-5.0 mm) is reported in docs/localization-notes.md, stage 3.
MERGE_MAX_SEP_MM = SEED_LENGTH_MM - 0.5
# The image must stay bright between the two centres: the ridge-sampled
# minimum (see _bridge_hu) has to reach background + this fraction of the
# dimmer fragment's peak above background.  Two separate objects dip toward
# background between them; one capsule does not.
MERGE_BRIDGE_FRAC = 0.35
# Each usable fragment axis must lie within this angle of the separation.
MERGE_MAX_ANGLE_DEG = 25.0
# Above this slice spacing the 3D PCA axis is unreliable (a capsule spans
# one or two slices; interpolated gap slices smear it along the normal);
# matches the thin-cut boundary of pipeline.seed_detection_params.  Coarse
# scans compare IN-PLANE axes with the in-plane part of the separation
# instead (slab-boundary fragments of a capsule lying near the slice plane),
# or accept a pair stacked along the slice normal with less than
# MERGE_MAX_INPLANE_MM in-plane offset (a capsule seen end-on).
MERGE_COARSE_SLICE_MM = 1.2
MERGE_MAX_INPLANE_MM = 1.5
# an in-plane trace counts as directional at or above this in-plane elongation
MERGE_MIN_INPLANE_ELONG = 1.5


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


def _weighted_lloyd(pts, w, k, n_iter=100):
    """Deterministic intensity-weighted k-means (Lloyd's algorithm).

    Two starts: k equal-weight slabs along the blob's 1st principal axis,
    and along its 2nd.  Two seeds lying side by side merge into a blob whose
    LONGEST direction can be the shared seed axis, so a start seeded only
    along the 1st axis would cut both seeds in half; the 2nd-axis start
    covers that case.  The start with the lower weighted SSE wins (ties ->
    the first).  Returns integer labels ``(n,)``, or ``None`` when neither
    start yields k non-empty clusters.
    """
    pts = np.asarray(pts, dtype=float)
    w = np.asarray(w, dtype=float)
    wsum = float(w.sum())
    c0 = (pts * w[:, None]).sum(axis=0) / wsum
    d = pts - c0
    cov = (d * w[:, None]).T @ d / wsum
    _evals, evecs = np.linalg.eigh(cov)
    best_lab, best_sse = None, np.inf
    for ax in (evecs[:, -1], evecs[:, -2]):
        proj = d @ ax
        order = np.argsort(proj, kind="stable")
        cw = np.cumsum(w[order]) / wsum
        lab = np.empty(len(pts), dtype=int)
        lab[order] = np.minimum((cw * k - 1e-9).astype(int), k - 1)
        cents = None
        ok = True
        for _ in range(n_iter):
            new = []
            for c in range(k):
                m = lab == c
                if not m.any():
                    ok = False
                    break
                new.append((pts[m] * w[m, None]).sum(axis=0) / w[m].sum())
            if not ok:
                break
            new = np.asarray(new)
            if cents is not None and np.allclose(new, cents, atol=1e-9):
                break
            cents = new
            d2 = ((pts[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            lab = np.argmin(d2, axis=1)
        if not ok or cents is None:
            continue
        d2 = ((pts[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        sse = float((w * d2[np.arange(len(pts)), lab]).sum())
        if sse < best_sse - 1e-12:
            best_lab, best_sse = lab.copy(), sse
    return best_lab


def _collinear_halves(parts, pts, w, guard_mm, max_angle_deg=MERGE_MAX_ANGLE_DEG,
                      min_linearity=SPLIT_ROD_LINEARITY):
    """True when a proposed split cuts ONE capsule into pieces.

    The parent blob is a rod (weighted PCA: sqrt(major / middle eigenvalue)
    >= ``min_linearity``) and two parts lie closer than ``guard_mm`` along
    its long axis (within ``max_angle_deg``).  Two distinct capsules lying
    end to end cannot have centres closer than the capsule length, so such
    parts are halves of one seed -- the configuration the fragment merge
    would rejoin.  Seeds side by side are unaffected: their merged blob is
    planar (major ~ middle; measured <= 1.4 at 2.5 mm), not a rod.
    """
    wsum = float(w.sum())
    c0 = (pts * w[:, None]).sum(axis=0) / wsum
    d0 = pts - c0
    evals, evecs = np.linalg.eigh((d0 * w[:, None]).T @ d0 / wsum)
    lin = float(np.sqrt(max(float(evals[2]), 0.0) / max(float(evals[1]), 1e-12)))
    if lin < float(min_linearity):
        return False
    axis = evecs[:, 2]
    cos_max = float(np.cos(np.radians(max_angle_deg)))
    cents = [(p * pw[:, None]).sum(axis=0) / pw.sum() for p, pw in parts]
    for a in range(len(cents)):
        for b in range(a + 1, len(cents)):
            s = cents[b] - cents[a]
            d = float(np.linalg.norm(s))
            if d < guard_mm and (d <= 1e-9 or abs(float(s @ axis)) / d >= cos_max):
                return True
    return False


def _split_merged_blobs(blobs, voxel_mm3, min_mm3=None, max_mm3=None,
                        weighted=False, halves_guard_mm=None):
    """Split blobs whose volume is a clean multiple of the lone-seed volume.

    Needs no tile geometry: identical capsules bloom to near-identical
    thresholded volumes, so a median estimates the lone-seed volume, and
    clustering an oversized blob's voxels separates the constituent seeds
    (observed when two tiles meet edge-to-edge and their seeds sit closer
    than the intra-tile spacing).

    ``blobs`` are ``(pts, w, vol_mm3[, source_blob])`` tuples; the result is
    a list of ``(pts, w, vol_mm3, source_blob)`` in which every part of a
    split blob keeps its parent's ``source_blob`` id (default: the blob's
    index), so later stages can tell split siblings from separate blobs.

    ``min_mm3`` / ``max_mm3``: the reference median is taken only over blobs
    inside this per-seed volume window, so oversize dense bone, plates and
    merged runs do not inflate it; with fewer than
    ``SPLIT_MIN_REFERENCE_BLOBS`` blobs in the window it falls back to all
    blobs (the historical behaviour).

    ``weighted=True`` changes how a flagged blob is divided, not which blobs
    are flagged: the number of parts is the blob's integrated excess
    intensity (sum of the weights, HU above threshold) over the median of
    that quantity, instead of its volume ratio -- overlapping blooms ADD in
    intensity, whereas their thresholded volume grows super-additively (two
    in-plane seeds 2.5 mm apart bloom into ~2.7 lone-seed volumes and were
    cut in 3) -- and the cut is made by :func:`_weighted_lloyd` (intensity
    weights, two deterministic starts) instead of k-means++ on positions.

    ``halves_guard_mm``: when given, a split of a rod-shaped blob is refused
    if two of its parts lie closer than this along the blob's long axis --
    the two halves of one capsule, see :func:`_collinear_halves`.  Only
    meaningful where the 3D axis is reliable (thin slices).
    """
    from scipy.cluster.vq import kmeans2

    norm = [(b[0], b[1], float(b[2]), int(b[3]) if len(b) > 3 else i)
            for i, b in enumerate(blobs)]
    if not norm:
        return []
    vols = np.array([b[2] for b in norm], dtype=float)
    masses = np.array([float(np.sum(b[1])) for b in norm], dtype=float)
    sel = np.ones(len(norm), dtype=bool)
    if min_mm3 is not None or max_mm3 is not None:
        lo = -np.inf if min_mm3 is None else float(min_mm3)
        hi = np.inf if max_mm3 is None else float(max_mm3)
        inside = (vols >= lo) & (vols <= hi)
        if int(inside.sum()) >= SPLIT_MIN_REFERENCE_BLOBS:
            sel = inside
    med = float(np.median(vols[sel]))
    med_mass = float(np.median(masses[sel]))
    if med <= 0:
        return norm
    out = []
    for (pts, w, vol_mm3, src), mass in zip(norm, masses):
        k = int(round(vol_mm3 / med))
        # k > 4 is not a run of touching seeds; it's large foreign metal
        # (plate, frame) that the tile-fitting stage rejects by geometry.
        if vol_mm3 <= SPLIT_FACTOR * med or k < 2 or k > 4:
            out.append((pts, w, vol_mm3, src))
            continue
        if weighted and med_mass > 0:
            k = int(round(mass / med_mass))
            if k < 2 or k > 4:
                out.append((pts, w, vol_mm3, src))
                continue
        try:
            if weighted:
                lab = _weighted_lloyd(pts, w, k)
                if lab is None:
                    raise ValueError("weighted split found no partition")
            else:
                _, lab = kmeans2(pts, k, minit="++", seed=0)
        except Exception:
            out.append((pts, w, vol_mm3, src))
            continue
        parts = [(pts[lab == c], w[lab == c]) for c in range(k)]
        if any(len(p) < MIN_PCA_VOXELS for p, _ in parts):
            out.append((pts, w, vol_mm3, src))  # degenerate split: keep the blob
            continue
        if halves_guard_mm is not None and _collinear_halves(
                parts, pts, w, float(halves_guard_mm)):
            out.append((pts, w, vol_mm3, src))  # one long capsule: keep it
            continue
        for p, pw in parts:
            out.append((p, pw, len(p) * voxel_mm3, src))
    return out


def _slice_normal_and_spacing(affine):
    """Unit slice normal (RAS) and the slice spacing measured along it."""
    a = np.asarray(affine, dtype=float)[:3, :3]
    n = np.cross(a[:, 0], a[:, 1])
    nn = float(np.linalg.norm(n))
    n = n / nn if nn > 0 else np.array([0.0, 0.0, 1.0])
    return n, float(abs(a[:, 2] @ n))


def _to_kji(inv_affine, pts):
    ijk = np.atleast_2d(pts) @ inv_affine[:3, :3].T + inv_affine[:3, 3]
    return ijk[:, ::-1]


def _bridge_hu(arr, inv_affine, a, b, step_mm=0.25):
    """Ridge-tolerant minimum HU along the segment ``a -> b`` (RAS).

    At each sample the brightest of the 8 voxels around the point (the cell
    the segment passes through) is taken, and the minimum of that along the
    segment is returned.  A straight line between two fragment CENTROIDS
    cuts the corner of an oblique capsule on a 2 mm grid; the cell maximum
    follows the capsule without reaching further than one voxel.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(9, int(np.ceil(np.linalg.norm(b - a) / step_mm)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    kji = _to_kji(inv_affine, a[None, :] * (1.0 - t) + b[None, :] * t)
    base = np.floor(kji).astype(int)
    shape = np.array(arr.shape)
    best = np.full(n, -np.inf)
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                idx = base + np.array([dk, dj, di])
                ok = np.all((idx >= 0) & (idx < shape), axis=1)
                v = np.full(n, -np.inf)
                v[ok] = arr[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
                best = np.maximum(best, v)
    if not np.isfinite(best).any():
        return -np.inf
    return float(best[np.isfinite(best)].min())


_SHELL_DIRS = None


def _local_background(arr, inv_affine, center, radii=(6.0, 8.0, 10.0)):
    """Median HU (nearest voxel) on spherical shells around ``center``.

    The radii start beyond half a capsule plus its bloom, so the median sees
    the surrounding material, not the seed.
    """
    global _SHELL_DIRS
    if _SHELL_DIRS is None:
        m = 48  # Fibonacci sphere
        i = np.arange(m) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / m)
        theta = np.pi * (1.0 + 5 ** 0.5) * i
        _SHELL_DIRS = np.stack([np.cos(theta) * np.sin(phi),
                                np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    pts = np.concatenate([np.asarray(center, float)[None, :] + r * _SHELL_DIRS
                          for r in radii])
    kji = np.round(_to_kji(inv_affine, pts)).astype(int)
    ok = np.all((kji >= 0) & (kji < np.array(arr.shape)), axis=1)
    if not ok.any():
        return 0.0
    kji = kji[ok]
    return float(np.median(arr[kji[:, 0], kji[:, 1], kji[:, 2]]))


def _blob_peak(arr, inv_affine, pts):
    """Maximum HU over a blob's voxels (looked up from their RAS centres)."""
    kji = np.clip(np.round(_to_kji(inv_affine, pts)).astype(int), 0,
                  np.array(arr.shape) - 1)
    return float(arr[kji[:, 0], kji[:, 1], kji[:, 2]].max())


def _inplane_axis(pts, w, normal, quant_cov):
    """Weighted in-plane principal axis and in-plane elongation of a blob."""
    P = np.eye(3) - np.outer(normal, normal)
    q = pts @ P
    wsum = w.sum()
    c = (q * w[:, None]).sum(axis=0) / wsum
    d = q - c
    cov = (d * w[:, None]).T @ d / wsum + P @ quant_cov @ P
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    # the smallest eigenvalue belongs to the (zero-variance) slice normal
    elong = float(np.sqrt(float(evals[-1]) / max(float(evals[-2]), 1e-12)))
    return evecs[:, -1], elong


def _merge_fragments(blobs, arr, affine, max_mm3=None,
                     max_sep_mm=MERGE_MAX_SEP_MM,
                     bridge_frac=MERGE_BRIDGE_FRAC,
                     max_angle_deg=MERGE_MAX_ANGLE_DEG,
                     coarse_slice_mm=MERGE_COARSE_SLICE_MM,
                     max_inplane_mm=MERGE_MAX_INPLANE_MM):
    """Rejoin candidates that are fragments of ONE capsule (rule-based).

    A pair is merged only if ALL hold:

    1. the intensity-weighted centres are < ``max_sep_mm`` apart (default
       4.0 mm = capsule length 4.5 mm minus a 0.5 mm margin: two distinct
       capsules sharing an axis cannot be closer than L);
    2. the image stays bright between them: :func:`_bridge_hu` >= bg +
       ``bridge_frac`` * (dimmer peak - bg), bg = local shell median;
    3. the separation runs along the capsule:

       * thin slices (<= ``coarse_slice_mm``): every fragment with a usable
         3D axis (>= MIN_PCA_VOXELS voxels) has it within ``max_angle_deg``
         of the separation, and at least one fragment has one;
       * coarse slices: either the separation is mostly along the slice
         normal with an in-plane offset < ``max_inplane_mm`` (a capsule
         along the slice normal, seen end-on); or every fragment with a
         directional in-plane trace (>= MIN_PCA_VOXELS voxels, in-plane
         elongation >= MERGE_MIN_INPLANE_ELONG) has its in-plane axis
         within ``max_angle_deg`` of the in-plane separation; or, when
         neither fragment has a directional trace, the two sit one slice
         apart (0.5-1.5 slice spacings along the normal).  These are the
         slab-boundary fragments of a capsule crossing the unimaged gap
         between two 1 mm slices exported every 2 mm, measured on the
         PostOp scan (docs/localization-notes.md, stage 3).

    Parts of one split blob (same ``source_blob``) are never re-merged; a
    merge whose union would exceed ``max_mm3`` (no longer one seed) or whose
    fragments would span ``max_sep_mm`` or more is refused.  Pairs are taken
    closest first (single-link agglomeration under that span cap), and a
    merged group is kept only if it is ISOLATED: no candidate outside it
    lies within ``max_sep_mm`` of a member (crowded neighbourhoods, e.g. two
    seeds side by side broken into four slab pieces, are left unmerged
    rather than risk fusing two real seeds).

    ``blobs``: ``(pts, w, vol_mm3, source_blob)`` tuples.  Returns
    ``(merged_blobs, n_fragments, log)``: ``merged_blobs`` keeps the tuple
    shape (``source_blob`` = smallest id of the group), ``n_fragments``
    counts the blobs behind each output, ``log`` is a tuple of dicts, one
    per accepted merge.
    """
    from scipy.spatial import cKDTree

    n = len(blobs)
    if n < 2:
        return list(blobs), [1] * n, ()
    affine = np.asarray(affine, dtype=float)
    inv = np.linalg.inv(affine)
    quant_cov = affine[:3, :3] @ np.diag([1.0 / 12.0] * 3) @ affine[:3, :3].T
    normal, dz = _slice_normal_and_spacing(affine)
    coarse = dz > float(coarse_slice_mm)
    cos_max = float(np.cos(np.radians(max_angle_deg)))

    centers = np.empty((n, 3))
    axes = np.empty((n, 3))
    for i, b in enumerate(blobs):
        centers[i], axes[i], _e = _measure(b[0], b[1], quant_cov)
    tree = cKDTree(centers)
    pairs = sorted(tree.query_pairs(float(max_sep_mm)),
                   key=lambda p: (float(np.linalg.norm(centers[p[0]] - centers[p[1]])), p))
    if not pairs:
        return list(blobs), [1] * n, ()

    def tree_ball(c):
        return [x for x in tree.query_ball_point(c, float(max_sep_mm))
                if float(np.linalg.norm(centers[x] - c)) < float(max_sep_mm)]

    cache = {}

    def _inplane(i):
        if ("ip", i) not in cache:
            pts, w = blobs[i][0], blobs[i][1]
            ax = None
            if len(pts) >= MIN_PCA_VOXELS:
                a, el = _inplane_axis(pts, w, normal, quant_cov)
                if el >= MERGE_MIN_INPLANE_ELONG:
                    ax = a
            cache[("ip", i)] = ax
        return cache[("ip", i)]

    def _peak(i):
        if ("pk", i) not in cache:
            cache[("pk", i)] = _blob_peak(arr, inv, blobs[i][0])
        return cache[("pk", i)]

    def _geometry(i, j):
        s = centers[j] - centers[i]
        d = float(np.linalg.norm(s))
        if d <= 1e-9:
            return True, "coincident"
        if not coarse:
            usable = [axes[x] for x in (i, j) if len(blobs[x][0]) >= MIN_PCA_VOXELS]
            if not usable:
                return False, "no usable axis"
            return all(abs(float(a @ s)) / d >= cos_max for a in usable), "axis"
        sn = abs(float(s @ normal))
        sp = s - float(s @ normal) * normal
        p = float(np.linalg.norm(sp))
        if p < float(max_inplane_mm) and sn >= p:
            # mostly along the slice normal: a capsule seen end-on, whose
            # in-plane trace has no direction to compare
            return True, "end-on"
        if p < 1e-9:
            return False, "no in-plane separation"
        usable = [a for a in (_inplane(i), _inplane(j)) if a is not None]
        if usable:
            # a directional trace must point along the separation (two
            # parallel seeds side by side point across it)
            return (all(abs(float(a @ sp)) / p >= cos_max for a in usable),
                    "in-plane axis")
        # neither fragment has a direction (each is the short piece of the
        # capsule inside one slab): accept only the slab-boundary geometry,
        # one slice apart along the normal
        return 0.5 * dz <= sn <= 1.5 * dz, "adjacent slices"

    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    log = []
    for i, j in pairs:
        gi, gj = find(i), find(j)
        if gi == gj:
            continue
        mi, mj = members[gi], members[gj]
        if {blobs[x][3] for x in mi} & {blobs[x][3] for x in mj}:
            continue                      # split siblings stay apart
        vol_union = sum(blobs[x][2] for x in mi + mj)
        if max_mm3 is not None and vol_union > float(max_mm3):
            continue
        span = max(float(np.linalg.norm(centers[a] - centers[b]))
                   for a in mi for b in mj)
        if span >= float(max_sep_mm):
            continue
        geo_ok, rule = _geometry(i, j)
        if not geo_ok:
            continue
        bg = _local_background(arr, inv, 0.5 * (centers[i] + centers[j]))
        need = bg + float(bridge_frac) * (min(_peak(i), _peak(j)) - bg)
        bridge = _bridge_hu(arr, inv, centers[i], centers[j])
        if bridge < need:
            continue
        parent[gj] = gi
        members[gi] = mi + mj
        del members[gj]
        log.append(dict(
            source_blobs=(int(blobs[i][3]), int(blobs[j][3])),
            sep_mm=round(float(np.linalg.norm(centers[j] - centers[i])), 3),
            rule=rule, bridge_hu=round(bridge, 1), need_hu=round(need, 1),
            bg_hu=round(bg, 1),
            centers=(tuple(np.round(centers[i], 2).tolist()),
                     tuple(np.round(centers[j], 2).tolist())),
            _pair=(i, j)))

    # Isolation: a merge stands only when its neighbourhood is unambiguous --
    # no candidate outside the group lies within max_sep_mm of a member.
    # Two seeds side by side (adjacent tiles) break into four slab pieces
    # that pass the pairwise tests crosswise; refusing crowded groups keeps
    # them as separate (possibly duplicate) candidates instead of fusing two
    # real seeds into one.
    for root in [g for g in members if len(members[g]) > 1]:
        mem = members[root]
        mset = set(mem)
        crowded = any(x not in mset for m in mem
                      for x in tree_ball(centers[m]))
        if crowded:
            for m in mem:
                parent[m] = m
                members[m] = [m]
            if root not in mset:
                del members[root]
    kept = []
    for entry in log:
        i, j = entry.pop("_pair")
        if find(i) == find(j) and len(members.get(find(i), [])) > 1:
            kept.append(entry)
    log = kept

    out, nfrag = [], []
    for root in sorted(members, key=lambda g: min(members[g])):
        idx = sorted(members[root])
        if len(idx) == 1:
            out.append(blobs[idx[0]])
            nfrag.append(1)
            continue
        out.append((np.concatenate([blobs[x][0] for x in idx]),
                    np.concatenate([blobs[x][1] for x in idx]),
                    float(sum(blobs[x][2] for x in idx)),
                    int(min(blobs[x][3] for x in idx))))
        nfrag.append(len(idx))
    return out, nfrag, tuple(log)


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
    cov_ras : ndarray (N, 3, 3), optional
        Per-seed position covariance in RAS mm^2 when a refinement step
        (``gtcore.seeds.refine``) estimated one; ``None`` otherwise.
    info : dict, optional
        Free-form per-run diagnostics (e.g. refinement status per seed).
    """

    mask: np.ndarray
    centers_ras: np.ndarray
    axes_ras: np.ndarray
    volumes_mm3: np.ndarray
    elongations: np.ndarray
    cov_ras: Optional[np.ndarray] = None
    info: Optional[dict] = None

    def __len__(self):
        return int(self.centers_ras.shape[0])

    def subset(self, keep) -> "SeedCandidates":
        """Candidates selected by a bool mask or index array; every per-seed
        field (including ``cov_ras``) is carried, the union ``mask`` and
        ``info`` are shared.  Use this instead of rebuilding the dataclass
        by hand so fields added later are never silently dropped."""
        keep = np.asarray(keep)
        if keep.dtype == bool:
            idx = np.flatnonzero(keep)
        else:
            idx = keep.astype(int).reshape(-1)
        info = self.info
        if info is not None:
            info = dict(info)
            for key, val in list(info.items()):
                arr = np.asarray(val) if isinstance(val, (list, np.ndarray)) else None
                if arr is not None and arr.ndim >= 1 and arr.shape[0] == len(self):
                    info[key] = arr[idx]
        return SeedCandidates(
            mask=self.mask,
            centers_ras=self.centers_ras[idx],
            axes_ras=self.axes_ras[idx],
            volumes_mm3=self.volumes_mm3[idx],
            elongations=self.elongations[idx],
            cov_ras=None if self.cov_ras is None else self.cov_ras[idx],
            info=info,
        )


def detect_seed_candidates(vol, hu_threshold=2000.0, min_mm3=0.2, max_mm3=120.0,
                           split_merged=True, split_window=True,
                           split_guard=True, split_weighted=True,
                           merge_fragments=False,
                           merge_max_sep_mm=MERGE_MAX_SEP_MM):
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
        Split blobs whose volume is a multiple of the lone-seed (median)
        volume into that many candidates by k-means (touching seeds, e.g.
        from adjacent tiles).  Needs at least 4 blobs to estimate the median.
    split_window : bool
        Take that median only over blobs inside ``[min_mm3, max_mm3]``
        (falls back to all blobs when fewer than 4 are inside), so oversize
        bone, plates and merged runs do not inflate the reference.
    split_guard : bool
        On thin slices (<= 1.2 mm), refuse to split a rod-shaped blob into
        parts closer than ``merge_max_sep_mm`` along its long axis: those
        are the two halves of one capsule (two capsules end to end cannot be
        closer than the 4.5 mm capsule length).  Measured on the DOE
        thin-cut (1 mm post-implant series): the 1.6x volume rule cuts 3
        single implant seeds in half with the all-blob median and 7 with
        the windowed one; the guard keeps them whole.
    split_weighted : bool
        Divide a flagged blob into round(excess-intensity ratio) parts with
        deterministic intensity-weighted Lloyd's (two starts, 1st and 2nd
        principal axis) instead of round(volume ratio) parts by unweighted
        k-means++ (``False``, the historical method).  Measured: identical
        on the synthetic head phantoms (0.7-2.8 mm) and on every real scan;
        two seeds side by side 2.5 mm apart that bloom into one blob are
        split within 0.2 mm where k-means++ cut them wrongly (1.5 mm) or in
        three.
    merge_fragments : bool
        Rejoin candidate pairs that are fragments of one capsule
        (:func:`_merge_fragments`; distance cap ``merge_max_sep_mm``,
        derived from the 4.5 mm capsule length).  Opt-in here; the pipeline
        decides per scan (``gtcore.pipeline.reconstruct``).

    Returns
    -------
    SeedCandidates, ordered by descending blob volume.  ``info`` carries
    per-candidate ``source_blob`` (id of the thresholded component it came
    from; parts of one split blob share it), ``split_k`` (1 = not split) and
    ``n_fragments`` (> 1 = merged), plus ``merge_log`` (tuple of dicts) when
    ``merge_fragments`` ran.
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
        blobs.append((pts, w, vol_mm3, lab))

    if not blobs:
        return empty

    if split_merged and len(blobs) >= 4:
        _n, dz = _slice_normal_and_spacing(affine)
        guard = (float(merge_max_sep_mm)
                 if split_guard and dz <= MERGE_COARSE_SLICE_MM else None)
        blobs = _split_merged_blobs(
            blobs, voxel_mm3,
            min_mm3=float(min_mm3) if split_window else None,
            max_mm3=float(max_mm3) if split_window else None,
            weighted=bool(split_weighted), halves_guard_mm=guard)
    # parts of a k-way split share their source id: count siblings
    src_count = {}
    for b in blobs:
        src_count[b[3]] = src_count.get(b[3], 0) + 1

    nfrag = [1] * len(blobs)
    merge_log = None
    if merge_fragments and len(blobs) >= 2:
        blobs, nfrag, merge_log = _merge_fragments(
            blobs, arr, affine, max_mm3=float(max_mm3),
            max_sep_mm=float(merge_max_sep_mm))

    # per-seed volume window, applied post-split so merged runs are first
    # separated into their constituents rather than discarded whole
    keep = [i for i, b in enumerate(blobs)
            if float(min_mm3) <= b[2] <= float(max_mm3)]
    blobs = [blobs[i] for i in keep]
    nfrag = [nfrag[i] for i in keep]
    if not blobs:
        return empty

    quant_cov = affine[:3, :3] @ np.diag([1.0 / 12.0] * 3) @ affine[:3, :3].T
    centers, axes, volumes, elongs = [], [], [], []
    for pts, w, vol_mm3, _src in blobs:
        center, axis, elong = _measure(pts, w, quant_cov)
        centers.append(center)
        axes.append(axis)
        volumes.append(vol_mm3)
        elongs.append(elong)

    centers = np.asarray(centers, dtype=float)
    axes = np.asarray(axes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    elongs = np.asarray(elongs, dtype=float)
    source = np.array([b[3] for b in blobs], dtype=int)
    split_k = np.array([src_count.get(b[3], 1) if nf == 1 else 1
                        for b, nf in zip(blobs, nfrag)], dtype=int)
    nfrag = np.asarray(nfrag, dtype=int)

    order = np.argsort(-volumes)
    info = {"source_blob": source[order], "split_k": split_k[order],
            "n_fragments": nfrag[order]}
    if merge_log is not None:
        info["merge_log"] = merge_log
    return SeedCandidates(
        mask=mask,
        centers_ras=centers[order],
        axes_ras=axes[order],
        volumes_mm3=volumes[order],
        elongations=elongs[order],
        info=info,
    )
