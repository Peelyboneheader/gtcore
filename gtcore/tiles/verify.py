"""Image check of inferred seeds (plan-localization stage 8).

The cover pass of :mod:`gtcore.tiles.auto` completes an L-shaped triplet to
a full tile and *infers* the 4th seed from the manufactured geometry
(``TilePose.inferred_seed_ras``).  That position is a model prediction: it
says where a seed should be, not that one is there.  This module goes back
to the RAW volume and asks the image.

Rule (rule-based, disclosed in the paper as such):

* ROI = voxels within ``roi_mm`` (2.5 mm) of the inferred position; shell =
  the next ``max(shell_mm, max voxel size)`` beyond it, so a coarse slab
  export always contributes shell voxels.  Voxels nearer to a *detected*
  candidate than to the inferred position are excluded from both (stage 2's
  Voronoi rule), so a tile-mate's bloom cannot vouch for a missing seed.
* background ``b`` = median of the shell, noise ``sigma`` = 1.4826 x MAD of
  the shell (both robust to a stray bright voxel).
* reference contrast ``C_ref`` = median over the detected candidates of
  their own (peak - shell background) in the same windows: what a seed
  looks like on THIS scan (partial volume included).
* **evidence** iff ``peak_ROI > b + max(k_sigma * sigma, 0.3 * C_ref)``.
  The ``k_sigma`` term is the conventional detection threshold; the
  ``0.3 * C_ref`` term demands a peak of at least ~a third of a typical
  seed's, so a noise spike or faint streak on a clean scan cannot pass.

With evidence, the position is refined with the stage-2 grey-level
centroid (:func:`gtcore.seeds.refine.grey_centroid`; orientation-free first
pass, the tile-mates' mean axis as the start) and the tile is recorded as
``"recovered"`` with the refined position and its covariance; otherwise
``"no image evidence"``.  The outcome lives in ``result.verification``
(tile_id -> dict); the tile stays *tentative* either way -- the check adds
evidence for the user, it never promotes a tile.

A seed on the superior cavity wall sits next to the air pocket (measured on
the 2.1 mm head phantom: 93 % of the shell at -1000 HU), where the stage-2
estimator's flat-background assumption fails (bimodal shell, negative
signed-weight sum, ``fallback:no_signal``).  The refinement then falls back
to the conventional **above-threshold intensity-weighted centroid** over
the ROI, weights ``v - threshold`` clipped at zero with the evidence
threshold itself as the floor (no extra constant), re-centred for up to 3
passes; covariance = the stage-2 noise term ``sigma^2 S / W^2`` plus the
voxel-quantization term ``s_a^2 / 12``.  ``refine_status`` says which
estimator produced the position (``"ok"`` = grey centroid,
``"roi_centroid:<stage-2 reason>"`` = the fallback).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["verify_inferred_seeds", "ROI_MM", "SHELL_MM", "K_SIGMA",
           "CONTRAST_FRACTION", "STATUS_RECOVERED", "STATUS_NO_EVIDENCE"]

ROI_MM = 2.5                    # radius of the image window at the inferred seed
SHELL_MM = 1.5                  # background shell thickness (>= one voxel)
K_SIGMA = 5.0                   # noise term of the evidence threshold
CONTRAST_FRACTION = 0.3         # of the median detected-seed contrast
MAD_TO_SIGMA = 1.4826
STATUS_RECOVERED = "recovered"
STATUS_NO_EVIDENCE = "no image evidence"


class _Windows:
    """Spherical ROI / shell voxel windows on one volume."""

    def __init__(self, vol, roi_mm, shell_mm):
        aff = np.asarray(vol.affine, dtype=float)
        self.arr = np.asarray(vol.array)
        self.M, self.t = aff[:3, :3], aff[:3, 3]
        self.Minv = np.linalg.inv(self.M)
        self.spacing = np.linalg.norm(self.M, axis=0)
        self.shape_ijk = np.array(self.arr.shape[::-1])
        self.r_roi = float(roi_mm)
        self.r_out = self.r_roi + max(float(shell_mm), float(self.spacing.max()))

    def window(self, center_ras, others=None):
        """``(pts_ras, vals, in_roi)`` of the ROI + shell at ``center_ras``,
        Voronoi-cut against ``others`` (M, 3) RAS."""
        c = np.asarray(center_ras, dtype=float).reshape(3)
        p = self.Minv @ (c - self.t)
        half = np.ceil(self.r_out / self.spacing).astype(int) + 1
        pc = np.round(p).astype(int)
        lo = np.clip(pc - half, 0, self.shape_ijk - 1)
        hi = np.clip(pc + half, 0, self.shape_ijk - 1)
        sub = self.arr[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        K, J, I = np.meshgrid(np.arange(lo[2], hi[2] + 1),
                              np.arange(lo[1], hi[1] + 1),
                              np.arange(lo[0], hi[0] + 1), indexing="ij")
        idx = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1).astype(float)
        e = (idx - p) @ self.M.T                       # RAS offsets
        d = np.linalg.norm(e, axis=1)
        keep = d <= self.r_out
        if others is not None and len(others):
            g = np.asarray(others, dtype=float).reshape(-1, 3) - c[None, :]
            for m in np.flatnonzero(np.linalg.norm(g, axis=1) < 2.0 * self.r_out):
                keep &= (e @ g[m]) <= 0.5 * float(g[m] @ g[m])
        sel = np.flatnonzero(keep)
        return (e[sel] + c[None, :], sub.reshape(-1)[sel].astype(float),
                d[sel] <= self.r_roi)


def _contrast(win, center, others):
    """``(peak, background, sigma_noise, n_roi, n_shell)`` of one window;
    NaNs when the ROI or the shell is empty."""
    _pts, vals, inr = win.window(center, others)
    roi, shell = vals[inr], vals[~inr]
    if roi.size == 0 or shell.size < 4:
        return float("nan"), float("nan"), float("nan"), int(roi.size), int(shell.size)
    b = float(np.median(shell))
    sig = MAD_TO_SIGMA * float(np.median(np.abs(shell - b)))
    return float(roi.max()), b, sig, int(roi.size), int(shell.size)


def _reference_contrast(win, centers):
    """Median (peak - background) over the detected candidates."""
    vals = []
    for i in range(len(centers)):
        others = np.delete(centers, i, axis=0)
        peak, b, _s, n_roi, _n_sh = _contrast(win, centers[i], others)
        if n_roi and np.isfinite(peak) and np.isfinite(b):
            vals.append(peak - b)
    return float(np.median(vals)) if vals else float("nan")


def verify_inferred_seeds(result, vol, cands, k_sigma=K_SIGMA, roi_mm=ROI_MM,
                          shell_mm=SHELL_MM, contrast_fraction=CONTRAST_FRACTION,
                          update_poses=False) -> Dict[int, dict]:
    """Look for image evidence at every inferred seed of ``result``.

    Parameters
    ----------
    result : TileFitResult / AutoFitResult
        Fitted tiles; only poses with ``inferred_seed_ras`` are examined
        (``all_tiles`` when present, else ``tiles``).
    vol : Volume
        The RAW HU volume the seeds were detected on (not the
        metal-inpainted one: inpainting removes exactly the signal sought).
    cands : SeedCandidates (or (N, 3) array)
        The detected candidates: excluded from the windows and the source
        of the reference contrast.
    k_sigma, roi_mm, shell_mm, contrast_fraction : float
        The rule's constants (module docstring).
    update_poses : bool
        Also move ``pose.inferred_seed_ras`` to the refined position of a
        recovered seed.  Default False: the geometric inference stays as
        reported and the refined position is only recorded.

    Returns
    -------
    dict
        ``tile_id -> {status, inferred_ras, peak_hu, background_hu,
        sigma_noise_hu, threshold_hu, ref_contrast_hu, n_roi, n_shell,
        refined_ras, cov_ras, refine_status, shift_mm}``; also stored as
        ``result.verification`` (empty when nothing was inferred).  The
        tile's ``confidence`` is never changed.
    """
    from ..seeds.refine import estimate_saturation, grey_centroid

    poses = result.all_tiles if hasattr(result, "all_tiles") else result.tiles
    todo = [p for p in poses if p.inferred_seed_ras is not None]
    out: Dict[int, dict] = {}
    result.verification = out
    if not todo:
        return out
    if hasattr(cands, "centers_ras"):
        centers = np.asarray(cands.centers_ras, dtype=float).reshape(-1, 3)
        axes = getattr(cands, "axes_ras", None)
        axes = None if axes is None else np.asarray(axes, dtype=float).reshape(-1, 3)
    else:
        centers = np.asarray(cands, dtype=float).reshape(-1, 3)
        axes = None
    win = _Windows(vol, roi_mm, shell_mm)
    c_ref = _reference_contrast(win, centers)
    saturation = estimate_saturation(vol.array)
    for pose in todo:
        x0 = np.asarray(pose.inferred_seed_ras, dtype=float).reshape(3)
        peak, b, sig, n_roi, n_shell = _contrast(win, x0, centers)
        rec = dict(status=STATUS_NO_EVIDENCE, inferred_ras=x0.copy(),
                   peak_hu=peak, background_hu=b, sigma_noise_hu=sig,
                   ref_contrast_hu=c_ref, n_roi=n_roi, n_shell=n_shell,
                   refined_ras=None, cov_ras=None, refine_status=None,
                   shift_mm=None)
        if n_roi == 0 or not np.isfinite(peak) or not np.isfinite(b):
            rec["threshold_hu"] = float("nan")
            rec["reason"] = "window off the volume"
            out[pose.tile_id] = rec
            continue
        terms = [float(k_sigma) * sig]
        if np.isfinite(c_ref):
            terms.append(float(contrast_fraction) * c_ref)
        thr = b + max(terms)
        rec["threshold_hu"] = float(thr)
        if not peak > thr:
            out[pose.tile_id] = rec
            continue
        # evidence: refine with the stage-2 grey centroid, started from the
        # tile-mates' mean axis (the first pass is orientation-free anyway)
        axis0 = None
        if axes is not None and len(pose.seed_indices):
            _c, a4 = pose.seed_points(centers, axes)
            axis0 = a4[-1]
        gc = grey_centroid(vol, x0, axis=axis0, others=centers,
                           others_axes=axes, saturation=saturation,
                           max_shift_mm=float(roi_mm))
        if gc.status == "ok":
            refined = np.asarray(gc.center_ras, dtype=float).copy()
            cov = np.asarray(gc.cov_ras, dtype=float).copy()
            refine_status = "ok"
        else:
            # air/tissue interface or another stage-2 fallback (module
            # docstring): above-threshold weighted centroid over the ROI
            refined, cov = _roi_centroid(win, x0, centers, thr, sig)
            refine_status = "roi_centroid:" + gc.status.replace("fallback:", "")
        rec.update(status=STATUS_RECOVERED, refined_ras=refined, cov_ras=cov,
                   refine_status=refine_status,
                   shift_mm=float(np.linalg.norm(refined - x0)),
                   n_saturated=int(gc.n_saturated))
        if update_poses:
            pose.inferred_seed_ras = np.asarray(refined, dtype=float)
        out[pose.tile_id] = rec
    return out


def _roi_centroid(win, x0, others, thr, sig, n_iter=3, tol_mm=0.05):
    """Above-threshold intensity-weighted centroid over the ROI (module
    docstring), re-centred up to ``n_iter`` times; stays at ``x0`` when no
    ROI voxel exceeds the threshold or the estimate leaves the ROI radius."""
    c = np.asarray(x0, dtype=float).copy()
    cov_noise = np.zeros((3, 3))
    for _ in range(max(1, int(n_iter))):
        pts, vals, inr = win.window(c, others)
        w = np.clip(vals[inr] - thr, 0.0, None)
        W = float(w.sum())
        if W <= 0.0:
            break
        c_new = (pts[inr] * w[:, None]).sum(axis=0) / W
        if float(np.linalg.norm(c_new - x0)) > win.r_roi:
            break
        d = pts[inr] - c_new[None, :]
        cov_noise = (sig ** 2) * (d.T @ d) / W ** 2
        moved = float(np.linalg.norm(c_new - c))
        c = c_new
        if moved < tol_mm:
            break
    D = win.M / win.spacing
    quant = D @ np.diag(win.spacing ** 2 / 12.0) @ D.T
    return c, cov_noise + quant


def verification_summary(verification: Optional[Dict[int, dict]]) -> dict:
    """JSON-friendly digest for ``vol.meta["seed_verify"]``."""
    ver = verification or {}
    tiles = {}
    for tid, r in ver.items():
        d = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                d[k] = np.round(v, 4).tolist()
            elif isinstance(v, (np.floating, float)):
                d[k] = None if not np.isfinite(v) else round(float(v), 4)
            else:
                d[k] = v
        tiles[int(tid)] = d
    n_rec = sum(1 for r in ver.values() if r["status"] == STATUS_RECOVERED)
    return dict(n_checked=len(ver), n_recovered=n_rec,
                n_no_evidence=len(ver) - n_rec, tiles=tiles)
