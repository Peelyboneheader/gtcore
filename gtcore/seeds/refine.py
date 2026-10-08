"""Threshold-free sub-voxel seed localization (plan-localization stage 2).

Detection (:mod:`gtcore.seeds.detect`) finds each seed by a global HU
threshold and reports the HU-weighted mean of the *above-threshold* voxels.
That estimator is biased: the threshold cuts the blurred seed profile at a
level that depends on the scanner, the slice thickness and where the seed
falls in the voxel grid, so the centre moves when the threshold moves and is
pulled toward the voxel grid (the known bias of thresholded centroids in
particle and fiducial localization).

This module re-measures each detected candidate with the standard
**background-subtracted intensity-weighted centroid** over a fixed window.

Window (ROI), in voxel-axis millimetres, with ``sigma_eff,a^2 =
sigma_psf^2 + s_a^2/12`` (scanner blur plus the voxel box along axis ``a``):

* ``roi="capsule"`` (default): every voxel within ``D/2 + 3 sigma_eff,a`` of
  the seed's axis segment (``+/- L/2`` about the current centre, along the
  current axis estimate) -- a window matched to the capsule that holds the
  seed and 3 sigma of its blur;
* ``roi="ellipsoid"``: the orientation-free ellipsoid of radius
  ``L/2 + 3 sigma_eff,a`` that holds the capsule in ANY orientation.

Both are point-symmetric about the current centre, so for a seed profile
symmetric about the seed centre the true centre is a fixed point of the
re-centred iteration even when the window clips some blur tail (an axis error
costs signal, not bias).  The capsule is the default because the centroid's
sensitivity to background structure grows with the window: a background step
``db`` over half of the window shifts the centroid by about
``db * N * r / sum(w)`` (``N`` voxels, radius ``r``, i.e. ~ r^4), and the
capsule holds ~3x fewer voxels, all close to the seed.  Measured on the head
phantom (docs/localization-notes.md, stage 2): the cavity's air-fluid level
4 mm above a seed moved the ellipsoid centroid 0.96 mm at 2.1 mm slices.

Neighbour masking (``mask``) removes, from both the ROI and the background
shell, voxels closer to another candidate centre than to this one
(``"voronoi"``, the default) and/or voxels inside a capsule-shaped exclusion
around every other candidate, its axis +/- L/2 with radius
``D/2 + 2 sigma_eff`` (``"capsule"``; ``"both"``; ``"none"``).  Measured
(notes, stage 2): the Voronoi cut is the one needed -- without masking, two
seeds 7 mm apart end to end both fall back; the capsule exclusion alone
leaves a 0.74 mm error in that case at 2 mm slices; adding it to the Voronoi
cut changes no result on the head phantom and doubles the runtime.

Estimator:

* background ``b`` = median of a shell just outside the ROI (same masking);
* weights ``w = v - b`` SIGNED over the whole ROI.  They are deliberately not
  clipped at zero: clipping keeps only the positive half of the background
  noise, a pedestal that pulls the centroid toward the window centre, i.e.
  toward the threshold-based starting point -- re-introducing the threshold
  dependence this stage removes.  With signed weights the background noise
  contributes zero-mean terms, and the estimator is unbiased for any profile
  symmetric about the seed centre, including one clipped symmetrically by
  the scanner's HU ceiling (so saturated voxels are kept);
* the window is re-centred on the new estimate ``n_iter`` times (the first
  pass of the capsule window is the orientation-free ellipsoid, so a wrong
  detection axis cannot truncate the seed); the long axis that orients the
  next capsule window is the principal axis of the weighted covariance with
  weights ``max(v - b, 0)``, Sheppard-corrected by the voxel box
  (``detect._measure`` without the correction flips the axis of a seed lying
  across a thick-slab boundary into z).

Covariance (analytic; no constant is fitted to data), sum of:

* noise (first-order error propagation of the centroid under i.i.d. voxel
  noise): ``sigma_n^2 * sum_i (x_i - c)(x_i - c)^T / (sum_i w_i)^2``, with
  ``sigma_n`` the robust (1.4826 * MAD) spread of the background shell;
* sampling, per voxel axis ``a``: the variance, over a uniformly distributed
  sub-voxel position, of the discretization error of a centroid of
  box-integrated samples.  By the Poisson summation formula, for a seed
  profile ``f_a`` along ``a`` sampled by boxes of width ``s_a`` the error is
  a Fourier series in the sub-voxel phase, and

      var_a = s_a^2 / (2 pi^2) * sum_{k>=1} |F_a(k / s_a)|^2 / k^2,
      F_a(nu) = sinc(L |u_a| nu) * exp(-2 pi^2 sigma_a^2 nu^2),
      sigma_a^2 = sigma_psf^2 + (1 - u_a^2) D^2 / 16

  (``D^2/16``: the capsule cross-section's variance).  It is ``s_a^2/12``
  (a position uniform within the slab) for a seed much thinner than the slab
  and vanishes once the blurred seed spans several slabs.
  ``slab="bound"`` (default) evaluates it with ``|u_a| = 0``, the seed lying
  in the slab plane -- the largest value over orientations.  The through-slab
  axis component is exactly what a thick slice cannot resolve, and the
  axis-dependent ``slab="exact"`` was over-confident on the head phantom
  (z NEES 15-33 at 2.1-2.8 mm vs 0.5-0.7 for the bound).  ``slab="rule"`` is
  the protocol's switch (``s_a^2/12`` where ``L |u_a| + 2.5 sigma_eff,a <
  1.5 s_a``, else 0), kept for comparison only;
* background structure: ``delta delta^T`` with ``delta = S g / sum(w)`` the
  shift a planar background gradient ``g`` (least squares over the shell;
  ``S = sum (x_i - c)(x_i - c)^T``) induces under the constant-background
  model -- the first-order systematic of that model, carried as an
  uncertainty.

Voxel-axis terms are rotated into RAS with the affine's direction cosines.
Calibration is CHECKED with the normalized estimation error squared (NEES)
on synthetic data with truth, never fitted.

Interpolated slices (``vol.meta["interpolated_k"]``, the DICOM loader's
fill of z gaps) are linear blends of the measured slices around them and
carry no independent z information.  Their voxels are KEPT in the centroid
and the through-slice variance is inflated instead: whenever the window
touches an interpolated slice, the k-axis sampling variance becomes the
uniform-slab ``s_k^2/12`` (no sub-slab z information).  Keeping them is
not a convenience: linear interpolation hands each measured slice to its two
bracketing grid slices with weights that sum to one and positions that
average to the measured slice's true z, so the blend preserves the first
moment of the measured data; and on a loader grid whole runs of grid slices
can be interpolated (measured slices sitting between grid positions), so
excluding them removes the seed.  Measured on the harness G3 (PostOp-like
1 mm slabs at irregular positions on a 2 mm loader grid): excluding left the
3D error at 0.83 / 0.72 mm (detection 0.86 / 0.78, sparse / crowded) with
46 % fallbacks; keeping and inflating gave 0.54 / 0.49 mm.

Rule-based steps (disclosed, with sensitivity reported in the notes): the
3-sigma ROI and 2-sigma exclusion margins, the 1.5 mm background shell, and
the fallbacks to the detected centre --
``shift`` (the refined centre moved more than ``max_shift_mm``),
``close_neighbour`` (another candidate within one seed length, 4.5 mm:
usually two fragments of one seed, which the Voronoi cut would halve),
``no_signal`` (no positive weight), ``extended`` (the shell holds more than
half of the ROI's peak contrast: the object, or a background step, reaches
beyond the seed's 3-sigma window -- a plate, bone), ``background`` (the
planar-gradient shift ``|delta|`` exceeds ``max_bg_shift_mm`` = 0.1 mm: an
air level or bone edge inside the window), ``roi_truncated`` (the ROI runs
off the volume).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import geometry as _geom
from .detect import SeedCandidates, _measure

__all__ = [
    "PSF_SIGMA_MM",
    "GreyCentroid",
    "estimate_saturation",
    "sampling_variance",
    "seed_roi",
    "grey_centroid",
    "refine_seed_candidates",
]

PSF_SIGMA_MM = 0.45            # scanner blur (Gaussian sigma), head kernels
ROI_N_SIGMA = 3.0              # ROI margin beyond the capsule
EXCL_N_SIGMA = 2.0             # neighbour exclusion margin beyond its radius
BG_SHELL_MM = 1.5              # background shell thickness (>= one voxel)
MAX_BACKGROUND_SHIFT_MM = 0.1  # "background" fallback (module docstring)
MAX_SHELL_CONTRAST = 0.5       # "extended" fallback (module docstring)
SLAB_EXTENT_N_SIGMA = 2.5      # slab="rule": projected extent L|u_a| + 2.5 sigma
SLAB_EXTENT_FACTOR = 1.5       # ... smaller than 1.5 slabs -> add s_a^2/12
SAMPLING_N_HARMONICS = 64      # Fourier terms of the sampling variance
SATURATION_MIN_VOXELS = 5      # voxels at exactly max() that reveal a clip
MAD_TO_SIGMA = 1.4826          # Gaussian consistency factor of the MAD
MASK_MODES = ("both", "voronoi", "capsule", "none")
ROI_SHAPES = ("capsule", "ellipsoid")
SLAB_MODELS = ("bound", "exact", "rule")


def estimate_saturation(arr) -> Optional[float]:
    """The scanner's HU clip ceiling, or ``None`` when the data is not clipped.

    Reconstructions store HU in a fixed integer range (3071 HU on most
    scanners) and dense metal saturates at its top.  A clip leaves many
    voxels at exactly the same maximum value; an unclipped maximum is
    essentially unique.  ``SATURATION_MIN_VOXELS`` (5) identical maxima are
    taken as evidence of a ceiling.
    """
    a = np.asarray(arr)
    if a.size == 0:
        return None
    top = a.max()
    if int(np.count_nonzero(a == top)) >= SATURATION_MIN_VOXELS:
        return float(top)
    return None


def sampling_variance(spacing, axis_vox, psf_sigma_mm=PSF_SIGMA_MM,
                      length_mm=None, diameter_mm=None):
    """Per-voxel-axis variance of the centroid's discretization error
    (module docstring).  ``axis_vox`` is the seed's unit axis in the voxel
    frame (zeros give the orientation-free bound); returns ``(3,)`` mm^2."""
    L = float(_geom.SEED_LENGTH_MM if length_mm is None else length_mm)
    D = float(_geom.SEED_DIAMETER_MM if diameter_mm is None else diameter_mm)
    s = np.asarray(spacing, dtype=float).reshape(3)
    u = np.abs(np.asarray(axis_vox, dtype=float).reshape(3))
    sig2 = float(psf_sigma_mm) ** 2 + (1.0 - u ** 2) * D ** 2 / 16.0
    k = np.arange(1, SAMPLING_N_HARMONICS + 1, dtype=float)[:, None]
    nu = k / s[None, :]
    F = (np.sinc(L * u[None, :] * nu)
         * np.exp(-2.0 * np.pi ** 2 * sig2[None, :] * nu ** 2))
    return s ** 2 / (2.0 * np.pi ** 2) * (F ** 2 / k ** 2).sum(axis=0)


class _Grid:
    """Per-volume geometry shared by every seed of one refinement run."""

    def __init__(self, vol, psf_sigma_mm=PSF_SIGMA_MM):
        aff = np.asarray(vol.affine, dtype=float)
        self.arr = np.asarray(vol.array)
        self.M = aff[:3, :3]
        self.t = aff[:3, 3]
        self.Minv = np.linalg.inv(self.M)
        self.spacing = np.linalg.norm(self.M, axis=0)          # (si, sj, sk)
        self.D = self.M / self.spacing                         # direction cosines
        self.psf = float(psf_sigma_mm)
        self.sig_eff = np.sqrt(self.psf ** 2 + self.spacing ** 2 / 12.0)
        self.L = float(_geom.SEED_LENGTH_MM)
        self.Dseed = float(_geom.SEED_DIAMETER_MM)
        self.r_ell = self.L / 2.0 + ROI_N_SIGMA * self.sig_eff
        self.r_cap = self.Dseed / 2.0 + ROI_N_SIGMA * self.sig_eff
        self.r_excl = self.Dseed / 2.0 + EXCL_N_SIGMA * self.sig_eff
        self.quant_cov = self.M @ np.diag([1.0 / 12.0] * 3) @ self.M.T
        self.shape_ijk = np.array(self.arr.shape[::-1])
        nk = self.arr.shape[0]
        self.interp = np.zeros(nk, dtype=bool)      # interpolated k slices
        meta = getattr(vol, "meta", None) or {}
        ik = np.asarray(list(meta.get("interpolated_k", ()) or ()), dtype=int)
        self.interp[ik[(ik >= 0) & (ik < nk)]] = True

    def to_vox_mm(self, d_ras):
        """RAS offset(s) -> voxel-axis millimetres (``(..., 3)``, ijk order)."""
        return (np.asarray(d_ras) @ self.Minv.T) * self.spacing

    def axis_vox(self, axis_ras):
        u = self.to_vox_mm(np.asarray(axis_ras, dtype=float))
        n = np.linalg.norm(u)
        return u / n if n > 0 else np.array([0.0, 0.0, 1.0])

    def slab_var(self, axis_ras, model="bound"):
        if model == "bound":
            return sampling_variance(self.spacing, np.zeros(3), self.psf,
                                     self.L, self.Dseed)
        u = self.axis_vox(axis_ras)
        if model == "exact":
            return sampling_variance(self.spacing, u, self.psf, self.L,
                                     self.Dseed)
        extent = self.L * np.abs(u) + SLAB_EXTENT_N_SIGMA * self.sig_eff
        return np.where(extent < SLAB_EXTENT_FACTOR * self.spacing,
                        self.spacing ** 2 / 12.0, 0.0)

    def vox_to_ras_cov(self, var_ijk):
        return self.D @ np.diag(var_ijk) @ self.D.T

    def quantization_cov(self):
        """``s_a^2/12`` per voxel axis: the covariance reported for a centre
        known only to its voxel (every fallback)."""
        return self.vox_to_ras_cov(self.spacing ** 2 / 12.0)


@dataclass
class _Neighbourhood:
    pts: np.ndarray        # (n, 3) RAS of every kept voxel (ROI + shell)
    vals: np.ndarray       # (n,)
    kji: np.ndarray        # (n, 3) int
    in_roi: np.ndarray     # (n,) bool
    truncated: bool        # the window ran off the volume


def _as_others(others, others_axes):
    if others is None:
        return np.zeros((0, 3)), None
    oc = np.asarray(others, dtype=float).reshape(-1, 3)
    oa = None
    if others_axes is not None:
        oa = np.asarray(others_axes, dtype=float).reshape(-1, 3)
        if len(oa) != len(oc):
            raise ValueError("others_axes must match others")
    return oc, oa


def _capsule_rho2(grid, e, u, radius_vox):
    """Squared normalized distance of RAS offsets ``e`` to the segment
    ``[-L/2, L/2] * u``, per-axis radii ``radius_vox`` (voxel-axis mm)."""
    tt = np.clip(e @ u, -grid.L / 2.0, grid.L / 2.0)
    d = e - tt[:, None] * u[None, :]
    return ((grid.to_vox_mm(d) / radius_vox) ** 2).sum(axis=1)


def _neighbourhood(grid, center, axis, others_c, others_a, mask="voronoi",
                   bg_shell_mm=BG_SHELL_MM, roi="capsule"):
    c = np.asarray(center, dtype=float).reshape(3)
    p = grid.Minv @ (c - grid.t)                                 # ijk, continuous
    shell_t = np.maximum(float(bg_shell_mm), grid.spacing)
    if roi == "capsule":
        u = np.asarray(axis, dtype=float).reshape(3)
        u = u / max(np.linalg.norm(u), 1e-12)
        reach = grid.L / 2.0 * np.abs(grid.axis_vox(u)) + grid.r_cap + shell_t
    else:
        reach = grid.r_ell + shell_t
    half = np.ceil(reach / grid.spacing).astype(int) + 1
    pc = np.round(p).astype(int)
    lo, hi = pc - half, pc + half
    # only the ROI running off the volume biases the centroid; a shell cut
    # by the volume edge just estimates the background from fewer voxels
    half_roi = np.ceil(np.maximum(reach - shell_t, 0.0) / grid.spacing).astype(int)
    truncated = bool(np.any(pc - half_roi < 0)
                     or np.any(pc + half_roi > grid.shape_ijk - 1))
    lo = np.clip(lo, 0, grid.shape_ijk - 1)
    hi = np.clip(hi, 0, grid.shape_ijk - 1)
    sub = grid.arr[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
    K, J, I = np.meshgrid(np.arange(lo[2], hi[2] + 1),
                          np.arange(lo[1], hi[1] + 1),
                          np.arange(lo[0], hi[0] + 1), indexing="ij")
    idx = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1).astype(float)
    e = (idx - p) @ grid.M.T                                     # RAS offsets
    if roi == "capsule":
        rho_roi = _capsule_rho2(grid, e, u, grid.r_cap)
        rho_out = _capsule_rho2(grid, e, u, grid.r_cap + shell_t)
    else:
        dvox = (idx - p) * grid.spacing
        rho_roi = ((dvox / grid.r_ell) ** 2).sum(axis=1)
        rho_out = ((dvox / (grid.r_ell + shell_t)) ** 2).sum(axis=1)
    keep = rho_out <= 1.0
    if len(others_c) and mask != "none":
        g = others_c - c[None, :]
        span = 2.0 * float(np.max(reach)) + grid.L
        for m in np.flatnonzero(np.linalg.norm(g, axis=1) < span):
            if mask in ("both", "voronoi"):
                # closer to the other candidate than to this one
                keep &= (e @ g[m]) <= 0.5 * float(g[m] @ g[m])
            if mask in ("both", "capsule") and others_a is not None:
                uo = others_a[m] / max(np.linalg.norm(others_a[m]), 1e-12)
                keep &= _capsule_rho2(grid, e - g[m][None, :], uo,
                                      grid.r_excl) >= 1.0
    sel = np.flatnonzero(keep)
    pts = e[sel] + c[None, :]
    vals = sub.reshape(-1)[sel].astype(float)
    kji = np.stack([K.ravel()[sel], J.ravel()[sel], I.ravel()[sel]], axis=1)
    return _Neighbourhood(pts, vals, kji, rho_roi[sel] <= 1.0, truncated)


def _long_axis(pts, w, quant_cov, fallback):
    """Principal axis of the Sheppard-corrected weighted covariance.

    Box-sampled positions overstate a profile's variance by ``s_a^2/12`` on
    average (Sheppard's correction for grouped data).  Uncorrected, a seed
    lying flat across a 2.8 mm slab boundary -- its weight split between two
    slabs 2.8 mm apart -- has a larger through-slab spread (1.96 mm^2) than
    along its own 4.5 mm length (~1.9 mm^2) and its PCA axis flips into z
    (``detect._measure`` uses the uncorrected data covariance for the axis).
    """
    if len(pts) < 4 or not w.sum() > 0:
        return fallback
    c = (pts * w[:, None]).sum(axis=0) / w.sum()
    d = pts - c
    cov = (d * w[:, None]).T @ d / w.sum() - quant_cov
    evals, evecs = np.linalg.eigh(cov)
    if not evals[-1] > 0:
        _, axis, _ = _measure(pts, w, quant_cov)
        return axis
    return evecs[:, -1]


def _check_options(roi, slab, mask):
    if roi not in ROI_SHAPES:
        raise ValueError("roi must be one of %s" % (ROI_SHAPES,))
    if slab not in SLAB_MODELS:
        raise ValueError("slab must be one of %s" % (SLAB_MODELS,))
    if mask not in MASK_MODES:
        raise ValueError("mask must be one of %s" % (MASK_MODES,))


def seed_roi(vol, center, others=None, axis=None, *, others_axes=None,
             psf_sigma_mm=PSF_SIGMA_MM, mask="voronoi"):
    """Voxels of one seed's ROI, neighbour-masked (module docstring).

    Parameters
    ----------
    vol : Volume
    center : (3,) RAS mm -- the ROI centre (current seed estimate).
    others : (M, 3) RAS centres of the OTHER candidates, or ``None``.
    axis : (3,) this seed's axis.  Given: the capsule window along it;
        ``None``: the orientation-free ellipsoid.
    others_axes : (M, 3) axes of ``others`` (needed for the capsule
        exclusion; without them only the Voronoi cut applies).
    mask : "voronoi" (default), "both", "capsule" or "none".

    Returns
    -------
    (points_ras (n, 3), values (n,), kji (n, 3) int)
    """
    roi = "ellipsoid" if axis is None else "capsule"
    _check_options(roi, "bound", mask)
    grid = _Grid(vol, psf_sigma_mm)
    oc, oa = _as_others(others, others_axes)
    nb = _neighbourhood(grid, center, axis, oc, oa, mask=mask, roi=roi)
    r = nb.in_roi
    return nb.pts[r], nb.vals[r], nb.kji[r]


@dataclass
class GreyCentroid:
    """Result of :func:`grey_centroid` for one seed."""

    center_ras: np.ndarray
    axis_ras: np.ndarray
    cov_ras: np.ndarray            # (3, 3) mm^2
    background_hu: float
    sigma_noise_hu: float
    weight_sum: float              # sum of signed weights (HU x voxels)
    n_roi: int
    n_saturated: int               # ROI voxels at the clip ceiling
    status: str                    # "ok" or "fallback:<reason>"
    cov_noise: Optional[np.ndarray] = None
    slab_var_ijk: Optional[np.ndarray] = None
    bg_shift: Optional[np.ndarray] = None   # delta (module docstring), RAS mm


def _grey_centroid(grid, center, axis, others_c, others_a, *, n_iter=3,
                   bg_shell_mm=BG_SHELL_MM, mask="voronoi", roi="capsule",
                   slab="bound", saturation=None,
                   max_bg_shift_mm=MAX_BACKGROUND_SHIFT_MM, max_shift_mm=None):
    c0 = np.asarray(center, dtype=float).reshape(3)
    ax0 = (np.array([0.0, 0.0, 1.0]) if axis is None
           else np.asarray(axis, dtype=float).reshape(3))
    c, ax = c0.copy(), ax0.copy()
    nb, b, sig_n, wsum = None, 0.0, 0.0, 0.0

    def fail(reason, n_sat=0, bg_shift=None):
        n_roi = 0 if nb is None else int(nb.in_roi.sum())
        return GreyCentroid(c0, ax0, grid.quantization_cov(), b, sig_n, wsum,
                            n_roi, n_sat, "fallback:" + reason,
                            bg_shift=bg_shift)

    if len(others_c):
        # another candidate closer than one seed length: the two windows
        # overlap and the Voronoi cut runs through the signal of at least
        # one of them -- typically two fragments of ONE seed split by the
        # threshold (stage 3); the cut halves it and each half's centroid is
        # biased by up to L/4, so the detection is kept
        if float(np.linalg.norm(others_c - c0[None, :], axis=1).min()) < grid.L:
            return fail("close_neighbour")
    n_iter = max(1, int(n_iter))
    for it in range(n_iter):
        # the first pass of the capsule window runs orientation-free: a
        # detection axis that is badly wrong (a thick-slice seed split across
        # two slabs reports ~z) would otherwise truncate the seed's ends
        # and the PCA of the truncated signal can lock onto that wrong axis
        shape = "ellipsoid" if (roi == "capsule" and it == 0
                                and n_iter > 1) else roi
        nb = _neighbourhood(grid, c, ax, others_c, others_a, mask=mask,
                            bg_shell_mm=bg_shell_mm, roi=shape)
        if nb.truncated:
            return fail("roi_truncated")
        inr = nb.in_roi
        if inr.sum() < 4 or (~inr).sum() < 4:
            return fail("empty_roi")
        sv = nb.vals[~inr]
        b = float(np.median(sv))
        sig_n = MAD_TO_SIGMA * float(np.median(np.abs(sv - b)))
        w = nb.vals[inr] - b
        peak = float(w.max())
        wsum = float(w.sum())
        if not (peak > 0.0 and wsum > 0.0):
            return fail("no_signal")
        # judged on the final window only: an early window built on a poor
        # detection axis can hold a seed end in its shell, which the PCA
        # axis of that same pass then corrects
        if (it == n_iter - 1
                and float(np.abs(sv - b).max()) > MAX_SHELL_CONTRAST * peak):
            return fail("extended")
        pts = nb.pts[inr]
        c = (pts * w[:, None]).sum(axis=0) / wsum
        if (max_shift_mm is not None
                and float(np.linalg.norm(c - c0)) > float(max_shift_mm)):
            return fail("shift")
        ax = _long_axis(pts, np.clip(w, 0.0, None), grid.quant_cov, ax)

    v = nb.vals[nb.in_roi]
    n_sat = 0 if saturation is None else int(np.count_nonzero(v >= saturation))
    d = nb.pts[nb.in_roi] - c[None, :]
    S = d.T @ d
    cov_noise = sig_n ** 2 * S / wsum ** 2
    sh = ~nb.in_roi
    X = np.column_stack([np.ones(int(sh.sum())), nb.pts[sh] - c[None, :]])
    grad = np.linalg.lstsq(X, nb.vals[sh], rcond=None)[0][1:]
    bg_shift = S @ grad / wsum
    if (max_bg_shift_mm is not None
            and float(np.linalg.norm(bg_shift)) > float(max_bg_shift_mm)):
        return fail("background", n_sat, bg_shift)
    var_ijk = grid.slab_var(ax, slab)
    if grid.interp[np.unique(nb.kji[:, 0])].any():
        # the window touches an interpolated slice: no sub-slab z information
        var_ijk = var_ijk.copy()
        var_ijk[2] = max(var_ijk[2], grid.spacing[2] ** 2 / 12.0)
    cov = cov_noise + grid.vox_to_ras_cov(var_ijk) + np.outer(bg_shift, bg_shift)
    return GreyCentroid(c, ax, cov, b, sig_n, wsum, int(nb.in_roi.sum()), n_sat,
                        "ok", cov_noise=cov_noise, slab_var_ijk=var_ijk,
                        bg_shift=bg_shift)


def grey_centroid(vol, center, axis=None, others=None, n_iter=3,
                  bg_shell_mm=BG_SHELL_MM, *, others_axes=None,
                  psf_sigma_mm=PSF_SIGMA_MM, mask="voronoi", roi="capsule",
                  slab="bound", saturation=None,
                  max_bg_shift_mm=MAX_BACKGROUND_SHIFT_MM,
                  max_shift_mm=None) -> GreyCentroid:
    """Background-subtracted intensity-weighted centroid of one seed.

    See the module docstring for the estimator, its covariance and the
    fallbacks.  ``axis`` is the starting axis (the detection axis; ``None``
    starts from +z, which the PCA corrects after the first pass) and is
    returned unchanged on a fallback; ``others`` / ``others_axes`` are the
    other candidates (masked out of ROI and shell).  ``saturation`` (the
    clip ceiling, :func:`estimate_saturation`) is only counted.
    ``max_shift_mm`` (``None``: unchecked) stops the iteration as soon as the
    estimate leaves that radius around ``center``.  A fallback returns the
    input centre, the voxel-quantization covariance and
    ``status = "fallback:<reason>"``.
    """
    _check_options(roi, slab, mask)
    grid = _Grid(vol, psf_sigma_mm)
    oc, oa = _as_others(others, others_axes)
    return _grey_centroid(grid, center, axis, oc, oa, n_iter=n_iter,
                          bg_shell_mm=bg_shell_mm, mask=mask, roi=roi,
                          slab=slab, saturation=saturation,
                          max_bg_shift_mm=max_bg_shift_mm,
                          max_shift_mm=max_shift_mm)


def refine_seed_candidates(vol, cands: SeedCandidates, method="centroid",
                           max_shift_mm=1.5, psf_sigma_mm=PSF_SIGMA_MM, *,
                           n_iter=3, bg_shell_mm=BG_SHELL_MM, mask="voronoi",
                           roi="capsule", slab="bound",
                           max_bg_shift_mm=MAX_BACKGROUND_SHIFT_MM,
                           update_axes=False) -> SeedCandidates:
    """Re-localize every candidate on the RAW (not metal-inpainted) volume.

    Returns a new :class:`SeedCandidates` (every other field carried by
    ``subset``) with refined ``centers_ras``, ``cov_ras`` (N, 3, 3) and
    per-seed diagnostics in ``info``: ``refine_status`` ("ok" /
    "fallback:<reason>"), ``refine_shift_mm``, ``refine_axis_ras`` (the
    grey-level axis), ``refine_background_hu``, ``refine_sigma_noise_hu``,
    ``refine_bg_shift_mm``, ``refine_n_saturated``; plus ``refine_method``
    and ``saturation_hu`` (the detected clip ceiling or ``None``).

    ``axes_ras`` keeps the DETECTION axes unless ``update_axes=True``.  The
    grey-level axis is the more accurate one (head phantom median error
    8.9 vs 16.4 deg at 2.1 mm, 12.7 vs 23.9 deg at 2.8 mm), but the tile
    fitter's gates and scores were calibrated on detection axes, and handing
    it the new axes cost one correct tile partition out of five at 2.1 mm
    (analytic phantom) and at 2.8 mm (binary), while refined centres with
    detection axes gained one at 2.1 / 2.8 mm (binary) and 2.8 mm (analytic)
    -- re-tuning those gates is outside this stage.

    A candidate falls back to its detected centre and axis, with the
    voxel-quantization covariance ``s_a^2/12``, when the grey centroid is not
    usable (module docstring) or moves farther than ``max_shift_mm`` from the
    detection.
    """
    if method not in ("centroid",):
        raise ValueError("unknown refinement method %r" % (method,))
    _check_options(roi, slab, mask)
    n = len(cands)
    out = cands.subset(np.arange(n))
    info = dict(out.info or {})
    saturation = estimate_saturation(vol.array)
    grid = _Grid(vol, psf_sigma_mm)
    centers = np.array(cands.centers_ras, dtype=float).reshape(-1, 3)
    axes = np.array(cands.axes_ras, dtype=float).reshape(-1, 3)
    new_c, new_a, ref_a = centers.copy(), axes.copy(), axes.copy()
    cov = np.zeros((n, 3, 3))
    status = []
    shift, bg, sig = np.zeros(n), np.zeros(n), np.zeros(n)
    bgs = np.full(n, np.nan)
    nsat = np.zeros(n, dtype=int)
    for i in range(n):
        r = _grey_centroid(grid, centers[i], axes[i],
                           np.delete(centers, i, axis=0),
                           np.delete(axes, i, axis=0),
                           n_iter=n_iter, bg_shell_mm=bg_shell_mm, mask=mask,
                           roi=roi, slab=slab, saturation=saturation,
                           max_bg_shift_mm=max_bg_shift_mm,
                           max_shift_mm=max_shift_mm)
        st = r.status
        if st == "ok":
            new_c[i], cov[i] = r.center_ras, r.cov_ras
            ref_a[i] = r.axis_ras
            if update_axes:
                new_a[i] = r.axis_ras
            shift[i] = float(np.linalg.norm(r.center_ras - centers[i]))
        else:
            cov[i] = grid.quantization_cov()
        status.append(st)
        bg[i], sig[i], nsat[i] = r.background_hu, r.sigma_noise_hu, r.n_saturated
        if r.bg_shift is not None:
            bgs[i] = float(np.linalg.norm(r.bg_shift))
    info.update(
        refine_method=method,
        saturation_hu=saturation,
        refine_status=np.array(status, dtype=object),
        refine_shift_mm=shift,
        refine_background_hu=bg,
        refine_sigma_noise_hu=sig,
        refine_bg_shift_mm=bgs,
        refine_n_saturated=nsat,
        refine_axis_ras=ref_a,
    )
    out.centers_ras, out.axes_ras, out.cov_ras, out.info = new_c, new_a, cov, info
    return out
