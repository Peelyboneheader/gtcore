"""Physically integrated CT rendering of Cs-131 seeds (validation harness).

This is the *truth generator* of the seed-localization harness
(``docs/plan-localization.md``, stage 0).  It is deliberately independent of
every localization or fitting model in :mod:`gtcore.seeds` (no shared code):
a fitting model that shared its forward model with the renderer would be
graded on its own assumptions (the "inverse crime").

Forward model, per seed
-----------------------
1. **Exact capsule.**  The CS-1 Cs-131 seed is a titanium capsule 4.5 mm long
   and 0.8 mm across (:mod:`gtcore.geometry`).  It is modelled as a cylinder
   with **hemispherical end caps** of radius ``diameter / 2`` whose overall
   end-to-end length is ``length_mm`` (so the straight part is
   ``length - diameter`` = 3.7 mm and the volume is
   ``pi r^2 (L - 2r) + 4/3 pi r^3`` = 2.128 mm^3).  The same convention as
   :func:`gtcore.phantom.generate._paint_capsule`.
2. **Fine local grid in voxel-index space.**  Around each seed a box of whole
   voxels covering the capsule plus 4 PSF sigmas on every side (6-11 mm) is
   oversampled by ``m_a = ceil(|s_a| / fine_mm)`` per voxel axis, so every
   fine sample lies inside exactly one voxel footprint.  Each fine sample
   centre is marked in/out of the capsule.
3. **PSF.**  The in/out patch is Gaussian-blurred in mm (``psf_sigma_mm``
   scalar, or ``(sigma_xy, sigma_z)``) on the fine grid.
4. **Slab integration.**  The blurred fine patch is block-averaged by
   ``m_a`` per axis: each voxel receives the mean of the blurred image over
   its exact footprint (in-plane pixel and slice slab).  Blur and block
   average are separable, so they are applied axis by axis (blur along an
   axis, then average along it) -- identical to the full 3-D blur followed by
   the 3-D block average, at a fraction of the cost.
5. The patch, scaled by ``metal_hu`` (the capsule's contrast ABOVE the
   background, see below), is ADDED into the volume; patches of nearby seeds
   sum.  Then i.i.d. Gaussian noise (``noise_hu``) is added to the whole
   volume and the result is clipped at ``saturate_hu`` (the 12-bit scanner
   ceiling 3071 HU; ``None`` = no clip).

``metal_hu`` is a contrast: noise-free and unclipped,
``sum(v - background) * voxel_volume = metal_hu * capsule_volume``.

Calibration of ``metal_hu`` (2026-10-08)
----------------------------------------
Measured on the real scans (max HU within 2 mm of each detected seed):

* printed 8-tile phantom (Philips 0.59 x 0.59 x 1.0 mm, O-MAR): 31 of 32
  seeds clip at 3071 HU, the last peaks at 2862;
* PostOp CT (Siemens, 0.52 mm pixels, SliceThickness 1 mm, 64 slices
  present at irregular 1 mm multiples, rebuilt by the loader onto a 2.0 mm
  grid): ``docs/data-notes.md`` quotes 1500-1950 HU, which holds on the
  interpolated grid slices; the 22 tile-assigned seeds peak (+/- 1 mm) at a
  median 2936 HU on the measured slices, 4 of 22 at 3071.

With background 35 HU, PSF sigma 0.45 mm and noise 10 HU, one contrast cannot
reproduce both: for random orientations the 1 mm / 2 mm peak ratio is only
~1.3 (the 0.8 mm capsule cross-section, not the slab, limits a 1 mm peak), so
a contrast giving a 1700 HU median at 2 mm gives a 2200 HU median and NO
saturation at 0.59 x 0.59 x 1.0 mm.  The two scanners differ (kernel, MAR,
patient vs plastic), so two contrasts are calibrated, by bisection over 400
seeds at uniformly random sub-voxel offsets and axes:

* ``METAL_HU_POSTOP = 8380`` (= ``DEFAULT_METAL_HU``): median peak 1700 HU on
  0.5 x 0.5 x 2.0 mm;
* ``METAL_HU_PRINTED = 13500``: 31/32 of the seeds reach 3071 HU on
  0.59 x 0.59 x 1.0 mm, as on the printed phantom.

Check against the PostOp's measured slices (not the calibration target set
for this stage, so not the default): ``METAL_HU_POSTOP_MEASURED = 10900``
reproduces, on 0.52 x 0.52 x 1.0 mm, BOTH the median (2936 HU) and the
saturated fraction (18 %, real 4/22) with one parameter; it would put the
0.5 x 0.5 x 2.0 mm median at 2200 HU, i.e. the 1700 HU target is pessimistic
(lower contrast, harder detection) for this scanner.

Rendered peaks (``python scripts/validation_seed_localization.py
--calibrate``, 400 seeds per row, unclipped):

=====================================  ========  ==========  =========  =====
grid                                   contrast  median HU   P5-P95     >=3071
=====================================  ========  ==========  =========  =====
0.59 x 0.59 x 1.0                      postop    2203        1972-2382  0 %
0.59 x 0.59 x 1.0                      printed   3526        3152-3816  96 %
0.5 x 0.5 x 2.0                        postop    1696        1255-2247  0 %
0.5 x 0.5 x 2.0                        printed   2712        1994-3600  27 %
1 mm slabs, irregular gaps, 2 mm grid  postop    1776        730-2366   0 %
0.7 x 0.7 x 2.8                        postop    1202        874-1812   0 %
0.7 x 0.7 x 2.8                        printed   1913        1391-2893  4 %
=====================================  ========  ==========  =========  =====

The binary head phantom (8000 HU painted capsules) peaks at ~4000 HU on its
0.7 mm grid, i.e. in the saturating regime, so ``make_head_phantom(
seed_render="analytic")`` defaults to ``METAL_HU_PRINTED``.

Also here: :func:`thick_slices` (partial-volume slab averaging of an existing
volume, moved from ``scripts/validation_spacing.py``),
:func:`drop_and_interpolate` (the DICOM loader's gap filler applied to a
synthetic volume) and :func:`random_seed_layout`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage

from .. import geometry as _geom
from ..volume import Volume

__all__ = [
    "SEED_LENGTH_MM",
    "SEED_DIAMETER_MM",
    "DEFAULT_METAL_HU",
    "METAL_HU_POSTOP",
    "METAL_HU_PRINTED",
    "METAL_HU_POSTOP_MEASURED",
    "SATURATE_HU",
    "capsule_volume_mm3",
    "add_rendered_seeds",
    "render_seeds",
    "thick_slices",
    "drop_and_interpolate",
    "random_seed_layout",
]

SEED_LENGTH_MM = _geom.SEED_LENGTH_MM            # 4.5, end to end
SEED_DIAMETER_MM = _geom.SEED_DIAMETER_MM        # 0.8
SATURATE_HU = 3071.0                             # 12-bit CT ceiling
# Calibrated capsule contrasts above background (module docstring).
METAL_HU_POSTOP = 8380.0      # 0.5 x 0.5 x 2.0 mm: median peak 1700 HU
METAL_HU_PRINTED = 13500.0    # 0.59 x 0.59 x 1.0 mm: 31/32 seeds clip at 3071
# The PostOp's MEASURED 1 mm slices (not its 2 mm interpolated grid): 0.52 x
# 0.52 x 1.0 mm median peak 2936 HU, 18 % at 3071 (module docstring).
METAL_HU_POSTOP_MEASURED = 10900.0
DEFAULT_METAL_HU = METAL_HU_POSTOP
PSF_TRUNCATE = 4.0                               # sigmas kept on each side

Number = Union[float, int]


# ------------------------------------------------------------------ helpers
def capsule_volume_mm3(length_mm=SEED_LENGTH_MM, diameter_mm=SEED_DIAMETER_MM):
    """Volume of the hemispherically capped capsule (overall length L)."""
    r = 0.5 * float(diameter_mm)
    straight = max(0.0, float(length_mm) - 2.0 * r)
    return math.pi * r * r * straight + 4.0 / 3.0 * math.pi * r ** 3


def _axis_aligned_spacing(affine, rtol=1e-6):
    """Signed per-axis voxel step (i, j, k) of a diagonal affine; raises
    when the affine is rotated or permuted (not supported here)."""
    lin = np.asarray(affine, dtype=float)[:3, :3]
    diag = np.diag(lin).copy()
    off = lin - np.diag(diag)
    scale = float(np.abs(diag).max())
    if scale <= 0.0 or np.any(np.abs(diag) <= 0.0):
        raise ValueError("degenerate affine (zero spacing)")
    if float(np.abs(off).max()) > rtol * scale:
        raise ValueError("render_seeds needs an axis-aligned (diagonal) affine")
    return diag


def _psf_sigmas(psf_sigma_mm):
    """(sigma_i, sigma_j, sigma_k) in mm from a scalar or (sigma_xy, sigma_z)."""
    s = np.atleast_1d(np.asarray(psf_sigma_mm, dtype=float)).reshape(-1)
    if s.size == 1:
        return np.array([s[0], s[0], s[0]])
    if s.size == 2:
        return np.array([s[0], s[0], s[1]])
    if s.size == 3:
        return s.copy()
    raise ValueError("psf_sigma_mm must be a scalar, (sxy, sz) or (si, sj, sk)")


def _unit_rows(v):
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(n == 0.0):
        raise ValueError("seed axis of zero length")
    return v / n


def _seed_patch(center, axis, step, origin, sig_mm, length_mm, radius_mm,
                fine_mm):
    """Rendered unit-contrast patch of one seed.

    Returns ``(patch[k, j, i], (k0, j0, i0))`` where the patch covers whole
    voxels starting at index ``(k0, j0, i0)`` (may lie partly outside the
    volume; the caller clips).
    """
    # voxel axis a  <->  RAS axis a (diagonal affine); step may be negative
    abs_step = np.abs(step)
    half_straight = max(0.0, 0.5 * length_mm - radius_mm)
    extent = np.abs(axis) * half_straight + radius_mm + PSF_TRUNCATE * sig_mm
    c_idx = (center - origin) / step                       # continuous (i,j,k)
    e_idx = extent / abs_step
    v0 = np.floor(c_idx - e_idx + 0.5).astype(int)
    v1 = np.floor(c_idx + e_idx + 0.5).astype(int)
    m = np.maximum(1, np.ceil(abs_step / float(fine_mm) - 1e-9).astype(int))

    # fine sample centres (physical mm offsets from the seed centre) per axis
    offs = []
    for a in range(3):
        nv = int(v1[a] - v0[a] + 1)
        q = (np.arange(nv * m[a]) + 0.5) / m[a] - 0.5      # index units from v0
        idx = v0[a] + q
        offs.append(origin[a] + step[a] * idx - center[a])
    dx = offs[0][None, None, :]
    dy = offs[1][None, :, None]
    dz = offs[2][:, None, None]

    # exact capsule: distance to the axis segment <= radius
    proj = dx * axis[0] + dy * axis[1] + dz * axis[2]
    t = np.clip(proj, -half_straight, half_straight)
    d2 = dx * dx + dy * dy + dz * dz - 2.0 * t * proj + t * t
    patch = (d2 <= radius_mm * radius_mm).astype(np.float64)
    del proj, t, d2

    # separable blur + block average, axis by axis (array axes: k=0, j=1, i=2)
    # largest oversampling first: it shrinks the array the most
    for a in sorted(range(3), key=lambda a: -m[a]):
        arr_ax = 2 - a
        sig_fine = sig_mm[a] / (abs_step[a] / m[a])
        if sig_fine > 0.0:
            patch = ndimage.gaussian_filter1d(
                patch, sig_fine, axis=arr_ax, mode="constant", cval=0.0,
                truncate=PSF_TRUNCATE)
        if m[a] > 1:
            shp = list(patch.shape)
            nv = shp[arr_ax] // m[a]
            new_shape = shp[:arr_ax] + [nv, int(m[a])] + shp[arr_ax + 1:]
            patch = patch.reshape(new_shape).mean(axis=arr_ax + 1)
    return patch, (int(v0[2]), int(v0[1]), int(v0[0]))


# ------------------------------------------------------------------ renderer
def add_rendered_seeds(array, affine, centers, axes,
                       length_mm=SEED_LENGTH_MM, diameter_mm=SEED_DIAMETER_MM,
                       metal_hu=DEFAULT_METAL_HU, psf_sigma_mm=0.45,
                       fine_mm=0.1):
    """Add the slab-integrated, blurred seed patches into ``array`` in place.

    ``array`` is ``[k, j, i]`` float; ``affine`` must be axis-aligned.  No
    noise, no clipping (see :func:`render_seeds`).  Returns ``array``.
    """
    step = _axis_aligned_spacing(affine)
    origin = np.asarray(affine, dtype=float)[:3, 3]
    sig = _psf_sigmas(psf_sigma_mm)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    if centers.shape[0] == 0:
        return array
    axes = _unit_rows(axes)
    if axes.shape[0] != centers.shape[0]:
        raise ValueError("centers and axes disagree in length")
    radius = 0.5 * float(diameter_mm)
    shape = np.asarray(array.shape)
    for c, u in zip(centers, axes):
        patch, (k0, j0, i0) = _seed_patch(c, u, step, origin, sig,
                                          float(length_mm), radius,
                                          float(fine_mm))
        lo = np.array([k0, j0, i0])
        hi = lo + np.asarray(patch.shape)
        clo = np.maximum(lo, 0)
        chi = np.minimum(hi, shape)
        if np.any(chi <= clo):
            continue
        dst = tuple(slice(int(a), int(b)) for a, b in zip(clo, chi))
        src = tuple(slice(int(a - l), int(b - l))
                    for a, b, l in zip(clo, chi, lo))
        array[dst] += (float(metal_hu) * patch[src]).astype(array.dtype)
    return array


def render_seeds(shape_kji, affine, centers, axes,
                 length_mm=SEED_LENGTH_MM, diameter_mm=SEED_DIAMETER_MM,
                 metal_hu=DEFAULT_METAL_HU, background_hu=35.0,
                 psf_sigma_mm=0.45, noise_hu=10.0,
                 saturate_hu: Optional[float] = SATURATE_HU,
                 fine_mm=0.1, rng_seed=0) -> Volume:
    """Render seeds into a uniform background volume (module docstring).

    Parameters
    ----------
    shape_kji : (nk, nj, ni)
    affine : (4, 4) axis-aligned voxel-index -> RAS affine (any signs)
    centers, axes : (N, 3) seed centres (RAS mm) and long axes (any norm)
    length_mm, diameter_mm : capsule size (default 4.5 x 0.8 mm)
    metal_hu : capsule contrast above background (calibrated default)
    background_hu : uniform background (brain-like 35 HU)
    psf_sigma_mm : Gaussian PSF sigma, scalar or (sigma_xy, sigma_z)
    noise_hu : i.i.d. Gaussian noise SD added after the seeds (0 = none)
    saturate_hu : clip ceiling (3071 HU scanner limit); None = no clip
    fine_mm : supersampling step of the local fine grid
    rng_seed : noise seed (``np.random.default_rng``)

    Returns
    -------
    Volume (float32) with ``meta["seed_render"]`` recording the parameters.
    """
    shape = tuple(int(s) for s in shape_kji)
    arr = np.full(shape, float(background_hu), dtype=np.float64)
    add_rendered_seeds(arr, affine, centers, axes, length_mm=length_mm,
                       diameter_mm=diameter_mm, metal_hu=metal_hu,
                       psf_sigma_mm=psf_sigma_mm, fine_mm=fine_mm)
    arr = arr.astype(np.float32)
    if noise_hu and float(noise_hu) > 0.0:
        rng = np.random.default_rng(rng_seed)
        arr += rng.standard_normal(shape, dtype=np.float32) * np.float32(noise_hu)
    if saturate_hu is not None:
        np.minimum(arr, np.float32(saturate_hu), out=arr)
    meta = {
        "modality": "CT",
        "phantom": True,
        "seed_render": {
            "n_seeds": int(np.asarray(centers).reshape(-1, 3).shape[0]),
            "length_mm": float(length_mm), "diameter_mm": float(diameter_mm),
            "metal_hu": float(metal_hu), "background_hu": float(background_hu),
            "psf_sigma_mm": [float(s) for s in _psf_sigmas(psf_sigma_mm)],
            "noise_hu": float(noise_hu or 0.0),
            "saturate_hu": None if saturate_hu is None else float(saturate_hu),
            "fine_mm": float(fine_mm), "rng_seed": rng_seed,
        },
    }
    return Volume(arr, np.asarray(affine, dtype=float).copy(), meta)


# ------------------------------------------------------- volume degradations
def thick_slices(vol, factor):
    """Block-average along k: partial-volume simulation of thick slices.

    ``factor`` consecutive slices become one slab whose centre is the mean of
    the merged slice centres (trailing slices that do not fill a slab are
    dropped).  ``factor == 1`` returns ``vol`` itself.  Moved here from
    ``scripts/validation_spacing.py`` (which re-exports it).
    """
    if factor == 1:
        return vol
    nk = (vol.array.shape[0] // factor) * factor
    arr = vol.array[:nk].reshape(-1, factor, *vol.array.shape[1:]).mean(axis=1)
    affine = vol.affine.copy()
    affine[:3, 2] *= factor
    # new slice centres sit at the mean of the merged slice centres
    affine[:3, 3] += vol.affine[:3, 2] * (factor - 1) / 2.0
    return Volume(arr.astype(np.float32), affine, dict(vol.meta))


def drop_and_interpolate(vol, keep_k, grid="original"):
    """Keep only slices ``keep_k`` and refill the rest like the DICOM loader.

    The fill rule is the loader's own (:func:`gtcore.io.dicom.fill_slice_grid`,
    used by ``_load_series_resampled``): a grid slice within a quarter step of
    a kept slice is a copy of it, every other one is the linear interpolation
    between the bracketing kept slices.  As in the loader the output grid
    spans the first to the last kept slice (slices outside are cropped, the
    affine origin follows).

    grid : ``"original"`` keeps the input slice step, so every dropped slice
        is re-created by interpolation (e.g. keep every other slice of a 1 mm
        rendering -> a 1 mm grid whose odd slices are invented).
        ``"loader"`` uses the loader's own grid step, the median spacing of
        the kept slices (a regular keep pattern then gives a coarser grid with
        nothing interpolated; an irregular one, like the PostOp export,
        interpolates the off-grid slices).

    ``meta["interpolated_k"]`` lists the output k indices that were
    interpolated (same key as the loader), with ``slices_interpolated``,
    ``slices_present``, ``slices_on_grid`` and ``z_gap_interpolated``.
    """
    from ..io.dicom import fill_slice_grid

    nk = vol.array.shape[0]
    keep = np.unique(np.asarray(keep_k, dtype=int).reshape(-1))
    if keep.size < 2:
        raise ValueError("keep at least two slices")
    if keep[0] < 0 or keep[-1] >= nk:
        raise ValueError("keep_k out of range [0, %d)" % nk)
    z = keep.astype(float)                       # positions in input k units
    if grid == "original":
        dz = 1.0
    elif grid == "loader":
        dz = float(np.median(np.diff(z)))
    else:
        raise ValueError("grid must be 'original' or 'loader'")
    n_out = int(round((z[-1] - z[0]) / dz)) + 1
    zt = z[0] + dz * np.arange(n_out)
    out, interpolated = fill_slice_grid(z, vol.array[keep], zt, dz)

    affine = vol.affine.copy()
    affine[:3, 3] = vol.affine[:3, 3] + vol.affine[:3, 2] * z[0]
    affine[:3, 2] = vol.affine[:3, 2] * dz
    meta = dict(vol.meta)
    meta.update({
        "z_gap_interpolated": True,
        "slices_present": int(keep.size),
        "slices_on_grid": int(n_out),
        "slices_interpolated": int(interpolated.sum()),
        "interpolated_k": [int(k) for k in np.flatnonzero(interpolated)],
    })
    return Volume(out, affine, meta)


# ------------------------------------------------------------------ layouts
def random_seed_layout(n, rng, extent_mm, min_sep_mm=8.0,
                       max_nn_mm: Optional[float] = None,
                       max_tries: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    """Random seed centres and axes for the localization harness.

    Centres are continuous uniform positions inside the box
    ``[-extent/2, extent/2]`` (``extent_mm`` scalar or per-axis), so their
    sub-voxel offsets on any grid much finer than the box are uniform; axes
    are uniform on the sphere (normalized Gaussian vectors).

    * ``max_nn_mm is None`` (sparse): dart throwing, every pair at least
      ``min_sep_mm`` apart.
    * ``max_nn_mm`` given (crowded): sequential growth -- each new seed is
      placed at a random direction and a distance uniform in
      ``[min_sep_mm, max_nn_mm]`` from a random already-placed seed (and
      ``>= min_sep_mm`` from all others), so every seed has a neighbour in
      that window, like neighbouring seeds of adjacent tiles.

    ``rng`` is a ``np.random.Generator`` or an int seed.
    Returns ``(centers (n, 3), axes (n, 3))``.
    """
    rng = np.random.default_rng(rng) if not isinstance(
        rng, np.random.Generator) else rng
    half = 0.5 * np.broadcast_to(np.asarray(extent_mm, dtype=float), (3,))
    n = int(n)
    pts = np.zeros((0, 3))
    tries = 0
    while pts.shape[0] < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                "could not place %d seeds (placed %d) in a %s mm box with "
                "min separation %.1f mm" % (n, pts.shape[0],
                                            tuple(2 * half), min_sep_mm))
        if max_nn_mm is None or pts.shape[0] == 0:
            p = rng.uniform(-half, half)
            if max_nn_mm is not None:
                p = p * 0.25                     # start the cluster centrally
        else:
            parent = pts[rng.integers(pts.shape[0])]
            u = rng.standard_normal(3)
            u /= np.linalg.norm(u)
            p = parent + rng.uniform(min_sep_mm, max_nn_mm) * u
            if np.any(np.abs(p) > half):
                continue
        if pts.shape[0] and float(np.min(np.linalg.norm(pts - p, axis=1))) \
                < min_sep_mm:
            continue
        pts = np.vstack([pts, p])
    axes = rng.standard_normal((n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return pts, axes
