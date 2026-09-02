"""Synthetic post-implant head CT phantom with exact ground truth.

This module is the validation bed for the whole pipeline: real post-implant
scans have no ground truth, so every downstream stage (reconstruction,
segmentation, seed localization, tile fitting, TG-43 dose) is checked against
a head that we built ourselves and therefore know exactly.

Geometry conventions (identical to :mod:`gtcore.volume`)
-------------------------------------------------------
- ``array`` is indexed ``[k, j, i]``.
- ``i -> +x (Right)``, ``j -> +y (Anterior)``, ``k -> +z (Superior)``.
- The volume is an isotropic cube of side ``fov_mm`` centred on the RAS origin,
  so ``affine = diag(spacing, spacing, spacing, 1)`` with the translation set to
  ``-(n - 1) * spacing / 2`` on each axis.

Anatomy (concentric ellipsoids, outermost first)
------------------------------------------------
====================  ==================  ======
structure             semi-axes (mm)      HU
====================  ==================  ======
scalp / soft tissue   (78, 90, 82)        45
skull (6 mm shell)    outer = scalp - 5   900
CSF gap (2.5 mm)      inner skull surface 12
brain                 remainder           35
====================  ==================  ======

Surgery, layered on top of that:

- **Craniotomy** -- the skull is removed inside a cone of half-angle
  ``CRANIOTOMY_HALF_ANGLE_DEG`` about the entry direction ``ENTRY_DIR_U0`` and
  replaced by soft tissue. The cone is measured on the *ellipsoid-normalized*
  position ``unit((x/a, y/b, z/c))`` so the opening is a clean patch on the
  skull rather than a smeared oval.
- **Resection cavity** -- a lumpy ellipsoid whose wall radius is modulated by a
  handful of smooth directional bumps (see :class:`_CavityShape`). It holds
  fluid, with a gravity-dependent air pocket in its superior part.
- **Access tract** -- a fluid-filled cylinder from the cavity centre out along
  the entry direction through the craniotomy.
- **GammaTiles** -- ``n_tiles`` 4-seed tiles laid on the cavity wall, away from
  the entry corridor. Each of the four Cs-131 seeds is *individually*
  re-projected onto the lumpy wall, so the tile is a deformed (non-planar)
  quadrilateral, as a real collagen tile is once it conforms to the resection
  bed.

Imaging physics is deliberately minimal but in the right order: seeds are
painted at full metal HU, then a Gaussian PSF is applied, then i.i.d. Gaussian
noise. Optional qualitative FBP streaks can be layered on afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .. import geometry as _geom
from ..volume import Volume

__all__ = [
    "SeedTruth",
    "TileTruth",
    "PhantomTruth",
    "make_head_phantom",
    "SCALP_RADII",
    "SKULL_OUTER_RADII",
    "SKULL_INNER_RADII",
    "BRAIN_RADII",
    "CAVITY_RADII",
    "ENTRY_DIR_U0",
]

# ------------------------------------------------------------------ constants
HU_AIR = -1000.0
HU_CSF = 12.0
HU_FLUID = 18.0
HU_BRAIN = 35.0
HU_SOFT = 45.0
HU_BONE = 900.0
HU_SEED = 8000.0

SCALP_RADII = (78.0, 90.0, 82.0)
SKULL_STANDOFF_MM = 5.0            # scalp thickness above the outer table
SKULL_THICKNESS_MM = 6.0
CSF_GAP_MM = 2.5

SKULL_OUTER_RADII = tuple(r - SKULL_STANDOFF_MM for r in SCALP_RADII)
SKULL_INNER_RADII = tuple(r - SKULL_THICKNESS_MM for r in SKULL_OUTER_RADII)
BRAIN_RADII = tuple(r - CSF_GAP_MM for r in SKULL_INNER_RADII)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return v / n


ENTRY_DIR_U0 = _unit([0.45, 0.25, 0.86])
CRANIOTOMY_HALF_ANGLE_DEG = 16.0

CAVITY_RADII = (20.0, 18.0, 17.0)
CAVITY_DEPTH_MM = 22.0             # cavity centre sits this far inside the brain
CAVITY_N_BUMPS = 5
CAVITY_AMP_RANGE = (-0.10, 0.16)
CAVITY_WIDTH_RANGE = (0.10, 0.28)
CAVITY_AIR_OFFSET_MM = 4.0         # air above cavity-centre z + this

TRACT_RADIUS_MM = 7.0

# Manufactured tile geometry (citations in gtcore.geometry): seeds on a 10 mm
# pitch, so corners at +/- 5 mm in the tangent plane; the seed plane is 3.0 mm
# from the tissue-facing surface (hydrated 2.25-3.75 mm), so each seed centre
# sits 3 mm off the cavity wall, inside the lumen.
TILE_HALF_MM = _geom.SEED_PITCH_MM / 2.0        # 5.0
TILE_POLAR_RANGE_DEG = (100.0, 150.0)
SEED_LENGTH_MM = _geom.SEED_LENGTH_MM           # 4.5
SEED_INSET_MM = _geom.SEED_PLANE_OFFSET_MM      # 3.0

PSF_SIGMA_MM = 0.45


# --------------------------------------------------------------- truth records
@dataclass
class SeedTruth:
    """Exact pose of one Cs-131 seed."""

    seed_id: int
    tile_id: int
    center_ras: np.ndarray          # (3,) mm
    axis_ras: np.ndarray            # (3,) unit vector, seed long axis

    def __post_init__(self):
        self.center_ras = np.asarray(self.center_ras, dtype=float).reshape(3)
        self.axis_ras = np.asarray(self.axis_ras, dtype=float).reshape(3)


@dataclass
class TileTruth:
    """Exact pose of one GammaTile (4-seed full tile or 2-seed half tile)."""

    tile_id: int
    center_ras: np.ndarray          # (3,) mm -- centroid of its seeds
    normal_ras: np.ndarray          # (3,) unit, outward from cavity into the wall
    seed_ids: List[int] = field(default_factory=list)
    kind: str = "full"              # "full" (4 seeds) or "half" (2 seeds)

    def __post_init__(self):
        self.center_ras = np.asarray(self.center_ras, dtype=float).reshape(3)
        self.normal_ras = np.asarray(self.normal_ras, dtype=float).reshape(3)


@dataclass
class PhantomTruth:
    """Everything the generator knows that a real scan would not tell us.

    ``masks`` holds boolean arrays shaped like the volume (``[k, j, i]``),
    sampled **before** the PSF blur and noise:

    ``brain``
        brain ellipsoid minus cavity, tract and seeds.
    ``skull``
        bone shell after the craniotomy has been removed.
    ``cavity``
        resection-cavity lumen (fluid + air pocket) minus the seed voxels, so
        ``cavity`` and ``seeds`` are disjoint.
    ``seeds``
        union of the metal seed capsules.
    """

    seeds: List[SeedTruth] = field(default_factory=list)
    tiles: List[TileTruth] = field(default_factory=list)
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    cavity_center_ras: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.cavity_center_ras = np.asarray(
            self.cavity_center_ras, dtype=float
        ).reshape(3)


# ------------------------------------------------------------- lumpy wall model
class _CavityShape:
    """Star-shaped lumpy cavity wall: ``r(dir)`` for unit directions ``dir``.

    ``r(d) = r_ellipsoid(d) * (1 + sum_m amp_m * exp((d . d_m - 1) / w_m))``

    The bump sum is clipped to a single bump's amplitude range so that several
    overlapping bumps can never inflate the cavity out through the brain
    surface -- the phantom must stay anatomically valid for every seed.
    """

    def __init__(self, radii, rng, n_bumps=CAVITY_N_BUMPS):
        self.radii = np.asarray(radii, dtype=float).reshape(3)
        dirs = rng.standard_normal((n_bumps, 3))
        self.bump_dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        self.bump_amps = rng.uniform(CAVITY_AMP_RANGE[0], CAVITY_AMP_RANGE[1], n_bumps)
        self.bump_widths = rng.uniform(
            CAVITY_WIDTH_RANGE[0], CAVITY_WIDTH_RANGE[1], n_bumps
        )

    def radius(self, dirs):
        """Wall radius (mm) for unit direction(s); accepts ``(3,)`` or ``(N, 3)``."""
        d = np.asarray(dirs, dtype=np.float32)
        single = d.ndim == 1
        d = np.atleast_2d(d)
        r_ell = 1.0 / np.sqrt(((d / self.radii) ** 2).sum(axis=1))
        mod = np.zeros(d.shape[0], dtype=np.float64)
        for amp, bd, w in zip(self.bump_amps, self.bump_dirs, self.bump_widths):
            mod += amp * np.exp((d @ bd - 1.0) / w)
        np.clip(mod, CAVITY_AMP_RANGE[0], CAVITY_AMP_RANGE[1], out=mod)
        out = r_ell * (1.0 + mod)
        return float(out[0]) if single else out

    @property
    def max_radius(self):
        return float(self.radii.max() * (1.0 + CAVITY_AMP_RANGE[1]))


# -------------------------------------------------------------- grid utilities
def _axes(n, spacing):
    """1-D voxel-centre coordinates for one axis of an origin-centred cube."""
    return ((np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * spacing).astype(
        np.float32
    )


def _quad(az, ay, ax, radii):
    """``(x/a)^2 + (y/b)^2 + (z/c)^2`` as a float32 volume ``[k, j, i]``."""
    a, b, c = radii
    qx = (ax / a) ** 2
    qy = (ay / b) ** 2
    qz = (az / c) ** 2
    return qz[:, None, None] + qy[None, :, None] + qx[None, None, :]


def _sub_slice(axis, lo, hi):
    """Index slice covering ``[lo, hi]`` of a monotonically increasing axis."""
    i0 = int(np.searchsorted(axis, lo, side="left"))
    i1 = int(np.searchsorted(axis, hi, side="right"))
    i0 = max(0, min(i0, axis.size))
    i1 = max(i0, min(i1, axis.size))
    return slice(i0, i1)


def _box_slices(az, ay, ax, center, half_extent):
    """Sub-box slices ``(sk, sj, si)`` around a RAS centre."""
    cx, cy, cz = center
    h = half_extent
    return (
        _sub_slice(az, cz - h, cz + h),
        _sub_slice(ay, cy - h, cy + h),
        _sub_slice(ax, cx - h, cx + h),
    )


def _rel_coords(az, ay, ax, slices, center):
    """Per-axis offsets from ``center`` inside a sub-box."""
    sk, sj, si = slices
    return (
        (az[sk] - center[2]).astype(np.float32),
        (ay[sj] - center[1]).astype(np.float32),
        (ax[si] - center[0]).astype(np.float32),
    )


def _nearest_index(az, ay, ax, point):
    """Nearest voxel index ``(k, j, i)`` to a RAS point, clipped to the grid."""
    k = int(np.clip(np.round((point[2] - az[0]) / (az[1] - az[0])), 0, az.size - 1))
    j = int(np.clip(np.round((point[1] - ay[0]) / (ay[1] - ay[0])), 0, ay.size - 1))
    i = int(np.clip(np.round((point[0] - ax[0]) / (ax[1] - ax[0])), 0, ax.size - 1))
    return k, j, i


# ------------------------------------------------------------------ rasterizers
def _rasterize_cavity(shape_out, az, ay, ax, shape, center):
    """Boolean mask of the lumpy cavity lumen, rasterized in a sub-box."""
    mask = np.zeros(shape_out, dtype=bool)
    half = shape.max_radius + 2.0
    slices = _box_slices(az, ay, ax, center, half)
    dz, dy, dx = _rel_coords(az, ay, ax, slices, center)
    if dz.size == 0 or dy.size == 0 or dx.size == 0:
        return mask

    nk, nj, ni = dz.size, dy.size, dx.size
    rr = np.sqrt(
        dz[:, None, None] ** 2 + dy[None, :, None] ** 2 + dx[None, None, :] ** 2
    )
    np.maximum(rr, 1e-6, out=rr)

    # Single code path with the seed placement: normalize, then ask the shape.
    dirs = np.empty((nk * nj * ni, 3), dtype=np.float32)
    dirs[:, 0] = np.broadcast_to(dx[None, None, :], (nk, nj, ni)).reshape(-1)
    dirs[:, 1] = np.broadcast_to(dy[None, :, None], (nk, nj, ni)).reshape(-1)
    dirs[:, 2] = np.broadcast_to(dz[:, None, None], (nk, nj, ni)).reshape(-1)
    flat_rr = rr.reshape(-1)
    dirs /= flat_rr[:, None]

    wall = shape.radius(dirs).reshape(nk, nj, ni)
    del dirs
    mask[slices] = rr < wall
    del rr, wall
    return mask


def _paint_capsule(mask, az, ay, ax, center, axis, length, radius):
    """OR a capsule (cylinder with hemispherical caps) into ``mask``."""
    half_len = max(0.0, 0.5 * length - radius)
    a = np.asarray(center, dtype=float) - half_len * np.asarray(axis, dtype=float)
    b = np.asarray(center, dtype=float) + half_len * np.asarray(axis, dtype=float)

    slices = _box_slices(az, ay, ax, center, half_len + radius + 2.0)
    sk, sj, si = slices
    if (sk.stop - sk.start) and (sj.stop - sj.start) and (si.stop - si.start):
        zz = az[sk][:, None, None]
        yy = ay[sj][None, :, None]
        xx = ax[si][None, None, :]
        ab = b - a
        ab2 = float(ab @ ab)
        px = xx - a[0]
        py = yy - a[1]
        pz = zz - a[2]
        if ab2 > 0.0:
            t = (px * ab[0] + py * ab[1] + pz * ab[2]) / ab2
            np.clip(t, 0.0, 1.0, out=t)
            px = px - t * ab[0]
            py = py - t * ab[1]
            pz = pz - t * ab[2]
        d2 = px * px + py * py + pz * pz
        mask[slices] |= d2 <= radius * radius

    # A sub-voxel-thin seed can otherwise miss every voxel centre; guarantee at
    # least the voxel containing the seed centre so the truth mask is never empty.
    mask[_nearest_index(az, ay, ax, center)] = True


# ------------------------------------------------------------------ tile layout
def _frame_about(u0):
    """Two unit vectors completing a right-handed frame with ``u0``."""
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(helper @ u0)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(u0, helper))
    e2 = _unit(np.cross(u0, e1))
    return e1, e2


# Seeds of a newly placed tile must clear every already-placed seed by this
# much: two collagen tiles cannot occupy the same patch of wall, and a
# phantom whose tiles overlap has no meaningful truth partition (their CT
# blooms would merge as well).  Full tiles keep their historical first draw
# and only re-draw on a collision, so every phantom that never collided is
# reproduced bit-for-bit.
FULL_TILE_CLEARANCE_MM = 7.0
FULL_TILE_MAX_TRIES = 40
HALF_TILE_CLEARANCE_MM = 7.0
HALF_TILE_MAX_TRIES = 80


def _min_clearance(existing_seeds, centers):
    if not existing_seeds:
        return np.inf
    existing = np.asarray([s.center_ras for s in existing_seeds])
    return min(float(np.linalg.norm(existing - c, axis=1).min())
               for c in centers)


def _build_tiles(shape, cav_c, u0, n_tiles, rng, n_half_tiles=0):
    """Place ``n_tiles`` deformed 4-seed tiles (plus optional 2-seed half
    tiles) on the cavity wall.

    Both kinds are rejection-sampled so that no seed lands within the tile
    clearance of an already-placed seed (overlapping tiles are physically
    impossible and have no meaningful truth partition).  A full tile keeps
    its first draw whenever that draw already clears, so phantoms whose tiles
    never collided are reproduced bit-for-bit; half tiles are appended
    afterwards.
    """
    e1, e2 = _frame_about(u0)
    seeds = []
    tiles = []
    seed_id = 0
    for tile_id in range(n_tiles):
        best = None  # (clearance, centers, axes)
        for _ in range(FULL_TILE_MAX_TRIES):
            theta = np.deg2rad(rng.uniform(*TILE_POLAR_RANGE_DEG))
            phi = 2.0 * np.pi * tile_id / max(1, n_tiles) + rng.uniform(
                -0.4, 0.4
            ) * (2.0 * np.pi / max(1, n_tiles))
            d = _unit(
                np.cos(theta) * u0
                + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
            )
            w = cav_c + d * shape.radius(d)
            t1 = _unit(np.cross(d, u0))
            t2 = _unit(np.cross(d, t1))

            centers = []
            axes = []
            for sa in (-TILE_HALF_MM, TILE_HALF_MM):
                for sb in (-TILE_HALF_MM, TILE_HALF_MM):
                    # Each corner is re-projected onto the lumpy wall on its
                    # own: that is what deforms the tile away from a flat
                    # square.
                    dir_s = _unit(w + sa * t1 + sb * t2 - cav_c)
                    center = cav_c + dir_s * (shape.radius(dir_s) - SEED_INSET_MM)
                    axis = _unit(t1 - float(t1 @ dir_s) * dir_s)
                    centers.append(center)
                    axes.append(axis)
            clearance = _min_clearance(seeds, centers)
            if best is None or clearance > best[0]:
                best = (clearance, centers, axes)
            if clearance >= FULL_TILE_CLEARANCE_MM:
                break

        _, centers, axes = best
        ids = []
        for center, axis in zip(centers, axes):
            seeds.append(
                SeedTruth(
                    seed_id=seed_id,
                    tile_id=tile_id,
                    center_ras=center,
                    axis_ras=axis,
                )
            )
            ids.append(seed_id)
            seed_id += 1

        tile_center = np.mean(np.asarray(centers), axis=0)
        tiles.append(
            TileTruth(
                tile_id=tile_id,
                center_ras=tile_center,
                normal_ras=_unit(tile_center - cav_c),
                seed_ids=ids,
            )
        )

    # ---------------------------------------------------------- half tiles
    # A surgeon-cut 2x1 half tile carries one column of the seed grid: two
    # seeds 2 * TILE_HALF_MM apart along t2, axes along t1 -- exactly the
    # full-tile corner logic with sa fixed at 0.
    for half_idx in range(n_half_tiles):
        tile_id = n_tiles + half_idx
        best = None  # (min clearance, centers, axes)
        for _ in range(HALF_TILE_MAX_TRIES):
            theta = np.deg2rad(rng.uniform(*TILE_POLAR_RANGE_DEG))
            phi = rng.uniform(0.0, 2.0 * np.pi)
            d = _unit(
                np.cos(theta) * u0
                + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
            )
            w = cav_c + d * shape.radius(d)
            t1 = _unit(np.cross(d, u0))
            t2 = _unit(np.cross(d, t1))

            centers = []
            axes = []
            for sb in (-TILE_HALF_MM, TILE_HALF_MM):
                dir_s = _unit(w + sb * t2 - cav_c)
                center = cav_c + dir_s * (shape.radius(dir_s) - SEED_INSET_MM)
                axis = _unit(t1 - float(t1 @ dir_s) * dir_s)
                centers.append(center)
                axes.append(axis)

            clearance = _min_clearance(seeds, centers)
            if best is None or clearance > best[0]:
                best = (clearance, centers, axes)
            if clearance >= HALF_TILE_CLEARANCE_MM:
                break

        _, centers, axes = best
        ids = []
        for center, axis in zip(centers, axes):
            seeds.append(
                SeedTruth(
                    seed_id=seed_id,
                    tile_id=tile_id,
                    center_ras=center,
                    axis_ras=axis,
                )
            )
            ids.append(seed_id)
            seed_id += 1

        tile_center = np.mean(np.asarray(centers), axis=0)
        tiles.append(
            TileTruth(
                tile_id=tile_id,
                center_ras=tile_center,
                normal_ras=_unit(tile_center - cav_c),
                seed_ids=ids,
                kind="half",
            )
        )
    return seeds, tiles


# --------------------------------------------------------------------- streaks
def _add_metal_streaks(hu, seed_mask, spacing, amplitude=35.0):
    """Qualitative FBP metal streaks (NOT a physical beam-hardening model).

    A per-slice Radon transform of the metal mask is passed through a ``tanh``
    saturation, the residual is back-projected, and the result is scaled to a
    fixed HU amplitude. It reproduces the *appearance* of dark/bright bands
    radiating from dense seeds so segmentation code can be stress-tested; it is
    not quantitatively meaningful.
    """
    try:
        from skimage.transform import iradon, radon
    except Exception:
        return False

    slices = np.nonzero(seed_mask.any(axis=(1, 2)))[0]
    if slices.size == 0:
        return False
    theta = np.linspace(0.0, 180.0, 48, endpoint=False)
    ok = False
    for k in slices:
        try:
            metal = seed_mask[k].astype(np.float32)
            p = radon(metal, theta=theta, circle=False)
            scale = float(p.max())
            if scale <= 0.0:
                continue
            resid = scale * np.tanh(p / scale) - p
            art = iradon(
                resid, theta=theta, circle=False, filter_name="ramp",
                output_size=metal.shape[0],
            ).astype(np.float32)
            peak = float(np.abs(art).max())
            if peak <= 0.0:
                continue
            hu[k] += art * (amplitude / peak)
            ok = True
        except Exception:
            return ok
    return ok


# ------------------------------------------------------------------------- main
def make_head_phantom(
    spacing=0.7,
    n_tiles=3,
    noise_hu=4.0,
    streaks=False,
    rng_seed=0,
    fov_mm=200.0,
    n_half_tiles=0,
):
    """Build a synthetic post-implant head CT and its exact ground truth.

    Parameters
    ----------
    spacing : float
        Isotropic voxel size in mm.
    n_tiles : int
        Number of 4-seed GammaTiles on the cavity wall.
    n_half_tiles : int
        Number of surgeon-cut 2-seed half tiles, placed after the full tiles
        at rejection-sampled wall directions.  The default of 0 reproduces
        historical phantoms exactly (identical volume and truth).
    noise_hu : float
        Standard deviation of the additive Gaussian noise, in HU.
    streaks : bool
        Add qualitative FBP metal streaks (see :func:`_add_metal_streaks`).
    rng_seed : int
        Seed for ``np.random.default_rng``; identical seeds give identical
        volumes and identical truth.
    fov_mm : float
        Side of the cubic field of view in mm.

    Returns
    -------
    (Volume, PhantomTruth)
    """
    rng = np.random.default_rng(rng_seed)
    n = int(round(fov_mm / spacing))
    shape = (n, n, n)

    ax = _axes(n, spacing)          # x, along i
    ay = _axes(n, spacing)          # y, along j
    az = _axes(n, spacing)          # z, along k

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = spacing
    affine[:3, 3] = -(n - 1) * spacing / 2.0

    u0 = ENTRY_DIR_U0
    hu = np.full(shape, np.float32(HU_AIR), dtype=np.float32)

    # -- scalp + craniotomy cone -------------------------------------------
    q = _quad(az, ay, ax, SCALP_RADII)
    hu[q <= 1.0] = np.float32(HU_SOFT)
    # Angular deviation of the ellipsoid-normalized position from u0.
    a, b, c = SCALP_RADII
    num = (
        (az * (u0[2] / c))[:, None, None]
        + (ay * (u0[1] / b))[None, :, None]
        + (ax * (u0[0] / a))[None, None, :]
    )
    np.sqrt(q, out=q)
    q *= np.float32(np.cos(np.deg2rad(CRANIOTOMY_HALF_ANGLE_DEG)))
    cran_cone = num > q
    del num, q

    # -- skull shell --------------------------------------------------------
    q = _quad(az, ay, ax, SKULL_OUTER_RADII)
    skull = q <= 1.0
    hu[skull] = np.float32(HU_BONE)
    del q
    q = _quad(az, ay, ax, SKULL_INNER_RADII)
    inner = q <= 1.0
    hu[inner] = np.float32(HU_CSF)
    skull &= ~inner
    del q, inner

    # -- brain --------------------------------------------------------------
    q = _quad(az, ay, ax, BRAIN_RADII)
    brain = q <= 1.0
    hu[brain] = np.float32(HU_BRAIN)
    del q

    # -- craniotomy: bone flap out, soft tissue in --------------------------
    flap = skull & cran_cone
    hu[flap] = np.float32(HU_SOFT)
    skull &= ~cran_cone
    del flap, cran_cone

    # -- resection cavity ---------------------------------------------------
    cav_shape = _CavityShape(CAVITY_RADII, rng)
    brain_r_along_u0 = 1.0 / np.sqrt(((u0 / np.asarray(BRAIN_RADII)) ** 2).sum())
    cav_c = u0 * (brain_r_along_u0 - CAVITY_DEPTH_MM)

    cavity = _rasterize_cavity(shape, az, ay, ax, cav_shape, cav_c)
    hu[cavity] = np.float32(HU_FLUID)
    # Gravity-dependent air pocket in the superior part of the lumen.
    z_air = cav_c[2] + CAVITY_AIR_OFFSET_MM
    k_air = int(np.searchsorted(az, z_air, side="right"))
    if k_air < n:
        hu[k_air:][cavity[k_air:]] = np.float32(HU_AIR)

    # -- access tract -------------------------------------------------------
    tract_len = float(max(SCALP_RADII)) + 10.0 - float(np.linalg.norm(cav_c))
    tract = np.zeros(shape, dtype=bool)
    mid = cav_c + 0.5 * tract_len * u0
    half = 0.5 * tract_len + TRACT_RADIUS_MM + 2.0
    slices = _box_slices(az, ay, ax, mid, half)
    dz, dy, dx = _rel_coords(az, ay, ax, slices, cav_c)
    if dz.size and dy.size and dx.size:
        s = (
            (dz * u0[2])[:, None, None]
            + (dy * u0[1])[None, :, None]
            + (dx * u0[0])[None, None, :]
        )
        r2 = (
            dz[:, None, None] ** 2 + dy[None, :, None] ** 2 + dx[None, None, :] ** 2
        )
        r2 -= s * s
        inside = (s >= 0.0) & (s <= tract_len) & (r2 <= TRACT_RADIUS_MM ** 2)
        del s, r2
        # Only carve where there was tissue: never turn outside-air or the
        # cavity's air pocket into fluid.
        inside &= hu[slices] > -500.0
        tract[slices] = inside
        del inside
    hu[tract] = np.float32(HU_FLUID)

    # -- tiles and seeds ----------------------------------------------------
    seeds, tiles = _build_tiles(
        cav_shape, cav_c, u0, n_tiles, rng, n_half_tiles=n_half_tiles
    )
    seed_radius = max(0.4, 0.75 * spacing)
    seed_mask = np.zeros(shape, dtype=bool)
    for s in seeds:
        _paint_capsule(
            seed_mask, az, ay, ax, s.center_ras, s.axis_ras,
            SEED_LENGTH_MM, seed_radius,
        )
    hu[seed_mask] = np.float32(HU_SEED)

    # -- imaging physics ----------------------------------------------------
    sigma_vox = PSF_SIGMA_MM / float(spacing)
    if sigma_vox > 0.0:
        ndimage.gaussian_filter(hu, sigma=sigma_vox, output=hu, mode="nearest")
    if streaks:
        _add_metal_streaks(hu, seed_mask, spacing)
    if noise_hu and noise_hu > 0.0:
        hu += rng.standard_normal(shape, dtype=np.float32) * np.float32(noise_hu)

    # -- truth --------------------------------------------------------------
    brain &= ~cavity
    brain &= ~tract
    brain &= ~seed_mask
    cavity_lumen = cavity & ~seed_mask
    del cavity, tract

    truth = PhantomTruth(
        seeds=seeds,
        tiles=tiles,
        masks={
            "brain": brain,
            "skull": skull,
            "cavity": cavity_lumen,
            "seeds": seed_mask,
        },
        cavity_center_ras=cav_c,
    )
    meta = {"phantom": True, "modality": "CT", "n_tiles": int(n_tiles)}
    if n_half_tiles:
        meta["n_half_tiles"] = int(n_half_tiles)
    vol = Volume(hu, affine, meta)
    return vol, truth
