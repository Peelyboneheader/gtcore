"""Inter-seed and tile-carrier interference for the TG-43 dose engine.

Why this module exists
----------------------
TG-43U1 superposition -- what :func:`gtcore.dose.engine.compute_dose_grid`
does by default -- adds up single-seed dose distributions, each computed for
**one seed alone at the centre of an unbounded water phantom**.  A GammaTile
implant violates that assumption twice over:

1. **Inter-seed attenuation.**  Every other seed in the implant is a
   4.5 x 0.8 mm titanium can wrapped around a ceramic pellet.  At the Cs-131
   effective photon energy (~30.4 keV) titanium is an aggressive absorber
   (mu ~ 45 cm^-1), so a seed sitting in the line of sight removes a large
   fraction of the primary fluence behind it.  With four seeds on a 10 mm
   grid and tiles laid edge to edge on a cavity wall, those shadows fall
   inside the treated volume.
2. **The collagen carrier.**  The 20 x 20 x 4 mm bioresorbable tile is
   *less* dense than the water TG-43 assumes, so it attenuates less -- rays
   crossing a carrier are slightly *hotter* than the water-only calculation
   says, partially offsetting the seed shadows.

Both effects are first order in the same quantity -- the excess attenuation
of the material relative to the water it displaces -- so they are modelled
together here, as one line-of-sight Beer-Lambert correction applied to the
per-seed dose rate before superposition::

    D(p) = sum_s  Drate_TG43(p; s) * T(p; s) * S_K,s * tau

    T(p; s) = exp( - sum_j (mu_j - mu_water) * l_j(s -> p) )

``l_j`` is the length of the segment from seed ``s`` to field point ``p``
that lies inside occluder ``j``, computed analytically (segment vs finite
cylinder, segment vs oriented box).  The source seed's own capsule is
excluded: its self-absorption is already baked into the measured TG-43
anisotropy function ``F(r, theta)``.

Scope, and honesty about the model
----------------------------------
This is a **primary-fluence** correction, the standard first-order treatment
of interseed attenuation.  It deliberately does not model:

- scatter rebuilding dose behind an occluder (a full Monte-Carlo or
  grid-based-Boltzmann job).  Ignoring it makes the correction an
  *over*-estimate of the shadow depth, increasingly so with distance, where
  the scatter fraction grows;
- spectral hardening through titanium (a single effective mu is used);
- the tissue/air/bone heterogeneity of the head itself, which is a separate
  correction of comparable size and is not attempted anywhere in ``gtcore``.

Every coefficient below is an explicit, named, overridable constant with its
provenance recorded, in the same spirit as the ``S_K`` handling in
``engine.py``; see ``docs/interference-notes.md``.  The carrier density in
particular is *provisional* and should be replaced with a vendor or
Monte-Carlo value before absolute doses are quoted.

Geometry conventions match the rest of ``gtcore``: positions in RAS
millimetres, attenuation coefficients in cm^-1, path lengths converted
mm -> cm exactly once, where the exponent is formed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CS131_EFFECTIVE_KEV",
    "MU_WATER_CM1",
    "MU_TITANIUM_CM1",
    "MU_SEED_CORE_CM1",
    "TILE_CARRIER_DENSITY_G_CM3",
    "CAPSULE_OUTER_RADIUS_MM",
    "CAPSULE_WALL_MM",
    "CAPSULE_LENGTH_MM",
    "ACTIVE_LENGTH_MM",
    "TILE_THICKNESS_MM",
    "TILE_HALF_EXTENTS_MM",
    "SeedCapsule",
    "TileCarrier",
    "InterferenceModel",
    "interference_report",
]

# ------------------------------------------------------------------- physics
#: Effective photon energy of Cs-131 [keV].  Cs-131 decays by electron
#: capture to Xe-131, emitting Xe K x-rays at 29.5-34.4 keV; the
#: fluence-weighted mean quoted for TG-43 work is ~30.4 keV.
CS131_EFFECTIVE_KEV = 30.4

#: Linear attenuation coefficient of liquid water at 30 keV [cm^-1].
#: NIST XCOM mu/rho = 0.3756 cm^2/g, rho = 1.000 g/cm^3.
MU_WATER_CM1 = 0.3756

#: Titanium capsule wall at 30 keV [cm^-1].  NIST XCOM mu/rho ~ 9.90 cm^2/g,
#: rho = 4.506 g/cm^3 -> 44.6 cm^-1.  Titanium's K edge is at 4.97 keV, well
#: below the Cs-131 spectrum, so a single effective mu is defensible here.
MU_TITANIUM_CM1 = 44.6

#: Seed core (Cs-adsorbed alumina ceramic pellet) at 30 keV [cm^-1].
#: Al2O3 mass-weighted mu/rho = 0.5293*1.128 + 0.4707*0.378 = 0.775 cm^2/g;
#: an effective pellet density of 3.0 g/cm^3 (dense alumina is 3.97, the
#: sintered CS-1 pellet is porous) gives 2.33 cm^-1.
MU_SEED_CORE_CM1 = 2.33

#: PROVISIONAL effective density of the collagen tile carrier [g/cm^3],
#: treated as density-scaled water.  This is a *dry-sponge* figure, and it is
#: the single least trustworthy number in this module.
#:
#: It matters more than it looks.  Seeds sit on the carrier's mid-plane, so
#: rays travelling within the plane of a tile -- exactly the directions where
#: the neighbouring seeds and the high-dose region are -- run up to 20 mm
#: lengthwise through it.  At 0.30 g/cm^3 that is a +15% dose boost near the
#: prescription isodose, i.e. *larger, and opposite in sign, to the interseed
#: shadow it is meant to partially offset*.  An implanted carrier soaked in
#: blood and CSF is probably far closer to water-equivalent (density 1.0,
#: where this whole term vanishes).
#:
#: Because the answer swings between "+15%" and "nothing" on a number nobody
#: here has measured, carriers are **opt-in**: ``InterferenceModel.
#: from_implant`` leaves them out unless asked.  Replace this with vendor or
#: Monte-Carlo data before enabling them.  See docs/interference-notes.md.
TILE_CARRIER_DENSITY_G_CM3 = 0.30

# ------------------------------------------------------------------ geometry
#: IsoRay Proxcelan CS-1 Rev2 capsule: 0.8 mm outer diameter x 4.5 mm long.
CAPSULE_OUTER_RADIUS_MM = 0.40
#: Titanium wall thickness [mm] (0.05 mm is the CS-1 nominal wall).
CAPSULE_WALL_MM = 0.05
#: Physical capsule length [mm]; matches ``TG43Engine._CAP_HALF_CM * 2``.
CAPSULE_LENGTH_MM = 4.5
#: TG-43 *active* length [mm]; matches ``TG43Engine._L`` (0.40 cm).
ACTIVE_LENGTH_MM = 4.0
#: GammaTile carrier thickness [mm]; seeds sit on its mid-plane.
TILE_THICKNESS_MM = 4.0
#: In-plane half extents [mm] of the carrier footprint, per tile kind.
TILE_HALF_EXTENTS_MM = {"full": (10.0, 10.0), "half": (5.0, 10.0)}

_TINY = 1.0e-12
_MM_TO_CM = 0.1
#: Cap on the (points x capsules) scratch arrays the bounding-sphere prune
#: builds, in elements.  4M float64 is ~32 MB per array, so capsules are
#: pruned in blocks and peak memory no longer scales with the implant size.
_PRUNE_BLOCK_ELEMENTS = 4_000_000


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < _TINY:
        raise ValueError("cannot normalize a zero vector")
    return v / n


def _orthonormal_tangent(normal, hint=None):
    """A unit vector perpendicular to ``normal``, using ``hint`` if usable."""
    n = _unit(normal)
    if hint is not None:
        h = np.asarray(hint, dtype=float).reshape(3)
        t = h - float(h @ n) * n
        if np.linalg.norm(t) > 1.0e-6:
            return t / np.linalg.norm(t)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(helper @ n)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    t = np.cross(n, helper)
    return t / np.linalg.norm(t)


# ---------------------------------------------------------------- occluders
@dataclass
class SeedCapsule:
    """One seed modelled as coaxial cylinders: titanium wall around a core.

    The path through the wall is obtained as (path through the outer
    cylinder) minus (path through the inner cylinder), which is exact for
    coaxial cylinders and costs one extra intersection.
    """

    center_ras: np.ndarray
    axis_ras: np.ndarray
    outer_radius_mm: float = CAPSULE_OUTER_RADIUS_MM
    wall_mm: float = CAPSULE_WALL_MM
    length_mm: float = CAPSULE_LENGTH_MM
    mu_wall_cm1: float = MU_TITANIUM_CM1
    mu_core_cm1: float = MU_SEED_CORE_CM1

    def __post_init__(self):
        self.center_ras = np.asarray(self.center_ras, dtype=float).reshape(3)
        self.axis_ras = _unit(self.axis_ras)
        self.outer_radius_mm = float(self.outer_radius_mm)
        self.wall_mm = float(self.wall_mm)
        self.length_mm = float(self.length_mm)
        if self.outer_radius_mm <= 0.0 or self.length_mm <= 0.0:
            raise ValueError("capsule dimensions must be positive")
        if not 0.0 < self.wall_mm < self.outer_radius_mm:
            raise ValueError("wall_mm must lie in (0, outer_radius_mm)")

    @property
    def core_radius_mm(self):
        return self.outer_radius_mm - self.wall_mm

    @property
    def core_length_mm(self):
        return max(self.length_mm - 2.0 * self.wall_mm, 0.0)

    @property
    def bounding_radius_mm(self):
        return math.hypot(self.outer_radius_mm, 0.5 * self.length_mm)


@dataclass
class TileCarrier:
    """The collagen tile body, modelled as an oriented box (a flat slab).

    Real tiles drape onto the cavity wall, so this is a planar approximation
    of a gently curved 20 x 20 x 4 mm slab.  Over a 20 mm span on a cavity of
    ~15 mm radius the sagitta is well under a millimetre, small compared with
    the 4 mm thickness the ray actually integrates through.
    """

    center_ras: np.ndarray
    normal_ras: np.ndarray
    axis_ras: np.ndarray = None
    half_extents_mm: tuple = (10.0, 10.0)
    thickness_mm: float = TILE_THICKNESS_MM
    mu_cm1: float = field(
        default_factory=lambda: MU_WATER_CM1 * TILE_CARRIER_DENSITY_G_CM3)

    def __post_init__(self):
        self.center_ras = np.asarray(self.center_ras, dtype=float).reshape(3)
        n = _unit(self.normal_ras)
        t1 = _orthonormal_tangent(n, self.axis_ras)
        t2 = np.cross(n, t1)
        self.normal_ras = n
        self.axis_ras = t1
        #: rows are the box's orthonormal axes, matching ``_half``.
        self.frame = np.vstack([t1, t2, n])
        h = tuple(float(x) for x in self.half_extents_mm)
        self.thickness_mm = float(self.thickness_mm)
        if len(h) != 2 or min(h) <= 0.0 or self.thickness_mm <= 0.0:
            raise ValueError("carrier needs positive (u, v) half extents and "
                             "a positive thickness")
        self.half_extents_mm = h
        self.half = np.array([h[0], h[1], 0.5 * self.thickness_mm])

    @property
    def bounding_radius_mm(self):
        return float(np.linalg.norm(self.half))

    @classmethod
    def from_tile(cls, tile, carrier_density_g_cm3=TILE_CARRIER_DENSITY_G_CM3,
                  thickness_mm=TILE_THICKNESS_MM):
        """Build a carrier from a placed or inferred tile.

        Accepts anything exposing ``normal_ras`` plus either ``seed_centers``
        (a :class:`gtcore.interact.PlacedTile`) or ``center_ras`` (the
        phantom's ``TileTruth``); ``axis_ras`` and ``kind`` are used when
        present.  The carrier is centred on the seed centroid because seeds
        sit on the mid-plane of the 4 mm slab.
        """
        seeds = getattr(tile, "seed_centers", None)
        if seeds is not None and np.asarray(seeds).size:
            center = np.asarray(seeds, dtype=float).reshape(-1, 3).mean(axis=0)
        else:
            center = np.asarray(getattr(tile, "center_ras"), dtype=float)
        kind = str(getattr(tile, "kind", "full"))
        half = TILE_HALF_EXTENTS_MM.get(kind, TILE_HALF_EXTENTS_MM["full"])
        return cls(
            center_ras=center,
            normal_ras=getattr(tile, "normal_ras"),
            axis_ras=getattr(tile, "axis_ras", None),
            half_extents_mm=half,
            thickness_mm=thickness_mm,
            mu_cm1=MU_WATER_CM1 * float(carrier_density_g_cm3),
        )


# ----------------------------------------------- segment/solid intersections
def segment_cylinder_length_mm(origin, targets, center, axis, radius_mm,
                               half_length_mm):
    """Length [mm] of each segment ``origin -> targets[i]`` inside a cylinder.

    Finite cylinder of ``radius_mm`` and ``half_length_mm`` about ``center``
    with unit ``axis``.  Vectorized over ``targets`` (P, 3); returns (P,).
    Purely analytic: the radial quadratic intersected with the axial slab,
    clipped to the segment parameter range [0, 1].
    """
    o = np.asarray(origin, dtype=float).reshape(3)
    tgt = np.atleast_2d(np.asarray(targets, dtype=float))
    d = tgt - o[None, :]
    seg_len = np.sqrt(np.einsum("ij,ij->i", d, d))

    rel = o - np.asarray(center, dtype=float).reshape(3)
    a = np.asarray(axis, dtype=float).reshape(3)

    d_a = d @ a
    o_a = float(rel @ a)
    d_perp = d - d_a[:, None] * a[None, :]
    o_perp = rel - o_a * a

    A = np.einsum("ij,ij->i", d_perp, d_perp)
    B = 2.0 * (d_perp @ o_perp)
    C = float(o_perp @ o_perp) - radius_mm * radius_mm

    n = tgt.shape[0]
    t_lo = np.zeros(n)
    t_hi = np.ones(n)
    empty = np.zeros(n, dtype=bool)

    # radial constraint
    parallel = A <= _TINY
    disc = B * B - 4.0 * A * C
    hits = (~parallel) & (disc > 0.0)
    A_safe = np.where(parallel, 1.0, A)
    sq = np.sqrt(np.maximum(disc, 0.0))
    r_lo = (-B - sq) / (2.0 * A_safe)
    r_hi = (-B + sq) / (2.0 * A_safe)
    t_lo = np.where(hits, np.maximum(t_lo, r_lo), t_lo)
    t_hi = np.where(hits, np.minimum(t_hi, r_hi), t_hi)
    empty |= (~parallel) & (~hits)          # misses the infinite cylinder
    empty |= parallel & (C > 0.0)           # runs alongside it, outside

    # axial slab constraint
    ax_par = np.abs(d_a) <= _TINY
    d_a_safe = np.where(ax_par, 1.0, d_a)
    s0 = (-half_length_mm - o_a) / d_a_safe
    s1 = (half_length_mm - o_a) / d_a_safe
    s_lo = np.minimum(s0, s1)
    s_hi = np.maximum(s0, s1)
    t_lo = np.where(ax_par, t_lo, np.maximum(t_lo, s_lo))
    t_hi = np.where(ax_par, t_hi, np.minimum(t_hi, s_hi))
    empty |= ax_par & (abs(o_a) > half_length_mm)

    span = np.where(empty, 0.0, np.maximum(t_hi - t_lo, 0.0))
    return span * seg_len


def segment_box_length_mm(origin, targets, center, frame, half_extents_mm):
    """Length [mm] of each segment ``origin -> targets[i]`` inside an OBB.

    ``frame`` is a (3, 3) array whose ROWS are the box's orthonormal axes and
    ``half_extents_mm`` the matching (3,) half sizes.  Standard slab method,
    vectorized over ``targets``.
    """
    o = np.asarray(origin, dtype=float).reshape(3)
    tgt = np.atleast_2d(np.asarray(targets, dtype=float))
    d = tgt - o[None, :]
    seg_len = np.sqrt(np.einsum("ij,ij->i", d, d))

    rel = o - np.asarray(center, dtype=float).reshape(3)
    R = np.asarray(frame, dtype=float).reshape(3, 3)
    h = np.asarray(half_extents_mm, dtype=float).reshape(3)

    o_l = rel @ R.T
    d_l = d @ R.T

    n = tgt.shape[0]
    t_lo = np.zeros(n)
    t_hi = np.ones(n)
    empty = np.zeros(n, dtype=bool)

    for k in range(3):
        dk = d_l[:, k]
        par = np.abs(dk) <= _TINY
        dk_safe = np.where(par, 1.0, dk)
        s0 = (-h[k] - o_l[k]) / dk_safe
        s1 = (h[k] - o_l[k]) / dk_safe
        s_lo = np.minimum(s0, s1)
        s_hi = np.maximum(s0, s1)
        t_lo = np.where(par, t_lo, np.maximum(t_lo, s_lo))
        t_hi = np.where(par, t_hi, np.minimum(t_hi, s_hi))
        empty |= par & (abs(o_l[k]) > h[k])

    span = np.where(empty, 0.0, np.maximum(t_hi - t_lo, 0.0))
    return span * seg_len


# -------------------------------------------------------------------- model
class InterferenceModel:
    """Line-of-sight interference (attenuation) model for one implant.

    Parameters
    ----------
    capsules : sequence of SeedCapsule
        One per seed, **in the same order as the seed arrays passed to**
        :func:`gtcore.dose.compute_dose_grid`; index ``s`` is skipped when
        tracing rays out of seed ``s``.
    carriers : sequence of TileCarrier
        Collagen tile bodies.  A seed's own carrier is *not* skipped: the ray
        really does leave through it, and TG-43's water assumption does not
        account for that.
    mu_water_cm1 : float
        Attenuation of the displaced medium.  Every occluder contributes
        ``(mu_material - mu_water) * path``, so a carrier lighter than water
        legitimately yields a transmission slightly above 1.
    max_range_mm : float or None
        Rays longer than this get transmission 1.  A seed contributes far
        less than a percent of the 25% isodose at that range, so a shadow
        there changes nothing visible and tracing it is wasted work.
        ``None`` disables the cutoff.
    line_samples : int
        Number of ray origins spread over the TG-43 active length.  1 puts
        the origin at the seed centre (fast, hard shadow edges); an odd
        number >= 3 averages transmission over the active line and gives the
        physically correct penumbra at shadow boundaries.
    """

    def __init__(self, capsules, carriers=(), mu_water_cm1=MU_WATER_CM1,
                 max_range_mm=40.0, line_samples=1):
        self.capsules = list(capsules)
        self.carriers = list(carriers)
        self.mu_water_cm1 = float(mu_water_cm1)

        rng_mm = float("inf") if max_range_mm is None else float(max_range_mm)
        if rng_mm <= 0.0:
            raise ValueError("max_range_mm must be positive (or None)")
        self.max_range_mm = rng_mm

        n_samples = int(line_samples)
        if n_samples < 1:
            raise ValueError("line_samples must be >= 1")
        self.line_samples = n_samples

        if self.capsules:
            self._cap_centers = np.vstack([c.center_ras for c in self.capsules])
            self._cap_bounds = np.array([c.bounding_radius_mm
                                         for c in self.capsules])
        else:
            self._cap_centers = np.zeros((0, 3))
            self._cap_bounds = np.zeros(0)

        # Ray-origin offsets along the active line, symmetric and centred.
        if self.line_samples == 1:
            offs = np.zeros(1)
        else:
            offs = np.linspace(-1.0, 1.0, self.line_samples)
        self._line_offsets = offs * (0.5 * ACTIVE_LENGTH_MM)

    # ---------------------------------------------------------- construction
    @classmethod
    def from_implant(cls, seed_centers, seed_axes, tiles=None,
                     carrier_density_g_cm3=TILE_CARRIER_DENSITY_G_CM3,
                     include_carriers=False, **kwargs):
        """Build a model from the same seed arrays the dose grid will use.

        Seed capsules are always modelled: their geometry and composition are
        known and the effect is a shadow, so leaving them out overestimates
        dose.

        Tile carriers are **off by default** even when ``tiles`` is supplied.
        Their contribution scales directly with an unmeasured density
        (``TILE_CARRIER_DENSITY_G_CM3``) and, at the dry-sponge value, is
        both larger than the interseed shadow and opposite in sign -- a
        correction that could plausibly be nothing at all is not something to
        apply silently.  Pass ``include_carriers=True`` (with a density you
        can defend) to enable them.

        ``tiles`` is a sequence of placed or inferred tiles: anything
        :meth:`TileCarrier.from_tile` accepts.
        """
        centers = np.asarray(seed_centers, dtype=float).reshape(-1, 3)
        axes = np.asarray(seed_axes, dtype=float).reshape(-1, 3)
        if centers.shape != axes.shape:
            raise ValueError("seed_centers and seed_axes must both be (N, 3)")
        capsules = [SeedCapsule(c, a) for c, a in zip(centers, axes)]

        carriers = []
        if include_carriers and tiles is not None and len(tiles):
            carriers = [
                TileCarrier.from_tile(
                    t, carrier_density_g_cm3=carrier_density_g_cm3)
                for t in tiles
            ]
        return cls(capsules, carriers, **kwargs)

    # ------------------------------------------------------------ validation
    def validate_against(self, seed_centers, tol_mm=1.0e-6):
        """Raise unless this model's capsules match ``seed_centers`` in order.

        The whole correction hinges on index ``s`` of the model being the
        same seed as index ``s`` of the dose calculation; otherwise the wrong
        capsule is skipped and a seed shadows itself.  Checked loudly rather
        than assumed.
        """
        centers = np.asarray(seed_centers, dtype=float).reshape(-1, 3)
        if len(self.capsules) != centers.shape[0]:
            raise ValueError(
                "interference model has %d capsules but the dose calculation "
                "has %d seeds" % (len(self.capsules), centers.shape[0]))
        if centers.shape[0] and not np.allclose(self._cap_centers, centers,
                                                atol=tol_mm):
            raise ValueError("interference model capsule centres do not match "
                             "the seed centres passed to the dose engine")
        return True

    # ---------------------------------------------------------- transmission
    def transmission(self, source_index, points_ras):
        """Transmission factor for rays from seed ``source_index`` to points.

        Parameters
        ----------
        source_index : int
            Index into ``capsules`` of the emitting seed; its own capsule is
            excluded.
        points_ras : (P, 3) array
            Field points, RAS mm.

        Returns
        -------
        (P,) ndarray
            Multiplicative factor on the TG-43 dose rate.  Below 1 in a seed
            shadow, slightly above 1 through a sub-water-density carrier.
        """
        idx = int(source_index)
        cap = self.capsules[idx]
        pts = np.atleast_2d(np.asarray(points_ras, dtype=float))
        return self._transmission(pts, cap.center_ras, cap.axis_ras, idx)

    def transmission_cached(self, source_index, points_ras, offsets,
                            distances):
        """As :meth:`transmission`, reusing already-computed ray geometry.

        ``offsets`` is ``points_ras - seed_centre`` and ``distances`` its
        row norms.  The dose grid computes both anyway to get ``r`` and
        ``theta``, so handing them over avoids a second (P, 3) subtraction
        and norm per seed per chunk.
        """
        idx = int(source_index)
        cap = self.capsules[idx]
        pts = np.atleast_2d(np.asarray(points_ras, dtype=float))
        return self._transmission(pts, cap.center_ras, cap.axis_ras, idx,
                                  offsets=offsets, distances=distances)

    def transmission_from_point(self, origin_ras, points_ras, axis_ras=None):
        """Transmission from an arbitrary origin; no occluder is skipped."""
        pts = np.atleast_2d(np.asarray(points_ras, dtype=float))
        origin = np.asarray(origin_ras, dtype=float).reshape(3)
        axis = None if axis_ras is None else _unit(axis_ras)
        return self._transmission(pts, origin, axis, None)

    # ------------------------------------------------------------------ core
    def _transmission(self, pts, origin, axis, skip_index,
                      offsets=None, distances=None):
        """Shared implementation.

        ``offsets`` / ``distances`` are optional precomputed ``pts - origin``
        and its norm: the dose grid already has both, and recomputing a
        (P, 3) difference once per seed is not free.
        """
        n_pts = pts.shape[0]
        out = np.ones(n_pts, dtype=float)
        if n_pts == 0 or (not self.capsules and not self.carriers):
            return out

        if offsets is None:
            offsets = pts - origin[None, :]
        if distances is None:
            distances = np.sqrt(np.einsum("ij,ij->i", offsets, offsets))

        # Only points within range are traced; the rest keep T = 1.
        if np.isfinite(self.max_range_mm):
            near = distances <= self.max_range_mm
            if not near.any():
                return out
            near_pts = pts[near]
            near_off = offsets[near]
            near_dist = distances[near]
        else:
            near = slice(None)
            near_pts, near_off, near_dist = pts, offsets, distances

        # Ray origins along the active line (a single one = the seed centre).
        if self.line_samples > 1 and axis is not None:
            origins = [origin + off * axis for off in self._line_offsets]
        else:
            origins = [origin]

        depth = np.empty((len(origins), near_pts.shape[0]), dtype=float)
        for oi, org in enumerate(origins):
            if len(origins) == 1:
                off, dist = near_off, near_dist
            else:
                off = near_pts - org[None, :]
                dist = np.sqrt(np.einsum("ij,ij->i", off, off))
            depth[oi] = self._optical_depth(org, near_pts, off, dist,
                                            skip_index)

        # Average the TRANSMISSION over the line, not the optical depth:
        # exp() is convex, so averaging depths would systematically
        # over-attenuate the penumbra.
        out[near] = np.exp(-depth).mean(axis=0)
        return out

    def _optical_depth(self, origin, pts, offsets, distances, skip_index):
        """Excess optical depth (dimensionless) for one ray origin."""
        depth = np.zeros(pts.shape[0], dtype=float)

        # -- seed capsules: cheap bounding-sphere prune (one matrix product),
        # exact path only on the survivors.  Shadow subsets are a small
        # fraction of any grid, so this keeps an O(P x N) exact trace from
        # dominating.  The prune is conservative: a ray can only touch the
        # capsule if it passes within the capsule's bounding sphere and the
        # SEGMENT overlaps that sphere's span along the ray.
        #
        # Capsules are pruned in blocks so the (P x N) scratch arrays stay
        # bounded: a 100-seed implant on a full chunk would otherwise
        # allocate hundreds of megabytes per source seed.
        n_pts = pts.shape[0]
        block = max(1, int(_PRUNE_BLOCK_ELEMENTS // max(1, n_pts)))
        for j0 in range(0, len(self.capsules), block):
            j1 = min(len(self.capsules), j0 + block)
            rel = self._cap_centers[j0:j1] - origin[None, :]   # (B, 3)
            d_cc = np.sqrt(np.einsum("ij,ij->i", rel, rel))    # (B,)
            safe = np.maximum(distances, _TINY)[:, None]
            proj = (offsets @ rel.T) / safe                    # (P, B) on-ray
            perp2 = (d_cc * d_cc)[None, :] - proj * proj
            radii = self._cap_bounds[None, j0:j1]
            cand = ((perp2 < radii * radii)
                    & (proj + radii > 0.0)
                    & (distances[:, None] > proj - radii))
            del proj, perp2

            for jj in range(j1 - j0):
                j = j0 + jj
                if skip_index is not None and j == skip_index:
                    continue
                sel = np.flatnonzero(cand[:, jj])
                if sel.size == 0:
                    continue
                cap = self.capsules[j]
                tgt = pts[sel]
                outer = segment_cylinder_length_mm(
                    origin, tgt, cap.center_ras, cap.axis_ras,
                    cap.outer_radius_mm, 0.5 * cap.length_mm)
                core = segment_cylinder_length_mm(
                    origin, tgt, cap.center_ras, cap.axis_ras,
                    cap.core_radius_mm, 0.5 * cap.core_length_mm)
                wall = np.maximum(outer - core, 0.0)
                depth[sel] += _MM_TO_CM * (
                    (cap.mu_wall_cm1 - self.mu_water_cm1) * wall
                    + (cap.mu_core_cm1 - self.mu_water_cm1) * core)

        # -- tile carriers: few and large, so a prune would rarely fire;
        # trace them exactly for every point instead.
        for car in self.carriers:
            path = segment_box_length_mm(origin, pts, car.center_ras,
                                         car.frame, car.half)
            depth += _MM_TO_CM * (car.mu_cm1 - self.mu_water_cm1) * path

        return depth

    # ------------------------------------------------------------------ info
    def __repr__(self):                                   # pragma: no cover
        return ("InterferenceModel(%d capsules, %d carriers, "
                "max_range_mm=%.1f, line_samples=%d)"
                % (len(self.capsules), len(self.carriers),
                   self.max_range_mm, self.line_samples))

    def describe(self):
        """Serializable summary, stored in the dose Volume's metadata."""
        return {
            "model": "primary-fluence Beer-Lambert, excess mu vs water",
            "n_capsules": len(self.capsules),
            "n_carriers": len(self.carriers),
            "mu_water_cm1": self.mu_water_cm1,
            "mu_capsule_wall_cm1": float(
                self.capsules[0].mu_wall_cm1) if self.capsules else None,
            "mu_capsule_core_cm1": float(
                self.capsules[0].mu_core_cm1) if self.capsules else None,
            "mu_carrier_cm1": [float(c.mu_cm1) for c in self.carriers],
            "max_range_mm": self.max_range_mm,
            "line_samples": self.line_samples,
        }


# ------------------------------------------------------------------- report
def interference_report(dose_free, dose_corrected, mask=None, level_cgy=None):
    """Compare an uncorrected dose grid with an interference-corrected one.

    Parameters
    ----------
    dose_free, dose_corrected : Volume or ndarray
        Same-shaped dose grids, the second computed with an
        :class:`InterferenceModel`.
    mask : ndarray of bool, optional
        Restrict the statistics, e.g. to the cavity or a target volume.
    level_cgy : float, optional
        Also restrict to voxels at or above this dose in the *uncorrected*
        grid -- the honest place to quote a correction, since a percentage
        change out in the 1 cGy tail means nothing clinically.

    Returns
    -------
    dict
        ``n_voxels``, ``mean_ratio``, ``median_ratio``, ``p05_ratio``,
        ``p95_ratio``, ``min_ratio``, ``max_ratio`` and
        ``mean_percent_change``.  Ratios are corrected / uncorrected.
    """
    a = np.asarray(getattr(dose_free, "array", dose_free), dtype=float)
    b = np.asarray(getattr(dose_corrected, "array", dose_corrected),
                   dtype=float)
    if a.shape != b.shape:
        raise ValueError("dose grids must have the same shape")

    sel = np.ones(a.shape, dtype=bool) if mask is None \
        else np.asarray(mask, dtype=bool)
    if sel.shape != a.shape:
        raise ValueError("mask must match the dose grid shape")
    sel = sel & (a > 0.0)
    if level_cgy is not None:
        sel = sel & (a >= float(level_cgy))

    n = int(sel.sum())
    nan = float("nan")
    if n == 0:
        return {"n_voxels": 0, "mean_ratio": nan, "median_ratio": nan,
                "p05_ratio": nan, "p95_ratio": nan, "min_ratio": nan,
                "max_ratio": nan, "mean_percent_change": nan}

    ratio = b[sel] / a[sel]
    return {
        "n_voxels": n,
        "mean_ratio": float(ratio.mean()),
        "median_ratio": float(np.median(ratio)),
        "p05_ratio": float(np.percentile(ratio, 5.0)),
        "p95_ratio": float(np.percentile(ratio, 95.0)),
        "min_ratio": float(ratio.min()),
        "max_ratio": float(ratio.max()),
        "mean_percent_change": float(100.0 * (ratio.mean() - 1.0)),
    }
