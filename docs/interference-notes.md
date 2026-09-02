# Tile interference in the dose engine (2026-09-01)

`gtcore/dose/interference.py` adds the effect the TG-43U1 formalism cannot
express: the seeds and tile carriers of a GammaTile implant sitting in each
other's way. This note records the model, where every number comes from, what
was measured, and what is still a guess.

## The gap being closed

TG-43U1 is defined for **one seed alone at the centre of an unbounded water
phantom**. A multi-seed plan is then built by superposition, which silently
assumes that adding seed B to the plan does not change what seed A's photons
do on the way to a field point. It does. At the Cs-131 effective energy each
seed is a titanium can, and the collagen carrier is not water.

Two consequences, opposite in sign:

| Effect | Sign | Cause |
|---|---|---|
| Inter-seed attenuation | dose **down** | Ti capsule, mu ~ 45 cm^-1 at 30 keV |
| Carrier displacement | dose **up** | collagen is less dense than water |

## Model

One line-of-sight Beer-Lambert factor per (seed, field point), applied to the
TG-43 dose rate before superposition:

```
D(p) = sum_s  Drate_TG43(p; s) * T(p; s) * S_K,s * tau

T(p; s) = exp( - sum_j (mu_j - mu_water) * l_j(s -> p) )
```

`l_j` is the length of the segment from seed `s` to point `p` inside occluder
`j`, computed analytically — segment vs finite cylinder for capsules, segment
vs oriented box for carriers. Both primitives are verified against brute-force
numerical sampling of the segment in `tests/test_dose_interference.py`, with
no shared code.

Choices worth stating:

- **The excess coefficient is `mu_material - mu_water`, not `mu_material`.**
  The occluder displaces water that TG-43 already accounted for. This is what
  lets a sub-water-density carrier legitimately produce `T > 1`.
- **A seed never shadows itself.** Its self-absorption is already inside the
  measured anisotropy function `F(r, theta)`; counting it twice would be a
  double correction. Enforced by index, and the index alignment between the
  model and the dose call is *validated*, not assumed — a misalignment would
  silently make each seed shadow itself.
- **A seed's own carrier is not skipped.** The ray really does leave through
  the collagen, and nothing in TG-43 accounts for that.
- **The capsule is two coaxial cylinders**, wall and core, so the thin
  titanium wall is integrated at its own mu rather than smeared over the
  capsule. The wall path is (outer cylinder path) - (inner cylinder path).
- **Transmission is averaged over the active line, not optical depth.**
  With `line_samples > 1` the ray origin is spread over the 4 mm active
  length. `exp` is convex, so `mean(exp(-tau)) >= exp(-mean(tau))`: averaging
  depths would under-report dose at every partially shadowed point. Pinned by
  a test.

### What it does not model

This is a **primary-fluence** correction — the standard first-order treatment
of interseed attenuation, and the same thing a full Monte-Carlo run would
improve on. It ignores:

- **Scatter refilling the shadow.** The correction therefore *overestimates*
  shadow depth, increasingly so with distance as the scatter fraction grows.
- **Spectral hardening** through titanium; a single effective mu is used.
- **Patient heterogeneity** (bone, air, brain). That is a separate correction
  of comparable size, attempted nowhere in `gtcore`.

## Coefficients and their provenance

At the Cs-131 effective photon energy of 30.4 keV (Xe K x-rays, 29.5–34.4
keV, fluence-weighted):

| Material | mu/rho (cm^2/g) | rho (g/cm^3) | mu (cm^-1) | Source |
|---|---|---|---|---|
| Water | 0.3756 | 1.000 | 0.376 | NIST XCOM at 30 keV |
| Titanium | 9.90 | 4.506 | 44.6 | NIST XCOM; K edge 4.97 keV, far below the spectrum |
| Al2O3 core | 0.775 | 3.0 (assumed) | 2.33 | Mass-weighted Al + O; sintered CS-1 pellet is porous, so below the 3.97 of dense alumina |
| Collagen carrier | 0.3756 | **0.30 (guess)** | 0.113 | Density-scaled water; see below |

Geometry: capsule 0.8 mm outer diameter x 4.5 mm, 0.05 mm titanium wall;
carrier 20 x 20 x 4 mm (half tile 10 x 20 x 4), seeds on its mid-plane. A test
asserts these match the engine's own capsule constants, so the occluder and
the geometry-factor clamp can never drift apart.

## The carrier density is the weak point — and carriers are off by default

The carrier term scales directly with `TILE_CARRIER_DENSITY_G_CM3`, and that
number is a dry-collagen-sponge guess.

It matters more than its size suggests. Seeds sit on the carrier's mid-plane,
so rays travelling *within* the plane of a tile — exactly the directions where
the neighbouring seeds and the high-dose region are — run up to 20 mm
lengthwise through it. Mean dose change at or above 25% of a 60 Gy
prescription, from `scripts/validation_interference.py`:

| Carrier density (g/cm^3) | Flat 3-tile grid | Phantom truth implant |
|---|---|---|
| capsules only (no carriers) | **-0.27%** | **-0.65%** |
| 0.15 | +16.77% | +19.12% |
| 0.30 (the current guess) | +13.43% | +15.21% |
| 0.50 | +9.21% | +10.30% |
| 0.75 | +4.29% | +4.60% |
| 1.00 (water-equivalent) | -0.27% | -0.65% |

So the carrier correction is both **larger than the interseed shadow it is
meant to partially offset, and opposite in sign** — and it collapses to
exactly nothing if the implanted, blood- and CSF-soaked carrier is
water-equivalent, which is entirely plausible. The answer is set almost
entirely by a number nobody here has measured.

A correction that swings between "+15%" and "nothing" on an unmeasured number
must not be applied silently. Therefore:

- `InterferenceModel.from_implant(...)` builds **capsules only**;
  `include_carriers=True` is required to add carriers, and a test asserts that
  default cannot be flipped unnoticed.
- The interactive planner never enables carriers.

**To close this out:** get the implanted (hydrated) carrier density from the
vendor or from a Monte-Carlo model of the tile, set the constant, and re-run
`scripts/validation_interference.py`.

## Measured behaviour (capsules only)

Twelve seeds, 2 mm dose grid, 50 mm pad, run two ways: a flat 3-tile grid
(exactly known geometry) and the phantom's ground-truth implant (real
wall-conformed, non-coplanar seed poses).

| Quantity | Flat 3-tile grid | Phantom truth implant |
|---|---|---|
| Mean dose change, all voxels above 0 | -0.03% | -0.07% |
| Mean dose change at >= 25% rx (1500 cGy) | -0.27% | -0.65% |
| Mean dose change at >= 100% rx (6000 cGy) | -0.48% | -0.60% |
| 5th-percentile ratio at >= 25% rx | 0.982 | 0.962 |
| Worst single voxel | 0.840 | 0.850 |
| Runtime (tabulated kernel) | 0.35 s -> 0.42 s (+20%) | 0.40 s -> 0.46 s (+14%) |

Re-measured 2026-09-01 after the capsule geometry moved to the TG-43U1S2
Appendix A11 values (0.412 mm outer radius, 0.0555 mm Ti wall, from 0.40 /
0.05): every figure above is unchanged at the quoted precision (worst
voxels 0.8404 and 0.8504), i.e. the ~11 % thicker wall deepens the shadows
by less than the table's rounding.

Sub-percent on volume-averaged metrics with local shadows of 15% is what the
published interseed-attenuation literature reports for low-energy permanent
implants, and a test pins the correction inside that band. The conformed
implant shows a slightly larger effect than the flat grid: draped tiles put
more seeds in each other's line of sight than a plane does.

## Sampling caveat

The umbra behind a capsule is roughly its 0.8 mm width, magnified by the
source-occluder-point geometry. A 2 mm dose grid therefore **undersamples
individual shadows**: the volume statistics above are sound, but a single
voxel's value in a penumbra is grid-dependent. Use `line_samples=5` or higher,
and a finer grid, before quoting a point dose in a shadow.

## Performance

Naively this is `O(points x seeds x occluders)`. Two things keep it cheap:

- `max_range_mm` (default 40 mm) skips rays whose dose is orders of magnitude
  below any prescription isodose. Beyond it the seed contributes far less than
  a percent of the 25% isodose, so a shadow there changes nothing visible.
- Capsules get a bounding-sphere prune done as a single matrix product per
  (seed, chunk), and the exact intersection runs only on the points that
  survive — a small fraction of any grid, since shadows are narrow. Carriers
  are few and large, so they are traced exactly for every point instead.

## Flagging tiles that shadow each other

The planner already flags tiles whose collagen footprints collide
(`gtcore.interact.find_overlapping_tiles`, red tint plus a warning). That is a
geometric question. Underneath it sits a dosimetric one this module can now
answer: **tiles that never touch can still stand in each other's line of
fire**, and nothing in the geometry shows it.

`tile_shadowing(tiles)` evaluates the dose at each tile's own
prescription-depth points — 5 mm into the tissue behind each seed, where
GammaTile prescribes — twice: once as plain TG-43, once with exactly one
other tile's capsules attenuating. The relative drop is that ordered pair's
shadowing. Because only one occluding tile is live at a time the loss is
*attributable* rather than a lump sum, and the rows sum to the all-capsules
total (a test pins this, and it is what catches an over-eager prune).

`find_shadowing_tiles(tiles, threshold_pct=2.0)` reduces that to the same
shape `find_overlapping_tiles` returns, plus the percentage, so the planner
flags both the same way: an overlap is a `WARNING` (two sheets cannot occupy
one space), shadowing is a `NOTE` (the plan is legal but costs dose).

The sweep is skipped in three cases, all for latency rather than
correctness:

- **mid-drag**, since that path budgets tens of milliseconds and the sweep
  costs hundreds;
- **above 16 tiles**, where the pairwise cost stops being interactive;
- **during startup adoption of the fitted implant.** The planner opens with
  the tiles recovered from the scan already on the board, and running the
  sweep there would spend up to half a second of launch latency answering a
  question the user has not asked. The first real edit runs it.

### What it finds, and what it correctly ignores

| Layout | Worst pair | Flagged at 2%? |
|---|---|---|
| Two coplanar tiles on one wall, 22 mm apart | < 0.1% | no |
| Phantom truth implant, 3 conformed tiles | 0.56% | no |
| One tile stacked 4 mm behind another | 5.7% | **yes** |

The quiet result on good geometry is the important one. Seeds sit on the
wall and the prescription point is 7 mm behind it, so a ray from any seed to
any such point leaves the seed plane immediately and clears the coplanar
capsules. A flag that fired on every ordinary plan would be worthless; this
one fires when a tile is genuinely parked in another's way.

Shadowing is **directional** — A blocking B's rays to depth is a different
statement from the reverse — so `loss` is not symmetric, and the pair
listing reports the worse direction. The diagonal is a tile's *self*
shadowing, its own four seeds shading one another, which is real and is
usually the largest single entry (0.6–0.9% on the phantom implant).

### Cost

A pairwise sweep of dose evaluations: about 0.15 s at three tiles and 0.4 s
at eight. Two things keep it from growing as T²: the tabulated kernel, and an
exact segment-versus-sphere prune that skips the dose evaluation for tile
pairs genuinely out of each other's way. The prune has to consider rays from
*every* seed in the implant, not just the tile's own — the dose ratio is over
the total — which is the subtlety that makes a naive bounding-sphere test on
the tile pair alone silently wrong.
