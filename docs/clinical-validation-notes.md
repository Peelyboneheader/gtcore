# Clinical validation notes — first case (DOE^JOHN, post-implant CT, 2023-10-26)

Reproduce with:

```
python scripts/validation_clinical.py <CT dir> <RTSTRUCT> <RTDOSE>
```

Inputs: the 204-slice post-implant head CT (0.52 x 0.52 x 1.0 mm), the
treatment-planning-system RTSTRUCT (physician contours) and RTDOSE (1 x 1 x
1.67 mm plan grid, Gy). All three share one frame of reference. The planner
reaches the same numbers interactively via `gt plan <CT dir> --hrctv`, `T`
then `U`.

## 1. What the TPS calls the target

| TPS structure | Volume | Meaning |
|---|---|---|
| `Cav_Post` | 35.1 cc | physician's post-op resection cavity |
| `CTV_GT` | 19.0 cc | the GammaTile clinical target: a *partial* 5 mm rind — 0 % inside the cavity, every voxel ≤ 5.2 mm from it, but only 56 % of the full 5 mm rind (33.9 cc) |
| `CTV_Post` | 23.4 cc | a second target, reaching 10.6 mm from the cavity |
| `Seeds` | 30 contours | 30 contoured seed positions; the RTDOSE itself has **32** local maxima, i.e. the plan has 32 sources = 8 full tiles (two seeds were never contoured) |

The clinical plan's `CTV_GT` D90 is **6365 cGy** (V100 94 %), i.e. the 63 Gy
Jacob remembered. The same plan on the *full* 5 mm rind of the physician's
cavity gives D90 **4593 cGy**, V100 66 %: nearly half of the rind was left
out of `CTV_GT` by the physician (untiled wall, dura / brain surface, or
eloquent cortex), and that choice, not dosimetry, is what lifts the clinical
D90 from 46 Gy to 64 Gy. An automatic HR-CTV that takes the whole rind must
therefore be compared with the 46 Gy figure, not the 64 Gy one.

Is `CTV_GT` "cavity + 5 mm clipped to the brain"? No: the 5 mm expansion of
`Cav_Post` clipped to the TPS `Brain` contour is 35.4 cc and contains 99 % of
`CTV_GT`, but `CTV_GT` fills only 53 % of it (Dice 0.69). The clinical target
was trimmed by hand beyond any geometric rule, so no automatic expansion can
reproduce it exactly; the geometric definition Jacob specified -- the cavity
expanded by 5 mm -- is what `gtcore.dose.hrctv` builds, and its clinical
counterpart is the full-rind row (4593 cGy), with `CTV_GT` as the upper
reference.

## 2. Seed detection

| | |
|---|---|
| TPS seeds / detected | 30 / 31 |
| detected → nearest TPS seed | median 0.23 mm, p95 4.4 mm, max 8.7 mm |
| TPS seeds with no detection within 2 mm | 1 |

Of the 31 detections, 29 match contoured seeds (median 0.23 mm, max 0.9 mm)
and 2 are the uncontoured sources; one contoured seed (2 mm outside the
cranial-interior mask under the craniotomy) is dropped by the vault filter.
Tile fitting recovers 6 of the 8 full tiles; the 8 leftover seeds form no
square quads (crumpled or stacked tiles the bent-tile model does not
explain) -- a limitation to report.

## 3. Dose engine vs the TPS

TG-43 (gtcore, consensus TG-43U1S2 data, 3.5 U per seed) over the TPS seed
positions, against the RTDOSE at `CTV_GT` voxels more than 5 mm from any seed:

| ratio TG-43 / RTDOSE | value |
|---|---|
| median | 0.931 |
| p5 – p95 | 0.777 – 1.030 |

The 7 % deficit is NOT a seed-strength difference: it is the two sources the
`Seeds` contour omits. With all 32 sources the engine at the nominal 3.5 U
matches the RTDOSE to 1 % (implied 3.54 U). The 3.76 U row below is what you
get by scaling the 30 contoured seeds up to the missing dose, kept only to
show the size of the effect:

| D90 on `CTV_GT` (cGy) | 3.5 U | 3.76 U | RTDOSE |
|---|---|---|---|
| TPS seed positions | 5790 | 6221 | 6365 |
| gtcore detected seeds | 6080 | 6532 | — |

So the engine reproduces the clinical D90 within 1-2 % given the full source
list. gtcore's own 31 detections at 3.5 U land 4.5 % low on `CTV_GT` (6080
vs 6365 cGy), consistent with the one dropped seed. The assay S_K should
still be a planner input (it is already a parameter of `compute_dose_grid`).

## 4. Cavity segmentation and the HR-CTV

The original rule — "the low-density component the seed cloud touches" —
returned **193 cc**: cavity + peri-cavity oedema + both lateral ventricles,
with HR-CTV D90 of 66 cGy. The seed-sheet rule (`gtcore.segment.cavity`,
grow from the seed hull through low density with a geodesic reach, clip 3 mm
beyond the seed sheet, 3 mm closing) gives:

| reach | phantom Dice | clinical cavity | Dice vs `Cav_Post` | covers / inside |
|---|---|---|---|---|
| 12 mm | 0.808 | 37.7 cc | 0.677 | 70 % / 65 % |
| **14 mm** (default) | **0.851** | **40.5 cc** | **0.654** | 70 % / 61 % |
| 16 mm | 0.886 | 43.4 cc | 0.632 | 71 % / 57 % |
| 20 mm | 0.924 | 48.8 cc | 0.590 | 71 % / 51 % |

Coverage of the physician's cavity saturates at 71 % whatever the reach: the
remaining 29 % is cavity content above the fluid threshold (clot, debris)
that the physician contoured across and the intensity rule cannot. Growth
beyond ~12 mm only adds oedema. 14 mm balances this case against the
synthetic phantom (one-sided implant, whose far wall needs the reach).

HR-CTV (5 mm rind of the gtcore cavity, `gtcore.dose.hrctv`, 49.9 cc at
reach 14) under the clinical RTDOSE: D90 **2863 cGy**, V100 38 %. The full rind of
the physician's own cavity gets 4593 cGy / 66 % from the same dose, so about
two thirds of the gap to `CTV_GT` is the partial-rind definition and one
third is our cavity being 5 cc too big and 10 cc misplaced. Open items, in
order of value:

1. Let the HR-CTV follow the clinical convention: score only the rind
   within reach of the implant (the tiled wall ± a margin), or accept a
   physician-edited cavity. This alone moves the automatic D90 from the
   46 Gy regime to the 64 Gy one.
2. Cavity contents above 26 HU (clot) — the 29 % of `Cav_Post` never
   reached. A closing-free fill of the region enclosed by the seed sheet
   would capture it.
3. Widen the vault filter's dilation so a seed just under the craniotomy is
   kept (the dropped contoured seed); read S_K from the assay certificate.
4. A pure-geometry cavity (seed hull dilated 7 mm) scores Dice 0.82 /
   covers 89 % of `Cav_Post` on this case, better than the intensity-grown
   rule; worth testing on the phantom and the next case before switching.

## 5. Earlier intermediate numbers (superseded)

A first version of the sheet rule (radial normals, reach 12, brain-envelope
mask kept) gave a 21 cc cavity and HR-CTV D90 5805 cGy / V100 88 %, which
looked like the clinical 63 Gy. That agreement was accidental: the cavity
covered only 49 % of `Cav_Post` and the rule collapsed to 2 cc on the
synthetic phantom (seeds there sit 1 mm outside the wall, and the envelope
cut the cavity under the craniotomy). The current rule fixes both and the
honest comparison is section 4.
