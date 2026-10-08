# Localization upgrade — measurement notes

Plan: `docs/plan-localization.md` (approved 2026-10-08). Every number here
carries the command, seed list and commit hash that produced it. Negative
results stay in. A stage becomes the default only after its pre-declared gate
(plan, "Verification") passes on the synthetic harness AND the real-data
proxies do not regress.

## Baseline (stage 0, re-measured; replaces output/validation_spacing.csv of 2026-09-01)

## Stage gates

| Stage | Gate | Result | Commit | Shipped as |
|---|---|---|---|---|
| 3 | side-by-side seeds 2.5–3.5 mm apart still split | pass: 6/6 fixed configurations within 0.21 mm (`tests/test_seeds_merge_split.py`); 90/90 random pairs within 0.19 mm with the weighted split (k-means++: 86/90, worst 1.53 mm) | 6eb4aa4 | split: windowed median + halves guard + weighted split, all default-on |
| 3 | gap-slice fragment → 1 candidate | pass on the PostOp slab geometry (1 mm slices every 2 mm): 29/200 random seeds fragment, 0 stay fragmented at 4.0 mm, merged centre error 0.53 mean / 1.50 max. **Not met** on 2 mm slabs with interpolated gap slices (G3): there the fragments are tip pairs 4.2–4.7 mm apart, beyond the 4.0 mm cap (see stage 3, limits) | 6eb4aa4 | merge: pipeline default on for coarse scans with slices thinner than their spacing; opt-in in `detect_seed_candidates` |
| 3 | 0 false splits among 30 noise blobs | pass (10 seeds + 30 one/two-voxel specks at 2200 HU, all `split_k == 1`) | 6eb4aa4 | — |

## Stage 3 — merge/split repair (agent LM, branch loc/mergesplit)

Code: `gtcore/seeds/detect.py` (`_split_merged_blobs`, `_weighted_lloyd`,
`_collinear_halves`, `_merge_fragments`, `detect_seed_candidates(...,
split_window, split_guard, split_weighted, merge_fragments,
merge_max_sep_mm)`), `gtcore/pipeline.py` (`merge_fragments_default`,
`reconstruct(..., merge_fragments=None)`, `vol.meta["seed_merge"]`).
Tests: `tests/test_seeds_merge_split.py` (17 tests, ~2 s), scenes in
`tests/merge_split_scenes.py`. Real-data script:
`scripts/localization_mergesplit_realdata.py`; sweep:
`scripts/localization_mergesplit_sweep.py`. All numbers below: commit
6eb4aa4 unless stated, `PYTHONPATH=.` from the repo root.

### Step 1 — PostOp diagnosis (before any tuning)

`python scripts/localization_mergesplit_realdata.py diagnose --merge off`
(baseline detection is bit-identical to 3cf35af: 731 raw blobs, 53 in-vault
candidates). Count-free suggest (`fit_tiles_prior(..., ImplantPrior())`,
the planner path): 4 supported + 2 tentative tiles, **5 unassigned**
in-implant candidates (#4, #19, #24, #35, #37), 26 clutter.

Two facts the earlier notes did not have:

* The PostOp DICOM `SliceThickness` is **1.0 mm at 2.0 mm spacing**: the
  slices are 1 mm slabs with a 1 mm unimaged gap between neighbours. At the
  implant the slices k = 58–62 and 65 are present, 63–64 interpolated
  (52/89 interpolated overall, re-derived from the slice positions with the
  loader's rule).
* `DOEJOHNPOSTCT` is **not a pre-implant scan**: it is the complete
  contiguous 1 mm thin-cut ("STEALTH 1.0 Hr40", 204 slices, acquired
  2023-10-26 09:09:16) of the **same post-implant acquisition** as the
  PostOp export (PostOp series time 09:11:14, same in-plane origin and
  0.5195 mm pixels, all 64 PostOp slice positions coincide exactly with
  thin-cut slices; different kernel, mean |ΔHU| ≈ 40). Its seeds reproduce
  the PostOp supported tiles to ≤ 0.2 mm (6 supported tiles; the "6 chance
  quads" recorded for it in docs/data-notes.md are the implant). It is used
  below as a thin-cut reference for the PostOp merges.

| Unassigned ↔ nearest | \|d\| (mm) | separation RAS (mm) / voxel (i, j, k) | along normal / in-plane (mm) | axes (angle to separation) | volume (mm³) | peak HU | bridge: straight trilinear min / ridge (cell-max) min vs need (HU) | slices k (interpolated?) | one thresholded blob? | thin-cut seed (DOE) |
|---|---|---|---|---|---|---|---|---|---|---|
| #4 ↔ #37 | 3.39 | (−0.12, 2.70, −2.05) / (0.24, −5.20, −1.02) | 2.05 / 2.70 | #4 (−0.02, −0.11, 0.99) 47°, smeared along z by interpolated slice 63; #37 fallback (0, 0, 1) (2 voxels) | 9.72 / 1.08 | 2896 / 1768 | 603 (bg −2, frac 0.34) / 1224 vs 455 | 62.26 / 61.23 (both present) | no (labels 609 / 598) | one seed, axis (−0.01, −0.91, 0.42): separation within 13° of it; fragments 1.38 / 2.05 mm from it |
| #35 ↔ #34 (#34 in tentative T4) | 3.01 | (2.01, 0.87, −2.07) / (−3.87, −1.67, −1.03) | 2.07 / 2.19 | both (0, 0, 1) (#35 fallback, 3 voxels) | 1.62 / 2.16 | 1831 / 2178 | 897 (bg 24, frac 0.48) / 1557 vs 656 | 62.03 / 61.00 (both present) | no (611 / 601) | one seed lying in-plane, axis (−0.97, −0.26, 0): the in-plane separation runs along it; fragments 1.44 / 1.57 mm from it |
| #19 ↔ #3 (tile) | 9.98 | — | — | — | 6.48 / 9.72 | 3069 / 2963 | −47 | 58.24 / 61.00 | no | single seed 0.12 mm from #19, unassigned in the thin-cut fit too |
| #24 ↔ #37 | 7.16 | — | — | — | 5.94 / 1.08 | 3053 / 1768 | −316 | 59.49 / 61.23 | no | single seed 0.20 mm from #24, unassigned in the thin-cut fit too |

HU patches (`vol.array[k, j-7:j+8, i-6:i+7]`, k = 60–64) show the
mechanism directly: #4 is a capsule lying along j in slice 62 (rows j
184–190, 2700–2900 HU) whose anterior tip dips into slice 61 (#37, j
183–184, 1200–1300 HU), the voxel that would connect them sitting at
~1200 HU next to a −700 HU streak; #34/#35 are the two halves of one
in-plane capsule straddling the 61/62 slab boundary (a diagonal trace in
each slice, the left half brighter in 61, the right half in 62).

**Verdict.** Both pairs are *slab-boundary fragments of one capsule*: a
seed within ~25° of the slice plane crossing the unimaged 1 mm gap between
two present slices. They are not z-fragments of a seed along z (in-plane
offset 2.2–2.7 mm, not < 1.5 mm), not a k-means over-split (separate
thresholded blobs; no blob on PostOp is split at all because the all-blob
median is 1.08 mm³ = 2 voxels and every k ≥ 2 split would be degenerate),
and not two seeds (the thin-cut shows one seed at each pair). #19 and #24
are real single seeds. The 3D PCA axes are useless for these fragments
(< 4-voxel fallback, interpolated-slice smear), which is why the merge's
coarse-scan geometry test uses in-plane traces and slab adjacency instead
of 3D axes, and why the bridge is sampled ridge-tolerantly: the straight
centroid-to-centroid line of #4/#37 cuts the corner of the oblique capsule
and its trilinear minimum (frac 0.34) would fail a 0.35 rule that the
image clearly satisfies.

### What shipped, and the defaults

| Piece | Rule (disclosed) | Default | Evidence for the default |
|---|---|---|---|
| windowed split reference (`split_window`) | median over blobs inside `[min_mm3, max_mm3]` (fallback: all blobs if < 4 inside) | on | bit-identical on PostOp, printed phantom, tile-free scan and the 20 synthetic head-phantom cases (rng 0–4 × 0.7/1.4/2.1/2.8 mm) without the guard too; on the DOE thin-cut alone it LOWERS the median 4.59 → 4.32 mm³ and, without the guard, cuts 7 single implant seeds in half (legacy cuts 3) — hence the guard. Side effect: oversize bone-like blobs (1.6–4.5 × the lone-seed volume) are also cut into seed-sized pieces (test `test_reference_median_ignores_oversize_blobs`), left to the shape/vault/tile stages |
| halves guard (`split_guard`, slices ≤ 1.2 mm) | refuse a split when the parent blob is a rod (√(λ₁/λ₂) ≥ 2, weighted PCA) and two parts lie closer than `MERGE_MAX_SEP_MM` = L − 0.5 = 4.0 mm along its axis (≤ 25°): two capsules end to end cannot be closer than L | on | DOE thin-cut: the halves splits (parts 1.66–1.87 mm apart) are all refused; 0 split blobs within 40 mm of the implant (legacy 3, window-only 7). Linearity: lone seeds 2.3–3.1 (0.59 mm grid), DOE halved seeds 2.43–3.23, merged side-by-side pairs ≤ 1.37 |
| weighted split (`split_weighted`) | k = round(Σ(HU − thr) / median of that), deterministic weighted Lloyd's from equal-weight slabs along the 1st and the 2nd principal axis, lower weighted SSE wins | on (was specified opt-in until measured better) | 90 random side-by-side pairs on 0.59×0.59×1.0 (seed 7, d = 2.5/3.0/3.5): 90/90 within 0.5 mm, worst 0.19 mm; k-means++ with volume-ratio k 86/90, worst 1.53 mm (at 2.5 mm the bloom bridge makes a pair ~2.7 lone volumes → cut in 3; or cut across both seeds). Identical to the historical method on all real scans and the 20 head-phantom cases. Without the guard it over-splits much brighter single seeds (mass ratio ≥ 3) — the guard catches every case seen |
| fragment merge (`merge_fragments`) | pair < 4.0 mm; ridge bridge ≥ bg + 0.35 (dimmer peak − bg), bg = median HU on 6/8/10 mm shells; geometry: thin — every usable 3D axis ≤ 25° of the separation; coarse — end-on (in-plane < 1.5 mm and ≥ in-plane along the normal), or every directional in-plane trace (≥ 4 voxels, in-plane elongation ≥ 1.5) ≤ 25° of the in-plane separation, or (no directional trace) one slice apart; never split siblings; union ≤ `max_mm3`; group span < 4.0 mm; group isolated (no other candidate within 4.0 mm of a member) | `detect_seed_candidates`: off. `reconstruct`: on iff slices > 1.2 mm AND DICOM slice thickness ≤ 0.75 × spacing (unimaged gaps); logged in `vol.meta["seed_merge"]` | PostOp (1.0/2.0 mm): fixes both pairs, nothing else in the implant changes. Synthetic G2s (that geometry): fixes 29/29 fragmented seeds. G3 (contiguous 2 mm + interpolation): fixes 0, fuses 8/180 close pairs → off. G1/thin: nothing to fix (0/200 fragment) → off. Unknown thickness (NRRD, synthetic) → off, so every synthetic pipeline test is unchanged |

`SeedCandidates.info` now always carries per-candidate `source_blob`
(thresholded component id; split siblings share it), `split_k` and
`n_fragments`, plus `merge_log` (tuple of dicts) when the merge ran.

### Synthetic results

Scenes (`tests/merge_split_scenes.py`): supersampled 4.5 mm line source,
anisotropic Gaussian PSF, 20 HU noise, 3071 HU clip. G1 0.59×0.59×1.0
(PSF 0.6 mm, amplitude 5000 → clipped, thr 2000); G2s 0.5×0.5×2.0 with
1.0 mm slices (PSF 0.6 in-plane / 0.3 along z, amplitude 3500, thr 1200 →
in-plane peaks ≈ 3000 HU like PostOp); G3 0.5×0.5×2.0, 2 mm slabs,
alternate slices replaced by the neighbour mean (amplitude 4200).

**Sensitivity of `MERGE_MAX_SEP_MM`** —
`python scripts/localization_mergesplit_sweep.py --singles 200 --pairs 60`
(seed 2026, 15.8 s). "False merges" = merges whose two fragments belong to
different capsules (nearest axis segment), counted over 60 adversarial
close pairs per kind (collinear end to end 4.6–5.6 mm, parallel side by
side 2–4 mm, random 2.5–5.5 mm, never overlapping); unmerged fragment
centroids sit 0.6–2.0 mm from their seed.

| grid | cap (mm) | lone seeds fragmented (no merge) | still fragmented after merge | merged centre error mean / max (mm) | false merges: collinear / parallel / random (of 60 pairs each) |
|---|---|---|---|---|---|
| G1 | 3.0–5.0 | 0 / 200 | 0 | — | 0 / 0 / 0 (every cap) |
| G2s | 3.0 | 29 / 200 | 11 | 0.30 / 0.77 | 1 / 4 / 0 |
| G2s | 3.5 | 29 / 200 | 2 | 0.46 / 1.16 | 3 / 9 / 0 |
| G2s | **4.0** | 29 / 200 | **0** | 0.53 / 1.50 | 5 / 9 / 3 |
| G2s | 4.5 | 29 / 200 | 0 | 0.53 / 1.50 | 6 / 15 / 6 |
| G2s | 5.0 | 29 / 200 | 0 | 0.53 / 1.50 | 7 / 15 / 6 |
| G3 | 3.0 | 4 / 200 | 4 | — | 3 / 0 / 0 |
| G3 | 3.5 | 4 / 200 | 4 | — | 4 / 1 / 0 |
| G3 | 4.0 | 4 / 200 | 4 | — | 7 / 1 / 0 |
| G3 | 4.5 | 4 / 200 | 4 | — | 10 / 2 / 0 |
| G3 | 5.0 | 4 / 200 | 4 | — | 13 / 2 / 0 |

Reading: on the PostOp geometry 14.5 % of seeds fragment, all at 2.5–3.8
mm (= one slab spacing / sin(elevation) for capsules 25–55° out of plane);
4.0 mm (L − 0.5) is the smallest cap that rejoins every one, and false
merges grow past L (4.5–5.0). The residual false merges at 4.0 (17/180
adversarial pairs, 9 %) are two distinct seeds stacked so that each shows a
single piece in adjacent slabs; no measured image feature separates them
from one capsule's two pieces (perpendicular RMS 0.17–0.37 vs 0.17–0.27
mm, end gap 0.7–2.0 vs 0.9–3.0 mm, union extent 4.0–5.1 vs ≤ 4.0 mm on
synthetic but 4.65–4.76 mm for the two real PostOp merges). In a real
implant such close distinct pairs only occur at tile junctions/overlaps,
while fragments affect ~15 % of seeds on this geometry. On G3 the merge
has no benefit at any cap ≤ 4.0 (its fragments are the two tips of a
42–61° capsule centred on an interpolated slice, 4.2–4.7 mm apart, two
grid slices apart) and only adds false merges — the reason the pipeline
default excludes contiguous-slab scans.

Other synthetic checks: noise specks (10 seeds + 30 one/two-voxel specks
at 2200 HU, default 0.2 mm³ floor) → 0 splits, all seeds within 0.5 mm;
the existing 7 mm end-to-end pair test (`test_seeds_unit.py`) still
splits; on the 20 head-phantom cases the merge changes one (rng 4 at
2.1 mm: 13 → 12 candidates, 12 true seeds) but is not enabled there by the
pipeline (no slice-thickness tag).

## Real-data proxies (no truth)

Commit 6eb4aa4, `python scripts/localization_mergesplit_realdata.py
check <scan> --merge default` (`diagnose` for PostOp), each scan run alone.

| Scan | Merge default | Before (3cf35af detection) | After (6eb4aa4) |
|---|---|---|---|
| PostOp (0.52×0.52×2.0, 1 mm slices) | on ("1.00 mm slices every 2.00 mm leave unimaged gaps") | 731 raw → 53 in-vault; 4 supported + 2 tentative tiles; **5 unassigned**, 26 clutter; thin-cut seeds #16 and #23 each hit by two candidates | 30 merges head-wide (20 in-plane-axis, 9 adjacent-slice, 1 end-on), mostly bone/streak fragments; 3 merged groups touch in-vault candidates (the two pairs above and clutter #41, 60 mm from the implant) → 701 raw → 50 in-vault; the 4 supported tiles **bit-identical** (residuals 0.39/0.42/0.65/0.71 mm), both tentative tiles kept (the one containing #34 now uses the merged seed; its inferred seed moves 1.1 mm); **3 unassigned**, 25 clutter; no thin-cut seed hit twice; implant "confirmed" unchanged |
| printed 8-tile phantom (0.59×0.59×1.0) | off (thin) | 32/32 seeds, 8/8 tiles | 32/32 seeds, 8/8 tiles, detection bit-identical; forcing the merge on rejoins 0 pairs |
| tile-free printed (0.68×0.68×1.0) | off (thin) | 0 candidates, not present | 0 candidates, not present |
| DOE thin-cut (0.52×0.52×1.0) | off (thin) | 74 raw (24 from splits, 3 of them halves of implant seeds) → 31 in-vault; 6 supported tiles (residuals 0.18/0.21/0.29/0.51/0.64/0.67 mm), 7 unassigned; "confirmed" | 57 raw (no splits) → 31 in-vault; 6 supported tiles, the one that used a half-seed now uses the whole seed (residual 0.64 → 0.47 mm); 7 unassigned, all whole seeds; "confirmed" unchanged |

PostOp unassigned 5 → 3, not the planned ≤ 2: the 3 left are real seeds
(thin-cut seeds 1.36, 0.12 and 0.20 mm away), and the thin-cut fit leaves
the same three unassigned, so no detection repair can place them; the
plan's "≤ 2" counted each duplicate pair as two leftovers, but #34 was
already in a tentative tile. Thin-cut check of the merges: the 34/35
candidate moves from 1.44/1.57 mm to **0.32 mm** from its thin-cut seed;
the 4/37 candidate from 1.38/2.05 to 1.36 mm (its remaining ~1 mm z offset
is #4's smear into interpolated slice 63, not the merge). Merge cost on
PostOp: detection 0.26 → 0.29 s.

## With-truth track

The DOE thin-cut (see stage 3) is a same-acquisition 1 mm reference for
the PostOp 2 mm export: 64/64 PostOp slices are a re-reconstruction of
thin-cut slice positions, so PostOp seed positions can be scored against
thin-cut seeds (subject to the thin-cut's own ~0.2–0.4 mm localization
error). At 6eb4aa4: supported/tentative PostOp seeds → nearest thin-cut
seed median 0.34 mm (0.47 before the merge), max 2.41 mm.

## Open decisions

- **DOE relabel** (for the coordinator / Jacob): `DOEJOHNPOSTCT` is the
  complete thin-cut of the PostOp acquisition, not a pre-implant negative
  control. docs/data-notes.md and the "calcification chance quads"
  limitation in `pipeline.assess_implant` rest on that mislabel; there is
  currently no true pre-implant scan on this machine.
- Merge on contiguous-slab exports with interpolated gaps (G3, and the
  harness G3/G4): fragments are two grid slices apart and 4.2–4.7 mm
  apart; reaching them needs `meta["interpolated_k"]` (adjacency among
  present slices) and a cap above L, which the sweep shows costs false
  merges. Not shipped in 6eb4aa4.
- Residual merge ambiguity (9 % of adversarial close pairs on the PostOp
  geometry) is physical; the stage-7 two-capsule fit with BIC is the
  principled resolver.

## Runs log

| Date | Command | Seeds | Commit | Wall time |
|---|---|---|---|---|
| 2026-10-08 | `scripts/localization_mergesplit_realdata.py diagnose --merge off` (PostOp, baseline) | — | detection = 3cf35af | ~25 s (cached pickle) |
| 2026-10-08 | `scripts/localization_mergesplit_realdata.py diagnose --merge default` (PostOp) | — | 6eb4aa4 | 26 s |
| 2026-10-08 | `scripts/localization_mergesplit_realdata.py check phantom8 --merge default` (and `--merge on`) | — | 6eb4aa4 | 27 s |
| 2026-10-08 | `scripts/localization_mergesplit_realdata.py check tilefree --merge default` | — | 6eb4aa4 | 28 s |
| 2026-10-08 | `scripts/localization_mergesplit_realdata.py check doe --merge default [--legacy-split]` | — | 6eb4aa4 | 32 s / 26 s |
| 2026-10-08 | `scripts/localization_mergesplit_sweep.py --singles 200 --pairs 60` | 2026 | 6eb4aa4 | 15.8 s |
| 2026-10-08 | `pytest tests/test_seeds_merge_split.py tests/test_seeds_unit.py tests/test_integration.py tests/test_tiles_cover.py tests/test_implant_assessment.py tests/test_localization_plumbing.py` | — | 6eb4aa4 | 33 s, 51 passed |
