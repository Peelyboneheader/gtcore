# Automatic tile creation from the seed cloud (feature/tile-autogen)

Validation notes for `n_full_tiles="auto"`: the tile configuration is
inferred from the detected seeds alone, no implant count. Campaign script:
`scripts/validation_autogen.py` (writes `output/validation_autogen.csv`
and the score-saturation figure `output/validation_autogen.png`).

## What was built (plan `docs/plan-tile-autogen.md`)

| Step | Module | Idea |
|---|---|---|
| 0 | `gtcore/geometry.py` | Single cited source of the manufactured geometry: 20×20×4 mm tile, 2×2 seeds at 10 mm pitch with 5 mm margins, seed plane **3.0 mm** from the tissue face (hydrated 2.25–3.75 mm). The code used 2.0 mm before. |
| 1 | `gtcore/tiles/model.py` | Rigid nominal tile + Kabsch pose fit over the square's symmetries; seed axes resolve the 90° ambiguity; optional similarity scale. |
| 2 | `gtcore/tiles/auto.py` | Count-free search: exact best disjoint selection for every tile count, BIC-like penalty λ = 3.5 per tile, score curve + "evidence supports n=…" summary. Half tiles opt-in (clutter pairs are indistinguishable from real halves by geometry alone; they are reported as `half_candidates`). |
| 3 | `gtcore/tiles/deform.py` | Developable bent-tile model: double-arc seed sheet whose geodesic pitch follows the parallel-surface metric of the 3 mm offset (isometric 10 mm on the collagen). Explains wall-conformed chord contraction *and* a folded tile. Both normal signs are tried because a square quad is coplanar and only the seed-axis tilts see the bowl. |
| 4 | `gtcore/tiles/surface.py` | Stick-to-surface fit when a cavity mesh exists (3 dof over the planner's conformer): wall footprint, `detachment_mm` (seeds vs the 3 mm offset surface) and `agreement_mm` (free vs surface fit) as verdicts. |
| 5 | pipeline / CLI / planner | `reconstruct(n_full_tiles="auto")`, `gt view --tiles auto`, `gt plan --suggest`, planner key `g`; suggested tiles are ordinary placed tiles (drag, rotate, delete, live re-conform). `scripts/validation_autogen.py`. |

## Results (2026-09-02)

### Synthetic phantom, no count input (60 cases: rng 0–5 × 1–5 tiles × 0.8 / 1.2 mm)

| spacing | count correct | exact partition | centre err mean / max | normal err mean | time / case |
|---|---|---|---|---|---|
| 0.8 mm | 30 / 30 | 30 / 30 | 0.10 / 0.22 mm | 1.7° | 0.25 s |
| 1.2 mm | 24 / 30 | 24 / 30 | 0.13 / 0.27 mm | 3.2° | 0.30 s |

All six 1.2 mm misses are seed-detection misses (11 of 12 seeds found; the
3-seed remainder cannot be a tile for any method, and the count-based fit
fails identically there). Auto mode therefore matches the counted fit on
every case while needing no count. Decoys (4–8 random candidates) and
near-square junk quads never inflate n (tests).

### Physical 8-tile printed phantom (Philips 0.59×0.59×1.0 mm, O-MAR)

- 32 seeds → **n = 8**, "no further tile candidate"; the crumpled tile
  (sides squeezed to 4.8 / 5.9 mm) is recovered by the deformable tier at
  0.46 mm rms with a 288° fold (bending energy 0.066), flagged `degraded`
  — no count fallback needed.
- Per-tile deformable residuals 0.26–0.87 mm; folds 37–145° on the regular
  tiles (1 mm slices, PCA axes on bloomed capsules).
- Partition caveat: two adjacent tiles share a pair of seeds (25, 31) only
  4 mm apart. The deformable scoring prefers `{8,16,24,25}` + `{26,29,30,31}`
  (rms 0.78 / 0.87 mm, the second a near-perfect 9.1 mm square) over the
  earlier count-based `{25,26,29,30}` + `{8,16,24,31}` (1.11 / 1.47 mm).
  There is no independent truth for this pair; the two readings were
  essentially tied under the old score as well.
- No cavity mesh exists for this scan (printed shell, no skull), so the
  surface tier does not run here.

### Real post-op CT (2 mm export with gaps, 27-seed cluster)

- 53 in-vault candidates → **n = 4** full tiles (residuals 0.39–0.71 mm,
  folds 14–51°), "no further tile candidate": the same 4 tiles the counted
  fit returned for any requested count ≥ 4. Half-tile candidates are
  reported, not counted (true count still unknown).
- Cavity segmentation is empty on this export, so no surface verdicts.

### Score-saturation curves

`output/validation_autogen.png`: marginal score of the n-th tile vs n. On
every case the marginal gain sits far above λ = 3.5 up to the true n and
then drops to zero (no disjoint candidate) or below λ.

## Cover pass: explain every seed (2026-09-02, Jacob's request)

Pressing `T` on the PostOp CT gave 3-4 tiles for a 27-seed cluster and
said nothing about the other seeds.  The count-free search above is
conservative *by construction* (thin-cut gates, penalty 3.5) and on a 2 mm
gappy export real tiles fail the gates or score below the penalty; a tile
with one seed missed by detection was not modelled at all.  Rules agreed
with Jacob: when the OR count is unknown assume **0 half tiles**; **1- and
3-seed tiles do not exist** (a triplet is a 4-seed tile with a detection
miss); explain every seed as well as the geometry allows and *say* what
is left.

| Piece | Where | What |
|---|---|---|
| `ImplantPrior` | `tiles/auto.py`, `gt plan --tiles/--half/--seeds` | OR counts as optional trusted inputs; the planner status line always shows the assumption in force. `fit_tiles_prior` = count-constrained fit when known (cover pass only for the shortfall), auto otherwise. |
| spacing tolerance | `spacing_tolerance(spacing)` | `tol` = max voxel dimension in mm (>= 1). The *supported* selection keeps its thin-cut calibration; only the cover-pass gates scale. Above 1.2 mm the seed PCA axes are degenerate (single-slice pancakes), so the cover pass neither gates nor fits on axes. |
| relaxed quads | `_cover_pass` | loose tier re-run on the leftovers inside the implant region (25 mm of a supported tile) with rms <= 1.2·tol, E <= 0.20, similarity <= 3·tol, penalty `LAMBDA_COVER = 1.0`. |
| triplet completion | `_enumerate_triplets` | L-shaped triplets (arms 6-12.5 mm ± (tol-1), corner 65-115°) completed by the parallelogram rule, then the 4th corner re-placed where the bent-tile model fitted to the 3 real seeds predicts it (3 fixed-point rounds); rms <= 1.0·tol; score handicap 1.5 vs a quad; skipped when a detected candidate already occupies the corner (that quad was judged upstream). |
| reporting | `AutoFitResult` | `tentative_tiles` (confidence `"tentative"`, `inferred_seed_ras`), `unassigned_indices` (in-implant leftovers), `clutter_indices` (far), `capped`. `tiles` / `n_selected` unchanged. No supported tile -> no implant region -> nothing tentative (clutter-only scans stay at 0). |
| planner | `planner.py` | orange thin outline = tentative, hollow orange sphere = inferred seed, magenta = unassigned detected seed; status: `suggested 6 tile(s): 4 supported + 2 tentative`. |
| pipeline | `reconstruct(n_seeds_expected=)` | with `--seeds K` on a coarse scan: HU floor lowered 1000 -> 900 -> 800 while the in-implant count rises toward K without flooding (<= K+8, <= 60); logged in `vol.meta["seed_search"]`. |

Results: synthetic phantom (0.8 mm, 3 tiles, rng 2) with any one of the 12
seeds deleted -> 2 supported + 1 tentative tile, inferred seed within
0.3-2.1 mm of the truth seed, every seed explained.  Phantom thick-sliced
to 1.4 mm (rng 1): calibrated selection 2 of 3 tiles, cover pass recovers
the third (axis-free), nothing unassigned; at 2.1 mm one seed is lost by
detection and the tile comes back by triplet completion.  Real PostOp CT:
see `docs/data-notes.md`.

## Known limits / next

- The inferred seed of a triplet is a model prediction (bent tile fitted
  to three seeds), good to ~1-2 mm on a conformed tile; it is drawn hollow
  and never enters the supported set.
- A triplet's bent-tile residual is not discriminative (three points
  against a pose + scale + two-curvature model fit to ~0.05 mm on the
  PostOp case); the arm-length / corner-angle windows and the occupied-
  corner rule carry the rejection, which is why the tier is tentative.
- Split-blob duplicates on coarse scans (two candidates ~3 mm apart from
  one seed, PostOp seeds 34/35 and 4/37) stay unassigned rather than being
  merged; a coarse-scan merge step (candidates closer than a seed length
  that no tile can use) is the next detection improvement.
- Half tiles in auto mode need either the OR's confirmation
  (`n_half_tiles` non-zero / `--half 1`) or, with a cavity mesh, a surface
  verdict on the reported `half_candidates` (not yet wired).
- The deformable fit costs ~0.1 s per quad; the loose tier is bounded by a
  closed-form similarity prefilter and only runs on quads that fail the
  standard chord gates.
- The count-based `fit_tiles(n_full=int)` path is unchanged (classic gates,
  optional degraded completion); auto mode is the deformable-scored path.
