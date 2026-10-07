# Scout 7.3 — does the exact MILP scale on reduced instances?

Branch `plan/scout-milp-bound` (base `da71f76`). Evidence only; nothing here
is merged. Script: `scripts/scout_milp_bound.py`; raw rows:
`output/scout_milp/results.csv` (269 rows), full and condensed tables:
`output/scout_milp/summary_table.md`.

## Question

At what instance size (C candidates x M target points, N tiles) does the
section 3 E4 MILP (binary x_c, y_m; coverage rows sum_c D[c,m] x_c >= rx y_m;
conflict rows; sum x = N; max sum w_m y_m) reach proven optimality or a
<= 2 % gap with HiGHS in 60 s / 300 s, and when it does not, how tight are
the cheap bounds? Pre-declared criterion (plan section 7.3): the baseline is
adequate if on h = 4 mm, 2-3 spins, M <= 1000 it proves optimality or reaches
gap <= 2 % within 60 s for N <= 8 on all 3 cavities; otherwise an alternative
bound is worth promoting if it comes within 5 % of the greedy/SA incumbent.

## Method

- Cavities: `make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=k)`, k = 1..3;
  mesh from `truth.masks["cavity"]` via `segment.surface.mask_to_mesh`
  (areas 4178 / 4561 / 4444 mm2, volumes 24.6 / 27.8 / 26.8 cm3).
- Candidates: farthest-point anchors on the mesh vertices at spacing h, spins
  {0} / {0, 45} / {0, 30, 60} deg about the local normal, `snap_to_wall` +
  `conform_tile` (full tiles only); reject if any seed is > 1.5 mm from its
  3 mm wall offset (0 rejections on every instance).
- Conflicts: `find_overlapping_tiles` on the whole candidate list (its own
  bounding-sphere prefilter; on these ~18 mm-radius cavities nearly every pair
  survives it, so cost is O(C^2): 42 s at C = 486, 135 s at C = 786).
  Conflict density 40-43 % of all pairs.
- Influence: `dose_at_points` (cGy, total decay, nominal S_K) on the +5 mm
  shell (`shell_points`, 6.1-6.7 k vertices), full matrix cached, then an
  area-weighted stratified subsample of M points (FPS picks, each shell
  vertex's area assigned to its nearest pick). `exact=True` was used: the
  tabulated kernel was 30x slower per call here (1.7 s vs 0.05 s for 20 tiles
  x 6152 points) and agrees to 1e-5.
- No single tile reaches rx = 6000 cGy anywhere on the +5 mm shell (max
  4500-5260 cGy), so every covered point needs >= 2 tiles; the "N largest
  single-candidate coverages" bound is identically 0 and was replaced by the
  pointwise N-best bound (point m counts if its N largest single-candidate
  doses sum to >= rx) — which is 1.0 everywhere, i.e. uninformative.
- Formulations: (a) pairwise rows, (b) greedy edge-clique-cover rows (every
  conflict edge in >= 1 clique; 362 cliques of up to 32 candidates at
  C = 207), (c) both, (d) clique + "k-of-the-strong" rows
  k y_m <= sum_{c: D[c,m] > delta_k} x_c (delta_k = min_{j<k} (rx - S_j)/(N - j),
  S_j = sum of the j largest doses at m). Solver `scipy.optimize.milp`
  (HiGHS), `mip_rel_gap` 1e-6, `time_limit` 60 or 300 s, `disp` False.
- Bounds: LP relaxation (integrality 0) of (a), (b), (d); pointwise N-best.
- Incumbents: greedy forward selection by hard V100 (ties, which are universal
  until a point is covered, broken by weighted shortfall reduction, then
  lowest id) + first-improvement 1-swap local search ("greedy+LS").
- Alternative reference tried: a depth-first enumeration branch-and-bound over
  conflict-free N-subsets (`exact_bb`), pruning with
  gain <= sum of the r largest w({m uncovered : D[c,m] >= (rx - d_m)/r}) over
  still-allowed candidates (r tiles left); the last tile is read off directly.
- Hardware: AMD 12-core desktop; runs were launched as 4-8 concurrent
  single-threaded processes, so wall times carry ~10-20 % contention noise.

### Parameters

| h mm | spins | C (seed 1/2/3) | conflict pairs | build s (cand + conflicts) |
|---|---|---|---|---|
| 8 | 1 | 45 / 46 / 46 | 388-424 | ~1 |
| 6 | 1 | 69 / 76 / 72 | 1003-1066 | ~3 |
| 6 | 2 | 138 / 152 / 144 | 3975-4374 | ~7 |
| 4 | 2 | 324 / 330 / 332 | 20777-21966 | ~40 |
| 4 | 3 | 486 / 495 / 498 | 46891-49650 | ~55 |
| 3 | 3 | 786 (seed 1 only) | 130341 | ~150 (built, not solved) |

M in {100, 300, 1000}; N in {4, 6, 8}; rx = 6000 cGy; DETACHED 1.5 mm; FPS
seed 0; subsample seed 0.

## Results (condensed; V100 fractions of the +5 mm shell area)

MILP column: clique formulation, 60 s. Gap = (bound - incumbent)/incumbent as
HiGHS reports it. Full per-formulation rows (pair / both / clique_tight, LP
relaxations, 300 s runs, M = 100) are in `summary_table.md` / `results.csv`.

| seed | h | spins | C | M | N | greedy+LS | LP bound | MILP 60 s: status / inc / bound / gap | enum B&B 300 s: status / opt / time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 1 | 45 | 100 | 4 | 0.208 | 0.650 | opt / 0.269 / 0.269 / 0.00 | opt / 0.269 / 0 s |
| 1 | 8 | 1 | 45 | 100 | 6 | 0.208 | 0.969 | opt / 0.490 / 0.490 / 0.00 | opt / 0.490 / 0 s |
| 1 | 8 | 1 | 45 | 300 | 4 | 0.206 | 0.652 | limit / 0.258 / 0.512 / 0.99 | opt / 0.258 / 0 s |
| 1 | 8 | 1 | 45 | 300 | 6 | 0.206 | 0.971 | opt / 0.500 / 0.500 / 0.00 | opt / 0.500 / 0 s |
| 1 | 6 | 1 | 69 | 300 | 4 | 0.249 | 0.653 | limit / 0.249 / 0.575 / 1.31 | opt / 0.258 / 0 s |
| 1 | 6 | 1 | 69 | 300 | 6 | 0.485 | 0.972 | limit / 0.577 / 0.945 / 0.64 | opt / 0.577 / 0 s |
| 1 | 6 | 2 | 138 | 300 | 4 | 0.245 | 0.653 | limit / 0.247 / 0.580 / 1.34 | opt / 0.258 / 4 s |
| 1 | 6 | 2 | 138 | 300 | 6 | 0.544 | 0.973 | limit / 0.587 / 0.962 / 0.64 | opt / 0.610 / 14 s |
| 1 | 6 | 2 | 138 | 300 | 8 | 0.544 | 1.000 | opt / 1.000 / 1.000 / 0.00 | opt / 1.000 / 8 s |
| 1 | 4 | 2 | 324 | 300 | 4 | 0.253 | - | - | opt / 0.277 / 44 s |
| 1 | 4 | 2 | 324 | 300 | 6 | 0.582 | - | - | limit / 0.648 / 300 s |
| 1 | 4 | 2 | 324 | 300 | 8 | 1.000 | - | - | opt / 1.000 / 0 s |
| 1 | 4 | 3 | 486 | 300 | 4 | 0.275 | 0.654 | limit / 0.276 / 0.586 / 1.13 (300 s: 0.584 / 1.12) | opt / 0.285 / 160 s |
| 1 | 4 | 3 | 486 | 300 | 6 | 0.575 | 0.974 | limit / 0.566 / 0.966 / 0.71 | limit / 0.647 / 300 s |
| 1 | 4 | 3 | 486 | 300 | 8 | 0.993 | 1.000 | opt / 1.000 / 1.000 / 0.00 (9 s) | opt / 1.000 / 0 s |
| 1 | 4 | 3 | 486 | 1000 | 4 | 0.268 | 0.655 | limit / 0.125 / 0.589 / 3.70 | opt / 0.274 / 285 s (probe run, not in csv) |
| 1 | 4 | 3 | 486 | 1000 | 6 | 0.603 | 0.975 | limit / 0.481 / 0.967 / 1.01 | - |
| 1 | 4 | 3 | 486 | 1000 | 8 | 0.875 | 1.000 | opt / 1.000 / 1.000 / 0.00 (45 s) | - |
| 2 | 8 | 1 | 46 | 100 | 4 | 0.196 | 0.610 | limit / 0.196 / 0.395 / 1.01 | opt / 0.196 / 0 s |
| 2 | 8 | 1 | 46 | 100 | 6 | 0.549 | 0.910 | opt / 0.549 / 0.549 / 0.00 | opt / 0.549 / 0 s |
| 2 | 6 | 1 | 76 | 300 | 4 | 0.222 | 0.614 | limit / 0.222 / 0.536 / 1.41 | opt / 0.234 / 0 s |
| 2 | 6 | 1 | 76 | 300 | 6 | 0.477 | 0.913 | limit / 0.464 / 0.883 / 0.91 | opt / 0.528 / 2 s |
| 2 | 6 | 2 | 152 | 300 | 4 | 0.218 | 0.615 | limit / 0.204 / 0.546 / 1.67 | opt / 0.248 / 6 s |
| 2 | 6 | 2 | 152 | 300 | 6 | 0.476 | 0.914 | limit / 0.477 / 0.889 / 0.86 | opt / 0.546 / 49 s |
| 2 | 6 | 2 | 152 | 300 | 8 | 0.919 | 1.000 | opt / 1.000 / 1.000 / 0.00 | opt / 1.000 / 43 s |
| 2 | 4 | 2 | 330 | 300 | 4 | 0.240 | - | - | opt / 0.256 / 60 s |
| 2 | 4 | 2 | 330 | 300 | 6 | 0.488 | - | - | limit / 0.567 / 300 s |
| 2 | 4 | 2 | 330 | 300 | 8 | 0.773 | - | - | opt / 1.000 / 75 s |
| 2 | 4 | 3 | 495 | 300 | 4 | 0.235 | 0.616 | limit / 0.232 / 0.551 / 1.38 | opt / 0.259 / 237 s |
| 2 | 4 | 3 | 495 | 300 | 6 | 0.518 | 0.916 | limit / 0.453 / 0.894 / 0.97 | - |
| 2 | 4 | 3 | 495 | 300 | 8 | 0.957 | 1.000 | opt / 1.000 / 1.000 / 0.00 (41 s) | - |
| 2 | 4 | 3 | 495 | 1000 | 4 | 0.220 | 0.617 | limit / 0.176 / 0.554 / 2.15 | - |
| 2 | 4 | 3 | 495 | 1000 | 6 | 0.544 | 0.918 | limit / 0.002 / 0.898 / 459 | - |
| 2 | 4 | 3 | 495 | 1000 | 8 | 0.964 | 1.000 | limit / 0.002 / 1.000 / 457 | - |
| 3 | 8 | 1 | 46 | 100 | 4 | 0.156 | 0.624 | opt / 0.228 / 0.228 / 0.00 | opt / 0.228 / 0 s |
| 3 | 8 | 1 | 46 | 100 | 6 | 0.441 | 0.929 | opt / 0.521 / 0.521 / 0.00 | opt / 0.521 / 0 s |
| 3 | 6 | 1 | 72 | 300 | 4 | 0.195 | 0.624 | limit / 0.229 / 0.554 / 1.42 | opt / 0.234 / 0 s |
| 3 | 6 | 1 | 72 | 300 | 6 | 0.505 | 0.929 | limit / 0.545 / 0.907 / 0.66 | opt / 0.545 / 1 s |
| 3 | 6 | 2 | 144 | 300 | 4 | 0.195 | 0.625 | limit / 0.202 / 0.558 / 1.75 | opt / 0.252 / 4 s |
| 3 | 6 | 2 | 144 | 300 | 6 | 0.518 | 0.931 | limit / 0.505 / 0.911 / 0.80 | opt / 0.574 / 36 s |
| 3 | 6 | 2 | 144 | 300 | 8 | 0.724 | 1.000 | opt / 1.000 / 1.000 / 0.00 | opt / 1.000 / 25 s |
| 3 | 4 | 2 | 332 | 300 | 4 | 0.219 | - | - | opt / 0.264 / 67 s |
| 3 | 4 | 2 | 332 | 300 | 6 | 0.555 | - | - | limit / 0.585 / 300 s |
| 3 | 4 | 2 | 332 | 300 | 8 | 0.884 | - | - | opt / 1.000 / 2 s |
| 3 | 4 | 3 | 498 | 300 | 4 | 0.243 | 0.627 | limit / 0.227 / 0.563 / 1.48 | opt / 0.265 / 234 s |
| 3 | 4 | 3 | 498 | 300 | 6 | 0.551 | 0.932 | limit / 0.547 / 0.915 / 0.67 | - |
| 3 | 4 | 3 | 498 | 300 | 8 | 0.988 | 1.000 | limit / 0.980 / 1.000 / 0.02 | - |
| 3 | 4 | 3 | 498 | 1000 | 4 | 0.230 | 0.628 | limit / 0.151 / 0.565 / 2.73 | - |
| 3 | 4 | 3 | 498 | 1000 | 6 | 0.542 | 0.934 | limit / 0.002 / 0.916 / 377 | - |
| 3 | 4 | 3 | 498 | 1000 | 8 | 0.802 | 1.000 | limit / 0.002 / 1.000 / 489 | - |

Cross-check of the B&B: on the 1-spin instances it returns exactly the
HiGHS-proven optimum in all 8 cases where HiGHS closed (e.g. 0.269,
0.490, 0.500, 0.549, 0.228, 0.521) and is never below a HiGHS incumbent
elsewhere; it proves every C <= 76 instance in <= 2 s.

Formulation comparison (h = 4, 3 spins, M = 300, N = 4, 60 s, all three
cavities): pairwise, clique, both and clique+tight rows all end at the time
limit with dual bound 0.55-0.59 and gap 1.1-7.1; the clique form has the
best incumbents (0.23-0.28), pairwise/both/tight worse incumbents (0.07-0.23)
at the same bound. The LP relaxation is identical to 3 decimals for pairwise
and clique rows (e.g. 0.6546 vs 0.6543) and the k-of-the-strong rows move it
by <= 0.005 (0.627 -> 0.622 on seed 3). 300 s instead of 60 s moved the
seed 1 N = 4 bound from 0.586 to 0.584. N = 8 at h = 8 (1 spin) is
infeasible (no conflict-free 8-subset in 45 candidates).

## Verdict against the criterion

**The baseline MILP fails the pre-declared criterion.** On the reduced
instances (h = 4 mm, 3 spins, C ~ 490, M = 300 or 1000) HiGHS proves
optimality within 60 s only where the optimum is the trivial bound V100 = 1
(N = 8, 3 of 6 cases); for N = 4 and N = 6 it ends every run at the limit
with gaps of 67-460 % (not 2 %), and at M = 1000 it often has no useful
incumbent at all (0.002). This is not a size effect that a smaller instance
fixes: even C = 45-76, M = 300, N = 4 stays open at 60 s with gap ~ 1.0-1.8;
only C <= 46, M = 100 closes, and only on 2 of 3 cavities (17-33 s). 300 s does not help (bound moves by
0.002). The root cause is the relaxation, not the solver: no single tile
reaches rx, so coverage is a 2-3-tile additive event, and the LP can serve a
point with fractional "half tiles" from several neighbouring cliques; the
LP bound sits at 2.3-2.5x the proven optimum for N = 4 and 1.6-1.7x for N = 6, and
HiGHS's branching barely moves it.

**Cheap bounds are not within 5 % of the incumbent.** Pointwise N-best = 1.0
always; LP (any row set) = 0.61-0.65 for N = 4, 0.91-0.98 for N = 6; the
k-of-the-strong tightening is worth <= 0.5 pp. None of the section 7.3
alternatives based on relaxing the coverage rows (Lagrangian is bounded by
the same LP) can meet the 5 % criterion on this structure.

**What does work: enumeration branch-and-bound.** On h = 6 mm, 2 spins
(C = 138-152), M = 300, the DFS B&B proves the optimum for N = 4, 6, 8 on all
three cavities in 4-49 s (0.07-2.6 M nodes). On h = 4, 2-3 spins it proves
N = 4 in 44-237 s and N = 8 (V100 = 1) in 0-75 s, but N = 6 is open at
300 s with a bound of 1.0 (its node bound is weak at depth <= 3). Proven
optima are 4-29 % above greedy+LS (median ~ 10 %), so the optimality gap V3
wants to measure is real and large enough to need an exact reference.

## Recommendation for V3

- Do not use the HiGHS MILP as the V3 reference; keep it only as a sanity
  check on tiny cases (C <= 46, M = 100) and report its LP bound "as such"
  (it is a 1.6-2.5x overestimate, so it bounds nothing useful).
- Reduced instance for V3: **h = 6 mm, 2 spins, M = 300, N in {4, 6, 8}**,
  solved by the enumeration B&B (`exact_bb`), which proves optimality in
  < 1 min per (cavity, N). Add h = 4 mm / 2 spins for N = 4 (~1 min) as a
  discretization check. N = 6 at h = 4 needs a stronger node bound (e.g. a
  clique-group-aware top-r bound, or best-first search with the LS incumbent
  of each subtree) before it fits in 300 s; not pursued here.
- Report greedy+LS and SA gaps against those proven optima; on the current
  evidence greedy+LS sits 4-29 % below the optimum at N = 4-6.
- The MILP code path (clique rows are the formulation of choice if it is kept)
  should be documented as "incumbent finder with an uninformative bound".

## Commands

```
cd <worktree>; set PYTHONPATH=<worktree>
python scripts/scout_milp_bound.py --build-only --h 4 --spins 3
python scripts/scout_milp_bound.py --h 8 6 --spins 1 --M 100 300 --N 4 6 8 --forms clique --limits 60 --relax clique clique_tight --results output/scout_milp/res_F.csv
python scripts/scout_milp_bound.py --h 4 --spins 3 --M 300 --N 4 6 8 --forms clique --limits 60 --relax pair clique clique_tight --results output/scout_milp/res_A.csv   (seeds 1 2; res_A3.csv for seed 3)
python scripts/scout_milp_bound.py --h 4 --spins 3 --M 300 --N 4 --forms pair both clique_tight --limits 60 --results output/scout_milp/res_A2.csv
python scripts/scout_milp_bound.py --h 4 --spins 3 --M 1000 --N 4 6 8 --forms clique --limits 60 --relax clique --results output/scout_milp/res_B.csv
python scripts/scout_milp_bound.py --h 4 --spins 3 --seeds 1 --M 300 --N 4 8 --forms clique --limits 300 --results output/scout_milp/res_D.csv
python scripts/scout_milp_bound.py --h 6 --spins 2 --M 300 --N 4 6 8 --forms clique --limits 60 --relax clique --results output/scout_milp/res_E.csv
python scripts/scout_milp_bound.py --seeds K --h 6 4 --spins 2 --M 300 --N 4 6 8 --forms --bb 300 --results output/scout_milp/res_BBK.csv   (K = 1, 2, 3)
python scripts/scout_milp_bound.py --seeds K --h 4 --spins 3 --M 300 --N 4 --forms --bb 300 --results output/scout_milp/res_BB4|5|6.csv   (seed 1 also N 8)
python scripts/scout_milp_bound.py --seeds 1 2 3 --h 8 6 --spins 1 --M 100 300 --N 4 6 --forms --bb 60 --results output/scout_milp/res_X.csv
python scripts/scout_milp_bound.py --merge > output/scout_milp/summary_table.md
```

Provenance: the `clique_tight` rows produced before a sign fix in the
tightening rows (runs A seeds 1-2, E, F) were deleted from the per-run files
before merging; the B&B runs for N = 8 at h = 4 were re-run after capping
the node bound at the total weight (the uncapped version could not prune
against a V100 = 1 incumbent). Total compute ~55 min wall on 4-8 cores.

Commits on `plan/scout-milp-bound`: `85307d6` (script, doc, tables), followed
by the cross-check commit that adds the 1-spin B&B rows and this line.
