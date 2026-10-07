# optimize-notes — measurements and decisions for `gtcore.plan`

Companion to `docs/plan-tile-optimize.md`. Every number here carries the
command, seed, and commit hash that produced it (see "Runs log"). Negative
results stay in.

---

## Parameters

Every tunable is a named constant in `gtcore/plan/__init__.py` with the same
one-line justification as here (§5 of the plan). Sweeps that change a
default must update both.

| Constant | Value | Justification |
|---|---|---|
| `DEFAULT_H_MM` | 2.5 | Anchor spacing = 1/8 tile side: fine re-siting without an exploding candidate count (to be confirmed by V4). |
| `DEFAULT_N_SPINS_FULL` | 6 | Full tile spins 0, 15, …, 75°: the 2×2 seed grid has 90° symmetry. |
| `DEFAULT_N_SPINS_HALF` | 12 | Half tile spins 0, 15, …, 165°: a 10×20 strip has only 180° symmetry. |
| `DETACHED_MM` | 1.5 | Reject a candidate when a conformed seed is more than this off its 3 mm wall offset (half the 2.25–3.75 mm hydrated spread). |
| `DEFAULT_RX_CGY` | 6000 | GammaTile prescription: 60 Gy to the 5 mm shell. |
| `TARGET_SHELL_OFFSET_MM` | 5.0 | Default target = the +5 mm shell (clinical HR-CTV margin). |
| `M_OPT_MAX` | 4000 | Target points kept for optimization: C×M×4 bytes stays < ~100 MB at C ≈ 6000 and an objective evaluation stays ≈ 1 ms. |
| `TAU_FRACTION` | 0.05 | Soft-coverage sigmoid width τ = 0.05·rx: few-percent dose changes get a graded gain for annealing. |
| `LAMBDA_HOT` | 0.5 | P1 penalty per unit V200 above tolerance (plan §2 default). |
| `V200_TOL` | 0.10 | Tolerated V200 before the hot-spot penalty applies (plan §2). |
| `LAMBDA_OAR` | 1e3 | P1 penalty per cGy over an OAR limit: any violation dominates coverage. |
| `LOCAL_RADIUS_MM` | 10.0 | Local-search re-siting radius = half a tile side. |
| `SA_ALPHA` | 0.95 | Geometric cooling per sweep (plan §3 E3). |
| `SA_MOVES_PER_TILE_PER_SWEEP` | 50 | One sweep = 50·N proposed moves. |
| `SA_N_SWEEPS` | 40 | 40 sweeps at α = 0.95 cool T to ≈ 13 % of T0, past the freeze point. |
| `SA_N_RESTARTS` | 3 | One greedy start + two random feasible starts. |
| `MILP_TIME_LIMIT_S` | 60 | HiGHS limit per reduced instance; the bound is the reference when hit. |
| `TILE_AREA_CM2` | 4.0 | Manufacturer: a GammaTile is 2 × 2 cm (tile-count rule, plan §10). |
| `CONFLICT_GAP_MM` | 0.0 | Minimum edge gap beyond the planner's overlap rule (`find_overlapping_tiles`, threshold 1 mm). |

A1 module constants (`gtcore/plan/candidates.py`, `gtcore/plan/conflicts.py`;
implementation tunables, not optimizer parameters):

| Constant | Value | Justification |
|---|---|---|
| `candidates.DENSE_SAMPLES_PER_H2` | 20 | Dense surface samples per h² feeding farthest-point sampling: dense spacing ≈ h/4.5, well below the anchor spacing. |
| `candidates.DENSE_SAMPLES_MIN/MAX` | 500 / 200 000 | Bounds on the dense sample count (tiny / huge eligible regions). |
| `candidates.RAY_CHUNK` | 1000 | Grid points per batched fallback cast: bounds the transient (point, triangle) pair arrays (~2000 pairs per point on a 1 mm marching-cubes mesh). |
| `candidates.FALLBACK_LATERAL_TOL_MM` | 0.5 | Diagnostic only (`grid_fallback_flags`); not used for rejection (decision 10). |
| `candidates.VISIBLE_TOL_MM` | 0.5 | `visible_faces`: a first hit within this distance of the face centroid counts as the face (grazing an edge). |
| `candidates.ELLIPSOID_P` | 1.6075 | Knud Thomsen ellipsoid-area exponent (relative error < 1.1 %). |
| `candidates.MIN_FACES` | 4 | Fewer faces than a tetrahedron is a degenerate mesh (V8). |
| `conflicts.PLANNER_THRESHOLD_MM` | 1.0 | `find_overlapping_tiles`' default threshold; `gap_mm` adds to it (decision 11). |
| `conflicts.PAIR_CHUNK` | 512 | Candidate pairs per vectorized sample-distance batch (~30 MB transient). |
| `conflicts.ITEM_CHUNK` | 65 536 | (sample, triangle) pairs per batched point-triangle pass (~30 MB transient). |

---

## Go/no-go (§8)

_Not yet run._ Protocol: synthetic cavities, candidates at h = 3 mm, 6 spins,
influence on the +5 mm shell, greedy only; V100/D90 vs the uniform heuristic
and random feasible placements for N = 4, 6, 8.

| cavity (seed, volume) | N | V100 greedy | V100 uniform | V100 random (median / p95) | gain (pp) |
|---|---|---|---|---|---|

Verdict: _pending_ (≥ 3 pp → §3 as written; 1–3 pp → §7.7 framing; < 1 pp → stop, §7.1).

---

## V1 Unit / consistency

_Partly done._ Influence gate (§3 B) and weighted quantiles: below (A2).
Still pending: conflict symmetry and clique/pairwise agreement; greedy never
violates a conflict; SA reproducible from seed; MILP equals brute force on
the toy instance; the planner never flags an optimizer output as overlapping.

### A2 influence gate (§3 B)

`tests/test_plan_influence.py::test_influence_gate` — synthetic cavity
`make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)`, wall mesh from
`mask_to_mesh(truth.masks["cavity"], vol.affine)` (6152 vertices, watertight),
12 full-tile candidates at farthest-point anchors (spin 0, hand-built with
`conform_tile`; A1's `build_candidates` not used), target = the FULL +5 mm
shell (`TargetSet.from_shell`, M = 6152, no subsample, shell-area weights).
Row-sum (tabulated kernel) vs `compute_dose_grid(exact=True, spacing_mm=1.0)`
on mesh bounds ± 15 mm, trilinearly sampled at the same points
(`sample_doses`), both reduced with the same weighted metrics
(`objective.metrics_from_dose`). Five seeded draws (`default_rng(0)`) of
1–8 candidates:

| selection | V100 row / grid | D90 row / grid [cGy] |
|---|---|---|
| [0, 2, 3, 6, 8, 10, 11] | 0.6992 / 0.6991 | 4364 / 4367 |
| [2, 3, 4, 5, 6, 8, 10, 11] | 0.8475 / 0.8480 | 5677 / 5680 |
| [0, 1, 3, 5, 6, 8, 10] | 0.7273 / 0.7283 | 4830 / 4828 |
| [8] | 0.0000 / 0.0000 | 274 / 274 |
| [2, 4, 5, 6, 8] | 0.2526 / 0.2536 | 2652 / 2653 |

**Max deviation: |ΔV100| = 0.103 pp (gate 0.5 pp), |ΔD90| = 0.042 % of rx
(gate 1 %)**; max point-wise relative difference at points ≥ 0.5 rx is
5.2e-3 — the trilinear interpolation of the 1 mm grid, not the kernel
(tabulated vs exact rows agree to < 2e-3 in
`test_build_influence_matches_dose_at_points_rows`). Cheap flat-wall gate
(`test_influence_gate_flat_wall`, toy tiles, 400 equal-weight points at
z = +5): |ΔV100| = 0.000 pp, |ΔD90| = 0.033 % rx, point rel 2.9e-3.
Command: `python -m pytest -q -s tests/test_plan_influence.py`; seeds
phantom 1, draws 0; commit 5ea6f95. Whole module ≈ 13–20 s.

Weighted quantile: `objective.weighted_quantile` (weighted inverted CDF)
equals `numpy.percentile(method="inverted_cdf")` for unit weights over
n ∈ {1 … 1001} and 81 q values each, except where `n·q` overshoots an integer
by floating-point rounding (e.g. 25 × 0.28 = 7.000000000000001): numpy then
steps to element k + 1, ours returns element k (the definition); tested
explicitly (`test_weighted_quantile_unit_weights_matches_numpy_inverted_cdf`).
A3 (`tests/test_plan_solvers.py`, 26 tests): greedy never violates a
conflict (both toys, every N up to the packing limit) and is deterministic;
`fixed` ids are kept; an infeasible N fails with `status="infeasible"` and a
reason naming "4 of 5"; feasibility-aware greedy packs N = 9 on the 60-toy
where plain greedy stalls at 7; local search never decreases the hard
objective, ends feasible and is a fixed point of itself; SA is reproducible
from its seed (selection and history identical), feasible, and ≥ greedy on
both toys; `sweep_n` greedy rows are monotone in V100; E5 and
`solve_continuous` never lower V100 and never introduce an overlap on the
flat wall with the real engine; the evaluator routes through
`Objective.hard` / `metrics` / `gains_all` when they are real.

_Pending (other branches)._ Influence gate (§3 B); weighted quantiles; conflict symmetry and
clique/pairwise agreement; greedy never violates a conflict; SA reproducible
from seed; MILP equals brute force on the toy instance; the planner never
flags an optimizer output as overlapping.

## V2 Synthetic benchmark

### Declared design (A5, 2026-10-07, written BEFORE any arm was run)

Script: `python scripts/validation_optimize.py --section v2` (`--quick` = seeds
1–2, scale 1.0, N ∈ {4, 8}, n_random = 10). Outputs `output/validation_optimize/v2_*.csv`,
`v2_coverage_vs_n.png`, `v2_configs.pkl`; every run appends to `runs.csv`.

**Cavity grid.** `make_head_phantom(spacing=1.0, n_tiles=8, rng_seed=s)` for
seeds s ∈ {1, 2, 3, 4, 5, 6} × radius scales {0.8, 1.0, 1.25} = 18 cavities.
The generator has no size parameter; the scale multiplies the module
constant `CAVITY_RADII` = (20, 18, 17) mm around the call (nothing in the
generator is edited). Volumes = cavity-mask voxels × 1 mm³ (seed voxels
excluded), measured on seeds 1–2:

| scale | radii (mm) | volume (mL) | wall area (mm²) | manufacturer-rule tiles |
|---|---|---|---|---|
| 0.8 | 16.0 / 14.4 / 13.6 | 12.4, 14.0 | 2678, 2914 | 7, 8 |
| 1.0 | 20.0 / 18.0 / 17.0 | 24.4, 27.6 | 4178, 4561 | 11, 12 |
| 1.25 | 25.0 / 22.5 / 21.3 | 47.8, 54.2 | 6527, 7123 | 17, 18 |

The full table (all 18) is printed by the script as the first V2 block. Every
phantom carries 8 truth tiles regardless of scale (the generator's own
rejection-sampled placement), so the 0.8-scale truth implants are dense and
the 1.25-scale ones sparse — stated, not corrected.

**N list.** N ∈ {4, 6, 8, 10, 12} (12 added because the manufacturer rule gives
11–12 tiles for the scale-1.0 cavities; coordinator note 2026-10-07). On the
0.8-scale cavities N = 10, 12 may be packing-infeasible: such arm rows are
recorded as FAILED with `n_placed`, never as a lower score.

**Target.** `TargetSet.from_shell(mesh, 5)` (+5 mm shell, shell-area weights),
rx = 6000 cGy. Discrete candidate grid h = 4 mm, 3 spins, M_opt = 1000
(coordinator default pending V4).

**Arms** (per cavity × N): `random` (n_random = 50 feasible draws from the
candidate set; median and 95th percentile of V100 / D90), `uniform`
(farthest-point anchors over the distinct eligible anchors, spin-0 candidate,
next FPS point when a pick conflicts; local implementation in the script),
`greedy`, `greedy+local`, `sa` (seed 1000 + cavity seed), `milp` (reduced
instance h = 4 mm / 3 spins / M = 1000 / `MILP_TIME_LIMIT_S`; coverage only),
`continuous` (A3's `solve_continuous`, wall-time budget = greedy + SA wall
time on the same instance; skipped while absent), `truth` (the phantom's
truth tiles re-draped by `conform_tile` at their own anchor, N = 8 fixed) and
`truth_raw` (the raw truth seeds, N = 8 fixed).

**Endpoints.** V100, D90, V150, V200 of the +5 mm shell vertices and the
weighted V100 of the full target, all from `final_report` (`compute_dose_grid`,
exact kernel, 1 mm, mesh bounds + 15 mm) — never from the influence matrix;
solver wall time; `n_placed`; overlap count (planner rule).

**Statistics.** Mean ± SD across cavities per (N, arm); paired differences
with t-based 95 % CI and Wilcoxon signed-rank p for SA − greedy, SA − uniform,
greedy − uniform, (greedy+local) − greedy, SA − random (median), continuous − SA,
continuous − uniform.

**PRIMARY ENDPOINT.** SA − uniform on +5 mm-shell V100 at N*, where N* is,
per cavity, the smallest N in the list at which the uniform arm first reaches
D90 ≥ rx (if never: N* = 12 and the cavity is flagged in `v2_primary.csv`).
Paired mean difference, 95 % CI, Wilcoxon p across the 18 cavities. No
post-hoc switching.

### Results

_Pending A1–A4._ Smoke run at the Phase 0 stubs (commit cd73082,
`--quick`, 2026-10-07): only the two truth arms could run — V100 / D90 of the
conformed truth tiles 0.658 / 3772 cGy (seed 1) and 0.634 / 3594 cGy (seed 2);
raw seeds 0.647 / 3703 and 0.622 / 3539. Eight 3.5 U truth tiles do not reach
rx at +5 mm on these phantoms, and the planner's overlap rule flags 13–16
pairs among the conformed truth tiles (the generator packs tiles by seed
clearance, not footprint). Both numbers are references for the optimizer
arms, not results.

## V3 Optimality gap

Script: `--section v3` (seeds 1–6, scale 1.0, N ∈ N_LIST) on the reduced
instance h = 4 mm / 3 spins / M = 1000: gap = (V100_SA − ref) / ref with
V100 from the influence matrix (what the MILP optimizes; coverage only) and
ref = MILP incumbent, or the MILP bound when `status == "time_limit"` (bound
normalized by the target weight if the solver reports it unnormalized —
open decision 11). Grid V100 of both selections, and of the continuous arm
(budget = greedy + SA), reported alongside. Figure `v3_gap.png`.

_Pending A3–A4._
_Pending for SA._ (SA − MILP)/MILP on V100 over reduced instances; MIP gap
stated when the time limit was hit.

### A4 MILP on the toy instance

`gtcore.plan.milp` (branch `plan/milp`, based on cd73082; measurements from
the A4 commit's working tree). Command:
`python -m pytest -q tests/test_plan_milp.py` (20 tests, 17 s) asserts every
status / optimum below; the timings are from direct `solve_milp` /
`brute_force` calls on the same `toy_instance(...)` arguments (rng_seed 0) in
one interpreter session; scipy 1.18.1 / HiGHS; AMD Ryzen 5 7600X, 32 GB,
single thread. Coverage only
(V100); hot-spot terms are not in the MILP (§3 E4). Brute force enumerates
every conflict-free subset with the same OAR rows.

| instance (`toy_instance` args) | C | M | pairs / cliques | rows / nnz | N | status | V100 (MILP = brute force) | bound | gap | time |
|---|---|---|---|---|---|---|---|---|---|---|
| `n_candidates=12, n_targets=150` (pitch 10) | 12 | 150 | 57 / 12 | 208 / 2076 | 1 | optimal | 0.3133 | 0.3133 | 0 | 0.06 s |
| same | | | | | 2 | optimal | 0.6733 | 0.6733 | 0 | 0.08 s |
| same | | | | | 3 | infeasible (max independent set = 2) | — | — | — | 0.01 s |
| `…, pitch_mm=15` | 12 | 150 | 29 / 4 | 180 / 2020 | 1 | optimal | 0.1800 | 0.1800 | 0 | 0.10 s |
| same | | | | | 2 | optimal | 0.4867 | 0.4867 | 0 | 1.1 s |
| same | | | | | 3 | optimal | 0.6800 | 0.6800 | 0 | 0.5 s |
| same | | | | | 4 | optimal | 0.8267 | 0.8267 | 0 | 0.04 s |
| `n_candidates=80, n_targets=300` (pitch 10) | 80 | 300 | 712 / 80 | 381 / 24 998 | 6 | time_limit (0.5 s) | incumbent 0.5467 | 0.9967 | 0.82 | 0.5 s, 0 nodes |
| same | | | | | 6 | time_limit (60 s) | incumbent 0.8700 | 0.9933 | 0.14 | 60 s, 17 440 nodes |

LP relaxation (`lp_bound`): 0.843 / 0.953 for the pitch-15 toy at N = 2 / 3
(MILP optimum 0.487 / 0.680) and 1.000 for the 80-candidate toy at N = 6 —
weak, as expected: fractional `x` spreads dose over every point.

Findings. (1) MILP = brute force on every feasible (instance, N), to 1e-9 on
V100; pairwise-only and clique+pairwise formulations give the same optimum;
pruning at 1e-3·rx drops nothing on the toy (inverse-square dose never falls
below 1e-3·rx within 250 mm) and at 0.05·rx drops 6 entries without changing
the optimum. (2) **Negative result for §7.3:** the 80-candidate toy at N = 6
is not solved in 60 s (gap 14 %); neither the LP bound (1.00) nor the MILP
dual bound (0.993) is within 5 % of the incumbent (0.870). The toy grid is
highly symmetric (one spin, regular 10 mm pitch), which is the worst case
for branch-and-bound; real reduced instances (coarser h, few spins) may
behave differently and V3 must report the gap per instance. See Open
decisions 12 for the candidate tightenings.

#### Pigeonhole cover cuts (coordinator follow-up, same commit series)

Rows `y_m ≤ Σ_{c : D[c,m] ≥ rx/N} x_c` (valid for both count forms: a covered
point gets ≥ rx from ≤ N candidates, so one gives ≥ rx/N) plus fixing
`y_m = 0` for points with no candidate at ≥ rx/N. Keyword `cover_cuts` on
`solve_milp` / `lp_bound` / `build_formulation`. Same command/seeds as above;
the toy80 rows are from an uncontended rerun (an earlier run overlapped a
pytest run and gave 6047 / 2817 nodes instead; node counts vary between HiGHS
runs at a time limit).

| instance, N | cuts | status | incumbent V100 | bound | gap | nodes | rows / nnz | fixed y=0 | LP bound |
|---|---|---|---|---|---|---|---|---|---|
| pitch-15 toy, N=1 | off / on | optimal / optimal | 0.1800 / 0.1800 | — | 0 / 0 | — | 162 / 244 | 0 / 34 | 0.5335 / **0.1800** |
| pitch-15 toy, N=2 | off / on | optimal / optimal | 0.4867 / 0.4867 | — | 0 / 0 | — | 162 / 288 | 0 / 12 | 0.8432 / **0.6046** |
| pitch-15 toy, N=3 | off / on | optimal / optimal | 0.6800 / 0.6800 | — | 0 / 0 | — | 162 / 312 | 0 / 0 | 0.9531 / 0.8973 |
| pitch-15 toy, N=4 | off / on | optimal / optimal | 0.8267 / 0.8267 | — | 0 / 0 | — | 162 / 312 | 0 / 0 | 0.9664 / 0.9659 |
| toy80, N=6, 0.5 s | off | time_limit | 0.5467 | 0.9967 | 0.82 | 0 | 381 / 24 998 | 0 | 1.0000 |
| toy80, N=6, 0.5 s | on | time_limit | 0.0067 | 1.0000 | 149 | 0 | 681 / 33 945 | 0 | 1.0000 |
| toy80, N=6, 60 s | off | time_limit | 0.8667 | 0.9967 | 0.150 | 8305 | 381 / 24 998 | 0 | 1.0000 |
| toy80, N=6, 60 s | on | time_limit | 0.7867 | 0.9933 | 0.263 | 3557 | 681 / 33 945 | 0 | 1.0000 |

Verdict. The optimum is unchanged everywhere (= brute force; test
`test_cover_cuts_do_not_change_optimum` covers N = 1–4, both count forms).
The cuts bite only when rx/N is large: on the pitch-15 toy they make the LP
bound exact at N = 1 and ~30 % tighter at N = 2, and fix 34 / 12 uncoverable
points. On toy80 at N = 6 (rx/6 is reached by many candidates per point) they
tighten nothing — the dual bound is the same, the LP bound stays 1.0 — and
the extra 300 rows cost node throughput, so the 60 s incumbent is worse
(0.787 vs 0.867). Defaults therefore: `COVER_CUTS = False` for `solve_milp`,
`LP_COVER_CUTS = True` for `lp_bound` (a valid row never loosens an LP). The
§7.3 negative result stands: no bound within 5 % of the incumbent on this toy.

#### Enumeration branch-and-bound (`solve_enumeration`, the §7.3 scout's `--bb`)

Port of the scout's exact DFS over conflict-free N-subsets (fixed candidate
order, every subset once; incremental conflict masks; OAR limits pruned at
each node; incumbent from `solve_greedy` when implemented, else `start` or
the first leaf). Node bound: with r tiles left and current dose d, a newly
covered m needs ≥ (rx − d_m)/r from one new tile, so gain ≤ sum of the r
largest w({m uncovered : D[c,m] ≥ (rx − d_m)/r}) over the allowed c. At the
time limit `bound` = max(incumbent, bound of every open node). `solve_milp`
now routes with `method="auto"`: enumeration when C ≤ 600 and N ≤ 8
(`ENUM_MAX_C`, `ENUM_MAX_N`), HiGHS otherwise; `method="enum"|"highs"` force.
Same instances / seed / hardware as above; direct calls, uncontended.

| instance | N | enumeration (order=degree) | nodes | time | HiGHS (60 s, no cuts) for comparison |
|---|---|---|---|---|---|
| pitch-10 toy (C=12) | 1 / 2 / 3 | optimal 0.3133 / 0.6733 / infeasible | 1 / 13 / 19 | < 0.01 s | same |
| pitch-15 toy (C=12) | 1 / 2 / 3 / 4 | optimal 0.1800 / 0.4867 / 0.6800 / 0.8267 | 1 / 13 / 45 / 64 | < 0.05 s | same (0.04–1.1 s) |
| toy80 (C=80, M=300) | 4 | **optimal 0.5833** | 37 772 | 1.0 s | 5 s: time_limit, incumbent 0.5433, bound 0.8833 |
| toy80 | 6 | **optimal 0.9033** | 1 304 394 | 34.2 s | 60 s: time_limit, incumbent 0.8667, bound 0.9967 (gap 0.15) |
| toy80 | 8 | **optimal 1.0000** | 509 530 | 10.0 s | — |
| toy80, 0.05 s limit | 6 | time_limit, incumbent 0.84, bound 1.00 | 1 958 (169 open) | 0.05 s | — |

`order="potential"` (the scout's weighted-dose-potential order) gives the
same optima with 63–71 nodes on the toys (table uses the default
"single-coverage, then degree" order). Both count forms (exact N and ≤ N)
match brute force on every toy case (`test_enumeration_equals_brute_force`).

Findings. The enumeration proves the toy80 optimum at N = 6 in 34 s where
HiGHS had stalled at 60 s with a 0.867 incumbent (4 % below the true
optimum 0.903) and a 0.997 bound; the true V3 gap of that HiGHS incumbent
was 0.04, not the reported 0.15. This matches the scout's finding on real
synthetic cavities (C = 138–498: HiGHS gaps 67–460 %, enumeration 4–237 s).
The HiGHS path stays as the fallback above the size limits and as an
incumbent finder; its bound must be reported as "loose" in V3. No synthetic
cavity run here: A1 (`build_candidates`) is not on main as of this commit
(`git log main` shows A2 and A3 merged only), so the V3 cavity rows wait for
sync point (1).

## V4 Discretization

Script: `--section v4` (seeds 1–3, scale 1.0, N = 8): h ∈ {4, 3, 2.5, 2} mm ×
n_spins ∈ {3, 6, 12}; greedy, SA and continuous (same budget) per cell;
candidate / influence build time; E5 `refine_continuous` gain on the SA
solution (grid V100 before / after).

_Pending A1–A3._

## V5 Sensitivity

Script: `--section v5` (seeds 1–3, scale 1.0, N = 8; arms truth / uniform /
greedy / SA). Perturbations at evaluation time: seed-plane offset 2.25 and
3.75 mm (`geometry.SEED_PLANE_OFFSET_RANGE_MM`; every seed shifted by
(offset − 3) mm along its tile's inward anchor normal — the per-seed local
normal is not stored on `PlacedTile`), S_K ± 5 % (`parameters["sk_per_seed_u"]`
of `final_report`), interference on (capsules only). Perturbation at solve
time: M_opt = 4000 instead of 1000 (greedy, SA re-solved). Reported: V100
spread (max − min over perturbations) per arm, and Kendall τ between the arm
ranking under the nominal evaluation and under each perturbation.

_Pending A1–A3._

## V6 Physical phantom and clinical case

Script: `python scripts/validation_optimize.py --section v6 [--clinical <dir>]`.
Pipeline result cached under `output/validation_optimize/cache/` (reconstruct
≈ 30 s).

**Wall and eligibility (definition).** The 8-tile printed phantom
(`3D-Printed Phantom-8tiles (223)`, 157 slices) has no cavity mask
(`result.cavity_mask` is empty: printed shell, no brain window), so the wall
is `result.meshes["body"]` — one watertight component that includes the
OUTER surface of the printed shell (53 863 mm², 474 mL enclosed, 72 396
faces). Eligible faces = faces whose centroid is the first hit of a ray from
the implant centroid (mean of the localized seeds; `plan.visible_faces`, or
the script's identical local ray test while A1's is a stub) AND whose
centroid lies within 35 mm of a detected seed (the printed cavity is a wall
patch, not a closed cavity; without the radius the visible set extends over
the whole inner shell). Target / HR-CTV = +5 mm shell of the eligible faces
only (shell vertices of eligible faces, one third of the adjacent eligible
shell-face areas as weights). The optimizer gets the same `eligible_faces`.

**Caveats.** (i) Localized seeds sit ≈ 1 mm off this mesh (the mesh is the
printed surface; the tiles were glued on it), while the conformer puts seeds
at 3 mm — arm (a′) runs the as-implanted tiles through `conform_tile` at
their own anchor / axis and reports what that substitution alone changes
before any optimization is compared. (ii) The "tissue" side of the +5 mm
shell is printed plastic; dose is TG-43 water. (iii) The vault filter is
skipped by the pipeline on this scan (no cranial interior), so the 32
localized seeds are taken as found. (iv) `recommend_tile_count` is reported
on the eligible area against the implanted 8.

Arms: (a) as-implanted (`tiles.auto.to_placed_tiles`, localized seeds),
(a′) as-implanted through the conformer, (b) uniform / greedy / SA / MILP /
continuous with N = 8, (c) minimum N by P2 (`sweep_n`, greedy, N ≤ 12).
Figure `v6_dvh.png`: weighted DVH of the eligible +5 mm shell per arm.

Endpoints for V6 are the WEIGHTED stats of the eligible +5 mm target (the
body mesh's own shell vertices include the outer surface of the printed
shell, so the unweighted shell numbers are kept only as `*_shellverts`).

Smoke run (2026-10-07, `--quick --section v6`, commit cd73082 + A5; arms
(b)/(c) skipped — A1–A4 stubs):

| quantity | value |
|---|---|
| mesh area / enclosed volume / faces | 53 863 mm² / 473.9 mL / 72 396 |
| visible from the implant centroid (local ray test) | 4651 mm² (7347 faces) — identical to "visible AND ≤ 35 mm of a seed" on this scan |
| eligible +5 mm target | 4210 points, 15 539 mm² (shell area, the weights' sum) |
| localized seeds / fitted tiles | 32 / 8 |
| recommended tiles (local rule, eligible area / 4 cm²) | **12** vs 8 implanted (`recommend_tile_count` pending A1; its ellipsoid estimate not available yet) |
| localized seeds off the mesh | mean 0.84 mm (max 2.6 mm) |
| (a) as-implanted, localized seeds | V100 0.654, D90 2349 cGy, V150 0.552, V200 0.418; 3 overlap pairs |
| (a′) same tiles through `conform_tile` (seeds to 3 mm; mean seed shift 3.94 mm, nearest-seed matching) | V100 0.671, D90 2345 cGy; 6 overlap pairs |
| conformer substitution alone | **+1.75 pp V100, −4 cGy D90** |

Eight tiles cover only ~65 % of the eligible +5 mm target at rx on this
phantom (the +5 mm shell of a 46.5 cm² patch is 155 cm² of shell; the
manufacturer rule asks for 12 tiles). The conformer's snap must use the
tile's seed centroid, not its anchor, on this closed shell mesh: the anchor
(3 mm behind the seed plane) lies inside the plastic, and the nearest mesh
point from there can be the outer surface (first smoke run: 8 mm "shift"
before this fix) — flagged for A1/A6 under open decision 17.

**Clinical case 2:** not on this machine — the `--clinical <path>` hook exists
and prints "not run"; RTSTRUCT HR-CTV handling is not written.

## V7 Runtime

_Partly done (A2: influence and objective)._ Pending: candidate build, each
solver, final reporting; per cavity size.

Hardware: AMD Ryzen (Family 25 Model 97, laptop, Windows 11), Python 3.12.10,
numpy 2.5.2, scipy 1.18.1; single thread apart from BLAS. Numbers fluctuate
by ~1.5–2× with background load; both ends of the observed range are given.

| quantity | measured | budget | test / command | seed | commit |
|---|---|---|---|---|---|
| influence, per candidate, M = 4000, tabulated kernel, warm engine | 0.9–1.6 ms (4 seeds / candidate) | ≤ 5 ms | `test_influence_per_candidate_time` (`pytest -s tests/test_plan_influence.py`) | 0 | 5ea6f95 |
| influence, per candidate, exact kernel | ≈ 2.8 ms | — | scratch benchmark of `dose_at_points(exact=True)` (same points) | 0 | 5ea6f95 |
| `TG43Engine` kernel table (one-off) | ≈ 0.1 s | — | same | — | — |
| `Objective.hard`, M = 4000, N = 8, mean of 100 calls | 0.04–0.14 ms | ≤ 1 ms (test asserts ≤ 2 ms) | `test_hard_timing_m4000_n8` (`pytest -s tests/test_plan_objective.py`) | 0 | 5ea6f95 |
| `objective.gains_all`, C = 2000, M = 4000, mean of 5 calls | 40–49 ms | ≤ 50 ms (test asserts ≤ 100 ms) | `test_gains_all_timing_c2000_m4000` | 0 | 5ea6f95 |
| `objective.soft_gains_all`, C = 2000, M = 4000 | 120–140 ms | — | scratch benchmark | 0 | 5ea6f95 |
| `Objective.gain` (one candidate, incremental) | 0.07–0.2 ms | — | scratch benchmark | 0 | 5ea6f95 |
| `Objective.metrics`, M = 4000 | 0.25–0.4 ms | — | scratch benchmark | 0 | 5ea6f95 |

Influence structure: one `dose_at_points` call per candidate (it sums the
candidate's seeds and chunks over points) — at ~1 ms it is 5× inside the
budget, so no restructuring onto the engine's per-seed internals was done.
`gains_all` cost is two `(C, M)` comparisons + two matmuls with the weight
vector, chunked by 256 rows (`GAINS_CHUNK_ROWS`); the float32 rows are
compared against float32 thresholds rounded *up* from `rx − base`, which is
bit-identical to the float64 comparison and avoids an 8 MB upcast per chunk
(52 → 40 ms).
_Pending for the real modules._ Candidate build, influence, each solver,
final reporting; per cavity size; hardware stated.

### A3 solvers (toy instance)

Branch `plan/solvers`, code base cd73082 (Phase 0) + the A3 commit that adds
this section. Hardware: AMD64 family 25 model 97 (Ryzen 7000 class), Python
3.12.10, single thread. Command (from the worktree root):
`python scratch/timing.py` = the sequence below, equivalently
`python -m pytest -q tests/test_plan_solvers.py` for the assertions. Seeds:
`toy_instance(rng_seed=0)`, SA `seed=0`. Objective: A3's `_HardEval` fallback
(A2's `Objective` methods are stubs on this branch); the toy dose is the
analytic inverse-square kernel, not the engine.

| instance | solver | wall | hard P1 | V100 | evaluations |
|---|---|---|---|---|---|
| toy C=30, N=4, M=200, 213 conflict pairs | greedy (feasibility-aware) | 0.002 s | 0.6625 | 0.805 | 121 |
| | greedy (plain) | 0.001 s | 0.6625 | 0.805 | 121 |
| | local search from greedy (r = 10 mm) | 0.005 s | 0.8175 | 0.890 | 85 |
| | SA, 40 sweeps × 3 restarts, 50·N moves/sweep | 1.59 s | 0.8300 | 0.920 | 19 469 |
| | SA without `candidates` (all moves global) | 1.38 s | 0.8300 | 0.920 | 24 306 |
| | `sweep_n` greedy N = 1..4 | 0.017 s | — | 0.805 at N = 4 | — |
| toy C=60, N=9, M=250, 504 conflict pairs | greedy (feasibility-aware) | 0.007 s | 0.6100 | 0.992 | 385 |
| | greedy (plain) | 0.003 s | **infeasible, 7 of 9 placed** | 0.884 | 307 |
| | local search from greedy | 0.004 s | 0.6640 | 1.000 | 29 |
| | SA, 40 × 3 | 3.60 s | 0.6760 | 1.000 | 28 105 |
| | SA without `candidates` | 2.43 s | 0.6760 | 1.000 | 31 371 |
| | `sweep_n` greedy N = 1..9 | 0.022 s | — | min N: D90 ≥ rx at 8, V100 ≥ 0.90 at 8 | — |

SA diagnostics: T0 = 0.0398 (C=30) / 0.0146 (C=60) in soft-objective units
(a fraction of the target), acceptance rates 0.26–0.28 per restart, all
three restarts reached the same best on both toys (the toy is small and
saturates). On C=60 the hard P1 value is dominated by the V200 penalty
(V100 = 1.0 but V200 ≈ 0.8 at nine tiles on a 250-point shell), so "better
hard" there means fewer hot points, not more coverage.

Real engine (tabulated kernel, `dose_at_points(exact=False)`), flat 60 mm
wall, 300 target points on the +5 mm plane, rx 3000 cGy (one tile reaches
≈ 4200 cGy at 5 mm, so rx 6000 would give V100 = 0 for any pair):

| run | wall | per tile-NM | evals / tile-NM | result |
|---|---|---|---|---|
| E5 `refine_continuous`, 2 tiles, 2 passes, max_iter 60, step 1 mm / 5° | 1.9 s | ≈ 0.45 s | ≈ 65 | V100 0.387 → 0.413, soft 0.385 → 0.402, 3 of 4 steps accepted, no overlap |
| `solve_continuous`, 3 × 3 candidate grid (15 mm pitch), N = 2, 3 starts × 2 passes, max_iter 60, step 2 mm / 10° | 7.7 s | 0.63 s | 60 | greedy start hard 0.423 → best 0.450 (starts: 0.450, 0.450, 0.293) |
| same, 2 starts × 1 pass, max_iter 30 | 1.1 s | 0.27 s | 54 | 0.423 → 0.450 |

One NM evaluation costs ≈ 10 ms at 300 target points: `conform_tile` ≈ 3 ms
on the 2160-face wall + `dose_at_points` ≈ 4 ms for 4 seeds + overhead. The
per-tile NM cost therefore scales with mesh density (ray casts) and with
`m_opt`; at the recommended h = 4 mm / 3 spins start grid and M = 1000 the
scout's ~25 s for N = 6 × 2 passes is consistent with ≈ 1 s per tile-NM.

Test module: `python -m pytest -q tests/test_plan_solvers.py` → 26 passed in
≈ 16 s (E5 and continuous tests use the real engine, the rest the toy).
Script: `python scripts/validation_optimize.py --section v7` → `v7_runtime.csv`,
`v7_hardware.json`: candidates / conflicts / influence (M = 4000) / greedy /
greedy+local / SA / MILP / continuous (with its budget) / `final_report`
(15 mm margin, and the full 50 mm + rind + shadowing report) at N = 8 for seed
1 at each scale. Hardware: Windows 11 (10.0.26200), AMD64 Family 25 Model 97
(AuthenticAMD), 12 logical cores, Python 3.x, numpy 2.x (exact strings in
`v7_hardware.json`).

_Pending A1–A4_ for the solver columns. Smoke run (cavity s1, 24.4 mL, 8
truth tiles / 32 seeds): `final_report` 15 mm margin 2.3 s; full 50 mm
margin + rind DVH + shadowing 21.5 s (grid 1 mm, ≈ 3.4 M voxels, exact
kernel). Printed phantom (body mesh, 32 seeds): ≈ 60 s per 15 mm report; the
visible-face ray test over 72 k faces ≈ 6 min once (cached under
`cache/printed_visible_*.npy`); `reconstruct` 30 s (cached).
_Candidate build and conflict graph measured (A1); influence, solvers and
final reporting pending._

### A1 candidates/conflicts

Command: `python scripts/plan_candidates_runtime.py` (from the repo root;
`rng_seed=0` for the anchors, phantom `rng_seed=1`), commit `8a98e7e`,
2026-10-07. Hardware: AMD64 Family 25 Model 97 (Ryzen 7000-class desktop),
single process; Python 3.12.10, numpy 2.5.2, trimesh 5.1.0 (pure-python
`ray_triangle` intersector, no embree). Cavity =
`make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)`,
`mask_to_mesh(truth.masks["cavity"], vol.affine)`: 12 300 faces, 4178 mm²,
watertight, 24.6 cm³, extents 40 × 36 × 33 mm. Hollow sphere = icosphere
r = 25 mm (inverted) + r = 35 mm as one watertight mesh (2560 faces);
`visible_faces(hollow, origin)` took 0.20 s and marked 1280/1280 inner and
0/1280 outer faces.

| run | faces / area mm² | enumerated | C | rejected | build s | s per 100 | conflict pairs | mean degree (density) | cliques | conflicts s |
|---|---|---|---|---|---|---|---|---|---|---|
| cavity seed 1, h = 3, 6 spins | 12300 / 4178 | 1560 | 1514 | hanging 46 | 20.1 | 1.29 | 503 631 | 665.3 (0.440) | 517 (max 176) | 99.6 |
| cavity seed 1, h = 2.5, 6 spins | 12300 / 4178 | 2328 | 2252 | hanging 76 | 26.3 | 1.13 | 1 097 002 | 974.2 (0.433) | 771 (max 237) | 202.0 |
| hollow sphere inner wall (E = `visible_faces`), h = 3, 6 spins | 2560 / 23137 (E = 7845) | 3084 | 3084 | none | 11.2 | 0.36 | 1 187 182 | 769.9 (0.250) | 1028 (max 50) | 464.6 |

Columns: *enumerated* = anchors × spins × kinds before rejection; *C* =
accepted candidates; *s per 100* = build seconds per 100 enumerated
candidates; *density* = mean degree / (C − 1); *cliques* = verified
cliques (max size).

Reading:
- The build meets the brief's target (≤ 2 s per 100 candidates) on all
  three: 1.1–1.3 s/100 on the 12 300-face marching-cubes cavity and 0.36
  s/100 on the 2560-face shell. `conform_tile` dominates (≈ 12 ms per
  accepted tile on the 1 mm cavity mesh, of which its own ray casts are the
  larger part); the batched fallback cast is ≈ 1–2 ms per enumerated
  candidate and spares `conform_tile` for the hanging ones. The same cavity
  measured 3.4 s/100 with the first implementation (per-tile trimesh ray
  replica, conforming before the hanging test).
- Rejections on the closed cavity are all "hanging" (3–4 %; tiles whose
  tangent-plane grid points miss the wall or sag > 12 mm at the lumpy
  bumps); none are detached at `DETACHED_MM` = 1.5.
- Conflict graphs are dense: a 20 mm tile on a 40 mm cavity conflicts with
  everything whose anchor lies within ≈ 25 mm, i.e. 44 % of all candidates.
  The pairwise stage runs at ≈ 0.1 ms per candidate pair that passes the
  bounding-sphere stage (≈ 1 M pairs at h = 2.5 → 200 s). The hollow sphere
  is the slow case (465 s for C = 3084) because its smooth icosphere wall
  triggers the `_footprint_surface` conditioning defect (Open decision
  14): ballooned footprints defeat the bounding-sphere prefilter, so
  nearly every pair reaches the vectorized stages. On the marching-cubes
  cavity the footprints are sane (bounding radius p90 14 mm).
- Cliques: 517 / 771 verified cliques on the cavity (one per anchor plus
  the reduced anchor neighbourhoods, deduplicated), the largest covering
  237 candidates (all spins of the ≈ 40 anchors nearest one anchor).

Unit-test-scale timings (same hardware, `pytest --durations`): sphere r =
25 mm (5120 faces), h = 6, 2 spins: 260 candidates in 3.9 s (1.5 s/100);
flat 60 mm wall, h = 5, 3 spins: 261 enumerated / 154 accepted in 0.9 s;
`build_conflicts` on 444 cavity candidates (h = 4, 3 spins; 98 007 close
pairs, 42 813 conflicts) 10.5 s.

## V8 Failure modes

Script: `python scripts/validation_optimize.py --section v8` →
`v8_failure_modes.csv`. Verdict PASS = an exception with a non-empty message,
or a `SolverResult` / report whose `status != "ok"` carries a reason; a plain
return (or an empty plan without a reason) is FAIL.

| case | function | outcome | verdict |
|---|---|---|---|
| N larger than any packing (icosphere r = 12 mm, N = 12) | `optimize`; `build_candidates` + `solve_greedy` | _stub_ | not run |
| tile wider than the wall patch (eligible = 10 mm disc on the flat wall) | `build_candidates`; `optimize` | _stub_ | not run |
| eligibility mask excluding > 95 % of the wall (sphere cap z > 24.3 mm of r = 25) | `optimize` | _stub_ | not run |
| degenerate mesh (3 vertices, 1 face) | `build_candidates`; `optimize` | _stub_ | not run |
| empty mesh | `build_candidates`; `optimize` | _stub_ | not run |
| empty mesh | `final_report` | raised `ValueError("final_report: mesh is empty or None")` | PASS |

(2026-10-07, commit cd73082 + A5 changes, `--quick --section v8`.)

---

## Open decisions

Phase 0 (interface freeze, 2026-10-07). Each was left open by the
directions; the simpler answer was taken and is flagged here for the branch
agents.

1. **Target weights are areas of the offset shell, not of the wall.**
   `TargetSet.from_shell` carries the mesh faces onto the shell vertices and
   uses one third of the adjacent *shell* face areas, so `total_weight`
   equals the shell area (the quantity V100 is a fraction of). Alternative:
   wall-face areas (44 % smaller on a 25 mm sphere). Chosen: shell areas.
2. **Subsample weights.** `TargetSet.subsample` draws by systematic PPS
   sampling and gives the drawn points equal weights `remaining/n_drawn`
   (certainty points keep theirs) so `total_weight` is preserved and V100 is
   an unbiased estimate. Alternative: keep the original weights (biased
   toward large-area vertices). Chosen: equal weights.
3. **`seeds_of` / `tiles_of` / `subset` order by ascending candidate id**,
   independent of the order ids appear in the selection; duplicates are
   dropped. Alternative: preserve selection order. Chosen: ascending (stable
   seed arrays for the engine regardless of solver bookkeeping).
4. **Toy conflict rule.** `tests/plan_fixtures.toy_instance` uses the
   planner's own `find_overlapping_tiles(threshold_mm=1.0)` instead of a
   corner-distance proxy, so abutting tiles on the 10 mm grid conflict (as
   the planner would flag them). Alternative: strict footprint overlap only.
   Chosen: the planner rule — toy solver tests then match V1's "never
   flagged" requirement. A1 must decide how `CONFLICT_GAP_MM` composes with
   the planner's 1 mm threshold.
5. **E5 `refine_continuous` is assigned to A3 (`plan/solvers`)** — §6 lists no
   owner. Alternative: A1 (it is conformer geometry). Chosen: A3.
6. **`recommend_tile_count` is assigned to A1 (`plan/candidates`)** — it is
   mesh-area geometry with an `eligible_faces` mask. Alternative: A6 (UI).
   Chosen: A1; A6 wires the pre-filled prompt.
7. **`ConflictGraph.count_pairs` is a method** (not a property) returning the
   number of unordered pairs (`nnz // 2`).
8. **`CandidateSet.from_tiles`** was added as a non-frozen convenience
   constructor (builds the array views from `PlacedTile`s). Branches may use
   it; the frozen fields are unchanged.
9. **`OptimizeReport.metrics_grid` keys** are shell offsets (floats) in
   memory and strings in JSON (`"5.0"`); CSV keys are dotted
   (`metrics_grid.5.0.D90`).

A2 (`plan/influence`, 2026-10-07):

10. **Weighted quantile = weighted inverted CDF** (smallest sorted value whose
    cumulative weight reaches `q·W`; D90 = `weighted_quantile(D, w, 0.10)`),
    a dose that actually occurs in the sample. Alternative: weighted linear
    interpolation (numpy's default `linear` for unit weights, what
    `dvh_stats` uses on grids). Chosen: inverted CDF — order statistics on
    weighted samples have no canonical interpolation, and the grid-side
    `dvh_stats` D90 differs from it only by one sample's spacing.
11. **`gains_all` / `soft_gains_all` are module functions**
    (`gtcore.plan.objective.gains_all(objective, selection, dose_vec=None)`),
    not `Objective` methods, because the frozen dataclass may not grow
    methods on a branch. Alternative: add them as methods at merge time
    (coordinator). Chosen: functions; A3's greedy / SA should import them.
12. **`make_objective` takes `rx_cgy` from `influence.rx_cgy`** unless
    overridden. Alternative: always `DEFAULT_RX_CGY`. Chosen: the matrix's
    prescription — a matrix built for 5000 cGy evaluated at 6000 would be a
    silent error.
13. **`gain` is incremental** (`hard(sel ∪ {c}) − hard(sel)` from the cached
    base dose; a candidate already in `sel` has gain 0) and agrees with the
    full recomputation to 1e-9 (tested). Alternative: recompute `hard` on
    the union (exactly consistent, 2× slower). Chosen: incremental.
14. **OAR penalties apply only to OARs with a limit in `oar_limits`**; other
    OAR sets are metrics-only (`oar_dmax_<name>` reported, no penalty).
    Alternative: require a limit per OAR. Chosen: metrics-only is useful for
    reporting a structure without constraining it.
15. **Influence rows one engine call per candidate** (`dose_at_points`,
    tabulated). Alternative: evaluate a (seeds × points) block through the
    engine's `_rate_tabulated_folded` and reduce by candidate (measured
    1.7 ms vs 1.1 ms per candidate in a scratch benchmark — the per-seed
    geometry, not the call overhead, dominates). Chosen: the public call.
16. **Gate compares the tabulated row-sum against the exact 1 mm grid
    sampled trilinearly** (what §3 G reports), on the full shell with
    shell-area weights, both reduced by the same weighted metrics.
    Alternative: compare against exact `dose_at_points` at the points (tests
    only the kernel table). Chosen: the grid — it is the reporting path and
    its interpolation error (5e-3 at ≥ 0.5 rx) is the number the gate exists
    to bound.
A3 (`plan/solvers`, 2026-10-07). Taken while A2's `Objective` methods were
still stubs; each is flagged for the merge.

10. **Evaluation routing.** `solvers._HardEval` computes V100/V150/V200/D90,
    the P1 hard value, the sigmoid soft value and the vectorized "add each
    candidate" values straight from `influence.dose`, `target.weights`, the
    OAR matrices and the objective's lambda fields. `solvers._Evaluator`
    uses `objective.hard` / `metrics` when they are real (probe: a call on
    the empty selection that raises `NotImplementedError` marks a stub) and
    `objective.gains_all(selection, dose_vec=None)` / `soft_gains_all` when
    present (`getattr`), else the helper. Rule: within one solver run every
    comparison comes from one source (the objective's methods only when
    `hard` AND `gains_all` are both real), so a local search cannot cycle on
    float32/float64 differences between two implementations; only the
    reported `objective` / `metrics` prefer the objective's methods.
    Alternative: always use the helper. Chosen: route when real.
    D90 convention in the helper: weighted lower quantile (smallest dose
    whose cumulative weight reaches 10 %), equal to
    `numpy.percentile(method="inverted_cdf")` for unit weights.
11. **Plain vs feasibility-aware greedy (coordinator scout, 4 cavities at
    h = 4 mm / 2 spins).** Plain max-gain greedy clusters tiles where the
    immediate gain is largest and runs out of compatible candidates (7 of
    8 on one cavity; at or below random feasible selections on the others).
    `solve_greedy(feasibility_aware=True)` (default) walks the ranked
    candidates and skips any whose addition leaves a greedy maximal
    independent set (min static degree first, truncated at the remaining
    need; `_packing_bound`) smaller than the tiles still to place; at most
    `GREEDY_MAX_BOUND_CHECKS = 200` candidates are checked before falling
    back to the top-ranked one (`extra["bound_fallbacks"]`). The bound is a
    lower bound on the true packing number, so a skip is conservative (it
    can skip a candidate that would in truth still fit). On the 60-candidate
    toy at N = 9 plain greedy places 7, the aware variant 9 (test
    `test_feasibility_aware_greedy_packs_where_plain_greedy_stalls`).
    Alternative: exact packing (MILP / clique cover) per step — too slow at
    C ≈ 6000. Chosen: greedy MIS bound; `feasibility_aware=False` keeps
    plain greedy for comparisons.
12. **Greedy tie-break is lexicographic** `(hard gain, soft gain, room,
    lowest id)`, `room` = number of candidates still compatible after the
    addition. Step 1 ties on hard gain for every candidate on real cavities
    (no single tile reaches rx on the +5 mm shell), so the soft gain decides;
    `room` only matters for exact (hard, soft) ties (zero-dose rows in the
    test). Alternative (§3 E1 as written): lowest id. Chosen: lexicographic,
    still deterministic.
13. **`solve_greedy_local`** (greedy then E2 local search) is the default
    heuristic follow-up; `sweep_n(solver="local")` runs it per N.
14. **SA T0 estimator.** 200 trial proposals from the start state (not
    applied); T0 = median of the uphill `|Δsoft|` / ln 2, so the median
    uphill move is accepted with probability 1/2. Fallbacks: no uphill move
    among the trials → median of the non-zero `|Δsoft|` / ln 2; all trials
    flat → T0 = 1e-3 (soft is a fraction of the target weight). The trial
    moves consume the seeded RNG, so they are part of the reproducible
    stream. Alternative: fixed T0 as a fraction of the soft value. Chosen:
    trial-move estimate; measured T0 0.015–0.040 on the toys.
15. **SA move mix without `candidates`.** The 60/20/20 local/spin/global mix
    needs anchors and `anchor_ids`; when `candidates` is None (or when
    `solve_sa` is called through the frozen `gtcore.plan.solve_sa` signature
    without it) local and spin moves degrade to global re-sites
    (`extra["degraded_to_global"] = True`). On the toys the answer did not
    change (same best on both), but ~20 % more evaluations were needed on
    C=60 (31 371 vs 28 105). Kinds are preserved by every move when kinds
    are known (candidates given); without them any candidate may replace
    any tile.
16. **Random feasible starts (SA restarts 1..k, `solve_continuous`).** A
    random permutation walk (take each candidate compatible with those
    taken so far whose kind is still needed) spreads tiles out; when N is
    near the packing limit (the 8 × 8 toy at N = 9 needs the exact 3 × 3
    lattice) 50 such walks can all fall short, so one bound-aware
    construction follows (random order, each pick checked with
    `_packing_bound`). If both fail, SA restarts from the start selection
    (`per_restart[i]["origin"]` says so) and `solve_continuous` reports
    `status="infeasible"`.
17. **`kinds_required` needs candidate kinds.** `Objective` carries no
    `CandidateSet`; the module-level `solvers.solve_greedy` takes
    `candidates=` (and `objective.candidates`, if a caller attaches one, is
    also read). Through the frozen `gtcore.plan.solve_greedy` signature,
    `kinds_required` without an attached `CandidateSet` raises `ValueError`.
    `kinds_required` counts the whole selection (fixed tiles included) and
    must sum to `n_tiles`.
18. **Sweep warm start that boxes itself in.** `sweep_n` warm-starts N from
    N−1 (greedy: `fixed`; local/sa/continuous: previous + one greedy
    addition); when that fails (no compatible candidate) the N is retried
    from scratch and the row is marked `warm_start=False` (the V100 curve
    is then monotone only over the warm-started rows); the sweep stops at
    the first N neither start can place. `status="time_limit"` /
    `"max_evals"` results are kept (they are feasible).
19. **`solve_continuous` (scout 7.4 promotion).** Multi-start coordinate
    descent NM over `(u, v, θ)` per tile, same acceptance rule as E5 (hard
    V100 must not drop, soft must improve, `find_overlapping_tiles` empty);
    starts = `start` / greedy(objective) / first random feasible, plus
    `n_starts − 1` random feasible; best final hard (then soft) wins. The
    reported objective and metrics are computed on the subsampled target
    (`m_opt`, default 1000) with the real engine, not the influence matrix,
    so they are not directly comparable with the discrete solvers' values
    (compare through `final_report`). `time_budget_s` is a strict deadline
    raised inside the NM callback; when it is hit the status is
    `"time_limit"` and reproducibility from the seed is no longer
    guaranteed. Returned tiles are continuous poses; `selection` is the
    winning start's candidate ids. Recommended start grid h = 4 mm, 3 spins
    (coordinator). Conflicts for random starts come from `conflicts`, else
    `objective.conflicts`, else `find_overlapping_tiles` on the candidate
    tiles (O(C²), fine for a few hundred candidates).
20. **`tests/test_plan_interface.py`** exempts the five A3 functions from
    `test_every_stub_raises_not_implemented` (set `IMPLEMENTED`); other
    branches add their names to the same set.
A6 integration (`plan/ui`, 2026-10-07; `gtcore/plan/api.py`, `gt optimize`,
planner keys `O` / `N`).

10. **Tile-count prompt is an in-scene numeric entry, not a dialog.**
    PyVista/VTK has no text-input widget, and a console `input()` would
    block the render loop and hide behind the window. Chosen: the status
    line becomes the prompt (`_CountPrompt`): digits / numpad digits edit the
    highlighted field, BackSpace deletes, `H` switches to the half-tile
    count, `M` flips replace/add, `S` cycles the solver, Enter runs, Esc
    cancels; the recommendation is pre-filled. Every other key binding is a
    no-op while the prompt is open (`_guarded`), so typing "8" cannot place,
    delete or undo anything; BackSpace is routed to the prompt before the
    delete-all binding. Alternative: a Tk dialog (extra dependency and a
    second event loop). Rejected.
11. **Replace vs add.** Two modes, `M` in the prompt. Default **replace**:
    this session's hand-placed (green) proposals are removed (one undo
    step) and the tiles fitted FROM THE SCAN (gold, `_adopted_ids`) stay as
    fixed obstacles — the implant is physically there. **add**: everything
    on the board stays fixed and N tiles are added. Alternative: always
    clear the board. Rejected: that silently discards the recovered implant.
12. **Fixed tiles go through the greedy solver only.** Each fixed tile
    becomes one extra pre-selected pseudo-candidate (its own influence row
    from `dose_at_points(exact=False)`, its own conflict row from the
    planner's `find_overlapping_tiles` rule, no conflicts between fixed
    tiles) and the augmented instance is passed to `solve_greedy(fixed=…)`.
    `solve_local` / `solve_sa` / `solve_milp` / `solve_continuous` have no
    fixed-set parameter, so `optimize(fixed_tiles=…, solver≠"greedy")` is a
    `ValueError`; the planner falls back to greedy and says so in the status
    line. Alternative: an `Objective` subclass carrying a base dose — would
    depend on A2's internals (whether `hard()` goes through `dose_of()`).
    Rejected for now; revisit after A2/A3 merge if the planner needs SA with
    a fixed implant.
13. **Planner status metrics are influence-style estimates.** The `O` / `N`
    keys call `optimize(report=False)` and score the board before/after
    with `api.evaluate_tiles` (tabulated-kernel dose at ≤ `M_OPT_MAX` points
    of the +5 mm shell) so a run stays interactive; when a report with
    dose-grid shell metrics is present the status line uses those and says
    "dose grid". `gt optimize` runs the full `final_report` unless
    `--no-report`.
14. **Candidate / influence caches** (`api.cached_candidates`,
    `api.cached_influence`): 4-entry LRUs keyed on `(id(mesh), faces.shape,
    h, n_spins, kinds, sha1(eligible mask), rng_seed)` and on `(id(candidates),
    target signature, rx)`; entries hold the mesh / candidate object so a
    recycled `id()` can never alias a new object. Candidate sampling uses
    `rng_seed=0` regardless of the solver seed so the cache is shared across
    seeds. `api.clear_cache()` drops both.
15. **`solver="continuous"`** (coordinator note): `solve_continuous` is
    looked up with `getattr` and raises `NotImplementedError` until A3 merges;
    it starts from the discrete greedy selection on a coarser default grid
    (`CONTINUOUS_H_MM` = 4 mm, `CONTINUOUS_N_SPINS` = 3, used only when the
    caller left `h_mm` / `n_spins` at the frozen defaults), takes
    `time_budget_s` (`--budget`, default 60 s), returns poses rather than
    ids (so `refine` is skipped and the output is re-verified with
    `find_overlapping_tiles`). "greedy" stays the planner default until V2.
16. **Kind counts for non-greedy solvers** are enforced only through the
    greedy warm start (`solve_local` / `solve_sa` take no `kinds_required`);
    `optimize` re-checks the counts, the selection size, duplicates and
    conflicts after every solver and raises `RuntimeError` on any mismatch
    (never fewer tiles as success, plan §4 V8). `status="time_limit"` (MILP)
    is accepted when the incumbent is feasible and complete.
A5 (plan/validation, 2026-10-07):

10. **`final_report` grid margin is clipped, not the spacing.** The 50 mm
    default margin on a 1 mm grid is kept as long as the grid stays ≤ 30 M
    voxels (`report.MAX_GRID_VOXELS`); otherwise the margin shrinks in 1 mm
    steps and a note records it. Alternative: coarsen the grid. Chosen:
    clip the margin (the shells never need more than ~12 mm).
11. **MILP bound units in V3.** `SolverResult.bound` is "the upper bound on
    the objective (coverage only)"; the script divides by the target's total
    weight when the bound exceeds 1, assuming Σ w_m y_m units. A4 should
    state the unit; if the bound is already a fraction nothing changes.
12. **V2 per-arm evaluation margin is 15 mm, not 50 mm.** The +10 mm shell
    needs ≈ 12 mm; the full 50 mm report (rind DVH, shadowing) is written
    for the primary-endpoint configurations only (and timed in V7).
    Alternative: 50 mm everywhere (≈ 8× the voxels per report). Chosen: 15 mm.
13. **Truth-tile arm anchoring.** Truth tiles are re-draped at
    `center + 3 mm · outward_normal` snapped to the mesh with the first seed
    axis as spin hint, which lands within 4 mm of the truth centre (tested);
    the raw-seed arm uses a flat footprint square so its overlap count is
    approximate.
14. **Cavity sizes by scaling `CAVITY_RADII`** around `make_head_phantom`
    (module constant, restored after the call) instead of adding a generator
    parameter. Alternative: edit the generator (out of A5's remit).
15. **`test_plan_interface.STUBS` no longer lists `final_report`** (it is
    implemented); other branches will drop their names the same way.
16. **`final_report` reads optional knobs from `parameters`**
    (`sk_per_seed_u`, `seed`, `shell_offsets_mm`, `rind_depth_mm`,
    `shadowing`) because the frozen signature has no S_K argument.
17. **Normal orientation / snapping on closed shell meshes (printed
    phantom).** `snap_to_wall` orients normals toward the mesh centroid and
    returns the nearest point on the WHOLE mesh; on `meshes["body"]` a point
    inside the plastic can snap to the outer surface. The validation script
    snaps the tile's seed centroid and orients the normal toward it. A1's
    `build_candidates` on such a mesh should anchor on the eligible (visible)
    faces only and orient normals toward the implant centroid; A6 should
    pass `visible_faces(...)` as `eligible_faces` for a body-mesh fallback.
18. **V6 endpoints are target-weighted** (eligible +5 mm shell), V2 endpoints
    are the +5 mm shell-vertex stats (`shell_report`) with the weighted
    value alongside — V2 keeps the declared definition; V6 cannot use the
    mesh's own shell vertices (outer surface included).
19. **`continuous` arm budget** = greedy + SA wall time measured on the same
    instance in the same run (coordinator request); when neither ran, the
    solver's own default budget applies and `time_budget_s` is recorded as
    empty.
A4 (`plan/milp`, 2026-10-07).

10. **MILP coverage-row pruning threshold = 1e-3·rx** (`milp.PRUNE_FRACTION`).
    An influence entry `D[c,m] ≤ 1e-3·rx` is dropped from row m. Dropping can
    only lower the row's left-hand side, so the MILP is conservative (never
    counts an uncovered point) and can miss a point whose true dose is in
    `[rx, rx + N·1e-3·rx)` — ≤ 2 % of rx at N = 20. Alternatives: 1e-2·rx
    (fewer nnz, up to 20 % of rx underestimated at N = 20 — too coarse for a
    reference) or no pruning (exact; the C×M matrix is dense anyway for
    inverse-square dose, so the saving is only real for the TG-43 kernel at
    long range). Chosen: 1e-3·rx, `prune_frac=0` available and tested equal
    on the toy; `extra["formulation"]["n_pruned"]` is reported.
11. **Clique rows are a tightening only; the constraint set equals the
    pairwise graph exactly.** Every clique is validated against `pairs`
    (`milp.validate_cliques`); a non-clique "clique" would cut a feasible
    selection, so it is dropped with a `UserWarning` and its pairs fall back
    to pairwise rows; every conflicting pair not inside a valid clique gets
    its own row `x_i + x_j ≤ 1`. Alternatives: trust A1's cliques without
    validation (cheaper, unsafe); cliques only, pairs ignored (would miss
    conflicts outside the anchor neighbourhoods); pairwise rows only (same
    optimum, weaker LP relaxation, ~2× the rows on the toy). Chosen:
    validated cliques + residual pairs; tested by enumerating every 2- and
    3-subset of the pitch-15 toy (cut ⇔ pairwise infeasible).
12. **MILP scaling (negative result, V3).** The 80-candidate toy at N = 6
    does not close in 60 s (gap 14 %) and the LP bound is vacuous (1.00).
    Candidate tightenings: (a) pigeonhole cover cuts
    `y_m ≤ Σ_{c: D[c,m] ≥ rx/N} x_c` (any N-subset covering m contains a
    candidate giving ≥ rx/N) — **implemented** (`cover_cuts`, with the
    weaker fixing `y_m = 0` when no candidate reaches rx/N); measured in V3:
    no gain on toy80 at N = 6, large LP-bound gain at N ≤ 2, so off for
    `solve_milp` and on for `lp_bound`; (b) presolve fixing `y_m = 0` when
    the N largest `D[·,m]` sum below rx (stronger than the implemented fix;
    not done); (c) symmetry breaking across spins of one anchor; (d)
    Lagrangian relaxation of the coverage rows (§7.3). Chosen: the plain
    formulation of §3 E4 for the MILP, bound reported as the reference at
    the time limit; A5 states the gap per instance and the coordinator
    decides on (b)–(d) from real reduced instances.
13. **`SolverResult.objective` from the MILP is the full P1 hard value**
    recomputed from the selection (V100 − λ_hot·max(0, V200 − v200_tol) −
    Σ λ_oar·max(0, Dmax − L)), via `Objective.hard` / `.metrics` when they are
    implemented and otherwise the same §2 definitions implemented locally
    in `milp.evaluate_selection` (D90 = smallest dose whose cumulative weight
    reaches 10 %). The solver's own value is `extra["milp_objective"]` (V100).
    Alternative: report V100 as `objective`. Chosen: P1 hard, so MILP rows
    compare with the other solvers in sweep tables; V3 compares on
    `extra["milp_objective"]`. The OAR limits are hard rows in the MILP
    (not penalties), so a MILP selection never carries an OAR penalty.
14. **`solve_milp(method="auto")` routes to the enumeration B&B for
    C ≤ 600 and N ≤ 8** (`milp.ENUM_MAX_C`, `milp.ENUM_MAX_N`; HiGHS beyond)
    after the §7.3 scout showed HiGHS cannot close real reduced instances
    and the enumeration can (see V3). The result keeps `solver="milp"` with
    `extra["method"]` = `"enum_bb"` / `"highs"` so callers of the frozen
    wrapper see one reference solver; `solve_enumeration` called directly
    reports `solver="enum_bb"`. Alternatives: always enumerate (unbounded
    worst case above ~600 candidates), always HiGHS (loose bound), or
    report `solver="enum_bb"` from `solve_milp` too (would split sweep
    tables by method). The size limits are from the scout's largest proven
    case (C = 498, N = 8); they are constants in `milp.py`, not in
    `__init__.py`, because they are not tunables of the problem.
    Candidate order default "degree" (single-coverage, then conflict
    degree) as directed; `order="potential"` (the scout's) is kept.
A1 (`plan/candidates`, 2026-10-07).

10. **Hanging detection replicates the conformer's ray cast instead of
    inferring it from the conformed tile.** The brief suggested flagging a
    grid point as a fallback when its recovered wall point (nearest mesh
    point of the conformed point) deviates laterally > 0.5 mm from the ±n
    ray. Measured: because `conform_tile` offsets along the *smooth
    interpolated* normal (not the anchor normal), true ray hits showed
    lateral deviations up to 1.9 mm (p99 1.26 mm) on the seed-1 cavity and
    2.3 mm on the flat wall's edge strip; the rule rejected 30 of 63 good
    tiles. Alternative kept as a diagnostic (`grid_fallback_flags`). Chosen:
    `count_ray_fallbacks` casts the same ±n rays from the same flat grid
    points with the same 12 mm sag rule (`interact._MAX_SAG_MM`), using
    trimesh's own narrow phase but a broad phase clipped to the sag limit
    (cKDTree ball of 12 mm + circumradius; a hit beyond 12 mm is a fallback
    anyway), batched for all (anchor, spin, kind) *before* conforming so
    hanging candidates never pay for `conform_tile`. Tested equal to the
    `mesh.ray.intersects_location` reference on cavity, sphere, flat wall,
    hollow shell and an all-miss tiny sphere (`count_ray_fallbacks_trimesh`).
11. **`CONFLICT_GAP_MM` composes additively**: conflict threshold =
    1 mm (planner) + `gap_mm`, i.e. exactly
    `find_overlapping_tiles([t_i, t_j], threshold_mm=1.0 + gap_mm)`.
    Alternative: `gap_mm` replacing the 1 mm. Chosen: additive, so gap 0
    is the planner's rule verbatim and gap > 0 only adds conflicts
    (tested monotone).
12. **Pairwise conflicts are the two-tile planner call, batched.** The
    multi-tile `find_overlapping_tiles(all_tiles)` uses one slack (max grid
    cell diagonal over *all* tiles passed) for every pair; the two-tile call
    uses the pair's own. The contract is the pair call; the two differ only
    through ballooned footprints (decision 14): 3 of 2268 pairs on the
    r = 25 mm icosphere, 0 of 3005 on the seed-1 cavity. Stage 1 (nearest
    sample per sample, normal agreement, exclusive upper bound as
    `cKDTree.query`) is vectorized over pair chunks; the exact stage is an
    element-for-element port of `interact._point_triangle_dist` with a
    batch axis (tested bit-identical), applied only to (sample, triangle)
    pairs whose three vertices lie within threshold + longest edge of the
    sample (a sound bound). Pair-by-pair equality with the planner call is
    tested on random near, far and uniform pairs at gap 0 and 3 mm.
13. **Cliques** = per sampled anchor (a) all candidates of that anchor (every
    spin and kind), reduced greedily to a true clique if the planner's rule
    ever disagrees (decision 14), and (b) a maximal clique grown greedily
    (nearest anchors first, ties by id) from the anchor's first candidate
    inside the neighbourhood of candidates whose anchor lies within their
    *own kind's* tile diagonal (28.3 mm full, 22.4 mm half). Every clique
    is verified against the pairwise matrix before it is listed, so the
    clique form can never forbid a pairwise-feasible selection (tested on
    random selections and greedy independent sets). Alternative: one clique
    per anchor neighbourhood without reduction (not a true clique on real
    geometry). Chosen: verified maximal cliques, deduplicated.
14. **Planner defect found, not fixed (out of A1's remit:
    `gtcore/interact.py`).** `interact._footprint_surface` fits a quadratic
    height field through the 4 corners, the seeds and the anchor. Every
    `conform_tile` placement has |u| = |v| at all fit points, so the u²−v²
    direction is near-null (6th singular value ≈ 1e-4 of the first) — above
    the `rcond=1e-6` cutoff, so it is *kept*, and on any curved wall the
    residual noise is amplified into coefficients of ±100s: the sampled
    footprint balloons (bounding radius median 38 mm, p90 140 mm, max
    1969 mm on the r = 25 mm icosphere at h = 5, 3 spins; the nominal
    corner radius is 14.1 mm). Consequences measured with the planner's own
    `find_overlapping_tiles`: 45 % of its overlaps on that sphere are between
    anchors > 32 mm apart (geometrically impossible); on the seed-1
    marching-cubes cavity only 0.5 % (its irregular fit points condition the
    fit). A one-fallback-corner tile on the flat wall balloons the same way
    (radius 330 mm) because the corner's z is off the plane. Scratch test
    (no code change): `rcond=1e-3` gives radius 12.6 mm for every sphere
    tile and 0 spurious far overlaps (vs 4158 of 9270), leaves the cavity
    unchanged, and does not cure the fallback-corner case (that needs
    rejecting the outlier point or every fallback). The conflict graph
    replicates the rule as instructed, so it inherits this until the
    planner is fixed; A1 tests state geometric facts (same anchor ⇒
    conflict; > 30 mm ⇒ no conflict) only for footprints with radius < 20 mm
    and assert that every exception carries the signature. Recommendation
    to the coordinator: change `rcond` to 1e-3 in `_footprint_surface` (and
    consider excluding fallback points from the fit); A1's graph then
    inherits the fix with no change.
15. **`n_rejected` is counted in candidate units** (`"ineligible"` = dropped
    anchors × spins × kinds; `"hanging"`, `"detached"`, `"conform_error"`
    per (anchor, spin, kind)), so the counts add up with `len(cs)` to the
    enumerated total. Alternative: anchors for `ineligible`. Chosen:
    candidate units.
16. **`CandidateSet.n_spins` records the spin count of the first kind in
    `kinds`** when both kinds are built with their different defaults (6 /
    12); the per-kind sets are recoverable from `spins_deg`.
17. **Anchor sampling** = seeded `trimesh.sample.sample_surface` restricted
    to eligible faces (20 samples per h²), then farthest-point sampling from
    the dense point nearest the eligible centroid until the largest gap is
    below h (anchors pairwise ≥ h apart; `method` records
    `farthest_point(sample_surface n=..., seed=...)`). Alternative:
    `sample_surface_even` (Poisson-disc rejection, no spacing guarantee).
    Chosen: FPS for the spacing guarantee and determinism.
18. **Spin step** = 90/n (full) or 180/n (half) when `n_spins` is given for
    both kinds, as the frozen docstring says; the axis hint is the global
    axis least aligned with the inward normal projected to the tangent
    plane, rotated by the spin (Rodrigues) — `tile.axis_ras` at spin θ is
    exactly the spin-0 axis rotated by θ (tested).

---

## Reviewer report (A7)

_To be pasted unedited._

---

## Not claimed

- TG-43 in water: no heterogeneity corrections, no tissue composition.
- Static cavity: no post-implant shrinkage or deformation modelled.
- Tiles modelled as non-overlapping although collagen may stack at edges.
- Surgeon reachability not modelled beyond the eligibility mask.

---

## Runs log

| date | command | seed(s) | commit | output | note |
|---|---|---|---|---|---|
| 2026-10-07 | `python -m pytest -q tests/test_plan_interface.py` | — | (Phase 0 commit) | 34 passed (full suite 470 passed, 205 s) | interface freeze |
| 2026-10-07 | `python -m pytest -q -s tests/test_plan_influence.py tests/test_plan_objective.py` | phantom 1, draws 0, toy 0–5 | 5ea6f95 | 31 passed; gate max ΔV100 0.103 pp, ΔD90 0.042 % rx; 0.9–1.6 ms / candidate; hard 0.04–0.14 ms; gains_all 40–49 ms | A2 influence + objective |
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider` | — | 5ea6f95 | 498 passed, 3 skipped (489 s, concurrent with the gate run) | A2 full suite |
| 2026-10-07 | `python -m pytest -q tests/test_plan_solvers.py` | toy rng 0; SA seeds 0, 1, 7 | cd73082 + A3 commit (plan/solvers) | 26 passed, ≈ 16 s | greedy / local / SA / sweep / E5 / continuous; timings in V7 "A3 solvers" |
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider` | — | cd73082 + A3 commit (plan/solvers) | 491 passed, 383 s (run alone; 616 s when two suites shared the CPU) | 491 collected (470 + 26 − 5 stub cases now implemented) |
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider` (plan/ui) | 0 | (A6 commit) | 509 passed, 303 s (+20 `test_plan_api.py`, +20 `test_planner_optimize.py`; A1–A5 functions monkeypatched with toy fakes) | A6 integration; off-screen VTK tests crash with 0x8007000e when several suites share the GPU — rerun alone |
| 2026-10-07 | `python scripts/validation_optimize.py --quick --section v8` | — | cd73082 + A5 | `v8_failure_modes.csv` | 9 cases stub / not run; `final_report(empty mesh)` raises ValueError (PASS) |
| 2026-10-07 | `python scripts/validation_optimize.py --quick` | 1, 2 | cd73082 + A5 | `v2_rows.csv` (truth arms only), `v6_*`, `v7_runtime.csv` | A1–A4 stubs: every solver arm skipped with reason; see V2/V6/V7 text |
| 2026-10-07 | `python scripts/validation_optimize.py --quick --section v6` | — | cd73082 + A5 | `v6_phantom.csv`, `v6_info.json`, `v6_dvh.png`, `v6_as_implanted*.json` | after the centre-snap fix; numbers in V6 (517 s incl. the ray test) |
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider` | — | cd73082 + A5 | 491 passed, 490 s | full suite with `test_plan_report.py` (10) and `test_validation_optimize_smoke.py` (12) |
| 2026-10-07 | `python -m pytest -q tests/test_plan_milp.py` | toy rng_seed 0 | (A4 commit, plan/milp) | 20 passed, 17 s (full suite 490 passed, 1 skipped, 385 s under load) | MILP = brute force on the toy; 80-candidate toy at N = 6 not closed in 60 s (gap 0.14) |
| 2026-10-07 | `python -m pytest -q tests/test_plan_milp.py` + direct `solve_milp(..., cover_cuts=…)` calls on `toy_instance(80, 300)` N = 6 | toy rng_seed 0 | (A4 cover-cut commit, plan/milp) | 22 passed, 32 s (full suite 491 passed, 1 skipped, 328 s) | pigeonhole cover cuts: optimum unchanged; LP bound exact at N = 1; no bound gain on toy80 N = 6, worse 60 s incumbent (0.787 vs 0.867) → off for the MILP, on for `lp_bound` |
| 2026-10-07 | `python -m pytest -q tests/test_plan_milp.py` + direct `solve_enumeration` calls on `toy_instance(12,150)` (pitch 10/15) and `toy_instance(80,300)` N = 4/6/8 | toy rng_seed 0 | (A4 enumeration commit, plan/milp) | 32 passed, 29 s (full suite 501 passed, 1 skipped, 267 s) | enumeration B&B = brute force on every toy case; proves toy80 N = 6 (0.9033) in 34 s where HiGHS stalled at 0.867 / 0.997; `solve_milp` auto-routes to it for C ≤ 600, N ≤ 8 |
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider tests/test_plan_candidates.py tests/test_plan_conflicts.py tests/test_plan_interface.py` | fixtures seeded 0 | 8a98e7e | 66 passed (32 new) | A1 candidates / conflicts |
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider` | — | 8a98e7e (working tree) | 502 passed, 278 s | full suite before the A1 commit |
| 2026-10-07 | `python scripts/plan_candidates_runtime.py` | anchors 0, phantom 1 | 8a98e7e | V7 table above (A1 candidates/conflicts) | cavity h = 3 / 2.5, hollow sphere |
