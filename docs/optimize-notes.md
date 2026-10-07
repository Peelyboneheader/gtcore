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

## V2 Synthetic benchmark

_Pending._ Primary comparison (declared): SA vs uniform on V100 at the N
where uniform first reaches D90 ≥ rx.

## V3 Optimality gap

_Pending._ (SA − MILP)/MILP on V100 over reduced instances; MIP gap stated
when the time limit was hit.

## V4 Discretization

_Pending._ Sweep h and n_spins; gain from E5.

## V5 Sensitivity

_Pending._ Seed-plane offset over `geometry.SEED_PLANE_OFFSET_RANGE_MM`,
target subsample density, S_K ± 5 %, interseed attenuation on.

## V6 Physical phantom and clinical case

_Pending._ 8-tile printed phantom (HR-CTV = +5 mm shell of the inner wall of
`meshes["body"]`), clinical case 2.

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

## V8 Failure modes

_Pending._ No feasible N-tile placement; tile wider than the wall patch;
eligibility mask excluding most of the wall; degenerate meshes — each must
fail loudly with a reason string.

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
