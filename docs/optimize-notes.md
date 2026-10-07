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
