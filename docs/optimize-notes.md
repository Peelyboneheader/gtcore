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

_Pending._ Influence gate (§3 B); weighted quantiles; conflict symmetry and
clique/pairwise agreement; greedy never violates a conflict; SA reproducible
from seed; MILP equals brute force on the toy instance; the planner never
flags an optimizer output as overlapping.

## V2 Synthetic benchmark

_Pending._ Primary comparison (declared): SA vs uniform on V100 at the N
where uniform first reaches D90 ≥ rx.

## V3 Optimality gap

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

## V4 Discretization

_Pending._ Sweep h and n_spins; gain from E5.

## V5 Sensitivity

_Pending._ Seed-plane offset over `geometry.SEED_PLANE_OFFSET_RANGE_MM`,
target subsample density, S_K ± 5 %, interseed attenuation on.

## V6 Physical phantom and clinical case

_Pending._ 8-tile printed phantom (HR-CTV = +5 mm shell of the inner wall of
`meshes["body"]`), clinical case 2.

## V7 Runtime

_Pending._ Candidate build, influence, each solver, final reporting; per
cavity size; hardware stated.

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
    Candidate tightenings, none implemented: (a) pigeonhole cover cuts
    `y_m ≤ Σ_{c: D[c,m] ≥ rx/N} x_c` (any N-subset covering m contains a
    candidate giving ≥ rx/N); (b) presolve fixing `y_m = 0` when the N
    largest `D[·,m]` sum below rx; (c) symmetry breaking across spins of one
    anchor; (d) Lagrangian relaxation of the coverage rows (§7.3). Chosen:
    the plain formulation of §3 E4, bound reported as the reference at the
    time limit; A5 states the gap per instance and the coordinator decides
    on (a)–(d) from real reduced instances.
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
| 2026-10-07 | `python -m pytest -q tests/test_plan_milp.py` | toy rng_seed 0 | (A4 commit, plan/milp) | 20 passed, 17 s (full suite 490 passed, 1 skipped, 385 s under load) | MILP = brute force on the toy; 80-candidate toy at N = 6 not closed in 60 s (gap 0.14) |
