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
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider` (plan/ui) | 0 | (A6 commit) | 509 passed, 303 s (+20 `test_plan_api.py`, +20 `test_planner_optimize.py`; A1–A5 functions monkeypatched with toy fakes) | A6 integration; off-screen VTK tests crash with 0x8007000e when several suites share the GPU — rerun alone |
