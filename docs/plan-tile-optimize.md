# feature/tile-optimize — automated GammaTile placement optimization

Directions for building and validating a tile-placement optimizer inside
`gtcore`. Read in full before touching code. Lives at
`docs/plan-tile-optimize.md`; measurements go to `docs/optimize-notes.md`
with the command, seed, and commit hash that produced them.

> Precedence note (2026-10-07, coordinator): the code already in the
> repository takes precedence over these directions wherever the two
> disagree. The optimizer is an opt-in module layered on top of the existing
> primitives; nothing below changes existing pipeline, planner, engine, or
> geometry behaviour. See §10 for the additions from the brief.

---

## 0. Goal and non-goals

**Goal.** Given a resection-cavity surface, a target, and a tile inventory,
propose a non-overlapping, wall-conformed tile configuration that maximizes
target coverage, with a bounded optimality gap against an exact reference,
and quantify the coverage gap between an as-implanted configuration
(recovered by `gtcore.tiles`) and the best achievable with the same number
of tiles.

**Non-goals.**
- Learned models (RL policy, neural surrogate, imitation). See §7.8.
- Heterogeneity corrections or any change to the TG-43 engine.
- Changes to `gtcore/geometry.py`.
- Changes to existing pipeline or planner defaults. The optimizer is opt-in.

---

## 1. Existing primitives (use; do not re-implement)

| Need | Primitive | Notes |
|---|---|---|
| Drape a tile at (anchor, spin) | `gtcore.interact.conform_tile(mesh, surface_point, inward_normal, axis_hint, kind)` | `PlacedTile` with conformed `seed_centers`, `seed_axes`, `corners_ras`, `anchor_ras`. |
| Nearest wall point + inward normal | `gtcore.interact.snap_to_wall` | Normal re-oriented toward the interior regardless of winding. |
| Overlap test used by the planner | `gtcore.interact.find_overlapping_tiles` | The optimizer's conflict definition must agree with this. |
| Dose at points | `gtcore.dose.engine.dose_at_points` | Exact by default; tabulated kernel for speed. |
| Dose grid for final reporting | `gtcore.dose.engine.compute_dose_grid` | All reported metrics come from this, never from the influence matrix. |
| Target shells | `gtcore.dose.dvh.shell_points`, `dvh_stats`, `shell_report` | Wall, +5 mm, +10 mm shells. |
| Rind / wall coverage on a grid | `gtcore.dose.metrics.rind_mask`, `wall_dose`, `surface_coverage` | Volumetric counterparts. |
| Ground-truth cavities | `gtcore.phantom.generate` | Lumpy cavity + wall-conformed truth tiles. |
| Fitted tiles → planner tiles | `gtcore.tiles.auto.to_placed_tiles` | How the as-implanted configuration enters the comparison. |
| 3-dof (u, v, θ) fit over the conformer | `gtcore.tiles.surface.fit_on_surface` | Reuse its parameterization for continuous refinement (§3 E5). |
| Shadowing check | `gtcore.dose.find_shadowing_tiles` | Dosimetric counterpart of the overlap flag. |

Conventions: RAS mm, arrays `[k, j, i]`, deterministic outputs for identical
inputs, tests run from the repo root, full suite green at every commit, no
new hard dependencies in the core (`scipy.optimize.milp` is allowed;
OR-Tools, torch, etc. are not).

---

## 2. Problem statement (reproduce in the module docstring)

**Given**
- cavity wall mesh *S* (trimesh) and eligibility region *E ⊆ S* (default
  all of *S*; callers may pass an exclusion mask);
- target sample set *T* = {(p_m, w_m)}: points with area weights. Default:
  vertices of the +5 mm shell weighted by vertex area (one third of adjacent
  face areas). Alternative when an RTSTRUCT exists: voxel centres of the
  clinical HR-CTV, equal weights;
- optional OAR sample sets *O_j* with limits *L_j*;
- prescription *rx* (cGy); per-seed *S_K* and total-decay integration as in
  the engine;
- inventory: *N_full*, *N_half* (fixed-N form) or free (minimum-N form).

**Decision variables** per tile *i*: anchor *a_i ∈ E*, spin *θ_i* about the
local normal (full: *θ ∈ [0, 90°)* by the 2×2 symmetry; half:
*[0, 180°)*), kind ∈ {full, half}. Seed poses follow deterministically from
`conform_tile`.

**Dose** is additive over seeds: *D(p) = Σ_i Σ_s d(p; seed_{i,s})*, TG-43U1S2
line source in water. Interseed attenuation off during optimization; may be
on for final reporting with the difference stated.

**Metrics**
- *V100(T)* = Σ_m w_m·[D_m ≥ rx] / Σ_m w_m — primary.
- *D90(T)* = weighted 10th percentile of {D_m}.
- *V150(T), V200(T)* — hot-spot measures.
- *Dmax(O_j)* or *D0.1cc(O_j)*.
- *N*.

**Constraints:** no two tiles conflict (hard, §3 C); anchors in *E* (hard);
count = inventory (fixed-N).

**Problem forms**
- **P1 (fixed N):** maximize
  *V100 + λ_tail·T10 − λ_hot·max(0, V200 − v200_tol) − Σ_j λ_oar·max(0, Dmax(O_j) − L_j)*,
  where *T10* = weighted mean of *min(D_m, rx)/rx* over the coldest 10 % of
  the target weight (the lower-tail / CVaR mean of the capped dose; *T10 ≤
  min(D90, rx)/rx*, and *T10 = 1* iff *V100 = 1*).
  Defaults λ_tail = 1.0 (added 2026-10-08, see `optimize-notes.md`
  "Objective: lower-tail term"; 0 restores the original pure-V100 form),
  λ_hot = 0.5, v200_tol = 0.10, λ_oar = 1e3 per cGy over limit.
  Every λ is a named, swept parameter.
- **P2 (minimum N):** smallest *N* with the P1 optimum satisfying
  *D90(T) ≥ rx* (also report the *V100 ≥ 0.90* criterion). Solved by
  sweeping P1 over *N* and reporting the full coverage-vs-N curve.
  Note: the two criteria are the same statement — *D90 ≥ rx* holds exactly
  when at most 10 % of the target weight is below rx, i.e. *V100 ≥ 0.90*
  (0 disagreements in the 327 V2 rows) — so the sweep reports one minimum
  N; the objective choice cannot move it, only the below-N* behaviour.

**Discretization.** Once anchors and spins are sampled, the problem is a
combinatorial selection with pairwise conflicts and a coverage objective,
which admits an exact MILP reference on reduced instances and therefore a
measured optimality gap. The discretization error is itself measured
(§4 V4).

---

## 3. Baseline algorithm (build in this order)

Subpackage `gtcore/plan/` (numpy/scipy/trimesh only): `candidates.py`,
`influence.py`, `conflicts.py`, `objective.py`, `solvers.py`, `milp.py`,
`sweep.py`, `report.py`. Entry points `gtcore.plan.optimize(...)` →
`(List[PlacedTile], OptimizeReport)` and `gtcore.plan.suggest_next(...)`.

### A. Candidate generation (`candidates.py`)
1. Sample anchors on *S* at approximately even spacing *h* (default 2.5 mm):
   even surface sampling or farthest-point sampling on a densified vertex
   set; record the method. Drop anchors outside *E*.
2. Spin set Θ: full tiles 6 spins (0°, 15°, …, 75°); half tiles 12 spins.
   Parameter `n_spins`.
3. For each (anchor, θ, kind): `snap_to_wall` → `conform_tile`. Reject if
   the conformer fell back to nearest-point for more than one grid point
   (tile hanging off an edge) or any conformed seed is more than
   `DETACHED_MM` (1.5 mm) from its 3 mm offset. Count rejections.
4. Output `CandidateSet`: anchors, spins, kinds, seed centres (C, 4, 3),
   seed axes, corner polygons (C, 4, 3), eligibility flags.

### B. Influence matrix (`influence.py`)
- *D[c, m]* = dose (cGy, total decay, nominal S_K) at target point *m* from
  candidate *c*'s seeds via `dose_at_points` with the tabulated kernel,
  chunked, float32. Same for each OAR set.
- Memory is C × M × 4 bytes. Subsample *T* to M_opt ≤ 4k points for
  optimization (stratified by area); always evaluate final configurations on
  the full *T* and a full dose grid (§3 G).
- **Gate (unit test):** for random selections of 1–8 candidates, V100 from
  the row-sum must agree with V100 from `compute_dose_grid` (exact kernel,
  1 mm, sampled at the same points) within 0.5 percentage points, and D90
  within 1 % of rx.

### C. Conflict graph (`conflicts.py`)
- Two candidates conflict if their conformed footprints overlap, defined
  exactly as `interact.find_overlapping_tiles` defines it (call the same
  geometry on the two `PlacedTile`s). Optional minimum edge gap *g*
  (default 0 mm; parameter).
- Store a sparse boolean matrix plus **clique constraints** for the MILP:
  for every anchor neighbourhood (candidates whose anchors lie within one
  tile diagonal), "at most one of these". Pairwise and clique forms must
  agree (test).
- Tests: symmetry; a candidate conflicts with its own other spins; two
  tiles with anchors > 30 mm apart on a flat wall never conflict.

### D. Objective (`objective.py`)
- Dose vector for a selection = `D[sel].sum(0)`. Metrics by weighted order
  statistics (`weighted_quantile`, tested against `numpy.percentile` for
  unit weights).
- **Soft** coverage for annealing: Σ_m w_m·σ((D_m − rx)/τ), τ = 0.05·rx.
  **Hard** V100 is what gets reported; record both.
- O(N·M) per evaluation; target ≤ 1 ms at M = 4k.

### E. Solvers (`solvers.py`, `milp.py`)
- **E1 Greedy forward selection.** Repeatedly add the non-conflicting
  candidate with the largest objective gain until N. Deterministic (ties →
  lowest id). Also the "suggest next tile" feature. Record the
  marginal-gain curve.
- **E2 Local search.** First-improvement over moves: (i) re-site tile *i*
  to a non-conflicting candidate with anchor within *r* = 10 mm;
  (ii) re-spin in place; (iii) re-site anywhere. Stop at a local optimum.
- **E3 Simulated annealing.** Metropolis on the soft objective; move mix
  60 % local re-site, 20 % spin, 20 % global re-site; T0 set so ~50 % of
  initial uphill moves are accepted (estimated from trial moves); geometric
  cooling α = 0.95 per sweep of 50·N moves; `n_sweeps`, `n_restarts`
  (greedy start + random feasible starts) as parameters; keep the best
  **hard** objective seen. Reproducible from `seed`; log best-so-far per
  sweep.
- **E4 Exact reference (MILP, `scipy.optimize.milp` / HiGHS).**
  x_c ∈ {0,1} select candidate; y_m ∈ {0,1} target point covered.
  Maximize Σ_m w_m y_m subject to
  Σ_c D[c,m]·x_c ≥ rx·y_m ∀m;
  Σ_{c∈Q} x_c ≤ 1 ∀ cliques Q;
  Σ_c x_c = N (fixed-N) or ≤ N;
  Σ_c D_j[c,o]·x_c ≤ L_j ∀ OAR points o.
  Hot-spot terms are not in the MILP; compare on coverage only and state
  it. Run on reduced instances (coarser h, fewer spins, subsampled M) with a
  time limit; record the solver's MIP gap. If the limit is hit, report the
  bound, not the incumbent, as the reference.
- **E5 Continuous refinement.** Polish the discrete solution by Nelder–Mead
  on each tile's (u, v, θ) through `conform_tile` (as `fit_on_surface`
  does), coordinate descent over tiles, conflicts re-checked after each
  accepted step. The gain is the discretization-error estimate (§4 V4).

### F. N-sweep (`sweep.py`)
Run P1 for N = 1 … N_max with the chosen solver; output V100, D90, V150,
V200, OAR Dmax, runtime per N, and the minimum N per criterion. Warm-start
N from N−1.

### G. Final reporting (`report.py`)
For any configuration (optimized, as-implanted, manual, random):
1. `compute_dose_grid`, exact kernel, 1 mm grid, cavity + 50 mm; optional
   second run with the interference model, difference reported.
2. Shell metrics via `shell_report` on the full *T*; rind DVH via
   `rind_mask` (5 mm) when a cavity mask exists; clinical HR-CTV DVH when an
   RTSTRUCT exists.
3. Conflicts re-verified with `find_overlapping_tiles`; shadowing listed
   via `find_shadowing_tiles`.
4. JSON + CSV with gtcore version, parameters, seed, wall-clock time.

### H. Integration (last; additive only)
- `gtcore.plan.optimize(mesh, target, inventory, ...)`,
  `gtcore.plan.suggest_next(mesh, target, placed_tiles, ...)`.
- CLI: `gt optimize <scan-or-phantom> --tiles N [--half K] [--min-n]
  [--solver greedy|sa|milp] [--seed S]`.
- Planner: one key for "optimize current inventory", one for "suggest next
  tile"; results are ordinary placed tiles (draggable, deletable, on the
  undo stack), tinted until touched; status line reports objective
  before/after.

---

## 4. Validation

Script `scripts/validation_optimize.py` → `output/validation_optimize/*.csv`,
`*.png`; every figure regenerated from a fixed seed list.

**V1 Unit/consistency** (`tests/test_plan_*.py`): influence gate (§3 B);
weighted quantiles; conflict symmetry and clique/pairwise agreement; greedy
never violates a conflict; SA reproducible from seed; MILP equals brute-force
enumeration on a toy instance; the planner never flags an optimizer output
as overlapping.

**V2 Synthetic benchmark.** Cavities from `gtcore.phantom` over a declared
grid of rng seeds × sizes (state the volumes). For each cavity and each N:
arms = random feasible (n_random draws; median and 95th percentile), uniform
heuristic (farthest-point anchors, spin 0), greedy, greedy + local search,
SA, MILP where feasible, and the phantom's truth tiles. Endpoints: V100,
D90, V150, V200 of the +5 mm shell; runtime. Report mean ± SD across
cavities, paired differences with 95 % CIs, Wilcoxon signed-rank for SA vs
greedy and SA vs uniform. Primary comparison, declared before running: SA vs
uniform on V100 at the N where uniform first reaches D90 ≥ rx.

**V3 Optimality gap.** (SA − MILP)/MILP on V100 over the reduced instances;
distribution across cavities and N; solver MIP gap stated when the time
limit was hit.

**V4 Discretization.** Sweep h and n_spins; report objective, runtime, and
the gain from E5. Defaults chosen from this table.

**V5 Sensitivity.** Re-evaluate optimized configurations under
perturbations the optimizer did not see: seed-plane offset over
`geometry.SEED_PLANE_OFFSET_RANGE_MM`, target subsample density, S_K ±5 %,
interseed attenuation on. Report the spread and whether arm rankings
survive.

**V6 Physical phantom and clinical case.** For the 8-tile printed phantom
and clinical case 2 (case 1 if a cavity surface exists): (a) as-implanted =
`tiles.auto` fit → `to_placed_tiles`; (b) optimized with the same N;
(c) minimum N by P2. Report (b − a) on the shells and, with an RTSTRUCT, on
the clinical HR-CTV. The as-implanted dose uses localized seeds; the
optimized dose uses conformed model seeds — run the as-implanted tiles
through the conformer and report what that substitution alone changes.

**V7 Runtime.** Candidate build, influence matrix, each solver, final
reporting; per cavity size; hardware stated.

**V8 Failure modes.** No feasible N-tile placement; tile wider than the
wall patch; eligibility masks excluding most of the wall; degenerate meshes.
Each fails loudly with a reason string, never silently returns a worse plan.

Rules: no number without the command that produced it; fixed seeds;
negative results stay in; if SA does not beat greedy, greedy is the default.

---

## 5. Reporting standards

- Primary endpoint declared before runs; no post-hoc switching.
- Every parameter (h, n_spins, τ, λ's, SA schedule, MILP limits) is a named
  constant with a one-line justification, collected in one table.
- Optimality claims only relative to the MILP bound; never "global optimum"
  for SA.
- Uncertainty everywhere: SD or CI across cavities; variance across SA
  restarts.
- A section in `docs/optimize-notes.md` titled "Not claimed": TG-43 in
  water; static cavity (no post-implant shrinkage); tiles modelled as
  non-overlapping although collagen may stack; surgeon reachability not
  modelled beyond the eligibility mask.

---

## 6. Parallel agents

Several agents work at once, each in its own git worktree and branch off
`main`; one coordinator merges.

**Phase 0 — interface freeze (one agent, before any parallel work).**
`gtcore/plan/__init__.py` with dataclasses and signatures only:
`CandidateSet`, `InfluenceMatrix`, `ConflictGraph`, `Objective`,
`SolverResult`, `OptimizeReport`; `build_candidates`, `build_influence`,
`build_conflicts`, `evaluate`, `solve_greedy`, `solve_local`, `solve_sa`,
`solve_milp`, `sweep_n`, `final_report`, `optimize`, `suggest_next`. Each
raises `NotImplementedError` with a docstring stating inputs, outputs,
units, complexity. Commit; all branches start here.

**Phase 1 — parallel build.**

| Agent | Branch | Owns | Must not touch |
|---|---|---|---|
| A1 Geometry | `plan/candidates` | `candidates.py`, `conflicts.py`, tests | `interact.py` beyond read-only calls |
| A2 Dose | `plan/influence` | `influence.py`, `objective.py`, the §3 B gate | the engine |
| A3 Heuristics | `plan/solvers` | `solvers.py` E1–E3, `sweep.py` | — |
| A4 Exact | `plan/milp` | `milp.py`, clique constraints | — |
| A5 Validation | `plan/validation` | `scripts/validation_optimize.py`, `report.py`, cavity grid, figures | solver internals |
| A6 Integration | `plan/ui` | CLI, planner keys, docs | solver internals |

Until A1/A2 land, A3–A6 develop against **stub fixtures** committed under
`tests/`: a flat 60 × 60 mm wall mesh and a sphere-cap mesh with analytic
dose, so their tests do not depend on the real modules.

**Sync points.** (1) A1 + A2 merged → A3/A4 rerun on real candidates; the
§3 B gate must pass before any solver result is recorded. (2) A3 merged →
A5 runs V2 greedy/SA. (3) A4 merged → V3. (4) A6 last. Merge order: A1,
A2, A3, A4, A5, A6.

**Reviewer agent (A7)**, independent of the builders: reads §2 and §0
only, not the implementation, and writes adversarial tests — a cavity where
greedy is provably suboptimal (constructed), a conflict the pairwise test
misses but the clique catches, an influence row that disagrees with the
engine, a seed that changes the SA answer. Re-derives V100/D90 for stored
configurations with its own minimal implementation. Its report goes in the
notes unedited.

**Rules for every agent.**
- Suite green at every commit; tests with every function.
- Measurements to `docs/optimize-notes.md` with command, seed, commit hash.
- No edits to `gtcore/dose/engine.py`, `gtcore/geometry.py`, or pipeline
  defaults. Additive hooks only.
- Unsettled design question → write it with two candidate answers under
  "Open decisions" in the notes, proceed with the simpler one, flagged.

**Scout agents** run concurrently with Phase 1, one per alternative in §7:
smallest experiment that tests it on the stub fixtures plus a few synthetic
cavities; one-page report (question, method, numbers, verdict). Scouts
never merge into the feature branch; the coordinator promotes an
alternative only on evidence.

---

## 7. Alternatives (each with its decision criterion, written before the experiment)

Run §8 first.

**7.1 Coverage objective flat or misleading.** Symptom: SA accepts many
moves with zero hard-objective change; V100 saturates for every arm.
Alternatives: optimize D90 directly; soft coverage with smaller τ; a
conformity-style index (coverage minus normalized hot volume). Criterion:
discriminates arms in V2 where V100 does not.

**7.2 Non-overlap too strict.** Collagen tiles may stack or overlap at
edges; the planner treats overlap as a caution. Alternative: overlap as a
penalized soft constraint with the shadowing check, or allowed below an
area fraction (e.g. 15 %). Criterion: higher V100 at equal V200 on at least
half the V2 cavities; otherwise keep the hard constraint.

**7.3 MILP does not scale on reduced instances.** Alternatives: Lagrangian
relaxation of the coverage rows (cheap upper bound); column generation over
anchors; LP-relaxation bound reported as such. Criterion: a bound within
5 % of the SA incumbent within the time limit.

**7.4 Discretization error dominates.** Symptom: E5 gains more than the
SA-vs-greedy difference. Alternatives: finer h with pruning of dominated
candidates; direct continuous optimization over (u, v, θ) per tile with
CMA-ES or multi-start Nelder–Mead on the full conformer; gradient-based
refinement using the analytic derivative of TG-43 dose with respect to seed
position (chain rule through tangent-plane offsets, finite differences
through the mesh projection). Criterion: matches or beats discrete + E5 at
equal wall time.

**7.5 Half tiles / cutting as a decision.** Treat "cut this tile" as part
of the inventory decision (two half candidates for one full). Criterion:
minimum N (P2) drops on narrow cavities.

**7.6 Multi-objective.** ε-constraint sweeps over OAR limit and hot-spot
tolerance → Pareto front per cavity. Supplementary unless it changes a
conclusion.

**7.7 Optimizer gain is small.** If SA beats the uniform heuristic by
< 2 percentage points of V100 on most synthetic cavities, the value lies in
minimum-N estimation and placement verification (P2, V6), which compare
directly to the manufacturer's surface-area tile calculator and to the
implanted count. Reframe; do not select cavities to inflate the gain.

**7.8 Learned models (separate feature, deferred).** Justified only if
(a) SA runtime exceeds an intraoperative budget on clinical-size cavities
after profiling, or (b) a learned objective is needed (e.g. a predictor of
post-implant cavity shrinkage). If (a), the first remedy is engineering
(coarser h with E5 polish, cached influence rows), not a neural policy.

---

## 8. Go/no-go experiment (before Phase 1 goes wide)

On a small set of synthetic cavities: candidates at h = 3 mm, 6 spins;
influence on the +5 mm shell; greedy only; V100/D90 against the uniform
heuristic and random feasible placements for N = 4, 6, 8. Numbers to the
notes.

- Gain ≥ 3 percentage points V100 over uniform on most cavities → proceed
  with §3 as written.
- Gain 1–3 → proceed; framing follows §7.7; SA/MILP effort reduced to what
  V3 needs.
- Gain < 1 → stop; revisit objective and target definition (§7.1) before
  any further build.

---

## 9. Definition of done

- `gtcore.plan` merged; suite green; test count recorded in README.
- `scripts/validation_optimize.py` regenerates every table and figure from
  seeds; summarized in `docs/optimize-notes.md` with commands and commit
  hashes.
- V1–V8 complete, negative results included.
- Outputs: problem statement + parameter table; coverage-vs-N figure with
  all arms; optimality-gap figure; discretization/sensitivity table;
  phantom + clinical as-implanted-vs-optimized figure; runtime table; the
  "Not claimed" section.
- Planner and CLI expose the feature opt-in.

---

## 10. Additions from the 2026-10-07 brief (coordinator)

- **Tile-count recommendation.** Before asking for N, the planner recommends
  a count from the cavity mesh using the manufacturer's rule: treatable
  surface area / 4 cm² per tile, rounded up. Source: GammaTile Cavity
  Surface Area Calculator (gammatile.com/hcp/medical-physics/surface-area-calculator):
  ellipsoid surface area from three diameters (medial-lateral,
  anterior-posterior, superior-inferior), minus "estimated surgical cavity
  contraction (%)" and "estimated surface area not requiring GammaTiles (%)"
  (skull interface, tissue not at oncologic risk), then "divide the tumor
  bed calculator result by 4, rounding up to the nearest whole number
  (GammaTile = 4 cm²)". With a measured mesh the area is the mesh area itself
  (the scan is post-resection, so contraction defaults to 0 %); the
  ellipsoid estimate from the cavity's principal extents is reported
  alongside for comparison with the pre-operative workflow. Both deduction
  percentages stay as parameters.
- **Planner flow.** An "optimize placement" key prompts for N with the
  recommendation pre-filled, then runs the optimizer and drops the tiles as
  ordinary placed tiles.
- **Test case.** The 8-tile printed phantom
  (`C:\Users\jacob\OneDrive\Documents\3D-Printed Phantom-8tiles (223)`),
  HR-CTV defined as the +5 mm shell of the cavity wall. This scan has no
  cavity mask (printed shell, no brain window); the wall is the inner
  surface of the printed shell (`meshes["body"]`, the planner's existing
  phantom-shell fallback) and the optimizer must accept such a mesh.
