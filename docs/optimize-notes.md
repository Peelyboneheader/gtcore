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
| 2026-10-07 | `python scripts/validation_optimize.py --quick --section v8` | — | cd73082 + A5 | `v8_failure_modes.csv` | 9 cases stub / not run; `final_report(empty mesh)` raises ValueError (PASS) |
| 2026-10-07 | `python scripts/validation_optimize.py --quick` | 1, 2 | cd73082 + A5 | `v2_rows.csv` (truth arms only), `v6_*`, `v7_runtime.csv` | A1–A4 stubs: every solver arm skipped with reason; see V2/V6/V7 text |
| 2026-10-07 | `python scripts/validation_optimize.py --quick --section v6` | — | cd73082 + A5 | `v6_phantom.csv`, `v6_info.json`, `v6_dvh.png`, `v6_as_implanted*.json` | after the centre-snap fix; numbers in V6 (517 s incl. the ray test) |
| 2026-10-07 | `python -m pytest -q -p no:cacheprovider` | — | cd73082 + A5 | 491 passed, 490 s | full suite with `test_plan_report.py` (10) and `test_validation_optimize_smoke.py` (12) |
