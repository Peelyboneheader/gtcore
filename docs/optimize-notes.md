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
