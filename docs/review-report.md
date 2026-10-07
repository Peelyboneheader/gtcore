# Independent review of `gtcore.plan` (agent A7)

Branch `plan/review`. Reviewer read §0, §2, §4 V1, §6 (A7) and §10 of
`docs/plan-tile-optimize.md` plus the Phase-0 *interface* (public
signatures, docstrings, dataclass fields of `gtcore.plan`); not §3, not any
`gtcore/plan/*` body. Reference implementation: `tests/review_reference.py`;
probes: `tests/test_plan_review.py` (skip, never fail, while a builder
function still raises `NotImplementedError`). Run from the worktree root:
`python -m pytest tests/test_plan_review.py -q -rs -s`.

## Question

Do the builders' influence rows, conflict/clique constraints, solvers and
reported metrics agree with an independent derivation written from §2 alone,
and do the adversarial constructions (greedy trap, clique over-constraint)
behave as §2 requires?

## Method

- **Reference metrics**: `dose_at_points(exact=True)` summed over seeds;
  V100/V150/V200 as weighted fractions with `>=`; D90 = weighted 10th
  percentile by the inverted-CDF rule (first sorted dose whose normalised
  cumulative weight reaches 0.10; no interpolation).
- **Reference conflict**: `len(find_overlapping_tiles([a, b])) > 0`.
- **Reference target**: +5 mm shell vertices, weight = one third of the
  adjacent face areas. Two readings of §2 exist (faces of the offset shell
  vs faces of the wall); see finding 1.
- **Brute force**: every feasible n-subset of <= 12 candidates, n <= 3,
  maximising V100 (ties: D90, then lowest ids).
- **Solver probes feed the solvers an `Objective` whose `InfluenceMatrix`
  (exact rows) and `ConflictGraph` (pairs from `find_overlapping_tiles`,
  one 2-clique per edge) are built from the reference**, so solvers are
  judged independently of the builders' influence/conflict code, which have
  their own probes.
- Constructions use a subdivided closed box whose top face is the wall
  (`snap_to_wall` orients normals toward the centroid, so seeds sit at
  z = -3 mm and the +5 mm shell at z = +5 mm; seed grid exactly +-5 mm).
- Phantom: `make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)`, cavity
  mesh from `truth.masks["cavity"]` (6152 vertices, 41.8 cm^2 wall,
  77.0 cm^2 +5 mm shell). Phantom probes use rx = 4000 cGy and N = 4: at
  6000 cGy four tiles give V100 ~ 0 on this cavity (the §10 rule asks for
  11 tiles) and the objective is flat; at 4000 cGy N = 4 reaches V100 ~ 0.5
  with V200 = 0.

## Probes and numbers

Run against main `5f0105a` (all six builder branches merged), plan/review
merge commit; `tests/test_plan_review.py`: **14 passed, 2 errors** (both
errors are the shared `optimized` fixture of the end-to-end `optimize`
probe, finding 0; pytest reports a failing fixture as an error); builders'
`tests/test_plan_interface.py`: 19 passed, 13 skipped.

(a) **Greedy provably suboptimal (N = 2) -- FIRED.** Flat wall; full tiles
A (x = -15), B (+15), C (0), decoy D (+45); A-B feasible, C conflicts with
both; target rows above A/C/B weighted 0.3/0.4/0.3, rx = 3000 cGy.
Certified by the reference: single-tile V100 = 0.30 / 0.30 / 0.40 / 0.00,
every feasible pair containing C has V100 <= 0.40, brute-force optimum
{A, B} V100 = 0.84 (V200 = 0 everywhere, so P1 = V100). Builders' greedy:
selection [C, D], V100 = 0.400, status ok (trapped exactly as constructed;
its feasibility-aware bound does not rescue it because D keeps the packing
bound satisfied). SA (seed 0): [A, B], V100 = 0.840 = brute force. MILP:
[A, B], V100 = 0.840, status "optimal". MILP with `cliques=[]` (pairs
only): feasible pair returned -- pairs are honoured without cliques.

(b) **Clique over-constraint -- DID NOT FIRE (correct behaviour).** §2:
"no two tiles conflict (hard)": the hard constraint is pairwise; a clique
inequality is a valid tightening only when every pair in the clique
conflicts under `find_overlapping_tiles`. Triple P (0, 0), Q (23, 0),
R (11.5, 23): pairwise feasible (3 mm edge gaps), anchor distances
23.0 / 25.7 / 25.7 mm < one tile diagonal 28.3 mm; all three is the unique
brute-force optimum at N = 3. `build_conflicts` on the triple: 0 pairs,
cliques = [] (no neighbourhood clique spans the triple); MILP, greedy and SA
on the builders' graph all return all three at N = 3. Dense 5 x 5 grid at
11.5 mm pitch: 72 true conflicting pairs, builders 72 pairs (symmetric, no
diagonal, identical set), 16 cliques of size 4, every clique a true clique
of the pairwise graph.

(c) **Influence rows -- agree.** 6 reference candidates on the phantom,
full 6152-point +5 mm shell (`m_opt` = M, no subsampling), points >= 2.5 mm
from every seed, tolerance 1 % of rx (40 cGy). Tabulated kernel: max
|influence - exact| = 0.05 cGy = 0.001 % of rx, best-fit scale
1.00000-1.00001. Exact kernel: 0.00 cGy. S_K recorded = 3.5 U.

(d) **SA seed sensitivity.** 24 reference candidates, N = 4, rx 4000.
seed 0 twice: identical selection [0, 1, 3, 5] (reproducible). seed 1: the
same selection [0, 1, 3, 5] -- the answer did NOT change with the seed on
this instance. Greedy: [8, 11, 14, 19], V100 0.4993; SA (both seeds): V100
0.5324, P1 objective 0.5324 >= greedy 0.4993. Both feasible, status ok.
(SA beats greedy by 3.3 pp here; the greedy-trap instance in (a) is the
constructed case where the gap is 44 pp.)

(e) **Re-derived metrics -- agree where the code runs.** `Objective.metrics`
on identical rows, selection [0, 5, 10, 15]: builders V100 0.4445, V150
0.0596, V200 0.0, D90 2309.36 vs reference 0.4445 / 0.0596 / 0.0000 /
2309.4 (weighted-quantile conventions agree to < 0.1 cGy). `solve_greedy`
result metrics: V100 0.4993, V150 0.0019, V200 0.0, D90 2029.64 vs
re-derived 0.4993 / 0.0019 / 0.0000 / 2029.6; `objective` 0.4993 = P1
re-derived. `optimize(greedy, N = 4)` grid-reported +5 mm V100/D90 vs the
reference on the full shell: NOT OBTAINABLE -- `optimize` raises (finding
0), so the grid-vs-exact and weighted-vs-unweighted comparison is open.

(f) **Planner consistency.** `solve_greedy` output (N = 4, reference
candidates): `find_overlapping_tiles` == [] and `feasible` True. `optimize`
output: NOT OBTAINABLE (finding 0).

## Findings

0. **DEFECT -- `optimize()` is unusable end-to-end.** Every call fails:
   `optimize(mesh, 4, rx_cgy=4000, solver=<greedy|sa|milp>, seed=0, h_mm=8)`
   and `n_half=1` all raise
   `ValueError: kinds_required needs candidate kinds: pass candidates= or
   attach a CandidateSet as objective.candidates` from
   `gtcore/plan/solvers.py` `solve_greedy` (called via
   `gtcore/plan/api.py:556`). Cause as seen from the interface: `api.optimize`
   passes `kinds_required={"full": n}` to the public
   `gtcore.plan.solve_greedy`, whose frozen signature has no `candidates`
   parameter and which does not attach the `CandidateSet` to the objective;
   the underlying implementation needs one of the two. Direct check:
   `solve_greedy(obj, 4)` -> ok; `solve_greedy(obj, 4, kinds_required=
   {"full": 4})` -> the same ValueError; `solve_greedy(obj, 4,
   kinds_required=..., candidates=cset)` -> `TypeError: unexpected keyword
   argument 'candidates'`. At the default h = 2.5 mm the call spends ~300 s
   building ~2250 candidates and the influence matrix before failing. The
   two review errors are `test_reported_metrics_rederived_on_full_shell`
   and `test_optimizer_output_never_flagged_by_planner` (fixture
   `optimized`, exact message above). Not fixed here, per the brief.


1. **Target-weight convention.** `TargetSet.from_shell` weights are one
   third of the face areas of the *offset shell* (sum 7696.6 mm^2 on the
   phantom), not of the wall mesh (4178.1 mm^2). §2 admits both readings;
   the ratio is not uniform (shell/wall per vertex: mean 1.89, range
   0.11-11.95 on this marching-cubes mesh), so it is not a pure rescaling.
   The reference adopts the shell reading as the literal one and keeps the
   wall reading as an option; the (e) probe prints both.
2. **Worktree import hazard.** With the editable install pointing at the
   main checkout, `import gtcore.plan` in a worktree that lacks
   `gtcore/plan/` silently imports main's copy while `gtcore` itself comes
   from the worktree. Any branch test run before merging the interface
   freeze was testing main's stubs, not its own tree.
3. The flat-wall conformer is exact on a subdivided box (seeds at +-5 mm,
   z = -3 mm); on an unsubdivided 12-triangle box the interpolated vertex
   normals pull the seed grid to +-4.75 mm. Constructions must subdivide.

## Verdict

Solvers, influence, conflicts/cliques and metric conventions pass every
independent probe: the greedy-suboptimal construction fires against the
builders' greedy exactly as designed (0.40 vs 0.84) and SA / MILP recover
the brute-force optimum; the clique over-constraint probe does not fire
(the builders' cliques are true cliques of the pairwise graph and the
pairwise graph equals `find_overlapping_tiles`); influence rows match the
exact engine to 0.05 cGy (0.001 % of rx) with the tabulated kernel; weighted
V100/V150/V200/D90 match the reference to < 0.1 cGy; SA is reproducible from
its seed and, on the phantom instance, seed-insensitive and 3.3 pp better
than greedy. One real defect: the integration entry point `optimize()`
raises for every solver (finding 0), so the end-to-end probes (e)/(f) on the
grid-reported metrics and on the planner-facing output could not run.
Final counts for `tests/test_plan_review.py`: 14 passed, 2 errors (one
fixture, finding 0), 0 skipped. Max influence deviation: 0.05 cGy (tabulated), 0.00 cGy (exact).

## Not checked

`optimize` end-to-end (finding 0): grid-reported +5 mm metrics vs the
weighted exact-point reference (and the unweighted `dvh_stats` vs weighted
§2 definition question), `report.overlaps`, and `optimize` wall time at the
default h = 2.5 mm (~300 s before the failure; the probe uses h = 4 mm).
OAR terms, half tiles, `refine_continuous`, `sweep_n`, `suggest_next` and
`recommend_tile_count` were outside the brief. The pytest traceback of the
failing probe printed `gtcore/plan/solvers.py` source for `solve_greedy`;
the reviewer did not otherwise read builders' code.
