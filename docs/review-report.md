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

(a) **Greedy provably suboptimal (N = 2).** Flat wall; full tiles A (x = -15),
B (+15), C (0), decoy D (+45); A-B feasible, C conflicts with both; target
rows above A/C/B weighted 0.3/0.4/0.3, rx = 3000 cGy. Certified by the
reference: single-tile V100 = 0.30 / 0.30 / 0.40 / 0.00, so any forward
greedy picks C first; every feasible pair containing C has V100 <= 0.40;
brute-force optimum {A, B} V100 = 0.84 (V200 = 0 everywhere, so P1 = V100).
Reference greedy: [C, D], V100 = 0.40.
Builders' greedy: PENDING. SA: PENDING. MILP: PENDING.
MILP with `cliques=[]` (pairs only): PENDING.

(b) **Clique over-constraint.** §2: "no two tiles conflict (hard)" -- the
hard constraint is PAIRWISE; a clique inequality is a valid tightening only
when every pair in the clique conflicts under `find_overlapping_tiles`.
Triple P (0, 0), Q (23, 0), R (11.5, 23): pairwise feasible (3 mm edge
gaps), anchor distances 23.0 / 25.7 / 25.7 mm < one tile diagonal 28.3 mm;
all three is the unique brute-force optimum at N = 3.
`build_conflicts` on the triple: PENDING. Solvers on the builders' graph at
N = 3: PENDING. Dense 5 x 5 grid (11.5 mm pitch): pairs vs
`find_overlapping_tiles` and every clique a true clique: PENDING.

(c) **Influence rows.** 6 reference candidates on the phantom, full +5 mm
shell (`m_opt` = M, no subsampling), points >= 2.5 mm from every seed,
tolerance 1 % of rx. Tabulated kernel: PENDING. Exact kernel: PENDING.

(d) **SA seed sensitivity.** 24 reference candidates, N = 4, rx 4000.
PENDING.

(e) **Re-derived metrics.** `Objective.metrics` vs reference on identical
rows: PENDING. `solve_greedy` result metrics: PENDING. `optimize(greedy,
N = 4)` grid-reported +5 mm V100/D90 vs reference on the full shell: PENDING.

(f) **Planner consistency.** `optimize` output and greedy output under
`find_overlapping_tiles`: PENDING.

## Findings so far (independent of the builders' code)

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

PENDING.

## Not checked

PENDING.
