# Scout 7.1 — is the hard V100 objective flat or misleading?

Evidence only (branch `plan/scout-objective`, not for merge). Script:
`scripts/scout_objective.py`; numbers below were produced at commit
**5cb6199** (script logic identical to 79c230e; results.csv byte-identical
between a from-scratch rebuild and the cached re-run) with

    python scripts/scout_objective.py        # from the worktree root

Runtime: ~4 min from scratch (per cavity: phantom + mesh 0.3 s, candidates
4–6 s, influence rows 29–34 s, conflicts 13–27 s; selection/greedy/random
~3 s total); 3 s when the four pickled instances in
`output/scout_objective/` already exist. Outputs: `results.csv`,
`random.csv`, `instances.csv`, `summary.md` in that directory.

## Question

§7.1 of `plan-tile-optimize.md`: does hard V100 on the +5 mm shell
(area-weighted vertices) discriminate between placement arms, or does it
saturate / go flat so that a different objective (soft sigmoid coverage,
D90, conformity-style) is needed?

**Decision criterion (written before running).** An alternative objective
is worth promoting only if, on at least 3 of 4 cavities, greedy under it
reaches a hard V100 ≥ 1 percentage point higher than greedy under hard
V100 itself at the same N, or if hard-V100 greedy shows ties at ≥ 30 % of
its steps (tie = ≥ 2 feasible candidates share the best hard-V100 marginal
gain).

## Method

Four synthetic cavities (`make_head_phantom`, rng_seed 1–4, cavity mesh
only; truth tiles ignored). Candidates: farthest-point-sampled anchors on
the mesh vertices, two spins per anchor, conformed with `conform_tile`
(kind "full"); rejected if any seed is > 1.5 mm from its 3 mm wall offset
(0 rejections occurred). Influence rows: `dose_at_points` (tabulated
kernel, total decay, default S_K) at every +5 mm shell vertex; the
optimizer sees a weight-proportional subsample of 2000 vertices with equal
weights; the tables report hard metrics re-evaluated on the full
area-weighted shell (subsample and full agree to ≤ 1.9 pp; max 1.82). Conflicts:
one `find_overlapping_tiles` call over the whole candidate list.
For N ∈ {4, 6, 8}: 200 random feasible selections (random order, add if
non-conflicting), and greedy forward selection (ties → lowest id) under
each objective. Spearman ρ is between each objective and hard V100 over
the 200 random selections. One arm was added after the first run:
hard V100 with ties broken by soft coverage (τ = 0.05 rx), to separate
"flat at step 1" from "misleading later".

| parameter | value | note |
|---|---|---|
| phantom spacing | 1.0 mm | build speed; mesh 6.2–6.9 k vertices, area 42–46 cm² |
| anchor spacing h | 4 mm | FPS on vertices → 155–173 anchors |
| spins | 0°, 45° | 310–346 candidates per cavity |
| detach limit | 1.5 mm | §3 A |
| overlap threshold | 1.0 mm | `find_overlapping_tiles` default; conflict density 0.37–0.43 |
| target | +5 mm shell, vertex-area weights | §2 default |
| M_opt | 2000 | weight-proportional subsample, equal weights |
| rx | 6000 cGy | |
| N | 4, 6, 8 | |
| random arms | 200 per (cavity, N) | |
| soft τ | 0.05, 0.02, 0.10 rx | §3 D default and both neighbours |
| conformity | V100 − 0.5·max(0, V200 − 0.10) | P1 defaults |
| rng seed | 20261007 (+ cavity/N offsets) | fixed |

## Numbers

Hard V100 (full +5 mm shell) of the greedy result; tiles actually placed in
brackets when greedy ran out of feasible candidates before N. "cavities
≥ +1 pp" counts cavities where the arm beats hard-V100 greedy by ≥ 1 pp.

**N = 4**

| arm | cav 1 | cav 2 | cav 3 | cav 4 | mean ΔV100 (pp) | cavities ≥ +1 pp | mean ρ vs hard |
|---|---|---|---|---|---|---|---|
| random (200): mean / max | 0.029 / 0.172 | 0.018 / 0.126 | 0.024 / 0.158 | 0.022 / 0.159 | – | – | – |
| hard_v100 | 0.223 | 0.202 | 0.227 | 0.204 | +0.0 | 0/4 | – |
| hard_lex_soft0.05 | 0.224 | 0.222 | 0.200 | 0.204 | −0.1 | 1/4 | 0.97 |
| soft_tau0.05 | 0.224 | 0.222 | 0.227 | 0.204 | +0.6 | 1/4 | 0.94 |
| soft_tau0.02 | 0.224 | 0.220 | 0.195 | 0.204 | −0.3 | 1/4 | 0.96 |
| soft_tau0.10 | 0.253 | 0.222 | 0.227 | 0.207 | +1.4 | 2/4 | 0.90 |
| d90 | 0.082 | 0.052 | 0.053 | 0.021 | −16.2 | 0/4 | −0.60 |
| conformity | 0.223 | 0.202 | 0.227 | 0.204 | +0.0 | 0/4 | 1.00 |

**N = 6**

| arm | cav 1 | cav 2 | cav 3 | cav 4 | mean ΔV100 (pp) | cavities ≥ +1 pp | mean ρ vs hard |
|---|---|---|---|---|---|---|---|
| random (200): mean / max | 0.458 / 0.554 | 0.353 / 0.469 | 0.385 / 0.514 | 0.329 / 0.494 | – | – | – |
| hard_v100 | 0.468 | 0.467 | 0.453 | 0.454 | +0.0 | 0/4 | – |
| hard_lex_soft0.05 | 0.484 | 0.462 | 0.492 | 0.454 | +1.3 | 2/4 | 1.00 |
| soft_tau0.05 | 0.484 | 0.489 | 0.512 | 0.454 | +2.4 | **3/4** | 0.97 |
| soft_tau0.02 | 0.484 | 0.483 | 0.553 | 0.454 | +3.3 | **3/4** | 0.99 |
| soft_tau0.10 | 0.507 | 0.489 | 0.504 | 0.449 | +2.7 | **3/4** | 0.94 |
| d90 | 0.412 | 0.373 | 0.416 | 0.327 | −7.9 | 0/4 | −0.49 |
| conformity | 0.468 | 0.467 | 0.453 | 0.454 | +0.0 | 0/4 | 1.00 |

**N = 8**

| arm | cav 1 | cav 2 | cav 3 | cav 4 | mean ΔV100 (pp) | cavities ≥ +1 pp | mean ρ vs hard |
|---|---|---|---|---|---|---|---|
| random (200): mean / max | 0.905 / 1.000 | 0.823 / 0.985 | 0.842 / 0.986 | 0.783 / 0.947 | – | – | – |
| hard_v100 | 0.681 [7] | 0.889 | 0.821 | 0.769 | +0.0 | 0/4 | – |
| hard_lex_soft0.05 | 0.664 [7] | 0.854 | 0.833 | 0.769 | −1.0 | 1/4 | 1.00 |
| soft_tau0.05 | 0.664 [7] | 0.793 | 0.860 | 0.769 | −1.9 | 1/4 | 0.99 |
| soft_tau0.02 | 0.664 [7] | 0.849 | 0.890 | 0.769 | +0.3 | 1/4 | 1.00 |
| soft_tau0.10 | 0.943 | 0.793 | 0.887 | 0.753 | +5.4 | 2/4 | 0.96 |
| d90 | 0.412 [6] | 0.373 [6] | 0.416 [6] | 0.870 | −27.2 | 1/4 | 0.93 |
| conformity | 0.681 [7] | 0.889 | 0.821 | 0.769 | +0.0 | 0/4 | 1.00 |

D90 (cGy, full shell) of the greedy result, selected arms:

| N | hard_v100 | soft_tau0.02 | soft_tau0.10 | d90 |
|---|---|---|---|---|
| 4 | 1520 / 1418 / 1190 / 1218 | 1407 / 1164 / 1273 / 1218 | 1230 / 1289 / 1187 / 1216 | 2317 / 2194 / 2133 / 2158 |
| 6 | 3074 / 2691 / 2688 / 2199 | 2740 / 2334 / 2623 / 2199 | 2668 / 2134 / 2286 / 2138 | 4084 / 3887 / 4110 / 3999 |
| 8 | 4938 / 5956 / 5529 / 4666 | 4215 / 5404 / 5864 / 4666 | 6257 / 5407 / 5920 / 4600 | 4084 / 3887 / 4110 / 5869 |

Ties: hard-V100 greedy had a tie at **12 of 71 steps (17 %)**, and every
one is step 1, where all 310–346 candidates tie at zero gain (no single
tile reaches 6000 cGy anywhere on the +5 mm shell; max single-tile shell
dose ≈ 4200 cGy). No later step tied. Hard V100 across the 200 random
selections takes 78–165 distinct values per (cavity, N). V200 on the +5 mm
shell never exceeded 0.0004 at N ≤ 8, so the conformity arm is identical to
hard V100 here.

## Verdict against the criterion

**Not met → do not promote an alternative objective.**

- Tie test: 17 % < 30 %, and the ties are the structural step-1 degeneracy
  (every first tile is worthless under a hard threshold), not flatness
  during selection. A lexicographic (hard, then soft) tie-break is the
  cheap fix and changes the step-1 choice only.
- ≥ 1 pp on ≥ 3/4 cavities: the three soft-τ arms meet it **only at N = 6**
  (+2.4 to +3.3 pp mean). At N = 4 they are 1–2/4 and at N = 8 they are
  1–2/4 with losses of up to −9.8 pp (cavity 2, τ = 0.05 and 0.10). The
  choice of τ moves the same cavity by ±5–10 pp with no consistent sign,
  i.e. the soft objective is a noisy hyperparameter, not a systematic
  improvement. D90 as an objective is harmful at N ≤ 6 (−8 to −16 pp, ρ
  negative) and stops early at N = 8. Conformity is inert on the +5 mm
  shell at these N.
- Discrimination: hard V100 ranks random arms the same way the soft
  objectives do (ρ = 0.90–1.00); it is not saturated or flat in the
  regime tested (V100 0.02–1.00).

**Side finding that matters more (for §8 / §3 E, not §7.1).** At N = 8,
greedy under *every* objective is at or below the random-selection
*mean* on cavity 1 (hard 0.68 vs random mean 0.91, max 1.00) and ran out of
feasible candidates after 7 tiles (D90 arm: after 6 on three cavities).
Greedy forward selection with a hard conflict graph at density ~0.4 is
myopic (it clusters tiles where the immediate V100 gain is largest, then
has nowhere left to place the rest), and random order spreads tiles
better. The §8 "greedy vs random" go/no-go will show greedy *losing* at
N = 8 unless greedy is made feasibility-aware (reject a candidate whose
addition leaves < N − k placeable tiles, or seed from a spread-out
heuristic) and/or followed by E2 local search. That is a solver issue;
the objective is fine.

Not claimed: anything about N > 8, finer h or more spins, OAR terms, the
wall (0 mm) or +10 mm shells, or the behaviour of SA on the soft objective
(the annealer's acceptance statistics are a different question from
greedy's path).
