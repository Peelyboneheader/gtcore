# feature/tile-autogen — automatic tile creation from seeds alone

Goal: when a scan yields only a seed cloud (no reliable implant count), the
algorithm proposes the tile configuration automatically — enumerating
combinations against the REAL manufactured geometry, rigid first, then
deformed (collagen folds and sticks to the cavity wall).

## Ground-truth geometry (manufacturer/literature; keep cited in code)
- Tile: 20 x 20 x 4 mm bioresorbable collagen square; surgeons may cut halves
  (20 x 10 mm, 2 seeds).
- Seeds: 2x2 array of Cs-131 CS-1 (capsule 4.5 x 0.8 mm), centers 10 mm
  apart, centered in the tile face (5 mm margin to each edge).
- Seed plane offset from the TISSUE-FACING surface: nominal 3.0 mm
  (hydrated spread 2.25-3.75 mm, ~N(3.0, 0.25)) — NOTE: gtcore currently
  uses 2.0 mm (SEED_WALL_OFFSET_MM); Step 0 fixes this.
- Source strength fixed at 3.5 U per seed on implant day.

## Step 0 — recalibrate constants (small PR, do first)
- SEED_WALL_OFFSET_MM 2.0 -> 3.0 everywhere (phantom/generate.py,
  interact.py conform, tests' tolerance bands widen to the 2.25-3.75 spread).
- Add the citations above as comments. Re-run full suite; expect a handful of
  tolerance updates, no logic changes.

## Step 1 — rigid nominal tile model (no deformation)
- `gtcore/tiles/model.py`: `RigidTile` — exact seed positions/axes and the
  20x20 (or 20x10) footprint in a canonical frame; `fit_rigid(seed_pts,
  seed_axes) -> (pose SE(3), rms)` via Kabsch on the 4 (or 2) candidate
  seeds vs the canonical 10 mm grid, trying the 8 symmetries of the square
  (4 rotations x reflection) and both half-tile columns.
- Acceptance: on synthetic rigid layouts, pose recovered to <0.1 mm / <1 deg;
  rms is the residual metric downstream steps reuse.

## Step 2 — count-free configuration search
- Reuse fit.py's pair/quad enumeration but WITHOUT trusted counts:
  enumerate all gate-passing quads/pairs, then select a disjoint subset by
  model selection, not count: score = sum(tile scores) - lambda_penalty *
  n_tiles (a BIC-like complexity penalty), sweep the assignment for n = 0..N/4
  and pick the n where marginal score gain saturates (we already observed
  saturation behavior on the real post-op scan: requesting 5 or 6 tiles kept
  returning 4).
- Output: TileFitResult + a per-n score curve so the UI can show "evidence
  supports n=4 (5th tile adds only X)". Expose in `reconstruct(...,
  n_full_tiles="auto")`.
- Acceptance: on the synthetic phantom (rng sweep) and the 8-tile physical
  scan, auto mode recovers the true n with no count input; on decoy tests it
  refuses to inflate n.

## Step 3 — deformation tier 1: bending WITHOUT a surface (fold model)
- Collagen bends but barely stretches: model the tile as developable —
  start with a single-crease hinge (two rigid half-planes joined at a fold
  line: 7 dof) and, if residuals demand, a 2x2 bilinear-normal patch with a
  bending-energy penalty and an ISOMETRY constraint (geodesic seed spacing
  stays 10 mm even when chords contract).
- `fit_deformable(seed_pts, axes) -> (pose, fold params, rms)`; accept a
  quad as one tile when the DEFORMED fit explains it with low bending energy
  (this subsumes fit.py's chord windows with actual physics, and should
  natively solve the crumpled tile-7 case that today needs the count
  fallback).
- Acceptance: phantom's wall-conformed tiles fit with rms < 0.5 mm; the
  8-tile scan's crumpled tile fits WITHOUT the degraded-completion path;
  random non-tile quads still rejected (bending energy explodes).

## Step 4 — deformation tier 2: stick-to-surface prior (when a cavity exists)
- When the cavity mesh is available, constrain the tile's tissue face to lie
  ON the mesh (offset 3 mm to the seed plane): fit becomes "developable
  patch on a known surface", a 3-dof problem (anchor uv + rotation) plus
  fold residuals. Cross-feed: agreement between the surface-constrained and
  free fits is a confidence signal; disagreement flags either segmentation
  error or tile detachment (both clinically interesting).
- Acceptance: on the synthetic phantom, surface-constrained fit matches
  truth poses at least as well as Step 3 alone, and localizes the tile's
  footprint on the wall for the planner/coverage metrics.

## Step 5 — integration + validation campaign
- `n_full_tiles="auto"` wired through pipeline/CLI/planner ("suggest tiles"
  button). Sweep: synthetic (rng x n_tiles x spacing), 8-tile physical scan,
  PostOp 27-seed cluster. Report per-case: n chosen vs truth, pose errors,
  and the score-saturation curves as a paper figure.
- Merge order: Step 0 alone first (touches shared constants; coordinate with
  main), then 1-2, then 3-5.

Conventions: RAS mm, arrays [k,j,i]; venv python
C:\Users\jacob\.venvs\gammatile\Scripts\python.exe; run tests from the
worktree ROOT (python -m pytest tests/ -q) so this checkout's gtcore is
imported; full suite must stay green at every step; no Claude attribution in
commits.
