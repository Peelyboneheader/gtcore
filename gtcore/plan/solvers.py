"""Heuristic solvers (section 3 E1-E3, E5). Owner: A3 Heuristics, branch ``plan/solvers``.

Must implement:

- ``solve_greedy(objective, n_tiles, fixed, kinds_required)``: E1 greedy
  forward selection, deterministic (ties -> lowest id), marginal-gain curve
  in ``extra``; also the engine behind ``suggest_next``.
- ``solve_local(objective, n_tiles, start, radius_mm, candidates)``: E2
  first-improvement local search (local re-site within ``radius_mm``,
  re-spin in place, global re-site), stop at a local optimum.
- ``solve_sa(objective, n_tiles, seed, n_sweeps, n_restarts, start,
  candidates)``: E3 Metropolis on the soft objective, move mix 60/20/20,
  T0 from trial moves, geometric cooling ``SA_ALPHA`` per sweep of
  ``SA_MOVES_PER_TILE_PER_SWEEP * N`` moves, best HARD objective kept,
  reproducible from ``seed``, best-so-far per sweep in ``history``.
- ``refine_continuous(mesh, candidates, selection, target, rx_cgy, ...)``:
  E5 Nelder-Mead polish over (u, v, theta) through ``conform_tile``
  (assignment to A3 is a Phase 0 decision; see docs/optimize-notes.md).

Greedy must never violate a conflict; SA must be reproducible from its seed.
Develop against ``tests/plan_fixtures.toy_instance`` until A1/A2 land.
"""
