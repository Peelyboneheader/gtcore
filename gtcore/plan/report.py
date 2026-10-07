"""Final reporting (section 3 G). Owner: A5 Validation, branch ``plan/validation``.

Must implement ``final_report(mesh, tiles, rx_cgy, target, cavity_mask,
cavity_affine, interference, grid_mm, margin_mm, solver_result, parameters)
-> OptimizeReport`` for any configuration (optimized, as-implanted, manual,
random):

1. ``compute_dose_grid``, exact kernel, 1 mm grid, cavity bounds + 50 mm;
   optional second run with the interference model, difference reported;
2. shell metrics via ``shell_report`` on the full target; rind DVH via
   ``rind_mask`` (5 mm) when a cavity mask exists; clinical HR-CTV DVH when
   an RTSTRUCT exists;
3. conflicts re-verified with ``find_overlapping_tiles``; shadowing listed
   via ``find_shadowing_tiles``;
4. JSON + CSV (``OptimizeReport.to_json`` / ``to_csv``) with gtcore version,
   parameters, seed, wall-clock time.

All reported metrics come from the dose grid, never from the influence
matrix.  Must not touch solver internals.
"""
