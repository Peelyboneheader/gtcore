"""N-sweep (section 3 F, problem form P2). Owner: A3 Heuristics, branch ``plan/solvers``.

Must implement ``sweep_n(mesh, target, n_max, rx_cgy, solver, seed,
candidates, influence, conflicts, **kw) -> SweepResult``: run P1 for
N = 1 .. n_max with the chosen solver, warm-started from N-1; per-N rows
with V100, D90, V150, V200, OAR Dmax, runtime; minimum N per criterion
(``"D90>=rx"``, ``"V100>=0.90"``).
"""
