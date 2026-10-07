"""Exact reference (section 3 E4). Owner: A4 Exact, branch ``plan/milp``.

Must implement ``solve_milp(objective, n_tiles, time_limit_s, exact_n,
mip_rel_gap) -> SolverResult`` with ``scipy.optimize.milp`` (HiGHS):

- ``x_c in {0,1}`` select candidate, ``y_m in {0,1}`` target point covered;
  maximize ``sum_m w_m y_m`` s.t. ``sum_c D[c,m] x_c >= rx y_m``, clique
  constraints ``sum_{c in Q} x_c <= 1``, ``sum_c x_c = N`` (or ``<= N``),
  OAR rows ``sum_c D_j[c,o] x_c <= L_j``;
- coverage only (no hot-spot term) -- state it in every comparison;
- always fill ``bound`` and ``mip_gap``; ``status="time_limit"`` when the
  limit is hit, in which case the bound is the reference, not the incumbent;
- test: equals brute-force enumeration on a toy instance
  (``tests/plan_fixtures.toy_instance``).

Owns the clique-constraint consumption; A1 produces the cliques.
"""
