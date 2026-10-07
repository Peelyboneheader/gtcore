"""Objective (section 3 D). Owner: A2 Dose, branch ``plan/influence``.

Must implement the ``Objective`` methods (``dose_of``, ``metrics``, ``hard``,
``soft``, ``gain``) plus ``make_objective(influence, conflicts, **weights)``
and ``evaluate(objective, selection)``:

- dose vector of a selection = ``D[sel].sum(0)``;
- metrics by weighted order statistics (``weighted_quantile``, tested
  against ``numpy.percentile`` for unit weights): V100, D90, V150, V200,
  Dmean and ``oar_dmax_<name>``;
- hard P1 objective ``V100 - lambda_hot * max(0, V200 - v200_tol)
  - sum_j lambda_oar * max(0, Dmax(O_j) - L_j)``;
- soft coverage for annealing ``sum_m w_m sigma((D_m - rx)/tau)``,
  ``tau = TAU_FRACTION * rx``;
- O(N * M) per evaluation, target <= 1 ms at M = 4k.
"""
