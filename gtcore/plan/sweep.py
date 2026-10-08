"""N-sweep (section 3 F, problem form P2). Owner: A3 Heuristics, branch ``plan/solvers``.

Implements :func:`sweep_n`: run P1 for N = 1 .. ``n_max`` with the chosen
solver, warm-started from N-1; per-N rows with V100, D90, V150, V200, OAR
Dmax, runtime; minimum N per criterion (``"D90>=rx"``, ``"V100>=0.90"``).

Warm starts
-----------
``greedy``: ``fixed`` = the previous N's selection (so the curve is monotone
non-decreasing in V100 by construction: the previous tiles stay).
``local``: greedy top-up of the previous selection, then local search.
``sa`` / ``continuous``: ``start`` = previous selection + one greedy addition
(for ``continuous`` the row metrics come from the refined tiles on the
subsampled target with the real engine, and the tiles are stored under
``result.extra["tiles"]``).
When a warm start boxes itself in (no compatible candidate left) the N is
retried from scratch (``row["warm_start"] = False``); the sweep stops at the
first N neither start can place (``status != "ok"``); that result is kept in
``results`` without a row.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import (
    DEFAULT_RX_CGY,
    CandidateSet,
    ConflictGraph,
    InfluenceMatrix,
    Objective,
    SolverResult,
    SweepResult,
    TargetSet,
)

CRITERIA = ("D90>=rx", "V100>=0.90")


def _row(n: int, r: SolverResult, rx: float) -> Dict[str, Any]:
    m = dict(r.metrics)
    row: Dict[str, Any] = {
        "N": int(n),
        "V100": float(m.get("V100", float("nan"))),
        "D90": float(m.get("D90", float("nan"))),
        "V150": float(m.get("V150", float("nan"))),
        "V200": float(m.get("V200", float("nan"))),
        "objective": float(r.objective),
        "runtime_s": float(r.runtime_s),
        "solver": r.solver,
        "status": r.status,
    }
    for k, v in m.items():
        if k.startswith("oar_dmax_"):
            row[k] = float(v)
    row["D90>=rx"] = bool(row["D90"] >= rx)
    row["V100>=0.90"] = bool(row["V100"] >= 0.90)
    return row


def sweep_n(mesh, target: TargetSet, n_max: int, rx_cgy: float = DEFAULT_RX_CGY,
            solver: str = "greedy", seed: int = 0,
            candidates: Optional[CandidateSet] = None,
            influence: Optional[InfluenceMatrix] = None,
            conflicts: Optional[ConflictGraph] = None, **kw) -> SweepResult:
    """P2: run P1 for N = 1 .. ``n_max`` with ``solver`` (``"greedy"``,
    ``"local"``, ``"sa"`` or ``"continuous"``), warm-started from N-1, and report the
    coverage-vs-N curve plus the minimum N per criterion.

    ``candidates`` / ``influence`` / ``conflicts`` are built with
    :func:`gtcore.plan.build_candidates` / :func:`build_influence` /
    :func:`build_conflicts` when None (``mesh`` and ``target`` are only
    needed then).  Extra ``kw`` are forwarded to the solver (e.g.
    ``n_sweeps``, ``n_restarts``, ``radius_mm``, ``max_evals``); objective
    weights may be passed as ``lambda_hot``, ``v200_tol``, ``lambda_oar``,
    ``tau_cgy``, ``lambda_tail``, ``tail_q``.  The two P2 criteria are the
    same statement (``D90 >= rx`` <=> at most 10 % of the target weight is
    below rx <=> ``V100 >= 0.90``); both are kept for the report.
    """
    from . import build_candidates, build_conflicts, build_influence
    from .solvers import solve_continuous, solve_greedy, solve_local, solve_sa

    solver = str(solver).lower()
    if solver not in ("greedy", "local", "sa", "continuous"):
        raise ValueError("solver must be 'greedy', 'local', 'sa' or 'continuous', got %r"
                         % (solver,))
    n_max = int(n_max)
    if n_max < 1:
        raise ValueError("n_max must be >= 1")

    if candidates is None:
        candidates = build_candidates(mesh, **{k: kw.pop(k) for k in
                                               ("h_mm", "n_spins", "kinds", "eligible_faces")
                                               if k in kw})
    if influence is None:
        influence = build_influence(candidates, target, rx_cgy=rx_cgy)
    if conflicts is None:
        conflicts = build_conflicts(candidates)

    obj_kw = {k: kw.pop(k) for k in ("lambda_hot", "v200_tol", "lambda_oar", "tau_cgy",
                                     "lambda_tail", "tail_q")
              if k in kw}
    objective = Objective(influence, conflicts, rx_cgy=float(rx_cgy), **obj_kw)

    rows: List[Dict[str, Any]] = []
    results: List[SolverResult] = []
    prev: np.ndarray = np.zeros(0, dtype=int)
    def run(n: int, start: np.ndarray) -> SolverResult:
        if solver == "greedy":
            return solve_greedy(objective, n, fixed=start, candidates=candidates, **kw)
        if solver == "local":
            return solve_local(objective, n, start, candidates=candidates, **kw)
        g = solve_greedy(objective, n, fixed=start, candidates=candidates)
        if not g.feasible:
            g.solver = solver
            g.reason = "greedy warm start failed: " + g.reason
            return g
        if solver == "sa":
            return solve_sa(objective, n, seed=int(seed) + n, start=g.selection,
                            candidates=candidates, **kw)
        tiles, r = solve_continuous(mesh, candidates, target, float(rx_cgy), n,
                                    seed=int(seed) + n, start=g.selection,
                                    objective=objective, conflicts=conflicts, **kw)
        r.extra["tiles"] = tiles          # continuous poses, not candidates
        return r

    for n in range(1, n_max + 1):
        r = run(n, prev)
        warm = True
        if not r.feasible and prev.size:
            # the warm start boxed itself in: retry from scratch before giving up
            r_cold = run(n, np.zeros(0, dtype=int))
            if r_cold.feasible:
                r_cold.extra["warm_start_failed"] = r.reason
                r, warm = r_cold, False
        results.append(r)
        if not r.feasible:            # "time_limit" / "max_evals" results are still used
            break
        row = _row(n, r, float(rx_cgy))
        row["warm_start"] = warm
        rows.append(row)
        prev = np.asarray(r.selection, dtype=int)

    min_n: Dict[str, Optional[int]] = {}
    for crit in CRITERIA:
        hits = [row["N"] for row in rows if row[crit]]
        min_n[crit] = int(min(hits)) if hits else None
    return SweepResult(rows=rows, min_n=min_n, results=results)
