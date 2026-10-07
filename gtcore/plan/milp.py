"""Exact reference (section 3 E4). Owner: A4 Exact, branch ``plan/milp``.

Mixed-integer formulation of the discretized placement problem, solved with
``scipy.optimize.milp`` (HiGHS).  Used as the *reference* for the heuristic
solvers (section 4 V3), not as a production solver: the worst case is
exponential and the intended inputs are reduced instances (coarser anchor
spacing, fewer spins, subsampled targets).

Formulation
-----------
Variables ``x_c in {0,1}`` (candidate ``c`` selected, ``C`` of them) and
``y_m in {0,1}`` (target point ``m`` covered, ``M`` of them).

maximize   ``sum_m w_m y_m / sum_m w_m``             (= V100 of the selection)

subject to
  coverage   ``sum_c D[c,m] x_c - rx y_m >= 0``       for every ``m``
  cliques    ``sum_{c in Q} x_c <= 1``                for every valid clique ``Q``
  pairs      ``x_i + x_j <= 1``                       for every conflicting pair
                                                      not inside a valid clique
  count      ``sum_c x_c = N``  (``exact_n``)  or  ``<= N``
  OAR        ``sum_c D_j[c,o] x_c <= L_j``            for every OAR ``j``, sample ``o``

**Coverage only.**  The hot-spot term of P1 (``lambda_hot * max(0, V200 -
v200_tol)``) is NOT in the MILP (it is not linear in ``x``); the OAR limits
enter as hard rows rather than as the P1 penalty.  Every comparison against
the MILP must therefore be on V100 (``extra["milp_objective"]``), while
``SolverResult.objective`` is the full P1 hard value recomputed from the
returned selection so results stay comparable with the other solvers.

**Conflict rows.**  The constraint set equals the pairwise conflict graph
exactly: every conflicting pair is forbidden either by a clique row that
contains both ends or by its own pairwise row, and no non-conflicting pair
is ever forbidden.  Cliques are only a *tightening* of the LP relaxation
(one row ``sum x <= 1`` over ``k`` candidates dominates the ``k(k-1)/2``
pairwise rows).  Every clique is validated against ``conflicts.pairs``
before use (:func:`validate_cliques`); a "clique" with a non-conflicting
pair would forbid a feasible selection, so such entries are dropped with a
``UserWarning`` and their pairs fall back to pairwise rows.

**Pruning.**  Coverage rows are sparse: an influence entry ``D[c,m]`` is
dropped when ``D[c,m] <= PRUNE_FRACTION * rx``.  Dropping entries can only
lower the left-hand side of a coverage row, so the MILP never counts a
point as covered that is not (it is *conservative*); it can miss a point
whose true dose lies in ``[rx, rx + N * PRUNE_FRACTION * rx)`` (at most
``N`` selected candidates each contributing at most ``PRUNE_FRACTION * rx``).
``extra["formulation"]["n_pruned"]`` records how many entries were dropped;
``prune_frac=0.0`` disables pruning (tested to give the same optimum on the
toy instance).

Status mapping (``SolverResult.status``): ``"optimal"`` (HiGHS proved
optimality within ``mip_rel_gap``), ``"time_limit"`` (limit hit; ``bound``
is the reference per section 3 E4, the incumbent -- if any -- is returned
as ``selection``), ``"infeasible"`` (no selection satisfies the rows; empty
selection, ``feasible=False``, a reason, never an exception), ``"error"``
(solver failure or a returned selection that violates the conflict graph).
``bound`` is always filled with the solver's dual bound normalized by the
total weight (an upper bound on V100; NaN when infeasible / error) and
``mip_gap`` with HiGHS's relative gap.

Helpers: :func:`lp_bound` (LP relaxation, an upper bound for the validation
agent when the MILP times out), :func:`brute_force` (exhaustive enumeration
for tiny instances; the V1 test oracle), :func:`build_formulation` (the
constraint matrices, for inspection).
"""
from __future__ import annotations

import itertools
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import milp as _scipy_milp

from . import MILP_TIME_LIMIT_S, ConflictGraph, Objective, SolverResult, _as_ids

PRUNE_FRACTION = 1e-3
# Coverage-row entries below 1e-3 * rx are dropped: with N <= ~20 tiles the
# left-hand side of a row is underestimated by at most 2 % of rx, and the
# MILP can only become conservative (never counts an uncovered point).

BRUTE_FORCE_MAX_SUBSETS = 200_000
# Enumeration guard for brute_force (C <= 15, N <= 4 gives at most 1365).

_HIGHS_STATUS = {0: "optimal", 1: "time_limit", 2: "infeasible",
                 3: "unbounded", 4: "error"}


# ------------------------------------------------------------- evaluation
def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Smallest value whose cumulative weight reaches ``q`` of the total."""
    if values.size == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    total = cum[-1]
    if total <= 0:
        return float(v[0])
    k = int(np.searchsorted(cum, q * total, side="left"))
    return float(v[min(k, v.size - 1)])


def _metrics_local(objective: Objective, ids: np.ndarray) -> Dict[str, float]:
    """Section 2 metrics computed directly from the influence matrix (used
    while ``Objective.metrics`` is a stub on this branch)."""
    inf = objective.influence
    rx = float(objective.rx_cgy)
    dose = inf.dose_of(ids)
    w = inf.target.weights
    total = float(w.sum())
    if total <= 0:
        frac = lambda mask: float("nan")  # noqa: E731
    else:
        frac = lambda mask: float(w[mask].sum() / total)  # noqa: E731
    out = {
        "V100": frac(dose >= rx),
        "V150": frac(dose >= 1.5 * rx),
        "V200": frac(dose >= 2.0 * rx),
        "D90": _weighted_quantile(dose, w, 0.10),
        "Dmean": float((w * dose).sum() / total) if total > 0 else float("nan"),
    }
    for name in inf.oar:
        od = inf.oar_dose_of(name, ids)
        out["oar_dmax_" + name] = float(od.max()) if od.size else 0.0
    return out


def _hard_local(objective: Objective, metrics: Dict[str, float]) -> float:
    """P1: ``V100 - lambda_hot * max(0, V200 - v200_tol) - sum_j lambda_oar *
    max(0, Dmax_j - L_j)`` from a metrics dict."""
    val = metrics["V100"] - objective.lambda_hot * max(0.0, metrics["V200"] - objective.v200_tol)
    for name, limit in objective.influence.oar_limits.items():
        dmax = metrics.get("oar_dmax_" + name)
        if dmax is not None:
            val -= objective.lambda_oar * max(0.0, dmax - float(limit))
    return float(val)


def evaluate_selection(objective: Objective, selection) -> Tuple[float, Dict[str, float]]:
    """``(hard, metrics)`` of ``selection``: ``Objective.hard`` / ``.metrics``
    when they are implemented (branch plan/influence), otherwise the local
    section 2 definitions above."""
    ids = _as_ids(selection, objective.influence.n_candidates)
    try:
        metrics = dict(objective.metrics(ids))
        hard = float(objective.hard(ids))
    except NotImplementedError:
        metrics = _metrics_local(objective, ids)
        hard = _hard_local(objective, metrics)
    return hard, metrics


# ---------------------------------------------------------------- cliques
def validate_cliques(conflicts: ConflictGraph, warn: bool = True
                     ) -> Tuple[List[np.ndarray], List[Tuple[int, str]]]:
    """Keep only the cliques that are true cliques of ``conflicts.pairs``.

    A clique row ``sum_{c in Q} x_c <= 1`` forbids every pair in ``Q``; if
    some pair in ``Q`` does not conflict, the row would cut a pairwise-feasible
    selection.  Returns ``(valid, dropped)`` with ``dropped`` a list of
    ``(index, reason)``; emits one ``UserWarning`` when anything is dropped
    (``warn=True``).  Cliques are deduplicated (sorted ids); size < 2 and
    out-of-range / repeated ids are dropped as well.
    """
    n = conflicts.n
    pairs = conflicts.pairs
    valid: List[np.ndarray] = []
    dropped: List[Tuple[int, str]] = []
    seen = set()
    for k, q in enumerate(conflicts.cliques):
        q = np.asarray(q, dtype=int).reshape(-1)
        if q.size and (q.min() < 0 or q.max() >= n):
            dropped.append((k, "id out of range"))
            continue
        qs = np.unique(q)
        if qs.size != q.size:
            dropped.append((k, "repeated id"))
            continue
        if qs.size < 2:
            dropped.append((k, "fewer than two members"))
            continue
        sub = pairs[qs][:, qs].toarray()
        off = ~np.eye(qs.size, dtype=bool)
        if not np.all(sub[off]):
            missing = int(off.sum() - sub[off].sum()) // 2
            dropped.append((k, "%d non-conflicting pair(s)" % missing))
            continue
        key = tuple(qs.tolist())
        if key in seen:
            dropped.append((k, "duplicate"))
            continue
        seen.add(key)
        valid.append(qs)
    if dropped and warn:
        warnings.warn(
            "ConflictGraph.cliques: dropped %d of %d clique(s) that are not "
            "cliques of `pairs` (%s); their pairs use pairwise rows instead"
            % (len(dropped), len(conflicts.cliques),
               "; ".join("#%d: %s" % d for d in dropped[:5])
               + ("; ..." if len(dropped) > 5 else "")),
            UserWarning, stacklevel=2)
    return valid, dropped


def _uncovered_pairs(conflicts: ConflictGraph, cliques: Sequence[np.ndarray]
                     ) -> np.ndarray:
    """``(P, 2)`` int: conflicting pairs ``i < j`` not inside any clique."""
    n = conflicts.n
    coo = sp.triu(conflicts.pairs, k=1).tocoo()
    if coo.nnz == 0:
        return np.zeros((0, 2), dtype=int)
    if cliques:
        rows = []
        cols = []
        for q in cliques:
            a, b = np.meshgrid(q, q, indexing="ij")
            rows.append(a.ravel())
            cols.append(b.ravel())
        covered = sp.coo_matrix(
            (np.ones(sum(r.size for r in rows), dtype=np.int8),
             (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        hit = np.asarray(covered[coo.row, coo.col]).reshape(-1) > 0
        keep = ~hit
    else:
        keep = np.ones(coo.nnz, dtype=bool)
    return np.column_stack([coo.row[keep], coo.col[keep]]).astype(int)


# ------------------------------------------------------------ formulation
def build_formulation(objective: Objective, n_tiles: int, exact_n: bool = True,
                      prune_frac: float = PRUNE_FRACTION, use_cliques: bool = True,
                      warn: bool = True) -> Dict[str, Any]:
    """Constraint matrices of the MILP (module docstring).

    Returns a dict: ``c`` (objective coefficients to MINIMIZE, i.e.
    ``-w/sum w`` on the ``y`` block and 0 on ``x``), ``A`` (CSR, rows
    stacked coverage / clique / pair / count / OAR), ``lb``, ``ub``,
    ``n_x``, ``n_y``, ``row_kind`` (one label per row) and ``stats``
    (row counts, nnz, ``n_pruned``, cliques used / dropped).  Variable
    order is ``[x_0 .. x_{C-1}, y_0 .. y_{M-1}]``, all bounded to ``[0, 1]``.
    """
    inf = objective.influence
    conflicts = objective.conflicts
    C, M = inf.n_candidates, inf.n_targets
    if conflicts.n != C:
        raise ValueError("conflict graph has %d candidates, influence %d" % (conflicts.n, C))
    rx = float(objective.rx_cgy)
    if rx <= 0:
        raise ValueError("rx_cgy must be positive")
    N = int(n_tiles)
    w = np.asarray(inf.target.weights, dtype=float)
    total_w = float(w.sum())
    stats: Dict[str, Any] = {"n_candidates": C, "n_targets": M}

    # objective: minimize -sum w_m y_m / sum w
    c = np.zeros(C + M, dtype=float)
    if total_w > 0:
        c[C:] = -w / total_w

    blocks = []
    lbs = []
    ubs = []
    kinds: List[str] = []

    # coverage rows: sum_c (D[c,m]/rx) x_c - y_m >= 0
    D = np.asarray(inf.dose, dtype=np.float64)
    keep = D > float(prune_frac) * rx
    n_pruned = int(D.size - keep.sum())
    cc, mm = np.nonzero(keep)
    cov = sp.coo_matrix((D[cc, mm] / rx, (mm, cc)), shape=(M, C)).tocsr()
    a_cov = sp.hstack([cov, -sp.identity(M, format="csr")], format="csr")
    blocks.append(a_cov)
    lbs.append(np.zeros(M))
    ubs.append(np.full(M, np.inf))
    kinds += ["coverage"] * M
    stats["n_pruned"] = n_pruned
    stats["n_rows_coverage"] = M

    # clique rows
    if use_cliques:
        cliques, dropped = validate_cliques(conflicts, warn=warn)
    else:
        cliques, dropped = [], []
    if cliques:
        rows = np.concatenate([np.full(q.size, k) for k, q in enumerate(cliques)])
        cols = np.concatenate(cliques)
        a_clq = sp.coo_matrix((np.ones(cols.size), (rows, cols)),
                              shape=(len(cliques), C)).tocsr()
        blocks.append(sp.hstack([a_clq, sp.csr_matrix((len(cliques), M))], format="csr"))
        lbs.append(np.full(len(cliques), -np.inf))
        ubs.append(np.ones(len(cliques)))
        kinds += ["clique"] * len(cliques)
    stats["n_cliques_used"] = len(cliques)
    stats["n_cliques_dropped"] = len(dropped)
    stats["cliques_dropped"] = dropped

    # pairwise rows for the pairs no clique covers
    pairs = _uncovered_pairs(conflicts, cliques)
    P = int(pairs.shape[0])
    if P:
        rows = np.repeat(np.arange(P), 2)
        cols = pairs.ravel()
        a_pair = sp.coo_matrix((np.ones(2 * P), (rows, cols)), shape=(P, C)).tocsr()
        blocks.append(sp.hstack([a_pair, sp.csr_matrix((P, M))], format="csr"))
        lbs.append(np.full(P, -np.inf))
        ubs.append(np.ones(P))
        kinds += ["pair"] * P
    stats["n_rows_pair"] = P
    stats["n_conflict_pairs"] = conflicts.count_pairs()

    # count row
    a_cnt = sp.hstack([sp.csr_matrix(np.ones((1, C))), sp.csr_matrix((1, M))], format="csr")
    blocks.append(a_cnt)
    lbs.append(np.array([float(N) if exact_n else 0.0]))
    ubs.append(np.array([float(N)]))
    kinds.append("count")

    # OAR rows: sum_c (D_j[c,o] / L_j) x_c <= 1 (no pruning: must stay exact)
    n_oar_rows = 0
    for name, mat in inf.oar.items():
        if name not in inf.oar_limits:
            continue
        limit = float(inf.oar_limits[name])
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != C:
            raise ValueError("OAR %r matrix must be (C, Mo)" % name)
        mo = mat.shape[1]
        if mo == 0:
            continue
        scale = limit if limit > 0 else 1.0
        a_oar = sp.csr_matrix(mat.T / scale)
        blocks.append(sp.hstack([a_oar, sp.csr_matrix((mo, M))], format="csr"))
        lbs.append(np.full(mo, -np.inf))
        ubs.append(np.full(mo, limit / scale))
        kinds += ["oar:" + str(name)] * mo
        n_oar_rows += mo
    stats["n_rows_oar"] = n_oar_rows

    A = sp.vstack(blocks, format="csr")
    lb = np.concatenate(lbs)
    ub = np.concatenate(ubs)
    stats["n_rows"] = int(A.shape[0])
    stats["nnz"] = int(A.nnz)
    stats["prune_frac"] = float(prune_frac)
    return {"c": c, "A": A, "lb": lb, "ub": ub, "n_x": C, "n_y": M,
            "row_kind": kinds, "stats": stats, "total_weight": total_w}


def _run_highs(form: Dict[str, Any], integral: bool, time_limit_s: float,
               mip_rel_gap: float):
    n = form["n_x"] + form["n_y"]
    options = {"disp": False, "time_limit": float(time_limit_s)}
    if integral:
        options["mip_rel_gap"] = float(mip_rel_gap)
    return _scipy_milp(
        c=form["c"],
        constraints=LinearConstraint(form["A"], form["lb"], form["ub"]),
        integrality=np.full(n, 1 if integral else 0, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options=options,
    )


def _feasible(objective: Objective, ids: np.ndarray, n_tiles: int, exact_n: bool) -> bool:
    ok_count = (ids.size == n_tiles) if exact_n else (ids.size <= n_tiles)
    return bool(ok_count and objective.conflicts.is_feasible(ids))


# ------------------------------------------------------------------- MILP
def solve_milp(objective: Objective, n_tiles: int,
               time_limit_s: float = MILP_TIME_LIMIT_S, exact_n: bool = True,
               mip_rel_gap: float = 1e-4, *, prune_frac: float = PRUNE_FRACTION,
               use_cliques: bool = True) -> SolverResult:
    """E4 exact reference: maximize weighted coverage (V100) with HiGHS.

    See the module docstring for the formulation.  Hot-spot terms are NOT in
    the MILP; ``extra["milp_objective"]`` is the solver's coverage value
    (V100 of the incumbent), ``objective`` the P1 hard value recomputed from
    ``selection``.  ``bound`` (upper bound on V100) and ``mip_gap`` are
    always filled; on ``status="time_limit"`` the bound, not the incumbent,
    is the reference (section 3 E4).  Infeasible instances (no ``n_tiles``
    subset satisfies the conflict / OAR rows) return ``status="infeasible"``
    with an empty selection and a reason; nothing is raised for them.

    Keyword-only extras (not in the frozen signature): ``prune_frac``
    (coverage-row pruning threshold as a fraction of rx; 0 disables) and
    ``use_cliques`` (False -> pairwise rows only; same optimum, weaker LP).
    """
    t0 = time.perf_counter()
    N = int(n_tiles)
    if N < 0:
        raise ValueError("n_tiles must be >= 0")
    C = objective.influence.n_candidates
    base_extra: Dict[str, Any] = {
        "exact_n": bool(exact_n), "time_limit_s": float(time_limit_s),
        "mip_rel_gap": float(mip_rel_gap), "prune_frac": float(prune_frac),
        "use_cliques": bool(use_cliques), "n_tiles": N,
    }

    def finish(ids, status, reason, bound, gap, extra, feasible=None):
        ids = np.asarray(ids, dtype=int).reshape(-1)
        if feasible is None:
            feasible = _feasible(objective, ids, N, exact_n)
        if ids.size or status in ("optimal", "time_limit"):
            hard, metrics = evaluate_selection(objective, ids)
        else:
            hard, metrics = float("nan"), {}
        extra = dict(base_extra, **extra)
        extra.setdefault("v100", metrics.get("V100", float("nan")))
        return SolverResult(
            selection=np.sort(ids), objective=hard, metrics=metrics,
            history=[(0, hard)], runtime_s=time.perf_counter() - t0,
            solver="milp", seed=None, status=status, reason=reason,
            bound=bound, mip_gap=gap, feasible=bool(feasible), extra=extra)

    if N > C:
        return finish([], "infeasible",
                      "n_tiles=%d exceeds the %d candidates" % (N, C),
                      float("nan"), float("nan"),
                      {"milp_objective": float("nan")}, feasible=False)
    if N == 0:
        return finish([], "optimal", "", 0.0, 0.0, {"milp_objective": 0.0},
                      feasible=True)

    try:
        form = build_formulation(objective, N, exact_n=exact_n,
                                 prune_frac=prune_frac, use_cliques=use_cliques)
        res = _run_highs(form, True, time_limit_s, mip_rel_gap)
    except Exception as exc:  # solver / formulation failure: report, do not raise
        return finish([], "error", "MILP failed: %s: %s" % (type(exc).__name__, exc),
                      float("nan"), float("nan"),
                      {"milp_objective": float("nan")}, feasible=False)

    hstat = int(getattr(res, "status", 4))
    status = _HIGHS_STATUS.get(hstat, "error")
    message = str(getattr(res, "message", ""))
    extra: Dict[str, Any] = {
        "formulation": form["stats"], "highs_status": hstat, "highs_message": message,
        "node_count": getattr(res, "mip_node_count", None),
    }
    x = getattr(res, "x", None)
    dual = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    bound = float(-dual) if dual is not None and np.isfinite(dual) else float("nan")
    gap = float(gap) if gap is not None else float("nan")

    if status == "infeasible":
        return finish([], "infeasible",
                      "HiGHS: no %s%d-candidate selection satisfies the conflict / OAR "
                      "rows (%s)" % ("" if exact_n else "<= ", N, message),
                      float("nan"), float("nan"), dict(extra, milp_objective=float("nan")),
                      feasible=False)
    if status == "unbounded":
        status = "error"
    if x is None:
        reason = ("time limit %.3g s hit before any incumbent" % time_limit_s
                  if status == "time_limit" else "HiGHS returned no solution (%s)" % message)
        return finish([], status, reason, bound, gap,
                      dict(extra, milp_objective=float("nan")), feasible=False)

    ids = np.flatnonzero(np.asarray(x[:C]) > 0.5)
    milp_obj = float(-res.fun) if res.fun is not None else float("nan")
    extra["milp_objective"] = milp_obj
    feasible = _feasible(objective, ids, N, exact_n)
    if not feasible:
        return finish(ids, "error",
                      "solver selection violates the conflict graph or the count "
                      "(size %d, requested %d)" % (ids.size, N), bound, gap, extra,
                      feasible=False)
    reason = ""
    if status == "time_limit":
        reason = ("time limit %.3g s hit: bound %.4f is the reference, incumbent "
                  "V100 %.4f (gap %.3g)" % (time_limit_s, bound, milp_obj, gap))
    elif status == "error":
        reason = "HiGHS status %d: %s" % (hstat, message)
    return finish(ids, status, reason, bound, gap, extra, feasible=True)


# --------------------------------------------------------------- LP bound
def lp_bound(objective: Objective, n_tiles: int, exact_n: bool = True,
             time_limit_s: float = MILP_TIME_LIMIT_S, *,
             prune_frac: float = PRUNE_FRACTION, use_cliques: bool = True) -> float:
    """Value of the LP relaxation (all integrality dropped): an upper bound
    on the MILP's V100, for the validation agent when the MILP times out.
    Weak in general (fractional ``x`` spread dose over many points and
    ``y_m`` takes fractional credit); the MILP's own dual bound at the time
    limit is tighter.  Returns NaN when the LP is infeasible (then the MILP
    is too) or fails.
    """
    N = int(n_tiles)
    C = objective.influence.n_candidates
    if N < 0 or N > C:
        return float("nan")
    if N == 0:
        return 0.0
    try:
        form = build_formulation(objective, N, exact_n=exact_n, prune_frac=prune_frac,
                                 use_cliques=use_cliques, warn=False)
        res = _run_highs(form, False, time_limit_s, 0.0)
    except Exception:
        return float("nan")
    if not getattr(res, "success", False) or res.fun is None:
        return float("nan")
    return float(-res.fun)


# ------------------------------------------------------------ brute force
def brute_force(objective: Objective, n_tiles: int, exact_n: bool = True,
                max_subsets: int = BRUTE_FORCE_MAX_SUBSETS) -> SolverResult:
    """Exhaustive enumeration for tiny instances (C <= 15, N <= 4 intended).

    Maximizes V100 over all selections of size ``n_tiles`` (``exact_n``) or
    ``<= n_tiles`` that violate no conflict and no OAR limit -- the same
    feasible set and objective as the MILP, so the two must agree on V100
    (ties are broken toward the lexicographically smallest id tuple, so the
    selections themselves may differ).  ``objective`` is the P1 hard value
    of the chosen selection, ``extra["milp_objective"]`` its V100 (same key
    as :func:`solve_milp` for direct comparison), ``bound`` the optimum and
    ``mip_gap`` 0.  Raises ``ValueError`` when the enumeration would exceed
    ``max_subsets`` subsets.
    """
    t0 = time.perf_counter()
    inf = objective.influence
    C = inf.n_candidates
    N = int(n_tiles)
    if N < 0:
        raise ValueError("n_tiles must be >= 0")
    sizes = [N] if exact_n else list(range(0, N + 1))
    n_total = sum(math.comb(C, k) for k in sizes if k <= C)
    if n_total > max_subsets:
        raise ValueError("brute_force: %d subsets exceed max_subsets=%d"
                         % (n_total, max_subsets))
    rx = float(objective.rx_cgy)
    D = np.asarray(inf.dose, dtype=np.float64)
    w = np.asarray(inf.target.weights, dtype=float)
    total_w = float(w.sum())
    P = objective.conflicts.pairs.toarray()
    oars = [(np.asarray(inf.oar[name], dtype=np.float64), float(inf.oar_limits[name]))
            for name in inf.oar if name in inf.oar_limits]

    best_val = -np.inf
    best_ids: Optional[Tuple[int, ...]] = None
    n_enum = 0
    n_feas = 0
    for k in sizes:
        if k > C:
            continue
        for ids in itertools.combinations(range(C), k):
            n_enum += 1
            sel = np.asarray(ids, dtype=int)
            if k >= 2 and P[np.ix_(sel, sel)].any():
                continue
            if any(mat[sel].sum(axis=0).max(initial=0.0) > limit for mat, limit in oars):
                continue
            n_feas += 1
            dose = D[sel].sum(axis=0) if k else np.zeros(D.shape[1])
            v100 = float(w[dose >= rx].sum() / total_w) if total_w > 0 else 0.0
            if v100 > best_val + 1e-12:
                best_val = v100
                best_ids = ids
    extra = {"n_enumerated": n_enum, "n_feasible": n_feas, "exact_n": bool(exact_n),
             "n_tiles": N}
    if best_ids is None:
        return SolverResult(
            selection=np.zeros(0, dtype=int), objective=float("nan"), metrics={},
            history=[], runtime_s=time.perf_counter() - t0, solver="brute_force",
            status="infeasible",
            reason="no %s%d-candidate selection satisfies the conflict / OAR rows "
                   "(%d subsets enumerated)" % ("" if exact_n else "<= ", N, n_enum),
            bound=float("nan"), mip_gap=float("nan"), feasible=False,
            extra=dict(extra, milp_objective=float("nan")))
    ids = np.asarray(best_ids, dtype=int)
    hard, metrics = evaluate_selection(objective, ids)
    return SolverResult(
        selection=ids, objective=hard, metrics=metrics, history=[(0, hard)],
        runtime_s=time.perf_counter() - t0, solver="brute_force", status="optimal",
        reason="", bound=best_val, mip_gap=0.0, feasible=True,
        extra=dict(extra, milp_objective=best_val, v100=best_val))


__all__ = ["PRUNE_FRACTION", "BRUTE_FORCE_MAX_SUBSETS", "solve_milp", "lp_bound",
           "brute_force", "build_formulation", "validate_cliques", "evaluate_selection"]
