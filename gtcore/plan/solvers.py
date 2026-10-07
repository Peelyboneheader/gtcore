"""Heuristic solvers (section 3 E1-E3, E5). Owner: A3 Heuristics, branch ``plan/solvers``.

Implements:

- :func:`solve_greedy`: E1 greedy forward selection, deterministic (ties ->
  lowest id), marginal-gain curve in ``extra``; also the engine behind
  ``suggest_next`` (``n_tiles = len(fixed) + 1``).
- :func:`solve_local`: E2 first-improvement local search (local re-site
  within ``radius_mm``, re-spin in place, global re-site), stops at a local
  optimum of the hard objective or at ``max_evals``.
- :func:`solve_sa`: E3 Metropolis on the soft objective, move mix 60/20/20,
  T0 from trial moves, geometric cooling ``SA_ALPHA`` per sweep of
  ``SA_MOVES_PER_TILE_PER_SWEEP * N`` moves, best HARD objective kept,
  reproducible from ``seed``, best-so-far per sweep in ``history``.
- :func:`refine_continuous`: E5 Nelder-Mead polish over (u, v, theta)
  through ``conform_tile`` (assignment to A3 is a Phase 0 decision; see
  docs/optimize-notes.md).

Evaluation plumbing
-------------------
``Objective.hard`` / ``soft`` / ``metrics`` are implemented on branch
``plan/influence`` (A2).  Until they land, :class:`_HardEval` computes the
same quantities directly from ``objective.influence`` (dose matrix, target
weights, OAR sets) and the objective's lambda fields.  :class:`_Evaluator`
routes every call: when the objective provides a real ``hard`` (and the
vectorized ``gains_all(selection, dose_vec=None)`` / ``soft_gains_all``),
those are used; otherwise the helper is.  Within one solver run all
*comparisons* come from a single source (so a local search cannot cycle on
float32-vs-float64 differences); only the reported ``objective`` /
``metrics`` of the result prefer the objective's own methods when real.

Greedy must never violate a conflict; SA must be reproducible from its seed.
Developed against ``tests/plan_fixtures.toy_instance``.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import (
    DEFAULT_RX_CGY,
    LAMBDA_HOT,
    LOCAL_RADIUS_MM,
    SA_ALPHA,
    SA_MOVES_PER_TILE_PER_SWEEP,
    SA_N_RESTARTS,
    SA_N_SWEEPS,
    TAU_FRACTION,
    V200_TOL,
    CandidateSet,
    Objective,
    SolverResult,
    TargetSet,
    _as_ids,
)
from ..interact import PlacedTile

_EPS = 1e-12
_CHUNK_ELEMS = 8_000_000      # rows per chunk of (C, M) temporaries ~ 64 MB float64


# ------------------------------------------------------------------ metrics
def _weighted_quantile(values, weights, q: float) -> float:
    """Weighted lower quantile: smallest ``v`` with ``cumweight(v) >= q * total``."""
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v = v[order]
    cum = np.cumsum(w[order])
    total = cum[-1]
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(cum, q * total, side="left"))
    return float(v[min(idx, v.size - 1)])


class _HardEval:
    """P1 / soft evaluation straight from an :class:`Objective`'s influence.

    All selection-dependent quantities are expressed through a dose vector
    ``(M,)`` and (optionally) per-OAR dose vectors so solvers can update them
    incrementally (``dose - D[i] + D[j]``) instead of re-summing rows.
    """

    def __init__(self, objective: Objective):
        inf = objective.influence
        self.D = inf.dose                                  # (C, M) float32
        self.C, self.M = self.D.shape
        w = np.asarray(inf.target.weights, dtype=float).reshape(-1)
        wsum = float(w.sum())
        self.wn = w / wsum if wsum > 0 else np.full(w.shape, 1.0 / max(w.size, 1))
        self.w = w
        self.rx = float(objective.rx_cgy)
        self.lam_hot = float(objective.lambda_hot)
        self.v200_tol = float(objective.v200_tol)
        self.lam_oar = float(objective.lambda_oar)
        self.tau = float(objective.tau_cgy) if objective.tau_cgy else TAU_FRACTION * self.rx
        self.oar = {name: np.asarray(mat) for name, mat in inf.oar.items()}
        self.oar_limits = {name: float(inf.oar_limits[name])
                           for name in self.oar if name in inf.oar_limits}
        self.chunk = max(1, _CHUNK_ELEMS // max(self.M, 1))

    # ---- dose vectors
    def dose_of(self, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=int).reshape(-1)
        if ids.size == 0:
            return np.zeros(self.M, dtype=np.float64)
        return self.D[ids].astype(np.float64).sum(axis=0)

    def oar_doses_of(self, ids) -> Dict[str, np.ndarray]:
        ids = np.asarray(ids, dtype=int).reshape(-1)
        out = {}
        for name, mat in self.oar.items():
            if ids.size == 0:
                out[name] = np.zeros(mat.shape[1], dtype=np.float64)
            else:
                out[name] = np.asarray(mat[ids], dtype=np.float64).sum(axis=0)
        return out

    # ---- penalties
    def oar_penalty(self, oar_doses: Dict[str, np.ndarray]) -> float:
        pen = 0.0
        for name, lim in self.oar_limits.items():
            vec = oar_doses.get(name)
            if vec is not None and vec.size:
                pen += self.lam_oar * max(0.0, float(vec.max()) - lim)
        return pen

    def oar_penalty_all(self, oar_doses: Dict[str, np.ndarray], ids=None) -> np.ndarray:
        """Penalty of ``base + c`` for every candidate ``c`` (or the rows ``ids``)."""
        n = self.C if ids is None else int(len(ids))
        pen = np.zeros(n, dtype=float)
        for name, lim in self.oar_limits.items():
            mat = self.oar[name]
            if mat.shape[1] == 0:
                continue
            rows = mat if ids is None else mat[ids]
            base = oar_doses.get(name)
            if base is None:
                base = np.zeros(mat.shape[1])
            dmax = (rows.astype(np.float64) + base[None, :]).max(axis=1)
            pen += self.lam_oar * np.maximum(0.0, dmax - lim)
        return pen

    def hot_penalty(self, v200) :
        return self.lam_hot * np.maximum(0.0, v200 - self.v200_tol)

    # ---- scalar objectives from a dose vector
    def coverage(self, dose: np.ndarray, level: float) -> float:
        return float(self.wn @ (dose >= level * self.rx))

    def hard_from_dose(self, dose: np.ndarray, oar_pen: float = 0.0) -> float:
        v100 = self.coverage(dose, 1.0)
        v200 = self.coverage(dose, 2.0)
        return v100 - float(self.hot_penalty(v200)) - oar_pen

    def soft_from_dose(self, dose: np.ndarray, oar_pen: float = 0.0) -> float:
        z = (dose - self.rx) / self.tau
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        v200 = self.coverage(dose, 2.0)
        return float(self.wn @ sig) - float(self.hot_penalty(v200)) - oar_pen

    def metrics_from_dose(self, dose: np.ndarray, oar_doses: Dict[str, np.ndarray]
                          ) -> Dict[str, float]:
        m = {
            "V100": self.coverage(dose, 1.0),
            "V150": self.coverage(dose, 1.5),
            "V200": self.coverage(dose, 2.0),
            "D90": _weighted_quantile(dose, self.w, 0.10),
            "Dmean": float(self.wn @ dose),
        }
        for name, vec in oar_doses.items():
            m["oar_dmax_" + name] = float(vec.max()) if vec.size else float("nan")
        return m

    # ---- vectorized "add each candidate" evaluation
    def _rows(self, ids):
        return range(self.C) if ids is None else np.asarray(ids, dtype=int).reshape(-1)

    def hard_all(self, base_dose: np.ndarray, oar_doses: Dict[str, np.ndarray],
                 ids=None) -> np.ndarray:
        """Hard objective of ``base + c`` for every candidate (or rows ``ids``)."""
        rows = None if ids is None else np.asarray(ids, dtype=int).reshape(-1)
        n = self.C if rows is None else rows.size
        out = np.empty(n, dtype=float)
        for i0 in range(0, n, self.chunk):
            sl = slice(i0, min(n, i0 + self.chunk))
            block = self.D[sl] if rows is None else self.D[rows[sl]]
            tot = block + base_dose[None, :].astype(np.float32)
            v100 = (tot >= self.rx) @ self.wn
            v200 = (tot >= 2.0 * self.rx) @ self.wn
            out[sl] = v100 - self.hot_penalty(v200)
        if self.oar_limits:
            out -= self.oar_penalty_all(oar_doses, rows)
        return out

    def soft_all(self, base_dose: np.ndarray, oar_doses: Dict[str, np.ndarray],
                 ids=None) -> np.ndarray:
        rows = None if ids is None else np.asarray(ids, dtype=int).reshape(-1)
        n = self.C if rows is None else rows.size
        out = np.empty(n, dtype=float)
        for i0 in range(0, n, self.chunk):
            sl = slice(i0, min(n, i0 + self.chunk))
            block = self.D[sl] if rows is None else self.D[rows[sl]]
            tot = block.astype(np.float64) + base_dose[None, :]
            z = np.clip((tot - self.rx) / self.tau, -60.0, 60.0)
            sig = 1.0 / (1.0 + np.exp(-z))
            v200 = (tot >= 2.0 * self.rx) @ self.wn
            out[sl] = sig @ self.wn - self.hot_penalty(v200)
        if self.oar_limits:
            out -= self.oar_penalty_all(oar_doses, rows)
        return out


def _is_real(objective: Objective, name: str) -> bool:
    """True when ``objective.<name>`` is implemented (not the Phase 0 stub)."""
    fn = getattr(objective, name, None)
    if fn is None:
        return False
    try:
        fn(np.zeros(0, dtype=int))
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


class _Evaluator:
    """Routes evaluations to the objective's own methods when they are real,
    else to :class:`_HardEval`.  See the module docstring for the rule."""

    def __init__(self, objective: Objective, candidates: Optional[CandidateSet] = None):
        self.obj = objective
        self.conflicts = objective.conflicts
        self.h = _HardEval(objective)
        self.n = self.h.C
        self.candidates = candidates if candidates is not None else getattr(
            objective, "candidates", None)
        self.kinds = (np.asarray(self.candidates.kinds, dtype=object)
                      if self.candidates is not None else None)
        self._real_hard = _is_real(objective, "hard")
        self._real_metrics = _is_real(objective, "metrics")
        self._gains_all = getattr(objective, "gains_all", None)
        self._soft_gains_all = getattr(objective, "soft_gains_all", None)
        self._use_obj_internal = bool(self._real_hard and self._gains_all is not None)
        self.n_evals = 0

    # ---- state
    def state(self, ids) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.h.dose_of(ids), self.h.oar_doses_of(ids)

    # ---- scalar
    def hard_internal(self, ids, dose=None, oar_doses=None) -> float:
        self.n_evals += 1
        if self._use_obj_internal:
            return float(self.obj.hard(np.asarray(ids, dtype=int)))
        if dose is None:
            dose, oar_doses = self.state(ids)
        if oar_doses is None:
            oar_doses = self.h.oar_doses_of(ids)
        return self.h.hard_from_dose(dose, self.h.oar_penalty(oar_doses))

    def soft_internal(self, dose, oar_doses) -> float:
        self.n_evals += 1
        return self.h.soft_from_dose(dose, self.h.oar_penalty(oar_doses))

    def hard_report(self, ids) -> float:
        ids = np.asarray(ids, dtype=int)
        if self._real_hard:
            return float(self.obj.hard(ids))
        return self.hard_internal(ids)

    def metrics_report(self, ids) -> Dict[str, float]:
        ids = np.asarray(ids, dtype=int)
        if self._real_metrics:
            return dict(self.obj.metrics(ids))
        dose, oar = self.state(ids)
        return self.h.metrics_from_dose(dose, oar)

    # ---- vectorized
    def hard_all(self, base_ids, base_dose, oar_doses, ids=None) -> np.ndarray:
        """Hard objective of ``base + c`` for candidates ``ids`` (None = all)."""
        if self._use_obj_internal:
            base_ids = np.asarray(base_ids, dtype=int)
            base = float(self.obj.hard(base_ids))
            g = np.asarray(self._gains_all(base_ids, dose_vec=base_dose), dtype=float)
            vals = base + g
            self.n_evals += self.n
            return vals if ids is None else vals[np.asarray(ids, dtype=int)]
        self.n_evals += self.n if ids is None else int(len(ids))
        return self.h.hard_all(base_dose, oar_doses, ids)

    def soft_all(self, base_ids, base_dose, oar_doses, ids=None) -> np.ndarray:
        if self._use_obj_internal and self._soft_gains_all is not None:
            base_ids = np.asarray(base_ids, dtype=int)
            base = float(self.obj.soft(base_ids))
            g = np.asarray(self._soft_gains_all(base_ids, dose_vec=base_dose), dtype=float)
            vals = base + g
            self.n_evals += self.n
            return vals if ids is None else vals[np.asarray(ids, dtype=int)]
        self.n_evals += self.n if ids is None else int(len(ids))
        return self.h.soft_all(base_dose, oar_doses, ids)


# ------------------------------------------------------------------ helpers
def _kinds_of(ev: _Evaluator, ids) -> Optional[np.ndarray]:
    if ev.kinds is None:
        return None
    return ev.kinds[np.asarray(ids, dtype=int)]


def _result(ev: _Evaluator, sel, solver: str, t0: float, n_tiles: int, **kw) -> SolverResult:
    sel = np.asarray(sorted(int(i) for i in sel), dtype=int)
    feasible = bool(ev.conflicts.is_feasible(sel)) and sel.size == int(n_tiles)
    kw.setdefault("status", "ok" if feasible else "infeasible")
    kw.setdefault("feasible", feasible)
    extra = kw.pop("extra", {})
    extra.setdefault("n_evals", int(ev.n_evals))
    return SolverResult(
        selection=sel,
        objective=ev.hard_report(sel) if sel.size else float("nan"),
        metrics=ev.metrics_report(sel) if sel.size else {},
        runtime_s=time.perf_counter() - t0,
        solver=solver,
        extra=extra,
        **kw,
    )


def _local_neighbourhoods(candidates: Optional[CandidateSet], radius_mm: float, n: int):
    """``(local, spin)``: per-candidate id arrays of candidates whose anchor
    lies within ``radius_mm`` and of candidates sharing the ``anchor_id``.
    Both None when ``candidates`` is None (moves degrade to global)."""
    if candidates is None:
        return None, None
    if len(candidates) != n:
        raise ValueError("candidates (%d) do not match the influence rows (%d)"
                         % (len(candidates), n))
    from scipy.spatial import cKDTree
    anchors = np.asarray(candidates.anchors, dtype=float)
    tree = cKDTree(anchors)
    local = [np.asarray(sorted(q), dtype=int)
             for q in tree.query_ball_point(anchors, float(radius_mm))]
    aid = np.asarray(candidates.anchor_ids, dtype=int)
    groups: Dict[int, np.ndarray] = {}
    for a in np.unique(aid):
        groups[int(a)] = np.flatnonzero(aid == a)
    spin = [groups[int(a)] for a in aid]
    return local, spin


# ------------------------------------------------------------------- greedy
GREEDY_MAX_BOUND_CHECKS = 200
# Feasibility-aware greedy walks the ranked candidates and checks the packing
# bound on at most this many before falling back to the top-ranked one.


def _packing_bound(mask: np.ndarray, nbr: List[np.ndarray], deg: np.ndarray,
                   need: int) -> int:
    """Lower bound on how many mutually compatible candidates ``mask`` holds:
    a greedy maximal independent set (min static degree first), truncated at
    ``need``.  ``>= need`` means ``need`` more tiles can certainly be packed."""
    if need <= 0:
        return 0
    avail = mask.copy()
    count = 0
    while count < need:
        ids = np.flatnonzero(avail)
        if ids.size == 0:
            break
        pick = int(ids[int(np.argmin(deg[ids]))])
        avail[nbr[pick]] = False
        avail[pick] = False
        count += 1
    return count


def _conflict_lists(conflicts) -> List[np.ndarray]:
    pairs = conflicts.pairs
    return [np.sort(pairs.indices[pairs.indptr[i]:pairs.indptr[i + 1]])
            for i in range(conflicts.n)]


def solve_greedy(objective: Objective, n_tiles: int, fixed: Sequence[int] = (),
                 kinds_required: Optional[Dict[str, int]] = None,
                 candidates: Optional[CandidateSet] = None,
                 feasibility_aware: bool = True) -> SolverResult:
    """E1 greedy forward selection (section 3 E1), feasibility-aware.

    Starting from ``fixed`` (candidate ids already placed, e.g. tiles the
    surgeon put down; never removed), repeatedly add the compatible candidate
    (``conflicts.compatible_mask``) that ranks first on the lexicographic key
    ``(hard gain, soft gain, room, lowest id)`` where ``room`` is the number
    of candidates still compatible after the addition (step 1 usually ties
    on hard gain because no single tile reaches rx on the shell; the soft
    gain then decides).  Stops at ``n_tiles`` or, when no compatible
    candidate remains, with ``status="infeasible"`` and a reason naming how
    many tiles were placed (never silently fewer tiles as ``"ok"``).

    ``feasibility_aware`` (default): a candidate is skipped when, after
    adding it, a greedy maximal independent set over the remaining
    compatible candidates (:func:`_packing_bound`) holds fewer than the
    tiles still needed -- plain greedy otherwise clusters tiles where the
    immediate gain is largest and runs out of room.  The bound is a lower
    bound, so skipping is conservative; when every checked candidate (at
    most ``GREEDY_MAX_BOUND_CHECKS``) fails, the top-ranked one is taken
    anyway (``extra["bound_fallbacks"]`` counts this).

    ``kinds_required`` (``{"full": n, "half": k}``) is the inventory mix of
    the WHOLE selection (fixed tiles included); its counts must sum to
    ``n_tiles``.  It needs candidate kinds: pass ``candidates`` or attach a
    ``CandidateSet`` as ``objective.candidates``.

    ``history`` holds ``(n_selected, best_hard_so_far)``; ``extra["gains"]``
    (alias ``"marginal_gain"``) the marginal hard gain of each added tile,
    ``extra["order"]`` the ids in the order they were added and
    ``extra["skipped"]`` the number of candidates skipped by the bound.
    """
    t0 = time.perf_counter()
    ev = _Evaluator(objective, candidates)
    n_tiles = int(n_tiles)
    if n_tiles < 0:
        raise ValueError("n_tiles must be >= 0")
    sel: List[int] = [int(i) for i in _as_ids(fixed, ev.n)]
    if len(sel) > n_tiles:
        raise ValueError("fixed has %d ids but n_tiles = %d" % (len(sel), n_tiles))
    if not ev.conflicts.is_feasible(sel):
        return _result(ev, sel, "greedy", t0, n_tiles, status="infeasible",
                       feasible=False,
                       reason="fixed selection violates a conflict",
                       extra={"gains": [], "marginal_gain": [], "order": list(sel)})

    needed: Optional[Dict[str, int]] = None
    if kinds_required:
        if ev.kinds is None:
            raise ValueError("kinds_required needs candidate kinds: pass candidates= "
                             "or attach a CandidateSet as objective.candidates")
        needed = {str(k): int(v) for k, v in kinds_required.items()}
        if sum(needed.values()) != n_tiles:
            raise ValueError("kinds_required counts sum to %d, n_tiles = %d"
                             % (sum(needed.values()), n_tiles))
        for k in _kinds_of(ev, sel):
            needed[str(k)] = needed.get(str(k), 0) - 1
        if any(v < 0 for v in needed.values()):
            raise ValueError("fixed selection exceeds kinds_required: %r" % (needed,))

    pairs = ev.conflicts.pairs
    nbr = _conflict_lists(ev.conflicts) if feasibility_aware else None
    dose, oar = ev.state(sel)
    cur = ev.hard_internal(sel, dose, oar)
    history: List[Tuple[int, float]] = [(len(sel), cur)]
    gains: List[float] = []
    order: List[int] = list(sel)
    skipped = 0
    fallbacks = 0
    status, reason = "ok", ""

    while len(sel) < n_tiles:
        mask_all = ev.conflicts.compatible_mask(sel)       # every kind
        mask = mask_all.copy()
        if needed is not None:
            allowed = [k for k, v in needed.items() if v > 0]
            mask &= np.isin(ev.kinds.astype(str), allowed)
        if not mask.any():
            status = "infeasible"
            reason = ("no compatible candidate%s left after placing %d of %d tiles"
                      % (" of the required kinds" if needed is not None else "",
                         len(sel), n_tiles))
            break
        rows = np.flatnonzero(mask)
        hard_v = ev.hard_all(sel, dose, oar, rows)
        soft_v = ev.soft_all(sel, dose, oar, rows)
        # room: candidates (any kind) still compatible once this one is added
        hit = np.asarray(pairs[rows].dot(mask_all.astype(np.int64))).reshape(-1)
        room = int(mask_all.sum()) - hit - 1
        # lexicographic rank: hard desc, soft desc, room desc, id asc (stable)
        rank = np.lexsort((rows, -room, -soft_v, -hard_v))
        choice = int(rank[0])
        remaining = n_tiles - len(sel) - 1
        if feasibility_aware and remaining > 0:
            deg = np.asarray(pairs.dot(mask_all.astype(np.int64))).reshape(-1)
            found = False
            for k in rank[:GREEDY_MAX_BOUND_CHECKS]:
                c = int(rows[k])
                m_c = mask_all.copy()
                m_c[nbr[c]] = False
                m_c[c] = False
                if _packing_bound(m_c, nbr, deg, remaining) >= remaining:
                    choice = int(k)
                    found = True
                    break
                skipped += 1
            if not found:
                fallbacks += 1
                choice = int(rank[0])
        j = int(rows[choice])
        best = float(hard_v[choice])
        gains.append(best - cur)
        cur = best
        sel.append(j)
        order.append(j)
        dose = dose + ev.h.D[j].astype(np.float64)
        for name, mat in ev.h.oar.items():
            oar[name] = oar[name] + np.asarray(mat[j], dtype=np.float64)
        if needed is not None:
            needed[str(ev.kinds[j])] -= 1
        history.append((len(sel), cur))

    return _result(ev, sel, "greedy", t0, n_tiles, status=status, reason=reason,
                   feasible=(status == "ok"), history=history,
                   extra={"gains": gains, "marginal_gain": gains, "order": order,
                          "fixed": [int(i) for i in _as_ids(fixed, ev.n)],
                          "feasibility_aware": bool(feasibility_aware),
                          "skipped": skipped, "bound_fallbacks": fallbacks})


def solve_greedy_local(objective: Objective, n_tiles: int,
                       candidates: Optional[CandidateSet] = None,
                       fixed: Sequence[int] = (), kinds_required=None,
                       radius_mm: float = LOCAL_RADIUS_MM, max_evals: int = 20000,
                       feasibility_aware: bool = True) -> SolverResult:
    """Greedy (E1) followed by first-improvement local search (E2): the
    default heuristic.  Returns the local-search result with the greedy
    result's diagnostics under ``extra["greedy"]``; when greedy is infeasible
    its result is returned unchanged."""
    g = solve_greedy(objective, n_tiles, fixed=fixed, kinds_required=kinds_required,
                     candidates=candidates, feasibility_aware=feasibility_aware)
    if not g.feasible:
        return g
    r = solve_local(objective, n_tiles, g.selection, radius_mm=radius_mm,
                    candidates=candidates, max_evals=max_evals)
    r.solver = "greedy+local"
    r.runtime_s += g.runtime_s
    r.extra["greedy"] = {"objective": g.objective, "selection": g.selection.tolist(),
                         "gains": g.extra.get("gains", []), "runtime_s": g.runtime_s}
    r.history = [(0, g.objective)] + list(r.history)
    return r


# ------------------------------------------------------------- local search
def solve_local(objective: Objective, n_tiles: int, start,
                radius_mm: float = LOCAL_RADIUS_MM,
                candidates: Optional[CandidateSet] = None,
                max_evals: int = 20000) -> SolverResult:
    """E2 first-improvement local search on the hard objective (section 3 E2).

    Tiles are visited in slot order; for each, the move blocks are tried in
    order (i) re-site to a compatible candidate whose anchor is within
    ``radius_mm`` (needs ``candidates.anchors``), (ii) re-spin in place
    (same ``anchor_ids``, needs ``candidates``), (iii) re-site anywhere.
    Each block is evaluated vectorized and its best improving candidate
    (ties -> lowest id) is taken at once; the loop repeats until a full pass
    makes no move (status ``"ok"``) or ``max_evals`` candidate evaluations
    were spent (status ``"max_evals"``).  Without ``candidates`` only block
    (iii) exists.  Moves preserve each tile's kind when kinds are known.

    ``start`` may have fewer than ``n_tiles`` ids: it is topped up greedily.
    ``history`` holds ``(pass, hard)`` after every accepted move.
    """
    t0 = time.perf_counter()
    ev = _Evaluator(objective, candidates)
    n_tiles = int(n_tiles)
    sel = [int(i) for i in _as_ids(start, ev.n)]
    if len(sel) > n_tiles:
        raise ValueError("start has %d ids but n_tiles = %d" % (len(sel), n_tiles))
    if not ev.conflicts.is_feasible(sel):
        return _result(ev, sel, "local", t0, n_tiles, status="infeasible",
                       feasible=False, reason="start selection violates a conflict")
    if len(sel) < n_tiles:
        g = solve_greedy(objective, n_tiles, fixed=sel, candidates=ev.candidates)
        if not g.feasible:
            g.solver = "local"
            g.reason = "greedy top-up of start failed: " + g.reason
            return g
        sel = [int(i) for i in g.selection]
        ev.n_evals += int(g.extra.get("n_evals", 0))

    local, spin = _local_neighbourhoods(ev.candidates, radius_mm, ev.n)
    all_ids = np.arange(ev.n)
    dose, oar = ev.state(sel)
    cur = ev.hard_internal(sel, dose, oar)
    history: List[Tuple[int, float]] = [(0, cur)]
    n_moves = 0
    move_counts = {"local": 0, "spin": 0, "global": 0}
    status, reason = "ok", ""
    n_pass = 0
    improved = True
    while improved and status == "ok":
        improved = False
        n_pass += 1
        for slot in range(len(sel)):
            i = sel[slot]
            rest = sel[:slot] + sel[slot + 1:]
            base = dose - ev.h.D[i].astype(np.float64)
            base_oar = {name: oar[name] - np.asarray(ev.h.oar[name][i], dtype=np.float64)
                        for name in oar}
            mask = ev.conflicts.compatible_mask(rest)
            mask[i] = False
            if ev.kinds is not None:
                mask &= ev.kinds == ev.kinds[i]
            blocks = []
            if local is not None:
                blocks.append(("local", local[i]))
                blocks.append(("spin", spin[i]))
            blocks.append(("global", all_ids))
            for name, pool in blocks:
                rows = pool[mask[pool]]
                if rows.size == 0:
                    continue
                if ev.n_evals + rows.size > int(max_evals):
                    status, reason = "max_evals", ("stopped after %d evaluations (max_evals=%d)"
                                                   % (ev.n_evals, int(max_evals)))
                    break
                vals = ev.hard_all(rest, base, base_oar, rows)
                k = int(np.argmax(vals))
                if vals[k] > cur + _EPS:
                    j = int(rows[k])
                    sel[slot] = j
                    dose = base + ev.h.D[j].astype(np.float64)
                    oar = {name: base_oar[name] + np.asarray(ev.h.oar[name][j], dtype=np.float64)
                           for name in oar}
                    cur = float(vals[k])
                    n_moves += 1
                    move_counts[name] += 1
                    improved = True
                    history.append((n_pass, cur))
                    break
            if status != "ok":
                break

    return _result(ev, sel, "local", t0, n_tiles, status=status, reason=reason,
                   feasible=bool(ev.conflicts.is_feasible(sel)) and len(sel) == n_tiles,
                   history=history,
                   extra={"n_moves": n_moves, "moves": move_counts, "passes": n_pass,
                          "radius_mm": float(radius_mm),
                          "degraded_to_global": local is None})


# ------------------------------------------------------- simulated annealing
class _SAState:
    """Mutable SA state: slot -> candidate, dose vectors, conflict counters."""

    def __init__(self, ev: _Evaluator, sel: Sequence[int], nbr: List[np.ndarray]):
        self.ev = ev
        self.sel = [int(i) for i in sel]
        self.dose, self.oar = ev.state(self.sel)
        self.blocked = np.zeros(ev.n, dtype=int)
        self.selected = np.zeros(ev.n, dtype=bool)
        for i in self.sel:
            self.blocked[nbr[i]] += 1
            self.selected[i] = True
        self.soft = ev.soft_internal(self.dose, self.oar)
        self.hard = ev.hard_internal(self.sel, self.dose, self.oar)

    def trial(self, slot: int, j: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, float]:
        i = self.sel[slot]
        D = self.ev.h.D
        dose = self.dose - D[i].astype(np.float64) + D[j].astype(np.float64)
        oar = {name: self.oar[name] - np.asarray(mat[i], dtype=np.float64)
               + np.asarray(mat[j], dtype=np.float64)
               for name, mat in self.ev.h.oar.items()}
        pen = self.ev.h.oar_penalty(oar)
        self.ev.n_evals += 1
        return dose, oar, self.ev.h.soft_from_dose(dose, pen), self.ev.h.hard_from_dose(dose, pen)


def _random_packing(conflicts, kinds: Optional[np.ndarray],
                    kinds_needed: List[Optional[str]], rng,
                    n_attempts: int = 50) -> Optional[List[int]]:
    """Random feasible selection with the kind multiset ``kinds_needed``.

    First ``n_attempts`` random-permutation walks (take each candidate that is
    compatible with those taken so far and whose kind is still needed: a
    random maximal packing, which spreads tiles out).  When the packing is
    tight (N near the maximum, where a random maximal packing rarely reaches
    N) one bound-aware construction follows: candidates in random order,
    each accepted only if :func:`_packing_bound` says the rest still fit.
    None when both fail.
    """
    n = len(kinds_needed)
    nbr = _conflict_lists(conflicts)

    def kind_of(c: int) -> Optional[str]:
        return None if kinds is None else str(kinds[c])

    def need_map() -> Dict[Optional[str], int]:
        need: Dict[Optional[str], int] = {}
        for k in kinds_needed:
            need[k] = need.get(k, 0) + 1
        return need

    def key_of(c: int, need) -> Optional[str]:
        k = kind_of(c)
        return k if k in need else None

    for _ in range(n_attempts):
        need = need_map()
        blocked = np.zeros(conflicts.n, dtype=bool)
        sel: List[int] = []
        for c in rng.permutation(conflicts.n):
            c = int(c)
            if blocked[c] or need.get(key_of(c, need), 0) <= 0:
                continue
            need[key_of(c, need)] -= 1
            sel.append(c)
            blocked[c] = True
            blocked[nbr[c]] = True
            if len(sel) == n:
                return sel

    # bound-aware construction (kinds ignored by the bound, enforced by the pick)
    need = need_map()
    mask = np.ones(conflicts.n, dtype=bool)
    deg = np.asarray(conflicts.pairs.dot(mask.astype(np.int64))).reshape(-1)
    sel = []
    while len(sel) < n:
        remaining = n - len(sel) - 1
        rows = np.flatnonzero(mask)
        rows = rows[[need.get(key_of(int(c), need), 0) > 0 for c in rows]]
        if rows.size == 0:
            return None
        chosen = None
        for c in rng.permutation(rows)[:GREEDY_MAX_BOUND_CHECKS]:
            c = int(c)
            m_c = mask.copy()
            m_c[nbr[c]] = False
            m_c[c] = False
            if _packing_bound(m_c, nbr, deg, remaining) >= remaining:
                chosen, mask = c, m_c
                break
        if chosen is None:
            return None
        need[key_of(chosen, need)] -= 1
        sel.append(chosen)
    return sel


def _random_feasible(ev: _Evaluator, kinds_needed: List[Optional[str]], rng,
                     n_attempts: int = 50) -> Optional[List[int]]:
    """:func:`_random_packing` over an evaluator's conflicts and kinds."""
    return _random_packing(ev.conflicts, ev.kinds, kinds_needed, rng, n_attempts)


def solve_sa(objective: Objective, n_tiles: int, seed: int = 0,
             n_sweeps: int = SA_N_SWEEPS, n_restarts: int = SA_N_RESTARTS,
             start=None, candidates: Optional[CandidateSet] = None,
             n_trial: int = 200, alpha: float = SA_ALPHA,
             moves_per_tile: int = SA_MOVES_PER_TILE_PER_SWEEP,
             radius_mm: float = LOCAL_RADIUS_MM) -> SolverResult:
    """E3 simulated annealing on the SOFT objective (section 3 E3).

    Moves (feasibility maintained: only candidates compatible with the other
    tiles, same kind): 60 % local re-site (anchor within ``radius_mm``;
    needs ``candidates``, else global), 20 % re-spin (same ``anchor_id``;
    needs ``candidates``, else global), 20 % global re-site.  Metropolis
    acceptance ``exp(delta_soft / T)``.  ``T0`` = median uphill ``|delta
    soft|`` over ``n_trial`` trial moves from the start / ln 2 (so ~50 % of
    the initial uphill moves are accepted); geometric cooling ``alpha`` per
    sweep of ``moves_per_tile * N`` moves.  Restart 0 starts from ``start``
    (topped up greedily when short) or greedy; the others from random
    feasible starts.  The best HARD objective seen anywhere is returned
    (ties keep the earlier one, so the result is never worse than the start).

    Reproducible from ``seed`` (``np.random.default_rng(seed)``).
    ``history`` = ``(restart, sweep, best_hard_so_far)`` per sweep;
    ``extra`` = per-restart best, T0, acceptance rates, move counts.
    """
    t0 = time.perf_counter()
    ev = _Evaluator(objective, candidates)
    n_tiles = int(n_tiles)
    if n_tiles <= 0:
        raise ValueError("n_tiles must be >= 1")
    rng = np.random.default_rng(int(seed))

    # ---- start
    start_ids = [int(i) for i in _as_ids(start if start is not None else (), ev.n)]
    if len(start_ids) > n_tiles:
        raise ValueError("start has %d ids but n_tiles = %d" % (len(start_ids), n_tiles))
    if not ev.conflicts.is_feasible(start_ids):
        return _result(ev, start_ids, "sa", t0, n_tiles, seed=int(seed), status="infeasible",
                       feasible=False, reason="start selection violates a conflict")
    if len(start_ids) < n_tiles:
        g = solve_greedy(objective, n_tiles, fixed=start_ids, candidates=ev.candidates)
        ev.n_evals += int(g.extra.get("n_evals", 0))
        if not g.feasible:
            g.solver = "sa"
            g.seed = int(seed)
            g.reason = "greedy start failed: " + g.reason
            return g
        start_ids = [int(i) for i in g.selection]

    # ---- neighbourhoods
    nbr = _conflict_lists(ev.conflicts)
    local, spin = _local_neighbourhoods(ev.candidates, radius_mm, ev.n)
    all_ids = np.arange(ev.n)
    degraded = local is None
    kinds_seq = [None] * n_tiles if ev.kinds is None else [str(k) for k in ev.kinds[start_ids]]

    def propose(st: _SAState):
        """``(slot, j, kind)`` or None when the chosen pool has no compatible candidate."""
        slot = int(rng.integers(n_tiles))
        i = st.sel[slot]
        u = rng.random()
        if u < 0.6:
            kind, pool = "local", (all_ids if degraded else local[i])
        elif u < 0.8:
            kind, pool = "spin", (all_ids if degraded else spin[i])
        else:
            kind, pool = "global", all_ids
        st.blocked[nbr[i]] -= 1                     # remove i virtually
        ok = (st.blocked[pool] == 0) & ~st.selected[pool]
        if ev.kinds is not None:
            ok &= ev.kinds[pool] == ev.kinds[i]
        st.blocked[nbr[i]] += 1
        rows = pool[ok]
        if rows.size == 0:
            return None
        return slot, int(rows[rng.integers(rows.size)]), kind

    def apply(st: _SAState, slot: int, j: int, dose, oar, soft, hard):
        i = st.sel[slot]
        st.blocked[nbr[i]] -= 1
        st.selected[i] = False
        st.blocked[nbr[j]] += 1
        st.selected[j] = True
        st.sel[slot] = j
        st.dose, st.oar, st.soft, st.hard = dose, oar, soft, hard

    # ---- T0 from trial moves at the start
    st0 = _SAState(ev, start_ids, nbr)
    uphill = []
    deltas = []
    for _ in range(int(n_trial)):
        p = propose(st0)
        if p is None:
            continue
        slot, j, _k = p
        _d, _o, soft_new, _h = st0.trial(slot, j)
        d = soft_new - st0.soft
        deltas.append(abs(d))
        if d < 0:
            uphill.append(-d)
    if uphill:
        T0 = float(np.median(uphill)) / math.log(2.0)
    elif deltas and max(deltas) > 0:
        T0 = float(np.median([d for d in deltas if d > 0])) / math.log(2.0)
    else:
        T0 = 1e-3            # flat landscape: a nominal temperature (soft is a fraction in [0, 1])
    T0 = max(T0, 1e-12)

    # ---- restarts
    best_sel = list(start_ids)
    best_hard = st0.hard
    start_hard = st0.hard
    history: List[Tuple[int, int, float]] = []
    per_restart = []
    acceptance = []
    move_counts = {"local": [0, 0], "spin": [0, 0], "global": [0, 0]}   # [proposed, accepted]
    n_restarts = max(1, int(n_restarts))
    moves_per_sweep = int(moves_per_tile) * n_tiles
    for r in range(n_restarts):
        if r == 0:
            st = st0
            origin = "start"
        else:
            rs = _random_feasible(ev, kinds_seq, rng)
            if rs is None:
                st = _SAState(ev, start_ids, nbr)
                origin = "start (no random feasible start found)"
            else:
                st = _SAState(ev, rs, nbr)
                origin = "random"
        if st.hard > best_hard + _EPS:
            best_hard, best_sel = st.hard, list(st.sel)
        r_best, r_best_sel = st.hard, list(st.sel)
        n_prop = n_acc = 0
        T = T0
        for s in range(int(n_sweeps)):
            for _ in range(moves_per_sweep):
                p = propose(st)
                if p is None:
                    continue
                slot, j, kind = p
                n_prop += 1
                move_counts[kind][0] += 1
                dose, oar, soft_new, hard_new = st.trial(slot, j)
                d = soft_new - st.soft
                if d >= 0 or rng.random() < math.exp(d / T):
                    apply(st, slot, j, dose, oar, soft_new, hard_new)
                    n_acc += 1
                    move_counts[kind][1] += 1
                    if hard_new > r_best + _EPS:
                        r_best, r_best_sel = hard_new, list(st.sel)
                    if hard_new > best_hard + _EPS:
                        best_hard, best_sel = hard_new, list(st.sel)
            history.append((r, s, best_hard))
            T *= float(alpha)
        per_restart.append({"restart": r, "origin": origin, "best_hard": r_best,
                            "selection": sorted(r_best_sel), "proposed": n_prop,
                            "accepted": n_acc})
        acceptance.append(n_acc / n_prop if n_prop else float("nan"))

    return _result(ev, best_sel, "sa", t0, n_tiles, seed=int(seed), history=history,
                   extra={"T0": T0, "T_final": T0 * float(alpha) ** int(n_sweeps),
                          "per_restart": per_restart, "acceptance_rate": acceptance,
                          "move_counts": move_counts, "n_sweeps": int(n_sweeps),
                          "moves_per_sweep": moves_per_sweep,
                          "degraded_to_global": degraded,
                          "start": sorted(start_ids), "start_hard": start_hard})


# ------------------------------------------------------ continuous refinement
class _Deadline(Exception):
    """Raised inside Nelder-Mead when the wall-time budget is exhausted."""


class _ContinuousCore:
    """Shared machinery of :func:`refine_continuous` and :func:`solve_continuous`:
    soft / hard coverage of PlacedTiles on a (subsampled) target with the real
    engine (tabulated kernel), and coordinate-descent Nelder-Mead over each
    tile's ``(u, v, theta)`` in its local tangent frame exactly as
    ``tiles.surface.fit_on_surface._place`` parameterizes it.
    """

    def __init__(self, mesh, target: TargetSet, rx_cgy: float, objective=None,
                 engine=None, sk_per_seed_u=None, m_opt: int = 1000, rng_seed: int = 0,
                 step_mm: float = 1.0, step_deg: float = 5.0, max_iter: int = 60,
                 deadline: Optional[float] = None):
        from ..dose.engine import TG43Engine
        self.mesh = mesh
        self.eng = engine if engine is not None else TG43Engine()
        self.sk = TG43Engine.DEFAULT_SK_U if sk_per_seed_u is None else float(sk_per_seed_u)
        self.rx = float(rx_cgy)
        self.lam_hot = float(objective.lambda_hot) if objective is not None else LAMBDA_HOT
        self.v200_tol = float(objective.v200_tol) if objective is not None else V200_TOL
        self.tau = (float(objective.tau_cgy) if objective is not None and objective.tau_cgy
                    else TAU_FRACTION * self.rx)
        sub, _idx = target.subsample(int(m_opt), rng_seed=int(rng_seed))
        self.pts = np.asarray(sub.points, dtype=float)
        w = np.asarray(sub.weights, dtype=float)
        self.w = w
        self.wn = w / w.sum() if w.sum() > 0 else np.full(w.shape, 1.0 / max(w.size, 1))
        self.step_mm, self.step_deg, self.max_iter = float(step_mm), float(step_deg), int(max_iter)
        self.deadline = deadline
        self.n_evals = 0
        self.nm_runs = 0
        self.nm_seconds = 0.0
        self.nm_evals = 0

    # ---- dose and objectives on the subsample
    def tile_dose(self, t: PlacedTile) -> np.ndarray:
        from ..dose.engine import dose_at_points
        self.n_evals += 1
        return np.asarray(dose_at_points(t.seed_centers, t.seed_axes, self.pts,
                                         sk_per_seed_u=self.sk, engine=self.eng, exact=False),
                          dtype=float)

    def hot(self, dose: np.ndarray) -> float:
        v200 = float(self.wn @ (dose >= 2.0 * self.rx))
        return self.lam_hot * max(0.0, v200 - self.v200_tol)

    def soft(self, dose: np.ndarray) -> float:
        z = np.clip((dose - self.rx) / self.tau, -60.0, 60.0)
        return float(self.wn @ (1.0 / (1.0 + np.exp(-z)))) - self.hot(dose)

    def hard(self, dose: np.ndarray) -> float:
        return float(self.wn @ (dose >= self.rx)) - self.hot(dose)

    def v100(self, dose: np.ndarray) -> float:
        return float(self.wn @ (dose >= self.rx))

    def metrics(self, dose: np.ndarray) -> Dict[str, float]:
        return {"V100": self.v100(dose),
                "V150": float(self.wn @ (dose >= 1.5 * self.rx)),
                "V200": float(self.wn @ (dose >= 2.0 * self.rx)),
                "D90": _weighted_quantile(dose, self.w, 0.10),
                "Dmean": float(self.wn @ dose)}

    def expired(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    # ---- one tile's Nelder-Mead
    def refine_tile(self, tile: PlacedTile, others: np.ndarray):
        """NM over (u, v, theta) for one tile with the others' dose fixed.
        Returns ``(new_tile, new_dose, x)``; raises :class:`_Deadline`."""
        from scipy.optimize import minimize
        from ..interact import _rodrigues, conform_tile, snap_to_wall

        surf0 = np.asarray(tile.anchor_ras, dtype=float)
        n0 = np.asarray(tile.normal_ras, dtype=float)
        t1_0 = np.asarray(tile.axis_ras, dtype=float)
        t2_0 = np.cross(n0, t1_0)
        t2_0 /= max(np.linalg.norm(t2_0), 1e-12)
        kind = tile.kind
        mesh = self.mesh

        def _place(x):
            anchor = surf0 + x[0] * t1_0 + x[1] * t2_0
            surf, n_in = snap_to_wall(mesh, anchor)
            hint = _rodrigues(t1_0, n0, float(x[2]))
            return conform_tile(mesh, surf, n_in, hint, kind=kind)

        def _cost(x):
            if self.expired():
                raise _Deadline()
            self.nm_evals += 1
            try:
                t = _place(x)
            except Exception:
                return 1e6
            return -self.soft(others + self.tile_dose(t))

        simplex0 = np.array([[0.0, 0.0, 0.0], [self.step_mm, 0.0, 0.0],
                             [0.0, self.step_mm, 0.0],
                             [0.0, 0.0, np.deg2rad(self.step_deg)]], dtype=float)
        t0 = time.perf_counter()
        try:
            sol = minimize(_cost, np.zeros(3), method="Nelder-Mead",
                           options=dict(maxiter=self.max_iter, maxfev=2 * self.max_iter,
                                        xatol=0.05, fatol=1e-6, initial_simplex=simplex0))
            new_tile = _place(sol.x)
            x = np.asarray(sol.x, dtype=float)
        finally:
            self.nm_runs += 1
            self.nm_seconds += time.perf_counter() - t0
        return new_tile, self.tile_dose(new_tile), x

    # ---- coordinate descent over all tiles
    def descend(self, tiles: List[PlacedTile], n_passes: int) -> Tuple[List[PlacedTile], Dict[str, Any]]:
        from ..interact import find_overlapping_tiles
        tiles = list(tiles)
        n = len(tiles)
        doses = [self.tile_dose(t) for t in tiles]
        total = np.sum(doses, axis=0) if n else np.zeros(self.pts.shape[0])
        info: Dict[str, Any] = {
            "hard_before": self.hard(total), "soft_before": self.soft(total),
            "V100_before": self.v100(total), "n_tiles": n,
            "overlaps_before": list(find_overlapping_tiles(tiles)),
        }
        accepted = 0
        rejected = {"v100": 0, "soft": 0, "overlap": 0, "nm_failed": 0}
        offsets = [np.zeros(3) for _ in range(n)]
        soft_cur, v100_cur = info["soft_before"], info["V100_before"]
        history: List[Tuple[int, int, float]] = []
        passes_done = 0
        deadline_hit = False
        try:
            for p in range(int(n_passes)):
                for i in range(n):
                    if self.expired():
                        raise _Deadline()
                    others = total - doses[i]
                    try:
                        new_tile, new_dose, x = self.refine_tile(tiles[i], others)
                    except _Deadline:
                        raise
                    except Exception:
                        rejected["nm_failed"] += 1
                        continue
                    new_total = others + new_dose
                    new_soft, new_v100 = self.soft(new_total), self.v100(new_total)
                    if new_v100 < v100_cur - _EPS:
                        rejected["v100"] += 1
                        continue
                    if new_soft <= soft_cur + 1e-9:
                        rejected["soft"] += 1
                        continue
                    trial = list(tiles)
                    trial[i] = new_tile
                    if find_overlapping_tiles(trial):
                        rejected["overlap"] += 1
                        continue
                    tiles, doses[i], total = trial, new_dose, new_total
                    soft_cur, v100_cur = new_soft, new_v100
                    offsets[i] = offsets[i] + x
                    accepted += 1
                    history.append((p, i, self.hard(total)))
                passes_done += 1
        except _Deadline:
            deadline_hit = True
        info.update({
            "hard_after": self.hard(total), "soft_after": soft_cur, "V100_after": v100_cur,
            "metrics_after": self.metrics(total), "accepted": accepted, "rejected": rejected,
            "history": history, "passes_done": passes_done, "deadline_hit": deadline_hit,
            "offsets_uv_mm_theta_rad": [o.tolist() for o in offsets],
            "overlaps_after": list(find_overlapping_tiles(tiles)),
        })
        return tiles, info


def refine_continuous(mesh, candidates: CandidateSet, selection, target: TargetSet,
                      rx_cgy: float = DEFAULT_RX_CGY, objective: Optional[Objective] = None,
                      n_passes: int = 2, step_mm: float = 1.0, step_deg: float = 5.0,
                      engine=None, max_iter: int = 60, m_opt: int = 1000,
                      sk_per_seed_u: Optional[float] = None, rng_seed: int = 0,
                      time_budget_s: Optional[float] = None
                      ) -> Tuple[List[PlacedTile], Dict[str, Any]]:
    """E5 continuous polish (section 3 E5): coordinate descent over the
    selected tiles, each tile refined by Nelder-Mead over ``(u, v, theta)``
    in its local tangent frame exactly as ``tiles.surface.fit_on_surface``
    parameterizes it (anchor = ``anchor0 + u t1 + v t2`` -> ``snap_to_wall``
    -> ``conform_tile`` with the axis hint rotated by ``theta`` about the
    original normal).

    Inside NM the objective is the soft coverage (sigmoid, ``tau``) minus the
    hot-spot penalty with the other tiles' dose cached; the moving tile's
    dose comes from ``dose_at_points(exact=False)`` on ``target`` subsampled
    to ``m_opt`` points.  A new pose is accepted only if hard V100 does not
    decrease, the soft objective improves, and
    ``find_overlapping_tiles(new_tiles) == []``.  ``time_budget_s`` (optional)
    aborts NM at the deadline and returns the tiles refined so far.

    Returns ``(tiles, info)``; ``info`` has ``hard_before/after``,
    ``soft_before/after``, ``V100_before/after``, ``accepted``,
    ``evaluations``, ``seconds``, per-tile offsets and NM cost.
    """
    t0 = time.perf_counter()
    deadline = None if time_budget_s is None else t0 + float(time_budget_s)
    core = _ContinuousCore(mesh, target, rx_cgy, objective=objective, engine=engine,
                           sk_per_seed_u=sk_per_seed_u, m_opt=m_opt, rng_seed=rng_seed,
                           step_mm=step_mm, step_deg=step_deg, max_iter=max_iter,
                           deadline=deadline)
    tiles, info = core.descend(list(candidates.tiles_of(selection)), n_passes)
    info.update({"evaluations": core.n_evals, "seconds": time.perf_counter() - t0,
                 "m_opt": int(core.pts.shape[0]), "n_passes": int(n_passes),
                 "step_mm": float(step_mm), "step_deg": float(step_deg),
                 "nm_runs": core.nm_runs, "nm_seconds": core.nm_seconds,
                 "nm_seconds_per_tile": core.nm_seconds / max(core.nm_runs, 1),
                 "nm_evals_per_tile": core.nm_evals / max(core.nm_runs, 1)})
    return tiles, info


def _conflicts_from_tiles(candidates: CandidateSet):
    """ConflictGraph from ``find_overlapping_tiles`` on the candidate tiles
    (fallback when no :class:`ConflictGraph` is supplied; O(C^2) footprint
    tests, fine for a few hundred candidates)."""
    import scipy.sparse as sp
    from . import ConflictGraph
    from ..interact import find_overlapping_tiles
    c = len(candidates)
    pairs = sp.lil_matrix((c, c), dtype=bool)
    for i, j in find_overlapping_tiles(candidates.tiles):
        pairs[i, j] = True
        pairs[j, i] = True
    return ConflictGraph(n=c, pairs=pairs.tocsr(), cliques=[], gap_mm=0.0)


def solve_continuous(mesh, candidates: CandidateSet, target: TargetSet, rx_cgy: float,
                     n_tiles: int, seed: int = 0, n_starts: int = 3, n_passes: int = 2,
                     time_budget_s: Optional[float] = None, start=None, engine=None,
                     step_mm: float = 2.0, step_deg: float = 10.0,
                     kinds_required: Optional[Dict[str, int]] = None,
                     objective: Optional[Objective] = None, conflicts=None,
                     max_iter: int = 60, m_opt: int = 1000,
                     sk_per_seed_u: Optional[float] = None
                     ) -> Tuple[List[PlacedTile], SolverResult]:
    """Direct continuous optimization (scout 7.4): multi-start coordinate
    descent Nelder-Mead over each tile's ``(u, v, theta)`` through
    ``conform_tile``, keeping the best hard V100.

    Starts: ``start`` (candidate ids) if given, else the feasibility-aware
    greedy solution of ``objective`` if given, else the first random feasible
    selection; plus ``n_starts - 1`` random feasible selections drawn from
    ``candidates`` (``np.random.default_rng(seed)``; conflicts from
    ``conflicts`` / ``objective.conflicts`` / ``find_overlapping_tiles`` on
    the candidate tiles).  Each start is refined exactly as
    :func:`refine_continuous` does (soft surrogate ``tau = 0.05 rx`` inside
    NM; a tile move is accepted only if hard V100 does not drop, soft
    improves and ``find_overlapping_tiles`` stays empty); the start with the
    best final hard objective (then soft) wins.  ``time_budget_s`` is a
    strict wall-time cap: NM is aborted at the deadline and the best so far
    returned (``status="time_limit"``; reproducibility from ``seed`` then
    depends on the machine).  Recommended discrete start grid: h = 4 mm,
    3 spins.

    Returns ``(tiles, result)``: the tiles are CONTINUOUS poses, not
    candidates; ``result.selection`` is the winning start's candidate ids,
    ``result.objective`` / ``metrics`` the hard objective / metrics of the
    refined tiles on the subsampled target with the real engine (not the
    influence matrix), ``history`` = best-so-far hard per ``(start, pass)``,
    ``extra`` = per-start best hard/soft, passes done, evaluations, seconds,
    NM cost per tile.
    """
    from . import ConflictGraph
    from ..interact import find_overlapping_tiles

    t0 = time.perf_counter()
    deadline = None if time_budget_s is None else t0 + float(time_budget_s)
    n_tiles = int(n_tiles)
    if n_tiles <= 0:
        raise ValueError("n_tiles must be >= 1")
    rng = np.random.default_rng(int(seed))
    C = len(candidates)
    if conflicts is None:
        conflicts = objective.conflicts if objective is not None else _conflicts_from_tiles(candidates)
    if not isinstance(conflicts, ConflictGraph) or conflicts.n != C:
        raise ValueError("conflicts must be a ConflictGraph over the %d candidates" % C)
    kinds = np.asarray(candidates.kinds, dtype=object)
    if kinds_required:
        if sum(int(v) for v in kinds_required.values()) != n_tiles:
            raise ValueError("kinds_required counts must sum to n_tiles")
        kinds_seq: List[Optional[str]] = []
        for k, v in kinds_required.items():
            kinds_seq += [str(k)] * int(v)
    else:
        kinds_seq = [None] * n_tiles

    # ---- starts
    starts: List[Tuple[str, List[int]]] = []
    if start is not None:
        ids = [int(i) for i in _as_ids(start, C)]
        if len(ids) != n_tiles or not conflicts.is_feasible(ids):
            raise ValueError("start must be a feasible selection of %d candidate ids" % n_tiles)
        starts.append(("start", ids))
    elif objective is not None:
        g = solve_greedy(objective, n_tiles, kinds_required=kinds_required, candidates=candidates)
        if g.feasible:
            starts.append(("greedy", [int(i) for i in g.selection]))
    n_random = max(0, int(n_starts) - len(starts))
    if not starts:
        n_random = max(1, n_random)
    for _ in range(n_random):
        sel = _random_packing(conflicts, kinds, kinds_seq, rng)
        if sel is None:
            break
        starts.append(("random", sel))
    if not starts:
        return [], SolverResult(selection=np.zeros(0, dtype=int), solver="continuous",
                                seed=int(seed), status="infeasible", feasible=False,
                                reason="no feasible %d-tile start found among %d candidates"
                                       % (n_tiles, C), runtime_s=time.perf_counter() - t0)

    core = _ContinuousCore(mesh, target, rx_cgy, objective=objective, engine=engine,
                           sk_per_seed_u=sk_per_seed_u, m_opt=m_opt, rng_seed=int(seed),
                           step_mm=step_mm, step_deg=step_deg, max_iter=max_iter,
                           deadline=deadline)
    best = None
    per_start = []
    history: List[Tuple[int, int, float]] = []
    best_hard = -np.inf
    deadline_hit = False
    for s_idx, (origin, ids) in enumerate(starts):
        if core.expired() and best is not None:
            deadline_hit = True
            per_start.append({"start": s_idx, "origin": origin, "selection": sorted(ids),
                              "skipped": "time budget exhausted"})
            continue
        tiles, info = core.descend(list(candidates.tiles_of(ids)), n_passes)
        deadline_hit = deadline_hit or info["deadline_hit"]
        key = (info["hard_after"], info["soft_after"])
        if best is None or key > best[0]:
            best = (key, s_idx, sorted(ids), tiles, info)
        # best-so-far hard after each pass of this start (a pass without an
        # accepted step carries the previous value)
        hard_pass = info["hard_before"]
        best_hard = max(best_hard, hard_pass)
        for p in range(max(info["passes_done"], 1)):
            steps = [h for (pp, _i, h) in info["history"] if pp == p]
            if steps:
                hard_pass = steps[-1]
            best_hard = max(best_hard, hard_pass)
            history.append((s_idx, p, best_hard))
        per_start.append({"start": s_idx, "origin": origin, "selection": sorted(ids),
                          "hard_before": info["hard_before"], "hard_after": info["hard_after"],
                          "soft_before": info["soft_before"], "soft_after": info["soft_after"],
                          "V100_before": info["V100_before"], "V100_after": info["V100_after"],
                          "passes_done": info["passes_done"], "accepted": info["accepted"],
                          "deadline_hit": info["deadline_hit"]})
        if info["deadline_hit"]:
            break

    _key, s_idx, ids, tiles, info = best
    overlaps = list(find_overlapping_tiles(tiles))
    status = "time_limit" if deadline_hit else "ok"
    result = SolverResult(
        selection=np.asarray(ids, dtype=int), objective=float(info["hard_after"]),
        metrics=dict(info["metrics_after"]), history=history,
        runtime_s=time.perf_counter() - t0, solver="continuous", seed=int(seed),
        status=status,
        reason=("wall-time budget of %.1f s hit; best refined start returned"
                % float(time_budget_s)) if deadline_hit else "",
        feasible=(not overlaps) and len(tiles) == n_tiles,
        extra={"per_start": per_start, "best_start": int(s_idx), "n_starts": len(starts),
               "evaluations": core.n_evals, "seconds": time.perf_counter() - t0,
               "nm_runs": core.nm_runs, "nm_seconds": core.nm_seconds,
               "nm_seconds_per_tile": core.nm_seconds / max(core.nm_runs, 1),
               "nm_evals_per_tile": core.nm_evals / max(core.nm_runs, 1),
               "overlaps": overlaps, "m_opt": int(core.pts.shape[0]),
               "step_mm": float(step_mm), "step_deg": float(step_deg),
               "n_passes": int(n_passes), "deadline_hit": deadline_hit,
               "tiles_are_continuous_poses": True})
    return tiles, result
