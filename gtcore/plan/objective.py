"""Objective (section 3 D). Owner: A2 Dose, branch ``plan/influence``.

Module-level implementations of the :class:`gtcore.plan.Objective` methods
(``dose_of``, ``metrics``, ``hard``, ``soft``, ``gain``), the vectorized
``gains_all`` / ``soft_gains_all`` fast paths greedy and annealing need,
``make_objective`` and ``evaluate``.  ``gtcore/plan/__init__.py`` binds the
method stubs to the functions here; the extra fast paths are reached as
``gtcore.plan.objective.gains_all(objective, selection)``.

Definitions (``D`` = dose vector of a selection, ``w`` = target weights,
``W = sum(w)``, ``rx`` = prescription)
-------------------------------------------------------------------------
- ``V100 = sum(w * [D >= rx]) / W``; ``V150``, ``V200`` likewise at 1.5 and
  2.0 rx; ``Dmean = sum(w * D) / W``; ``D90 = weighted_quantile(D, w, 0.10)``;
  ``oar_dmax_<name> = max`` of the OAR dose vector.
- ``tail_mean = sum over the coldest tail_q of the target weight of
  w * min(D, rx) / rx, divided by tail_q * W`` (the lower-tail / CVaR mean of
  the capped dose; in [0, 1]; ``<= min(D90, rx) / rx`` for ``tail_q = 0.10``,
  and 1 exactly when V100 = 1).  Continuous in D where V100 is a step, and
  it rewards lifting the coldest points, which V100 does not: with too few
  tiles V100 alone clusters them and leaves one cold patch (docs/
  optimize-notes.md, "Objective: lower-tail term").
- hard (P1): ``V100 + lambda_tail * tail_mean - lambda_hot * max(0, V200 -
  v200_tol) - sum_j lambda_oar * max(0, Dmax_j - L_j)`` over the OARs that
  have a limit in ``influence.oar_limits``.
- soft: ``sum(w * sigmoid((D - rx) / tau)) / W`` plus the same tail term
  minus the same penalties.
- gain / gains_all: incremental ``hard(sel + c) - hard(sel)`` from the cached
  dose vector of ``sel``; a candidate already in ``sel`` has gain 0.

Weighted quantile convention
----------------------------
``weighted_quantile(v, w, q)`` is the weighted *inverted CDF*: the smallest
value ``v_(i)`` (values sorted ascending) whose cumulative weight
``w_(1) + ... + w_(i)`` reaches ``q * W``.  With unit weights it equals
``numpy.percentile(v, 100 q, method="inverted_cdf")`` (``q = 0`` -> min,
``q = 1`` -> max), and it is a value that actually occurs in ``v`` -- the
natural reading of "D90 = the dose received by 90 % of the target area".
Zero-weight entries are ignored.

Performance (measured in ``tests/test_plan_objective.py``): ``hard`` is a
few tens of microseconds at M = 4000, N = 8; ``gains_all`` evaluates all
C = 2000 candidates at M = 4000 in a few tens of milliseconds through a
chunked ``(C, M)`` comparison and one matmul with the weights.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import expit

from . import InfluenceMatrix, ConflictGraph, Objective, _as_ids

GAINS_CHUNK_ROWS = 256
# Rows of the (C, M) temporaries evaluated at once in gains_all: 256 x 4000
# float64 = 8 MB per temporary, cache friendly and far below the 100 MB cap.

_OBJECTIVE_WEIGHT_KEYS = ("lambda_hot", "v200_tol", "lambda_oar", "tau_cgy", "rx_cgy",
                          "lambda_tail", "tail_q")


# ------------------------------------------------------------- quantiles
def weighted_quantile(values, weights, q: float) -> float:
    """Weighted inverted-CDF quantile of ``values`` at ``q`` in [0, 1].

    Smallest sorted value whose cumulative weight reaches ``q * sum(weights)``;
    equals ``numpy.percentile(values, 100 * q, method="inverted_cdf")`` for
    unit weights.  Zero-weight entries are ignored; raises ``ValueError``
    on empty input, all-zero weights or ``q`` outside [0, 1].
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must lie in [0, 1]")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    keep = w > 0
    v, w = v[keep], w[keep]
    if v.size == 0:
        raise ValueError("weighted_quantile of an empty / zero-weight set")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    # numpy's inverted_cdf virtual index is n * q computed in floating point;
    # cumulative unit weights are exact integers, so searchsorted on q * W
    # reproduces it exactly (the tiny relative slack absorbs the summation
    # rounding of non-integer weights without changing integer cases).
    thresh = q * total
    thresh -= 4.0 * np.finfo(float).eps * total
    i = int(np.searchsorted(cw, thresh, side="left"))
    return float(v[min(i, v.size - 1)])


def tail_rows(block, weights, rx_cgy: float, q: float) -> np.ndarray:
    """Lower-tail mean of the capped dose, one value per row of ``block``.

    For each row ``D`` (``(M,)`` or ``(k, M)``): the weighted mean of
    ``min(D, rx) / rx`` over the coldest ``q`` of the total target weight
    (the last point counted fractionally, so the result is continuous in
    ``D``).  Returns ``(k,)`` float64 in [0, 1]; 0 when the weights sum to 0.
    O(k M log M) (one argsort per row).
    """
    d = np.asarray(block, dtype=np.float64)
    if d.ndim == 1:
        d = d[None, :]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(w.sum())
    if d.shape[1] == 0 or total <= 0.0:
        return np.zeros(d.shape[0], dtype=float)
    rx = float(rx_cgy)
    d = np.minimum(d, rx) / rx
    order = np.argsort(d, axis=1, kind="stable")
    ds = np.take_along_axis(d, order, axis=1)
    ws = w[order]                                        # (k, M)
    cw = np.cumsum(ws, axis=1)
    k_w = float(q) * total
    # fraction of each point's weight that lies inside the coldest k_w
    frac = np.clip((k_w - (cw - ws)) / np.where(ws > 0, ws, 1.0), 0.0, 1.0)
    return (frac * ws * ds).sum(axis=1) / k_w


def tail_mean(dose, weights, rx_cgy: float, q: float = 0.10) -> float:
    """Scalar :func:`tail_rows` of one dose vector."""
    return float(tail_rows(np.asarray(dose, dtype=float).reshape(-1), weights, rx_cgy, q)[0])


def _tail_term(objective: Objective, d) -> np.ndarray:
    """``lambda_tail * tail_rows(d)`` (``(k,)``; zeros without the term)."""
    d = np.asarray(d)
    k = 1 if d.ndim == 1 else d.shape[0]
    if not objective.lambda_tail:
        return np.zeros(k, dtype=float)
    return objective.lambda_tail * tail_rows(d, objective.influence.target.weights,
                                             objective.rx_cgy, objective.tail_q)


# --------------------------------------------------------------- metrics
def metrics_from_dose(dose, weights, rx_cgy: float) -> Dict[str, float]:
    """``V100, V150, V200, D90, Dmean`` of a dose vector with weights."""
    d = np.asarray(dose, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    rx = float(rx_cgy)
    total = float(w.sum())
    if d.size == 0 or total <= 0.0:
        return {"V100": 0.0, "V150": 0.0, "V200": 0.0, "D90": 0.0, "Dmean": 0.0}
    return {
        "V100": float(w @ (d >= rx)) / total,
        "V150": float(w @ (d >= 1.5 * rx)) / total,
        "V200": float(w @ (d >= 2.0 * rx)) / total,
        "D90": weighted_quantile(d, w, 0.10),
        "Dmean": float(w @ d) / total,
    }


def _oar_penalty(objective: Objective, dmax: Dict[str, float]) -> float:
    """``sum_j lambda_oar * max(0, Dmax_j - L_j)`` over limited OARs."""
    pen = 0.0
    for name, lim in objective.influence.oar_limits.items():
        pen += objective.lambda_oar * max(0.0, float(dmax[name]) - float(lim))
    return pen


def _hot_penalty(objective: Objective, v200: float) -> float:
    return objective.lambda_hot * max(0.0, float(v200) - objective.v200_tol)


def _oar_dmax(objective: Objective, ids: np.ndarray) -> Dict[str, float]:
    inf = objective.influence
    out = {}
    for name in inf.oar:
        vec = inf.oar_dose_of(name, ids)
        out[name] = float(vec.max()) if vec.size else 0.0
    return out


def dose_of(objective: Objective, selection) -> np.ndarray:
    """Dose ``(M,)`` float64 [cGy] of ``selection`` (``influence.dose_of``)."""
    return objective.influence.dose_of(selection)


def metrics(objective: Objective, selection) -> Dict[str, float]:
    """``V100, V150, V200, D90, Dmean`` plus ``oar_dmax_<name>`` per OAR."""
    inf = objective.influence
    ids = _as_ids(selection, inf.n_candidates)
    d = inf.dose_of(ids)
    out = metrics_from_dose(d, inf.target.weights, objective.rx_cgy)
    for name, dmax in _oar_dmax(objective, ids).items():
        out["oar_dmax_" + name] = dmax
    return out


def _coverage_terms(objective: Objective, d: np.ndarray):
    """``(V100, V200)`` weighted fractions of a dose vector."""
    w = objective.influence.target.weights
    total = float(w.sum())
    if total <= 0.0:
        return 0.0, 0.0
    rx = objective.rx_cgy
    v100 = float(w @ (d >= rx)) / total
    v200 = float(w @ (d >= 2.0 * rx)) / total
    return v100, v200


def hard(objective: Objective, selection) -> float:
    """P1 objective ``V100 - lambda_hot max(0, V200 - tol) - OAR penalties``."""
    inf = objective.influence
    ids = _as_ids(selection, inf.n_candidates)
    d = inf.dose_of(ids)
    v100, v200 = _coverage_terms(objective, d)
    return v100 + float(_tail_term(objective, d)[0]) - _hot_penalty(objective, v200) \
        - _oar_penalty(objective, _oar_dmax(objective, ids))


def soft_coverage(objective: Objective, d: np.ndarray) -> float:
    """``sum(w * sigmoid((D - rx) / tau)) / W``."""
    w = objective.influence.target.weights
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    s = expit((np.asarray(d, dtype=float) - objective.rx_cgy) / objective.tau_cgy)
    return float(w @ s) / total


def soft(objective: Objective, selection) -> float:
    """Smooth surrogate: soft coverage minus the hot-spot and OAR penalties."""
    inf = objective.influence
    ids = _as_ids(selection, inf.n_candidates)
    d = inf.dose_of(ids)
    _v100, v200 = _coverage_terms(objective, d)
    return soft_coverage(objective, d) + float(_tail_term(objective, d)[0]) \
        - _hot_penalty(objective, v200) - _oar_penalty(objective, _oar_dmax(objective, ids))


def _f32_threshold(thr: np.ndarray) -> np.ndarray:
    """Smallest float32 >= ``thr`` elementwise, so that for a float32 dose row
    ``row >= thr32`` (pure float32) is exactly ``row >= thr`` (float64)."""
    t32 = np.asarray(thr, dtype=np.float64).astype(np.float32)
    low = t32 < thr
    if low.any():
        t32[low] = np.nextafter(t32[low], np.float32(np.inf))
    return t32


# ----------------------------------------------------------------- gains
def _base_state(objective: Objective, selection, dose_vec):
    """Normalized ids, base dose vector, base OAR vectors and base penalties."""
    inf = objective.influence
    ids = _as_ids(selection, inf.n_candidates)
    base = inf.dose_of(ids) if dose_vec is None else np.asarray(dose_vec, dtype=float).reshape(-1)
    if base.shape[0] != inf.n_targets:
        raise ValueError("dose_vec must have M = %d entries" % inf.n_targets)
    oar_base = {name: inf.oar_dose_of(name, ids) for name in inf.oar_limits}
    v100, v200 = _coverage_terms(objective, base)
    dmax_base = {name: (float(v.max()) if v.size else 0.0) for name, v in oar_base.items()}
    base_hard = v100 + float(_tail_term(objective, base)[0]) - _hot_penalty(objective, v200) \
        - _oar_penalty(objective, dmax_base)
    return ids, base, oar_base, base_hard


def _pen_rows(objective: Objective, oar_base, rows) -> np.ndarray:
    """OAR penalty of ``base + candidate`` for each candidate id in ``rows``."""
    inf = objective.influence
    pen = np.zeros(rows.size, dtype=float)
    for name, lim in inf.oar_limits.items():
        mat = inf.oar[name]
        if mat.shape[1] == 0:
            continue
        dmax = (mat[rows].astype(np.float64) + oar_base[name][None, :]).max(axis=1)
        pen += objective.lambda_oar * np.maximum(0.0, dmax - float(lim))
    return pen


def gains_all(objective: Objective, selection, dose_vec: Optional[np.ndarray] = None
              ) -> np.ndarray:
    """``(C,)`` float: ``hard(selection + [c]) - hard(selection)`` for every
    candidate ``c`` in one vectorized pass (0 for candidates already selected).

    ``dose_vec`` may supply the cached dose of ``selection`` (``dose_of``).
    Chunked over ``GAINS_CHUNK_ROWS`` rows so the temporaries stay small.
    """
    inf = objective.influence
    ids, base, oar_base, base_hard = _base_state(objective, selection, dose_vec)
    w = inf.target.weights
    total = float(w.sum())
    rx = objective.rx_cgy
    # candidate c covers point m  <=>  D[c, m] >= rx - base_m; the float32
    # thresholds reproduce the float64 comparison exactly (see _f32_threshold)
    thr100 = _f32_threshold(rx - base)
    thr200 = _f32_threshold(2.0 * rx - base)
    c_n = inf.n_candidates
    out = np.empty(c_n, dtype=float)
    for i0 in range(0, c_n, GAINS_CHUNK_ROWS):
        i1 = min(i0 + GAINS_CHUNK_ROWS, c_n)
        block = inf.dose[i0:i1]                                  # (k, M) float32 view
        if total > 0.0:
            v100 = ((block >= thr100[None, :]) @ w) / total
            v200 = ((block >= thr200[None, :]) @ w) / total
        else:
            v100 = np.zeros(i1 - i0)
            v200 = np.zeros(i1 - i0)
        hot = objective.lambda_hot * np.maximum(0.0, v200 - objective.v200_tol)
        tail = 0.0
        if objective.lambda_tail:
            tail = _tail_term(objective, block.astype(np.float64) + base[None, :])
        out[i0:i1] = v100 + tail - hot - _pen_rows(objective, oar_base, np.arange(i0, i1)) \
            - base_hard
    out[ids] = 0.0
    return out


def gain(objective: Objective, selection, candidate: int) -> float:
    """Incremental ``hard(selection + [candidate]) - hard(selection)``, O(M)."""
    inf = objective.influence
    c = int(candidate)
    if not (0 <= c < inf.n_candidates):
        raise IndexError("candidate id out of range [0, %d)" % inf.n_candidates)
    ids, base, oar_base, base_hard = _base_state(objective, selection, None)
    if c in ids:
        return 0.0
    w = inf.target.weights
    total = float(w.sum())
    rx = objective.rx_cgy
    row = inf.dose[c]
    if total > 0.0:
        v100 = float((row >= _f32_threshold(rx - base)) @ w) / total
        v200 = float((row >= _f32_threshold(2.0 * rx - base)) @ w) / total
    else:
        v100 = v200 = 0.0
    pen = float(_pen_rows(objective, oar_base, np.array([c]))[0])
    tail = float(_tail_term(objective, row.astype(np.float64) + base)[0])
    return v100 + tail - _hot_penalty(objective, v200) - pen - base_hard


def soft_gains_all(objective: Objective, selection, dose_vec: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """``(C,)`` float: ``soft(selection + [c]) - soft(selection)`` for every
    candidate (0 for selected ones); chunked like :func:`gains_all`."""
    inf = objective.influence
    ids, base, oar_base, _ = _base_state(objective, selection, dose_vec)
    w = inf.target.weights
    total = float(w.sum())
    rx, tau = objective.rx_cgy, objective.tau_cgy
    _v100, v200_b = _coverage_terms(objective, base)
    dmax_base = {name: (float(v.max()) if v.size else 0.0) for name, v in oar_base.items()}
    base_soft = soft_coverage(objective, base) + float(_tail_term(objective, base)[0]) \
        - _hot_penalty(objective, v200_b) - _oar_penalty(objective, dmax_base)
    thr200 = _f32_threshold(2.0 * rx - base)
    c_n = inf.n_candidates
    out = np.empty(c_n, dtype=float)
    for i0 in range(0, c_n, GAINS_CHUNK_ROWS):
        i1 = min(i0 + GAINS_CHUNK_ROWS, c_n)
        block = inf.dose[i0:i1]                                  # (k, M) float32 view
        tail = 0.0
        if total > 0.0:
            tot = block.astype(np.float64) + base[None, :]
            s = expit((tot - rx) / tau)
            cov = (s @ w) / total
            v200 = ((block >= thr200[None, :]) @ w) / total
            if objective.lambda_tail:
                tail = _tail_term(objective, tot)
        else:
            cov = np.zeros(i1 - i0)
            v200 = np.zeros(i1 - i0)
        hot = objective.lambda_hot * np.maximum(0.0, v200 - objective.v200_tol)
        out[i0:i1] = cov + tail - hot - _pen_rows(objective, oar_base, np.arange(i0, i1)) \
            - base_soft
    out[ids] = 0.0
    return out


# ----------------------------------------------------------- constructors
def make_objective(influence: InfluenceMatrix, conflicts: ConflictGraph,
                   **weights) -> Objective:
    """Construct an :class:`Objective`.

    ``weights`` may override ``lambda_hot``, ``v200_tol``, ``lambda_oar``,
    ``tau_cgy``, ``lambda_tail``, ``tail_q`` and ``rx_cgy``; ``rx_cgy`` defaults to ``influence.rx_cgy``
    (the prescription the matrix was built for).  Unknown keys raise
    ``TypeError``.
    """
    bad = set(weights) - set(_OBJECTIVE_WEIGHT_KEYS)
    if bad:
        raise TypeError("make_objective: unknown weights %s (allowed: %s)"
                        % (sorted(bad), list(_OBJECTIVE_WEIGHT_KEYS)))
    if conflicts is not None and conflicts.n != influence.n_candidates:
        raise ValueError("conflicts.n (%d) != influence candidates (%d)"
                         % (conflicts.n, influence.n_candidates))
    kw = dict(weights)
    kw.setdefault("rx_cgy", influence.rx_cgy)
    return Objective(influence=influence, conflicts=conflicts, **kw)


def evaluate(objective: Objective, selection) -> Dict[str, Any]:
    """``{"hard", "soft", "metrics", "feasible"}`` for ``selection``."""
    ids = _as_ids(selection, objective.influence.n_candidates)
    feasible = True
    if objective.conflicts is not None:
        feasible = bool(objective.conflicts.is_feasible(ids))
    return {
        "hard": float(hard(objective, ids)),
        "soft": float(soft(objective, ids)),
        "metrics": metrics(objective, ids),
        "feasible": feasible,
    }


__all__ = [
    "weighted_quantile", "tail_rows", "tail_mean", "metrics_from_dose", "dose_of", "metrics", "hard",
    "soft", "soft_coverage", "gain", "gains_all", "soft_gains_all",
    "make_objective", "evaluate", "GAINS_CHUNK_ROWS",
]
