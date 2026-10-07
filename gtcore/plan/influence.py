"""Influence matrix (section 3 B). Owner: A2 Dose, branch ``plan/influence``.

``build_influence(candidates, target, ...)`` evaluates, for every candidate
``c`` and every (subsampled) target point ``m``, the total-decay dose
``D[c, m]`` [cGy] at nominal S_K delivered by candidate ``c``'s own seeds,
through :func:`gtcore.dose.engine.dose_at_points` -- one engine call per
candidate, the tabulated kernel by default, stored as float32.  Dose is
additive over seeds, so the dose of a selection is the row-sum
``D[sel].sum(0)`` (:meth:`InfluenceMatrix.dose_of`), which is what every
objective evaluation and every solver uses.

Design notes
------------
- One ``dose_at_points`` call per candidate was measured at ~1.1 ms per
  candidate for M = 4000 target points with the tabulated kernel (~2.8 ms
  exact), well under the 5 ms budget of the plan, so the candidate loop is
  kept as the simplest correct structure; ``dose_at_points`` already sums
  over the candidate's seeds and chunks over points.  The expensive part of
  the engine is the one-off kernel table (~0.1 s), so a single
  :class:`~gtcore.dose.engine.TG43Engine` is reused across every call.
- The target is subsampled with :meth:`TargetSet.subsample` (weight-
  preserving systematic PPS sampling); ``target_index`` records which
  points were kept.  OAR sets are never subsampled.
- The section 3 B gate (row-sum vs ``compute_dose_grid`` on the same points)
  lives in ``tests/test_plan_influence.py::test_influence_gate``.

Never edit ``gtcore/dose/engine.py``.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from . import (
    DEFAULT_RX_CGY,
    M_OPT_MAX,
    CandidateSet,
    InfluenceMatrix,
    TargetSet,
)

KERNELS = ("tabulated", "exact")


def _points_of(obj) -> np.ndarray:
    """``(M, 3)`` float points from a :class:`TargetSet` or a raw array."""
    pts = obj.points if hasattr(obj, "points") else obj
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def candidate_doses(candidates: CandidateSet, points, sk_per_seed_u: float,
                    engine, exact: bool) -> np.ndarray:
    """``(C, M)`` float32 total-decay dose [cGy] of every candidate at ``points``.

    One :func:`dose_at_points` call per candidate over its valid seeds
    (``n_seeds[c]``); candidates without seeds give a zero row.
    """
    from ..dose.engine import dose_at_points

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    c_n = len(candidates)
    out = np.zeros((c_n, pts.shape[0]), dtype=np.float32)
    if pts.shape[0] == 0:
        return out
    for c in range(c_n):
        k = int(candidates.n_seeds[c])
        if k <= 0:
            continue
        out[c] = dose_at_points(candidates.seed_centers[c, :k],
                                candidates.seed_axes[c, :k], pts,
                                sk_per_seed_u=float(sk_per_seed_u),
                                engine=engine, exact=bool(exact))
    return out


def build_influence(candidates: CandidateSet, target: TargetSet,
                    rx_cgy: float = DEFAULT_RX_CGY,
                    sk_per_seed_u: Optional[float] = None, m_opt: int = M_OPT_MAX,
                    oars: Optional[Dict[str, TargetSet]] = None,
                    oar_limits: Optional[Dict[str, float]] = None, engine=None,
                    kernel: str = "tabulated", rng_seed: int = 0) -> InfluenceMatrix:
    """Dose [cGy] from every candidate at every (subsampled) target point.

    Parameters
    ----------
    candidates : CandidateSet
    target : TargetSet
        Full target; subsampled to at most ``m_opt`` points with
        ``target.subsample(m_opt, rng_seed)`` (pass ``m_opt >= len(target)``
        for no subsampling, as the section 3 B gate does).
    rx_cgy : float
        Prescription recorded on the matrix (metrics reference).
    sk_per_seed_u : float or None
        Air-kerma strength per seed [U]; None -> ``TG43Engine.DEFAULT_SK_U``.
    m_opt : int
        Maximum number of target points kept.
    oars : dict or None
        ``name -> TargetSet`` (or ``(Mo, 3)`` points); evaluated in full.
    oar_limits : dict or None
        ``name -> Dmax limit`` [cGy]; names must be keys of ``oars``.
    engine : TG43Engine or None
        Reused when given (the kernel table is the expensive part).
    kernel : {"tabulated", "exact"}
        ``dose_at_points(exact=False)`` or ``exact=True``.
    rng_seed : int
        Seed of the target subsample.

    Returns
    -------
    InfluenceMatrix with ``dose`` (C, M) float32, ``target`` the subsample,
    ``target_index`` (M,) into the full target, OAR rows, ``kernel`` and
    ``build_seconds``.  Complexity O(C * 4 * M) kernel evaluations.
    """
    from ..dose.engine import TG43Engine

    if kernel not in KERNELS:
        raise ValueError("kernel must be one of %r, got %r" % (KERNELS, kernel))
    t0 = time.perf_counter()
    sk = TG43Engine.DEFAULT_SK_U if sk_per_seed_u is None else float(sk_per_seed_u)
    eng = engine if engine is not None else TG43Engine()
    exact = kernel == "exact"

    sub, index = target.subsample(int(m_opt), int(rng_seed))
    dose = candidate_doses(candidates, sub.points, sk, eng, exact)

    oar_rows: Dict[str, np.ndarray] = {}
    for name, pts in (oars or {}).items():
        oar_rows[str(name)] = candidate_doses(candidates, _points_of(pts), sk,
                                              eng, exact)
    limits: Dict[str, float] = {}
    for name, lim in (oar_limits or {}).items():
        if str(name) not in oar_rows:
            raise KeyError("oar_limits names an OAR not in oars: %r" % (name,))
        limits[str(name)] = float(lim)

    return InfluenceMatrix(
        dose=dose, target=sub, target_index=np.asarray(index, dtype=int),
        rx_cgy=float(rx_cgy), sk_per_seed_u=float(sk), oar=oar_rows,
        oar_limits=limits, kernel=kernel,
        build_seconds=float(time.perf_counter() - t0),
    )


__all__ = ["build_influence", "candidate_doses", "KERNELS"]
