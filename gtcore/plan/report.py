"""Final reporting (section 3 G). Owner: A5 Validation, branch ``plan/validation``.

``final_report(mesh, tiles, rx_cgy, target, cavity_mask, cavity_affine,
interference, grid_mm, margin_mm, solver_result, parameters)
-> OptimizeReport`` for any configuration (optimized, as-implanted, manual,
random):

1. ``compute_dose_grid``, exact kernel, ``grid_mm`` spacing, bounds = mesh
   bounds +- ``margin_mm`` (the margin is clipped so the grid stays at or
   below ``MAX_GRID_VOXELS``; the bounds actually used are recorded);
   optional second run with the interference model (seed capsules only, as
   the planner does), the difference reported;
2. shell metrics via ``shell_report`` (wall, +5, +10 mm) plus weighted
   metrics on the FULL target sampled trilinearly from the grid; rind DVH
   via ``rind_mask`` (5 mm) when a cavity mask exists;
3. conflicts re-verified with ``find_overlapping_tiles``; shadowing listed
   via ``find_shadowing_tiles`` (a failure there becomes a note, never a
   crash);
4. ``OptimizeReport`` with gtcore version, parameters, seed, wall-clock
   time and per-stage runtime; ``to_json`` / ``to_csv`` write it out.

All reported metrics come from the dose grid, never from the influence
matrix.  Nothing here touches solver internals.

Weighted statistics: ``weighted_quantile`` is taken from
``gtcore.plan.objective`` (A2) when that module provides it, otherwise the
local fallback below (same convention: cumulative-weight midpoints,
linear interpolation).

Optional parameters read from ``parameters``:

``sk_per_seed_u``
    Air-kerma strength per seed [U] for the grid (default: the engine's
    ``DEFAULT_SK_U``).  Lets a sensitivity study scale S_K without touching
    the frozen signature.
``seed``
    Recorded as the report's ``seed`` when no ``solver_result`` carries one.
``shell_offsets_mm``
    Shell offsets for ``shell_report`` (default ``(0, 5, 10)``).
``rind_depth_mm``
    Rind thickness for ``rind_mask`` (default 5 mm).
``shadowing``
    ``False`` to skip the (dose-based) shadowing check.
"""
from __future__ import annotations

import functools
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import DEFAULT_RX_CGY, OptimizeReport, SolverResult, TargetSet, \
    TARGET_SHELL_OFFSET_MM
from ..interact import PlacedTile, find_overlapping_tiles, tiles_to_seed_arrays

__all__ = ["final_report", "weighted_stats", "weighted_quantile_fallback",
           "weighted_dvh_curve", "grid_bounds", "MAX_GRID_VOXELS", "DEFAULT_SHELL_OFFSETS_MM"]

MAX_GRID_VOXELS = 30_000_000
# Upper bound on the dose grid: 30 M float64 voxels is ~240 MB and a few
# minutes of exact-kernel time for a 40-seed implant; the margin is shrunk
# (never the spacing) to stay below it.

DEFAULT_SHELL_OFFSETS_MM = (0.0, 5.0, 10.0)
RIND_DEPTH_MM = 5.0


# ------------------------------------------------------------ small helpers
def weighted_quantile_fallback(values, weights, q) -> float:
    """Weighted ``q``-quantile (q in [0, 1]) by cumulative-weight midpoints.

    Sorts ``values``, places each at the midpoint of its cumulative weight
    interval (normalized to [0, 1]) and interpolates linearly; with unit
    weights this is within one sample of ``numpy.percentile``.  Empty input
    -> 0.0.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0 or w.sum() <= 0:
        return 0.0
    order = np.argsort(v, kind="stable")
    v = v[order]
    w = w[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return float(np.interp(float(q), cum, v))


def _weighted_quantile():
    """A2's ``weighted_quantile`` when present, else the local fallback."""
    try:
        from . import objective as _obj
        fn = getattr(_obj, "weighted_quantile", None)
    except Exception:                                   # pragma: no cover
        fn = None
    return fn if callable(fn) else weighted_quantile_fallback


def weighted_stats(doses, weights, rx_cgy: float) -> Dict[str, float]:
    """Weighted V100 / V150 / V200 (fractions of total weight), D90 / D50
    (weighted 10th / 50th percentile, cGy), Dmin / Dmax / Dmean (weighted),
    ``n`` and ``total_weight``.  Empty input -> zeros."""
    d = np.asarray(doses, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    rx = float(rx_cgy)
    if d.size == 0 or w.sum() <= 0:
        return {k: 0.0 for k in ("V100", "V150", "V200", "D90", "D50", "Dmin",
                                 "Dmax", "Dmean", "n", "total_weight")}
    tot = float(w.sum())
    wq = _weighted_quantile()
    return {
        "V100": float(w[d >= rx].sum() / tot),
        "V150": float(w[d >= 1.5 * rx].sum() / tot),
        "V200": float(w[d >= 2.0 * rx].sum() / tot),
        "D90": float(wq(d, w, 0.10)),
        "D50": float(wq(d, w, 0.50)),
        "Dmin": float(d.min()),
        "Dmax": float(d.max()),
        "Dmean": float((d * w).sum() / tot),
        "n": float(d.size),
        "total_weight": tot,
    }


def grid_bounds(mesh_bounds, margin_mm: float, grid_mm: float,
                max_voxels: int = MAX_GRID_VOXELS) -> Tuple[np.ndarray, float]:
    """``(bounds (2, 3), margin_used_mm)``: mesh bounds padded by ``margin_mm``,
    the margin reduced (never below 0) until the voxel count fits
    ``max_voxels`` at ``grid_mm`` spacing."""
    b = np.asarray(mesh_bounds, dtype=float).reshape(2, 3)
    margin = float(max(0.0, margin_mm))
    spacing = float(grid_mm)

    def n_vox(m):
        ext = (b[1] - b[0]) + 2.0 * m
        return int(np.prod(np.floor(ext / spacing + 1e-9).astype(int) + 1))

    while margin > 0.0 and n_vox(margin) > int(max_voxels):
        margin = max(0.0, margin - max(1.0, spacing))
    return np.vstack([b[0] - margin, b[1] + margin]), margin


@functools.lru_cache(maxsize=1)
def _git_commit() -> Optional[str]:
    """Short commit hash of the checkout containing this file (None if no git)."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5.0)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def weighted_dvh_curve(doses, weights, levels_cgy) -> np.ndarray:
    """Weighted cumulative DVH: fraction of total weight at >= each level."""
    d = np.asarray(doses, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    lv = np.asarray(levels_cgy, dtype=float).reshape(-1)
    if d.size == 0 or w.sum() <= 0:
        return np.zeros(lv.shape)
    order = np.argsort(d)
    d, w = d[order], w[order]
    tail = np.cumsum(w[::-1])[::-1]              # weight at >= d[i]
    idx = np.searchsorted(d, lv, side="left")
    out = np.where(idx < d.size, tail[np.minimum(idx, d.size - 1)], 0.0)
    return out / float(w.sum())


def _target_metrics(dose_vol, target: TargetSet, rx: float) -> Dict[str, Any]:
    from ..dose.dvh import DEFAULT_CURVE_FRACTIONS, sample_doses
    d = sample_doses(dose_vol, target.points)
    st = weighted_stats(d, target.weights, rx)
    st["name"] = target.name
    fr = np.asarray(DEFAULT_CURVE_FRACTIONS, dtype=float)
    st["curve_x"] = fr.copy()
    st["curve_y"] = weighted_dvh_curve(d, target.weights, fr * rx)
    return st


def _shell_entries(dose_vol, mesh, rx: float, offsets) -> Dict[float, Dict[str, Any]]:
    from ..dose.dvh import shell_report
    rep = shell_report(dose_vol, mesh, rx, offsets_mm=tuple(float(o) for o in offsets))
    out: Dict[float, Dict[str, Any]] = {}
    for off, entry in rep.items():
        row = dict(entry["stats"])
        row["curve_x"] = np.asarray(entry["curve_x"], dtype=float)
        row["curve_y"] = np.asarray(entry["curve_y"], dtype=float)
        out[float(off)] = row
    return out


# ------------------------------------------------------------------- main
def final_report(mesh, tiles: Sequence[PlacedTile], rx_cgy: float = DEFAULT_RX_CGY,
                 target: Optional[TargetSet] = None, cavity_mask=None,
                 cavity_affine=None, interference: bool = False,
                 grid_mm: float = 1.0, margin_mm: float = 50.0,
                 solver_result: Optional[SolverResult] = None,
                 parameters: Optional[Dict[str, Any]] = None) -> OptimizeReport:
    """Section 3 G report for ``tiles`` on ``mesh`` (see the module docstring).

    ``metrics_grid`` keys: the shell offsets (float, mm) -> ``dvh_stats`` of
    the shell vertices plus ``curve_x`` / ``curve_y``; ``"target"`` -> weighted
    stats on the full ``target`` (default the +5 mm shell with area weights);
    ``"rind"`` -> ``dose_metrics`` on the ``rind_mask`` when a cavity mask is
    given.  ``metrics_grid_interference`` mirrors it with the capsule
    interference model on.  ``parameters`` is copied and extended with
    ``grid_bounds_ras``, ``grid_margin_used_mm``, ``grid_mm``, ``grid_shape``,
    ``kernel``, ``sk_per_seed_u`` and ``git_commit``.
    """
    from ..dose.engine import TG43Engine, compute_dose_grid

    t_start = time.perf_counter()
    runtime: Dict[str, float] = {}
    notes: List[str] = []
    params: Dict[str, Any] = dict(parameters or {})
    rx = float(rx_cgy)
    tiles = list(tiles)

    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        raise ValueError("final_report: mesh is empty or None")
    if float(grid_mm) <= 0.0:
        raise ValueError("final_report: grid_mm must be positive")

    sk = params.get("sk_per_seed_u", None)
    if sk is None:
        sk = TG43Engine.DEFAULT_SK_U
    sk = float(sk)
    params["sk_per_seed_u"] = sk
    offsets = tuple(float(o) for o in params.get("shell_offsets_mm", DEFAULT_SHELL_OFFSETS_MM))
    params["shell_offsets_mm"] = list(offsets)
    rind_depth = float(params.get("rind_depth_mm", RIND_DEPTH_MM))
    do_shadowing = bool(params.get("shadowing", True))

    if target is None:
        target = TargetSet.from_shell(mesh, TARGET_SHELL_OFFSET_MM)

    centers, axes = tiles_to_seed_arrays(tiles)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    axes = np.asarray(axes, dtype=float).reshape(-1, 3)
    if centers.shape[0] == 0:
        notes.append("no tiles: dose grid is identically zero")

    # 1. dose grid ----------------------------------------------------------
    bounds, margin_used = grid_bounds(mesh.bounds, margin_mm, grid_mm)
    if margin_used < float(margin_mm) - 1e-9:
        notes.append("grid margin clipped from %g to %g mm to keep the grid "
                     "<= %d voxels" % (margin_mm, margin_used, MAX_GRID_VOXELS))
    params.update({"grid_mm": float(grid_mm), "grid_margin_mm": float(margin_mm),
                   "grid_margin_used_mm": float(margin_used),
                   "grid_bounds_ras": bounds.tolist(), "kernel": "exact",
                   "git_commit": _git_commit(), "rx_cgy": rx,
                   "n_seeds": int(centers.shape[0])})

    eng = TG43Engine()
    t0 = time.perf_counter()
    dose = compute_dose_grid(centers, axes, bounds, spacing_mm=float(grid_mm),
                             sk_per_seed_u=sk, engine=eng, exact=True)
    runtime["dose_grid"] = time.perf_counter() - t0
    params["grid_shape"] = list(int(s) for s in dose.array.shape)

    # 2. metrics --------------------------------------------------------------
    t0 = time.perf_counter()
    metrics: Dict[Any, Any] = _shell_entries(dose, mesh, rx, offsets)
    runtime["shells"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics["target"] = _target_metrics(dose, target, rx)
    runtime["target"] = time.perf_counter() - t0

    rind = None
    if cavity_mask is not None and cavity_affine is not None:
        t0 = time.perf_counter()
        try:
            from ..dose.metrics import dose_metrics, rind_mask
            rind = rind_mask(dose, np.asarray(cavity_mask, dtype=bool),
                             np.asarray(cavity_affine, dtype=float),
                             depth_mm=rind_depth)
            if rind.any():
                metrics["rind"] = dose_metrics(dose, rind, rx)
                metrics["rind"]["depth_mm"] = rind_depth
            else:
                notes.append("rind: cavity mask does not intersect the dose grid")
        except Exception as exc:                        # pragma: no cover
            notes.append("rind DVH failed: %s: %s" % (type(exc).__name__, exc))
        runtime["rind"] = time.perf_counter() - t0
    elif (cavity_mask is None) != (cavity_affine is None):
        notes.append("rind DVH skipped: both cavity_mask and cavity_affine are needed")

    metrics_interf = None
    if interference and centers.shape[0] > 0:
        t0 = time.perf_counter()
        try:
            from ..dose.interference import InterferenceModel
            model = InterferenceModel.from_implant(centers, axes)   # capsules only
            dose_i = compute_dose_grid(centers, axes, bounds, spacing_mm=float(grid_mm),
                                       sk_per_seed_u=sk, engine=eng, exact=True,
                                       interference=model)
            metrics_interf = _shell_entries(dose_i, mesh, rx, offsets)
            metrics_interf["target"] = _target_metrics(dose_i, target, rx)
            if rind is not None and rind.any():
                from ..dose.metrics import dose_metrics
                metrics_interf["rind"] = dose_metrics(dose_i, rind, rx)
            ref = metrics.get(TARGET_SHELL_OFFSET_MM, metrics["target"])
            alt = metrics_interf.get(TARGET_SHELL_OFFSET_MM, metrics_interf["target"])
            notes.append(
                "interference (seed capsules only, carriers off): +%g mm shell "
                "V100 %.4f -> %.4f (%+.2f pp), D90 %.0f -> %.0f cGy (%+.2f %%)"
                % (TARGET_SHELL_OFFSET_MM, ref["V100"], alt["V100"],
                   100.0 * (alt["V100"] - ref["V100"]), ref["D90"], alt["D90"],
                   100.0 * (alt["D90"] - ref["D90"]) / max(ref["D90"], 1e-9)))
        except Exception as exc:
            notes.append("interference run failed: %s: %s" % (type(exc).__name__, exc))
        runtime["interference"] = time.perf_counter() - t0
    elif interference:
        notes.append("interference requested but there are no seeds")

    # 3. conflicts / shadowing ----------------------------------------------
    t0 = time.perf_counter()
    overlaps = [(int(i), int(j)) for i, j in find_overlapping_tiles(tiles)]
    runtime["overlaps"] = time.perf_counter() - t0
    if overlaps:
        notes.append("%d overlapping tile pair(s) (planner caution): %r"
                     % (len(overlaps), overlaps))

    shadowing: List[Any] = []
    if do_shadowing and len(tiles) >= 2:
        t0 = time.perf_counter()
        try:
            from ..dose.interference import find_shadowing_tiles
            shadowing = [(int(i), int(j), float(p))
                         for i, j, p in find_shadowing_tiles(tiles, engine=eng)]
        except Exception as exc:
            notes.append("shadowing check failed: %s: %s" % (type(exc).__name__, exc))
        runtime["shadowing"] = time.perf_counter() - t0

    # 4. assemble -------------------------------------------------------------
    seed = params.get("seed", None)
    if seed is None and solver_result is not None:
        seed = solver_result.seed
    wall = time.perf_counter() - t_start
    runtime["total"] = wall
    report = OptimizeReport(
        tiles=tiles, solver=solver_result, parameters=params,
        candidate_stats=dict(params.get("candidate_stats", {})),
        metrics_influence=dict(solver_result.metrics) if solver_result is not None else {},
        metrics_grid=metrics, metrics_grid_interference=metrics_interf,
        overlaps=overlaps, shadowing=shadowing, runtime=runtime,
        seed=None if seed is None else int(seed), wall_clock_s=wall, notes=notes)
    return report
