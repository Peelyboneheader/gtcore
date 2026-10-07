"""Integration entry points (section 3 H). Owner: A6 Integration, branch ``plan/ui``.

:func:`optimize` and :func:`suggest_next` are the two functions the planner
and the ``gt optimize`` command call; ``gtcore.plan`` re-exports them with
the frozen signatures and forwards extra keywords here.  Everything
algorithmic is delegated to the other ``gtcore.plan`` functions
(``build_candidates``, ``visible_faces``, ``build_influence``,
``build_conflicts``, ``make_objective``, ``evaluate``, ``solve_greedy``,
``solve_local``, ``solve_sa``, ``solve_milp``, ``refine_continuous``,
``final_report``, ``recommend_tile_count``) looked up on the package at
call time, so a monkeypatched or still-stubbed function is honoured and a
``NotImplementedError`` from a branch that has not landed propagates with
its own message (the planner shows it in the status line).

What this module adds on top of the solvers
-------------------------------------------
* **Fixed tiles** (``optimize(..., fixed_tiles=...)``): tiles already on the
  board (the implant recovered from the scan, or everything placed so far)
  are kept as obstacles.  Each becomes one extra pre-selected candidate --
  its own influence row from ``dose_at_points(exact=False)`` and its own
  conflict row against the free candidates -- and the augmented instance is
  handed to ``solve_greedy(fixed=...)``.  Only greedy exposes a fixed set,
  so ``fixed_tiles`` with another solver is a ``ValueError`` (the planner
  falls back to greedy and says so).  Fixed tiles never conflict with each
  other (an overlapping board stays optimizable; the planner's amber
  caution already covers that case).
* **Base dose for ``suggest_next``**: the placed tiles are not candidates,
  so their dose on the influence target is evaluated directly and the
  compatible candidate with the largest hard-objective gain is picked with
  the same weighted P1 formula as ``Objective.hard`` (no OAR term).
* **Candidate / influence caches**: small LRUs so repeated planner calls
  on the same wall do not rebuild (see :func:`cached_candidates`).

Conventions: RAS mm, cGy, deterministic for identical inputs and seeds.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .. import plan as _plan
from ..interact import PlacedTile, find_overlapping_tiles, tiles_to_seed_arrays
from . import (
    CONFLICT_GAP_MM,
    DEFAULT_H_MM,
    DEFAULT_N_SPINS_FULL,
    DEFAULT_N_SPINS_HALF,
    DEFAULT_RX_CGY,
    DETACHED_MM,
    LAMBDA_HOT,
    LAMBDA_OAR,
    LOCAL_RADIUS_MM,
    M_OPT_MAX,
    MILP_TIME_LIMIT_S,
    SA_ALPHA,
    SA_MOVES_PER_TILE_PER_SWEEP,
    SA_N_RESTARTS,
    SA_N_SWEEPS,
    TARGET_SHELL_OFFSET_MM,
    TAU_FRACTION,
    TILE_AREA_CM2,
    V200_TOL,
    CandidateSet,
    ConflictGraph,
    InfluenceMatrix,
    OptimizeReport,
    SolverResult,
    TargetSet,
    TileCountRecommendation,
)

SOLVERS = ("greedy", "local", "sa", "milp", "continuous")

CONTINUOUS_H_MM = 4.0
CONTINUOUS_N_SPINS = 3
# ``solver="continuous"`` (A3's multi-start Nelder-Mead over continuous tile
# poses, ``solve_continuous``) only needs the candidate grid for its greedy
# first start, so a coarser 4 mm / 3-spin grid is used unless the caller
# passes other values (coordinator note, 2026-10-07).

CONTINUOUS_TIME_BUDGET_S = 60.0
# Default wall-time budget handed to ``solve_continuous`` (``--budget``).

PLANNER_OVERLAP_THRESHOLD_MM = 1.0
# Must equal ``gtcore.planner.OVERLAP_THRESHOLD_MM``: compatibility of a
# candidate with a placed tile is decided by the planner's own rule so the
# planner never flags an optimizer output as overlapping (section 4 V1).
# Not imported from the planner to keep this module free of the rendering
# front-end; a test pins the two values together.

CANDIDATE_CACHE_SIZE = 4
# Walls a session touches at once: the planner has one, a sweep script a
# handful.  Entries hold a strong reference to their mesh so an ``id()``
# can never be recycled onto a different mesh while the entry lives.

BOUNDING_MARGIN_MM = 1.0
# Slack added to the bounding-sphere prefilter before the exact footprint
# test (the footprint sampler pushes points 1 mm along the normal).


# ------------------------------------------------------------------ logging
def _logger(verbose: bool, log: Optional[Callable[[str], None]]):
    def say(msg: str):
        if verbose:
            print("[gtcore.plan] " + msg)
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass
    return say


# ------------------------------------------------------------------ metrics
def weighted_quantile(values, weights, q: float) -> float:
    """Weighted ``q``-quantile (``q`` in [0, 1]) of ``values``: the smallest
    value whose cumulative weight fraction reaches ``q``."""
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0 or w.sum() <= 0:
        return 0.0
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order])
    k = int(np.searchsorted(cum, float(q) * cum[-1], side="left"))
    return float(v[order][min(k, v.size - 1)])


def target_metrics(dose, weights, rx_cgy: float) -> Dict[str, float]:
    """``V100 / V150 / V200`` (weighted fractions), ``D90`` (weighted 10th
    percentile, cGy), ``Dmean`` and the P1 ``hard`` value (no OAR term) for
    one dose vector on a weighted target."""
    d = np.asarray(dose, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(w.sum())
    rx = float(rx_cgy)
    if d.size == 0 or total <= 0:
        return {"V100": 0.0, "V150": 0.0, "V200": 0.0, "D90": 0.0,
                "Dmean": 0.0, "hard": 0.0}
    v100 = float(w[d >= rx].sum() / total)
    v150 = float(w[d >= 1.5 * rx].sum() / total)
    v200 = float(w[d >= 2.0 * rx].sum() / total)
    return {
        "V100": v100, "V150": v150, "V200": v200,
        "D90": weighted_quantile(d, w, 0.10),
        "Dmean": float((w * d).sum() / total),
        "hard": v100 - LAMBDA_HOT * max(0.0, v200 - V200_TOL),
    }


def _sk_of(influence: Optional[InfluenceMatrix]) -> float:
    from ..dose.engine import TG43Engine
    sk = float("nan") if influence is None else float(influence.sk_per_seed_u)
    return sk if np.isfinite(sk) else float(TG43Engine.DEFAULT_SK_U)


def tiles_dose(tiles: Sequence[PlacedTile], points, sk_per_seed_u: Optional[float] = None
               ) -> np.ndarray:
    """Total-decay dose [cGy] of ``tiles`` at ``points`` (tabulated kernel)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centers, axes = tiles_to_seed_arrays(list(tiles))
    if centers.shape[0] == 0 or pts.shape[0] == 0:
        return np.zeros(pts.shape[0], dtype=float)
    from ..dose.engine import dose_at_points
    sk = _sk_of(None) if sk_per_seed_u is None else float(sk_per_seed_u)
    return np.asarray(dose_at_points(centers, axes, pts, sk_per_seed_u=sk,
                                     exact=False), dtype=float).reshape(-1)


def evaluate_tiles(mesh, tiles: Sequence[PlacedTile], rx_cgy: float = DEFAULT_RX_CGY,
                   target: Optional[TargetSet] = None, m_max: int = M_OPT_MAX,
                   rng_seed: int = 0) -> Dict[str, float]:
    """Influence-style metrics of any tile list on ``target`` (default the
    +5 mm shell, subsampled to ``m_max`` points): the planner's before /
    after readout, independent of the solver modules."""
    if target is None:
        target = TargetSet.from_shell(mesh, TARGET_SHELL_OFFSET_MM)
    sub, _idx = target.subsample(int(m_max), rng_seed=int(rng_seed))
    d = tiles_dose(tiles, sub.points)
    out = target_metrics(d, sub.weights, rx_cgy)
    out["n_points"] = float(len(sub))
    out["n_tiles"] = float(len(list(tiles)))
    out["target"] = sub.name
    return out


def _gains_over_base(dose_matrix, base, weights, rx_cgy: float, mask,
                     chunk: int = 256) -> Tuple[np.ndarray, float]:
    """Hard-objective gain of adding each masked candidate to ``base``.

    Returns ``(gains (C,), hard_before)``; unmasked candidates get ``-inf``.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(w.sum())
    rx = float(rx_cgy)
    base = np.asarray(base, dtype=float).reshape(-1)
    h0 = target_metrics(base, w, rx)["hard"]
    c = int(dose_matrix.shape[0])
    gains = np.full(c, -np.inf, dtype=float)
    ids = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(c))
    if total <= 0:
        gains[ids] = 0.0
        return gains, h0
    for i0 in range(0, ids.size, int(chunk)):
        sel = ids[i0:i0 + int(chunk)]
        tot = np.asarray(dose_matrix[sel], dtype=np.float64) + base[None, :]
        v100 = ((tot >= rx) @ w) / total
        v200 = ((tot >= 2.0 * rx) @ w) / total
        gains[sel] = v100 - LAMBDA_HOT * np.maximum(0.0, v200 - V200_TOL) - h0
    return gains, h0


# ------------------------------------------------------------ compatibility
def _bounding_radius_tile(tile: PlacedTile) -> Tuple[np.ndarray, float]:
    center = np.asarray(tile.corners_ras, dtype=float).mean(axis=0)
    pts = np.vstack([tile.corners_ras, tile.seed_centers,
                     tile.anchor_ras.reshape(1, 3)])
    return center, float(np.linalg.norm(pts - center[None, :], axis=1).max())


def _bounding_radii(candidates: CandidateSet) -> Tuple[np.ndarray, np.ndarray]:
    centers = candidates.corners.mean(axis=1)
    r_corners = np.linalg.norm(candidates.corners - centers[:, None, :], axis=2).max(axis=1)
    d_seeds = np.linalg.norm(candidates.seed_centers - centers[:, None, :], axis=2)
    r_seeds = np.where(np.isnan(d_seeds), 0.0, d_seeds).max(axis=1)  # NaN = padding
    r_anchor = np.linalg.norm(candidates.anchors - centers, axis=1)
    return centers, np.maximum(np.maximum(r_corners, r_seeds), r_anchor)


def compatible_with_placed(candidates: CandidateSet, placed: Sequence[PlacedTile],
                           threshold_mm: float = PLANNER_OVERLAP_THRESHOLD_MM
                           ) -> np.ndarray:
    """``(C,)`` bool: candidates overlapping NO placed tile under the
    planner's rule (``find_overlapping_tiles``), after a bounding-sphere
    prefilter so the exact footprint test only runs for close pairs."""
    c = len(candidates)
    mask = np.ones(c, dtype=bool)
    placed = list(placed)
    if not placed or c == 0:
        return mask
    centers, radii = _bounding_radii(candidates)
    for tile in placed:
        p_center, p_r = _bounding_radius_tile(tile)
        d = np.linalg.norm(centers - p_center[None, :], axis=1)
        close = np.flatnonzero(mask & (d <= radii + p_r + float(threshold_mm)
                                       + BOUNDING_MARGIN_MM))
        for k in close:
            if find_overlapping_tiles([tile, candidates.tiles[int(k)]],
                                      threshold_mm=float(threshold_mm)):
                mask[k] = False
    return mask


# ------------------------------------------------------------------- caches
_candidate_cache: "OrderedDict[tuple, tuple]" = OrderedDict()   # key -> (mesh, CandidateSet)
_influence_cache: "OrderedDict[tuple, tuple]" = OrderedDict()   # key -> (candidates, InfluenceMatrix)


def _eligible_key(eligible_faces):
    if eligible_faces is None:
        return None
    m = np.asarray(eligible_faces, dtype=bool).reshape(-1)
    return (int(m.size), int(m.sum()),
            hashlib.sha1(np.packbits(m).tobytes()).hexdigest())


def candidate_key(mesh, h_mm: float, n_spins: Optional[int], kinds: Sequence[str],
                  eligible_faces=None, rng_seed: int = 0) -> tuple:
    """Cache key of a candidate build: ``(id(mesh), faces.shape, h, n_spins,
    kinds, eligibility hash, rng_seed)``.  ``id(mesh)`` is only trusted while
    the cached entry still holds that very mesh object."""
    return (id(mesh), tuple(int(s) for s in np.asarray(mesh.faces).shape),
            float(h_mm), None if n_spins is None else int(n_spins),
            tuple(str(k) for k in kinds), _eligible_key(eligible_faces), int(rng_seed))


def cached_candidates(mesh, h_mm: float = DEFAULT_H_MM, n_spins: Optional[int] = None,
                      kinds: Sequence[str] = ("full",), eligible_faces=None,
                      rng_seed: int = 0) -> CandidateSet:
    """``build_candidates`` behind a small LRU keyed by :func:`candidate_key`.

    Repeated planner calls (O then N then N ...) on the same wall reuse the
    conformed candidates; a different mesh object, face count, spacing,
    spin count, kind set or eligibility mask rebuilds.  The entry keeps the
    mesh alive, so a recycled ``id()`` cannot alias a new mesh; the check
    ``entry.mesh is mesh`` makes that explicit.  :func:`clear_cache` empties
    both caches (e.g. after editing a mesh in place).
    """
    key = candidate_key(mesh, h_mm, n_spins, kinds, eligible_faces, rng_seed)
    hit = _candidate_cache.get(key)
    if hit is not None and hit[0] is mesh:
        _candidate_cache.move_to_end(key)
        return hit[1]
    cand = _plan.build_candidates(mesh, h_mm=float(h_mm), n_spins=n_spins,
                                  kinds=tuple(kinds), eligible_faces=eligible_faces,
                                  rng_seed=int(rng_seed))
    _candidate_cache[key] = (mesh, cand)
    while len(_candidate_cache) > CANDIDATE_CACHE_SIZE:
        _candidate_cache.popitem(last=False)
    return cand


def _target_key(target: TargetSet) -> tuple:
    pts = target.points
    return (str(target.name), int(len(target)), round(float(target.total_weight), 6),
            round(float(pts.sum()), 6) if len(target) else 0.0)


def cached_influence(candidates: CandidateSet, target: TargetSet,
                     rx_cgy: float = DEFAULT_RX_CGY, oars=None, oar_limits=None,
                     **kw) -> InfluenceMatrix:
    """``build_influence`` behind an LRU keyed on the candidate object, the
    target's signature (name, size, total weight, coordinate sum) and rx.
    Calls with OARs or extra keywords are never cached."""
    if oars or kw:
        return _plan.build_influence(candidates, target, rx_cgy=float(rx_cgy),
                                     oars=oars, oar_limits=oar_limits, **kw)
    key = (id(candidates), len(candidates), _target_key(target), float(rx_cgy))
    hit = _influence_cache.get(key)
    if hit is not None and hit[0] is candidates:
        _influence_cache.move_to_end(key)
        return hit[1]
    infl = _plan.build_influence(candidates, target, rx_cgy=float(rx_cgy))
    _influence_cache[key] = (candidates, infl)
    while len(_influence_cache) > CANDIDATE_CACHE_SIZE:
        _influence_cache.popitem(last=False)
    return infl


def clear_cache() -> None:
    """Drop every cached candidate set and influence matrix."""
    _candidate_cache.clear()
    _influence_cache.clear()


# -------------------------------------------------------------- fixed tiles
def _augment_with_fixed(candidates: CandidateSet, influence: InfluenceMatrix,
                        conflicts: ConflictGraph, fixed: Sequence[PlacedTile],
                        oars: Optional[Dict[str, TargetSet]] = None
                        ) -> Tuple[CandidateSet, InfluenceMatrix, ConflictGraph, np.ndarray]:
    """Append one pre-selected pseudo-candidate per fixed tile.

    Returns ``(candidates, influence, conflicts, fixed_ids)`` where the
    fixed ids are ``C .. C+k-1``.  Fixed tiles conflict with the free
    candidates that overlap them (planner rule) and never with each other.
    """
    fixed = list(fixed)
    c, k = len(candidates), len(fixed)
    if k == 0:
        return candidates, influence, conflicts, np.zeros(0, dtype=int)
    anchor_base = int(candidates.anchor_ids.max()) + 1 if c else 0
    cand = CandidateSet.from_tiles(
        list(candidates.tiles) + fixed,
        spins_deg=np.concatenate([candidates.spins_deg, np.zeros(k)]),
        anchor_ids=np.concatenate([candidates.anchor_ids, anchor_base + np.arange(k)]),
        eligible=np.concatenate([candidates.eligible, np.ones(k, dtype=bool)]),
        method=candidates.method, h_mm=candidates.h_mm, n_spins=candidates.n_spins,
        n_rejected=dict(candidates.n_rejected), wall_area_mm2=candidates.wall_area_mm2)

    sk = _sk_of(influence)
    rows = np.vstack([tiles_dose([t], influence.target.points, sk)
                      for t in fixed]).astype(np.float32)
    oar = {}
    for name, mat in influence.oar.items():
        if oars is not None and name in oars:
            extra = np.vstack([tiles_dose([t], oars[name].points, sk) for t in fixed])
        else:  # OAR points unavailable: fixed tiles contribute nothing there
            extra = np.zeros((k, mat.shape[1]))
        oar[name] = np.vstack([np.asarray(mat, dtype=np.float32),
                               extra.astype(np.float32)])
    infl = InfluenceMatrix(
        dose=np.vstack([np.asarray(influence.dose, dtype=np.float32), rows]),
        target=influence.target, target_index=influence.target_index,
        rx_cgy=influence.rx_cgy, sk_per_seed_u=influence.sk_per_seed_u, oar=oar,
        oar_limits=dict(influence.oar_limits), kernel=influence.kernel,
        build_seconds=influence.build_seconds)

    x = np.zeros((k, c), dtype=bool)
    for j, tile in enumerate(fixed):
        x[j] = ~compatible_with_placed(candidates, [tile])
    pairs = sp.bmat([[sp.csr_matrix(conflicts.pairs, dtype=bool), sp.csr_matrix(x.T)],
                     [sp.csr_matrix(x), sp.csr_matrix((k, k), dtype=bool)]],
                    format="csr", dtype=bool)
    conf = ConflictGraph(n=c + k, pairs=pairs, cliques=list(conflicts.cliques),
                         gap_mm=conflicts.gap_mm)
    return cand, infl, conf, c + np.arange(k)


# ---------------------------------------------------------------- optimize
def _check_result(res: SolverResult, n_total: int, candidates: CandidateSet,
                  conflicts: ConflictGraph, kinds_required: Dict[str, int],
                  solver: str) -> np.ndarray:
    """Raise ``RuntimeError`` unless ``res`` is a feasible, complete selection."""
    reason = (" (" + res.reason + ")") if res.reason else ""
    if res.status not in ("ok", "optimal", "time_limit"):
        raise RuntimeError("%s solver failed: status %s%s" % (solver, res.status, reason))
    sel = np.asarray(res.selection, dtype=int).reshape(-1)
    if not res.feasible:
        raise RuntimeError("%s solver reported an infeasible selection%s" % (solver, reason))
    if sel.size != int(n_total):
        raise RuntimeError("%s solver returned %d tiles, %d requested%s"
                           % (solver, sel.size, n_total, reason))
    if sel.size and (sel.min() < 0 or sel.max() >= len(candidates)):
        raise RuntimeError("%s solver returned candidate ids outside [0, %d)"
                           % (solver, len(candidates)))
    if np.unique(sel).size != sel.size:
        raise RuntimeError("%s solver selected a candidate twice" % solver)
    if not conflicts.is_feasible(sel):
        raise RuntimeError("%s solver selection violates a conflict" % solver)
    kinds = candidates.kinds[sel]
    for kind, want in kinds_required.items():
        got = int(np.sum(kinds == kind))
        if got != int(want):
            raise RuntimeError("%s solver returned %d %s tiles, %d requested"
                               % (solver, got, kind, want))
    return sel


def _check_tiles(res: SolverResult, tiles: List[PlacedTile], n_total: int,
                 kinds_required: Dict[str, int], solver: str) -> None:
    """Raise ``RuntimeError`` unless a pose-returning solver (``continuous``)
    delivered ``n_total`` non-overlapping tiles of the requested kinds."""
    reason = (" (" + res.reason + ")") if res.reason else ""
    if res.status not in ("ok", "optimal", "time_limit"):
        raise RuntimeError("%s solver failed: status %s%s" % (solver, res.status, reason))
    if not res.feasible:
        raise RuntimeError("%s solver reported an infeasible configuration%s"
                           % (solver, reason))
    if len(tiles) != int(n_total):
        raise RuntimeError("%s solver returned %d tiles, %d requested%s"
                           % (solver, len(tiles), n_total, reason))
    for kind, want in kinds_required.items():
        got = sum(1 for t in tiles if t.kind == kind)
        if got != int(want):
            raise RuntimeError("%s solver returned %d %s tiles, %d requested"
                               % (solver, got, kind, want))
    pairs = find_overlapping_tiles(tiles, threshold_mm=PLANNER_OVERLAP_THRESHOLD_MM)
    if pairs:
        raise RuntimeError("%s solver returned overlapping tiles %r" % (solver, pairs))


def optimize(mesh, n_full: int, n_half: int = 0, rx_cgy: float = DEFAULT_RX_CGY,
             target: Optional[TargetSet] = None, solver: str = "greedy", seed: int = 0,
             h_mm: float = DEFAULT_H_MM, n_spins: Optional[int] = None,
             eligible_faces=None, oars: Optional[Dict[str, TargetSet]] = None,
             oar_limits: Optional[Dict[str, float]] = None, refine: bool = False,
             report: bool = True, verbose: bool = False,
             fixed_tiles: Sequence[PlacedTile] = (),
             candidates: Optional[CandidateSet] = None,
             log: Optional[Callable[[str], None]] = None,
             time_budget_s: float = CONTINUOUS_TIME_BUDGET_S
             ) -> Tuple[List[PlacedTile], OptimizeReport]:
    """Section 3 H entry point: candidates -> influence -> conflicts ->
    objective -> solver -> optional E5 refine -> report.

    ``solver="continuous"`` runs A3's ``solve_continuous`` (multi-start
    Nelder-Mead over continuous poses) from the discrete greedy result on a
    coarser default grid (``CONTINUOUS_H_MM`` / ``CONTINUOUS_N_SPINS`` unless
    ``h_mm`` / ``n_spins`` are given explicitly); it returns tile poses, not
    candidate ids, so ``refine`` is a no-op for it.  Until that branch is
    merged the solver raises ``NotImplementedError`` like any other stub.

    Parameters beyond the frozen signature (all additive):

    fixed_tiles : sequence of PlacedTile
        Tiles already on the board, kept as obstacles and dose contributors
        (greedy only; see the module docstring).  They are NOT in the
        returned list but ARE in ``report.tiles`` and its metrics.
    candidates : CandidateSet or None
        Reuse a prebuilt set instead of :func:`cached_candidates`.
    log : callable or None
        Receives one line per stage (the planner's status line); ``verbose``
        prints the same lines.
    time_budget_s : float
        Wall-time budget for ``solve_continuous`` (ignored by the others).

    Returns ``(tiles, report)`` with ``tiles`` the ``n_full + n_half`` new
    :class:`PlacedTile` objects.  Raises ``ValueError`` for bad inputs or an
    empty candidate set and ``RuntimeError`` when the solver result is not
    a feasible selection of the requested size -- never fewer tiles as
    success (section 4 V8).
    """
    t_all = time.perf_counter()
    say = _logger(verbose, log)
    runtime: Dict[str, float] = {}
    try:
        n_full, n_half = int(n_full), int(n_half)
    except (TypeError, ValueError):
        raise ValueError("n_full / n_half must be integers (got %r, %r)" % (n_full, n_half))
    if n_full < 0 or n_half < 0 or n_full + n_half == 0:
        raise ValueError("need at least one tile: n_full=%d n_half=%d" % (n_full, n_half))
    solver = str(solver).lower()
    if solver not in SOLVERS:
        raise ValueError("unknown solver %r (choose from %s)" % (solver, ", ".join(SOLVERS)))
    fixed = list(fixed_tiles or ())
    if fixed and solver != "greedy":
        raise ValueError("fixed_tiles are honoured by the greedy solver only "
                         "(solve_local / solve_sa / solve_milp / solve_continuous take "
                         "no fixed set); got solver=%r with %d fixed tiles"
                         % (solver, len(fixed)))
    continuous = solver == "continuous"
    solve_continuous = getattr(_plan, "solve_continuous", None)
    if continuous and solve_continuous is None:
        raise NotImplementedError("solve_continuous: implemented on branch plan/solvers "
                                  "(not merged into this checkout yet)")
    if continuous and float(h_mm) == DEFAULT_H_MM and n_spins is None:
        h_mm, n_spins = CONTINUOUS_H_MM, CONTINUOUS_N_SPINS   # coarse start grid
    rx = float(rx_cgy)
    seed = int(seed)

    t0 = time.perf_counter()
    if target is None:
        target = TargetSet.from_shell(mesh, TARGET_SHELL_OFFSET_MM)
    runtime["target"] = time.perf_counter() - t0
    kinds = ("full",) + (("half",) if n_half > 0 else ())
    say("target %s: %d points, %.0f mm^2" % (target.name, len(target), target.total_weight))

    t0 = time.perf_counter()
    cand = candidates if candidates is not None else cached_candidates(
        mesh, h_mm=h_mm, n_spins=n_spins, kinds=kinds, eligible_faces=eligible_faces)
    runtime["candidates"] = time.perf_counter() - t0
    if len(cand) == 0:
        raise ValueError("no candidate placement survived on this wall "
                         "(rejected: %r)" % (dict(cand.n_rejected),))
    say("candidates: %d (%s, h %.1f mm, %d spins) in %.1f s"
        % (len(cand), cand.method, cand.h_mm, cand.n_spins, runtime["candidates"]))

    t0 = time.perf_counter()
    infl = cached_influence(cand, target, rx_cgy=rx, oars=oars, oar_limits=oar_limits)
    runtime["influence"] = time.perf_counter() - t0
    say("influence: %d x %d (%s) in %.1f s" % (infl.n_candidates, infl.n_targets,
                                               infl.kernel, runtime["influence"]))

    t0 = time.perf_counter()
    conf = _plan.build_conflicts(cand)
    runtime["conflicts"] = time.perf_counter() - t0
    say("conflicts: %d pairs, %d cliques in %.1f s"
        % (conf.count_pairs(), len(conf.cliques), runtime["conflicts"]))

    c_free = len(cand)
    cand_s, infl_s, conf_s, fixed_ids = _augment_with_fixed(cand, infl, conf, fixed, oars)
    n_fixed_full = sum(1 for t in fixed if t.kind == "full")
    n_fixed_half = len(fixed) - n_fixed_full
    kinds_required = {"full": n_full + n_fixed_full}
    if n_half > 0 or n_fixed_half > 0:
        kinds_required["half"] = n_half + n_fixed_half
    n_total = n_full + n_half + len(fixed)

    objective = _plan.make_objective(infl_s, conf_s, rx_cgy=rx)
    # the solvers read candidate kinds/anchors from the objective when present
    # (kinds_required needs them; local/SA moves use the anchors)
    objective.candidates = cand_s

    t0 = time.perf_counter()
    res = _plan.solve_greedy(objective, n_total, fixed=list(int(i) for i in fixed_ids),
                             kinds_required=kinds_required)
    if solver == "local":
        res = _plan.solve_local(objective, n_total, start=res.selection,
                                candidates=cand_s)
    elif solver == "sa":
        res = _plan.solve_sa(objective, n_total, seed=seed, start=res.selection,
                             candidates=cand_s)
    elif solver == "milp":
        res = _plan.solve_milp(objective, n_total)
    cont_tiles = None
    if continuous:
        cont_tiles, res = solve_continuous(
            mesh, cand_s, target, rx, n_total, seed=seed,
            time_budget_s=float(time_budget_s), start=res.selection,
            kinds_required=kinds_required)
        cont_tiles = list(cont_tiles)
    runtime["solver"] = time.perf_counter() - t0
    say("solver %s: objective %.4f, status %s in %.1f s"
        % (res.solver or solver, res.objective, res.status, runtime["solver"]))
    if continuous:
        sel = _check_tiles(res, cont_tiles, n_total, kinds_required, solver)
        new_ids = np.zeros(0, dtype=int)
        tiles = cont_tiles
    else:
        sel = _check_result(res, n_total, cand_s, conf_s, kinds_required, solver)
        new_ids = sel[sel < c_free]
        tiles = cand.tiles_of(new_ids)

    refine_info = None
    if refine and continuous:
        say("refine: skipped (continuous solver output is already continuous)")
    elif refine:
        t0 = time.perf_counter()
        tiles, refine_info = _plan.refine_continuous(mesh, cand, new_ids, target, rx_cgy=rx)
        tiles = list(tiles)
        runtime["refine"] = time.perf_counter() - t0
        say("refine: %r in %.1f s" % (refine_info, runtime["refine"]))

    all_tiles = fixed + list(tiles)
    params: Dict[str, Any] = {
        "n_full": n_full, "n_half": n_half, "n_fixed": len(fixed), "kinds": list(kinds),
        "rx_cgy": rx, "target": target.name, "target_points": int(len(target)),
        "target_shell_offset_mm": TARGET_SHELL_OFFSET_MM,
        "h_mm": float(h_mm),
        "n_spins": (DEFAULT_N_SPINS_FULL if n_spins is None else int(n_spins)),
        "n_spins_half": (DEFAULT_N_SPINS_HALF if n_spins is None else int(n_spins)),
        "detached_mm": DETACHED_MM, "conflict_gap_mm": CONFLICT_GAP_MM,
        "m_opt_max": M_OPT_MAX, "tau_fraction": TAU_FRACTION, "tau_cgy": TAU_FRACTION * rx,
        "lambda_hot": LAMBDA_HOT, "v200_tol": V200_TOL, "lambda_oar": LAMBDA_OAR,
        "solver": solver, "seed": seed, "local_radius_mm": LOCAL_RADIUS_MM,
        "sa_alpha": SA_ALPHA, "sa_moves_per_tile_per_sweep": SA_MOVES_PER_TILE_PER_SWEEP,
        "sa_n_sweeps": SA_N_SWEEPS, "sa_n_restarts": SA_N_RESTARTS,
        "milp_time_limit_s": MILP_TIME_LIMIT_S, "refine": bool(refine),
        "eligible_faces": (None if eligible_faces is None
                           else int(np.asarray(eligible_faces, dtype=bool).sum())),
        "oars": sorted(oars.keys()) if oars else [],
        "oar_limits": dict(oar_limits) if oar_limits else {},
        "planner_overlap_threshold_mm": PLANNER_OVERLAP_THRESHOLD_MM,
    }
    stats: Dict[str, Any] = {
        "n_candidates": int(len(cand)), "n_rejected": dict(cand.n_rejected),
        "method": cand.method, "h_mm": float(cand.h_mm), "n_spins": int(cand.n_spins),
        "wall_area_mm2": float(cand.wall_area_mm2),
        "n_conflict_pairs": int(conf.count_pairs()), "n_cliques": int(len(conf.cliques)),
        "influence_build_s": float(infl.build_seconds), "influence_kernel": infl.kernel,
        "n_fixed": len(fixed),
    }
    metrics_influence: Dict[str, Any] = dict(res.metrics) if res.metrics else {}
    if not metrics_influence:
        try:
            if continuous:
                metrics_influence = evaluate_tiles(mesh, all_tiles, rx, target=target)
            else:
                metrics_influence = dict(_plan.evaluate(objective, sel).get("metrics", {}))
        except NotImplementedError:
            metrics_influence = {}
    metrics_influence["hard"] = float(res.objective)
    params["time_budget_s"] = float(time_budget_s) if continuous else None

    if report:
        t0 = time.perf_counter()
        rep = _plan.final_report(mesh, all_tiles, rx_cgy=rx, target=target,
                                 solver_result=res, parameters=params)
        runtime["report"] = time.perf_counter() - t0
        if not rep.tiles:
            rep.tiles = list(all_tiles)
        if rep.solver is None:
            rep.solver = res
        merged = dict(params)
        merged.update(rep.parameters or {})
        rep.parameters = merged
        rep.candidate_stats = dict(rep.candidate_stats or {}, **stats)
        if not rep.metrics_influence:
            rep.metrics_influence = metrics_influence
        rep.runtime = dict(rep.runtime or {}, **runtime)
        say("report in %.1f s" % runtime["report"])
    else:
        rep = OptimizeReport(
            tiles=list(all_tiles), solver=res, parameters=params, candidate_stats=stats,
            metrics_influence=metrics_influence,
            overlaps=[(int(i), int(j)) for i, j in find_overlapping_tiles(
                all_tiles, threshold_mm=PLANNER_OVERLAP_THRESHOLD_MM)],
            runtime=runtime)
        rep.notes.append("report=False: metrics are influence-matrix estimates "
                         "on %s, not dose-grid values" % target.name)
    rep.seed = seed
    rep.wall_clock_s = time.perf_counter() - t_all
    rep.runtime.setdefault("total", rep.wall_clock_s)
    if refine_info is not None:
        rep.notes.append("refine_continuous: %r" % (refine_info,))
    if fixed:
        rep.notes.append("%d fixed tile(s) kept as obstacles; report.tiles and "
                         "its metrics include them, the returned list does not"
                         % len(fixed))
    say("done: %d tiles in %.1f s" % (len(tiles), rep.wall_clock_s))
    return list(tiles), rep


# ------------------------------------------------------------ suggest_next
def suggest_next(mesh, placed_tiles: Sequence[PlacedTile], rx_cgy: float = DEFAULT_RX_CGY,
                 target: Optional[TargetSet] = None, kind: str = "full",
                 h_mm: float = DEFAULT_H_MM, n_spins: Optional[int] = None,
                 eligible_faces=None, candidates: Optional[CandidateSet] = None
                 ) -> Tuple[PlacedTile, Dict[str, Any]]:
    """One greedy step given the current board.

    The placed tiles are not candidates: their dose on the influence target
    is evaluated directly (``dose_at_points(exact=False)``), every candidate
    of ``kind`` that overlaps none of them (planner rule, bounding-sphere
    prefilter) is scored by its hard-objective gain over that base, and the
    best one (ties -> lowest id) is returned with ``{"gain", "hard_before",
    "hard_after", "V100_before", "V100_after", "D90_before", "D90_after",
    "candidate_id", "n_candidates", "n_compatible", "n_placed", "seconds",
    "kind", "target"}``.  Raises ``ValueError`` when no compatible candidate
    exists (the reason names the counts).
    """
    t0 = time.perf_counter()
    kind = str(kind)
    if kind not in ("full", "half"):
        raise ValueError("kind must be 'full' or 'half', got %r" % (kind,))
    placed = list(placed_tiles or ())
    rx = float(rx_cgy)
    if target is None:
        target = TargetSet.from_shell(mesh, TARGET_SHELL_OFFSET_MM)
    cand = candidates if candidates is not None else cached_candidates(
        mesh, h_mm=h_mm, n_spins=n_spins, kinds=(kind,), eligible_faces=eligible_faces)
    if len(cand) == 0:
        raise ValueError("suggest_next: no candidate placement survived on this wall "
                         "(rejected: %r)" % (dict(cand.n_rejected),))
    infl = cached_influence(cand, target, rx_cgy=rx)
    pts, w = infl.target.points, infl.target.weights
    base = tiles_dose(placed, pts, _sk_of(infl))
    mask = (cand.kinds == kind) & cand.eligible & compatible_with_placed(cand, placed)
    n_compat = int(mask.sum())
    if n_compat == 0:
        raise ValueError("suggest_next: none of the %d %s candidates is compatible with "
                         "the %d placed tile(s) -- the wall is full or ineligible"
                         % (int(np.sum(cand.kinds == kind)), kind, len(placed)))
    gains, h0 = _gains_over_base(infl.dose, base, w, rx, mask)
    best = int(np.argmax(gains))          # first maximum = lowest id on ties
    tile = cand.tiles[best]
    m0 = target_metrics(base, w, rx)
    m1 = target_metrics(base + np.asarray(infl.dose[best], dtype=float), w, rx)
    info = {
        "gain": float(gains[best]), "hard_before": float(h0), "hard_after": float(m1["hard"]),
        "V100_before": m0["V100"], "V100_after": m1["V100"],
        "D90_before": m0["D90"], "D90_after": m1["D90"],
        "V200_before": m0["V200"], "V200_after": m1["V200"],
        "candidate_id": best, "n_candidates": int(len(cand)), "n_compatible": n_compat,
        "n_placed": len(placed), "seconds": time.perf_counter() - t0, "kind": kind,
        "target": infl.target.name, "n_target_points": int(infl.n_targets),
    }
    return tile, info


# ------------------------------------------------------- recommendation
def recommended_count(mesh, eligible_faces=None, **kw
                      ) -> Tuple[TileCountRecommendation, str]:
    """``recommend_tile_count`` plus a one-line status-bar string."""
    rec = _plan.recommend_tile_count(mesh, eligible_faces=eligible_faces, **kw)
    line = ("recommended %d tiles (treatable %.1f cm^2 at %g cm^2 per tile; "
            "ellipsoid estimate %d; manufacturer rule, see docs)"
            % (rec.n_tiles, rec.treatable_area_mm2 / 100.0, TILE_AREA_CM2,
               rec.n_tiles_ellipsoid))
    return rec, line


__all__ = [
    "SOLVERS", "PLANNER_OVERLAP_THRESHOLD_MM", "CANDIDATE_CACHE_SIZE",
    "weighted_quantile", "target_metrics", "tiles_dose", "evaluate_tiles",
    "compatible_with_placed", "candidate_key", "cached_candidates",
    "cached_influence", "clear_cache", "optimize", "suggest_next",
    "recommended_count",
]
