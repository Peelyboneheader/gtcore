"""Validation campaign for the opt-in placement optimizer ``gtcore.plan``
(docs/plan-tile-optimize.md section 4, V2-V8).

Outputs go to ``output/validation_optimize/`` (CSV tables, PNG figures,
``runs.csv`` log, pickled caches under ``cache/``); the script prints a
Markdown block ready to paste into ``docs/optimize-notes.md``.

PRIMARY ENDPOINT (declared before any run; no post-hoc switching)
-----------------------------------------------------------------
SA versus the uniform heuristic on V100 of the +5 mm shell (grid based,
``final_report``), paired across cavities, evaluated at N* -- the smallest N
in ``N_LIST`` at which the uniform heuristic first reaches D90 >= rx on that
cavity (if it never does, N* = max(N_LIST) and the cavity is flagged).
Statistic: paired mean difference with a t-based 95 % CI and a Wilcoxon
signed-rank p-value across the cavity grid.  Everything else in V2 is
secondary.

Cavity grid
-----------
``make_head_phantom`` (spacing 1.0 mm) has no size parameter; its lumpy
cavity is built from the module constant ``CAVITY_RADII`` (20, 18, 17 mm).
Sizes are therefore produced by scaling that constant (``SCALES``) around
the call, which touches nothing in the generator's code.  Grid = rng seeds
``SEEDS`` x ``SCALES``; the cavity volume (cavity-mask voxels x spacing^3,
seed voxels excluded) and wall area are stated per cavity in every table.
Every phantom carries 8 truth tiles (the "truth" arm).

Arms (V2)
---------
random      n_random feasible draws over the candidate set (median, p95)
uniform     farthest-point anchors, spin 0 (local implementation, below)
greedy      ``solve_greedy``
greedy+local ``solve_local`` started from greedy
sa          ``solve_sa``
milp        ``solve_milp(method="auto")`` on the reduced instance (h = 6 mm,
            2 spins, M = 300, 300 s), N in ``V3_N`` only
continuous  ``solve_continuous`` (A3, multi-start Nelder-Mead over continuous
            tile poses) with ``time_budget_s`` = greedy + SA wall time on the
            same instance (guarded: skipped when absent)
truth       the phantom's truth tiles through ``conform_tile``
truth_raw   the phantom's raw truth seeds (no conformer)

Every arm row records ``n_placed``; an arm that returns fewer than N tiles
or a conflicting selection is a FAILED row, never a lower score.

Conflict rule (validation-side mitigation, 2026-10-07): ``find_overlapping_tiles``
misses real footprint overlaps on curved walls (its quadratic footprint
reconstruction degenerates; reported), so A1's conflict graph is AUGMENTED
here with the geometric proxy (``augment_conflicts``: anchors closer than
18 mm or inter-tile seeds closer than 9 mm, normals agreeing) before any
solver runs, and every returned plan is re-checked with the same proxy.
``n_pairs_planner`` / ``n_pairs_added`` are recorded per instance.

All endpoints (V100, D90, V150, V200 on the +5 mm shell; weighted V100 on
the full target) come from ``final_report`` (``compute_dose_grid``, exact
kernel, 1 mm) -- never from the influence matrix.  The grid margin for the
per-arm evaluations is ``REPORT_MARGIN_MM`` (15 mm: enough for the +10 mm
shell); the primary-endpoint configurations get a second full report with
the 50 mm margin written to JSON.

Functions from ``gtcore.plan`` that are still Phase 0 stubs raise
``NotImplementedError``; every arm / section that needs one is skipped with
the reason logged, so the script runs against whatever has landed.

Usage
-----
    python scripts/validation_optimize.py --quick
    python scripts/validation_optimize.py --section v2 --seeds 1 2 3
    python scripts/validation_optimize.py --section v6 [--clinical <dicom dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import trimesh  # noqa: E402
from scipy import stats as sstats  # noqa: E402

import gtcore.plan as plan  # noqa: E402
from gtcore.interact import (PlacedTile, conform_tile, find_overlapping_tiles,  # noqa: E402
                             snap_to_wall)
from gtcore.plan import DEFAULT_RX_CGY, TargetSet, final_report  # noqa: E402

# ------------------------------------------------------------------ design
SEEDS = (1, 2, 3, 4, 5, 6)
SCALES = (0.8, 1.0, 1.25)
# CAVITY_RADII multipliers: ~12-14, 24-28 and 48-54 mL cavities (stated per run).
N_LIST = (4, 6, 8, 10, 12)
# The manufacturer rule gives 11-12 tiles for the scale-1.0 cavities (coordinator, 2026-10-07).
N_RANDOM = 20
# (declared 50; trimmed to 20 for the overnight budget, coordinator 2026-10-07)
N_TRUTH_TILES = 8
PHANTOM_SPACING_MM = 1.0
RX_CGY = DEFAULT_RX_CGY
GRID_MM = 1.0
REPORT_MARGIN_MM = 15.0
# Grid margin for the per-arm reports: the +10 mm shell plus trilinear
# support; the full 50 mm report is written for the primary configurations.
FULL_MARGIN_MM = 50.0
H_MM = 4.0
N_SPINS = 6
# V2 discrete grid: the coordinator's default was h = 4 mm / 3 spins; the quick V4
# run (2026-10-07) showed 3 spins cannot pack 8 tiles on the 24 mL cavity s1
# (enumeration proves infeasibility, 77 608 nodes) while h = 4 / 6 spins packs
# them at V100 0.999 -- so V2 uses 6 spins ("unless V4 says otherwise").
M_OPT = 1000
# Optimization-time target subsample (V5 compares 1000 vs 4000).
MILP_H_MM = 6.0
MILP_N_SPINS = 2
MILP_M_OPT = 300
MILP_TIME_LIMIT_S = 300.0
V3_N = (4, 6, 8)
# Reduced MILP instances (coordinator 2026-10-07): h = 6 mm / 2 spins / M = 300,
# N in V3_N, solve_milp(method="auto") -> enumeration branch-and-bound when
# C <= 600 and N <= 8 (the section 7.3 scout showed HiGHS cannot close these),
# 300 s limit; the bound is the reference when status == "time_limit".
V4_H = (4.0, 3.0, 2.5)
V4_SPINS = (3, 6, 12)
# h = 2 mm dropped from the overnight grid: build_conflicts is O(C^2) exact
# footprint tests (345 s at C = 2500; h = 2 / 6 spins would be ~6300 candidates).
V4_N = 8
V5_N = 8
V5_OFFSETS_MM = (2.25, 3.0, 3.75)
# Seed-plane offsets: geometry.SEED_PLANE_OFFSET_RANGE_MM endpoints + nominal (re-read below).
V6_PHANTOM_DIR = r"C:\Users\jacob\OneDrive\Documents\3D-Printed Phantom-8tiles (223)"
V6_SEED_RADIUS_MM = 35.0
V6_N = 8
V6_N_MAX = 12
SA_SEED_BASE = 1000

try:
    from gtcore.geometry import SEED_PLANE_OFFSET_RANGE_MM
    V5_OFFSETS_MM = (float(SEED_PLANE_OFFSET_RANGE_MM[0]), 3.0,
                     float(SEED_PLANE_OFFSET_RANGE_MM[1]))
except Exception:                                                     # pragma: no cover
    pass

OUT_DIR = os.path.join(ROOT, "output", "validation_optimize")
CAPTION = "(conflict rule = planner ∪ geometric proxy; n_pairs_added recorded per instance)"
ARM_ORDER = ("random", "uniform", "greedy", "greedy+local", "sa", "milp", "continuous",
             "truth", "truth_raw")
SOLVER_ARMS = ("uniform", "greedy", "greedy+local", "sa", "milp", "continuous")


class Skip(Exception):
    """An arm / section cannot run (stub, infeasible, missing data)."""


class ArmFailed(Exception):
    """An arm ran but produced an unusable plan (fewer than N tiles, a
    conflict, a non-ok status): recorded as a failed row, never as a score."""

    def __init__(self, msg, n_placed=0):
        super().__init__(msg)
        self.n_placed = int(n_placed)


def _log(msg: str) -> None:
    print("[validation_optimize] " + str(msg), flush=True)


def _try(name: str, fn: Callable[[], Any]) -> Tuple[Any, Optional[str]]:
    """Run ``fn``; return ``(result, None)`` or ``(None, reason)`` on
    NotImplementedError / Skip / any exception (logged, never fatal)."""
    try:
        return fn(), None
    except NotImplementedError as exc:
        reason = "stub: %s" % exc
    except Skip as exc:
        reason = "skipped: %s" % exc
    except ArmFailed as exc:
        reason = "FAILED (n_placed=%d): %s" % (exc.n_placed, exc)
    except Exception as exc:                                  # noqa: BLE001
        reason = "error: %s: %s" % (type(exc).__name__, exc)
        _log(traceback.format_exc().strip().splitlines()[-1])
    _log("%s -> %s" % (name, reason))
    return None, reason


def _solve_milp(objective, n, time_limit_s):
    """``solve_milp(method="auto")``: the ``gtcore.plan.solve_milp`` wrapper does
    not forward ``method`` (reported to the coordinator), so the module
    function is called directly when it accepts it."""
    import inspect
    try:
        from gtcore.plan import milp as _milp
        fn = _milp.solve_milp
        if "method" in inspect.signature(fn).parameters:
            return fn(objective, n, time_limit_s=time_limit_s, method="auto")
    except ImportError:
        pass
    return plan.solve_milp(objective, n, time_limit_s=time_limit_s)


def _solve_continuous():
    """A3's ``solve_continuous``: from ``gtcore.plan`` when exported, else from
    ``gtcore.plan.solvers`` (not re-exported at the time of writing)."""
    fn = getattr(plan, "solve_continuous", None)
    if fn is None:
        try:
            from gtcore.plan import solvers as _solvers
            fn = getattr(_solvers, "solve_continuous", None)
        except ImportError:
            fn = None
    return fn


def row_status(reason: Optional[str]) -> Tuple[str, int]:
    """``("failed", n_placed)`` for an ArmFailed reason, else ``("skipped", 0)``."""
    if reason and reason.startswith("FAILED"):
        try:
            n = int(reason.split("n_placed=")[1].split(")")[0])
        except Exception:                                             # pragma: no cover
            n = 0
        return "failed", n
    return "skipped", 0


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "?"
    except Exception:                                                 # pragma: no cover
        return "?"


def hardware() -> Dict[str, Any]:
    return {"platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "cores": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__}


# --------------------------------------------------------------- statistics
def mean_sd(x) -> Tuple[float, float, int]:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def paired_ci(a, b, alpha: float = 0.05) -> Dict[str, float]:
    """Paired difference ``a - b``: mean, SD, n and a t-based CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = int(d.size)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0,
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        t = float(sstats.t.ppf(1.0 - alpha / 2.0, n - 1))
        half = t * sd / np.sqrt(n)
    else:
        half = float("nan")
    return {"mean": m, "sd": sd, "n": n, "ci_lo": m - half, "ci_hi": m + half}


def wilcoxon_p(a, b) -> float:
    """Wilcoxon signed-rank p for paired ``a`` vs ``b``; 1.0 when every
    difference is zero or fewer than 2 pairs are available."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size < 2 or np.all(d == 0.0):
        return 1.0
    try:
        return float(sstats.wilcoxon(d).pvalue)
    except ValueError:
        return 1.0


def kendall_tau(x, y) -> float:
    """Kendall tau between two score vectors (rank agreement); nan if < 2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    tau = sstats.kendalltau(x[ok], y[ok]).statistic
    return float(tau) if tau is not None else float("nan")


# ---------------------------------------------------------- local heuristics
def farthest_point_order(points, start: Optional[int] = None) -> np.ndarray:
    """Farthest-point sampling order over ``points`` (all of them).

    Start = the point farthest from the centroid when ``start`` is None.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if start is None:
        start = int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    order = [int(start)]
    dist = np.linalg.norm(pts - pts[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0.0 and len(order) > 1:
            # all remaining points coincide with chosen ones; keep ids ascending
            rest = sorted(set(range(n)) - set(order))
            order.extend(rest)
            break
        order.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.asarray(order, dtype=int)


def uniform_heuristic(candidates, conflicts, n_tiles: int, kind: str = "full"
                      ) -> np.ndarray:
    """Uniform baseline: farthest-point anchors with spin 0.

    FPS over the distinct eligible anchors (one representative per
    ``anchor_id``: the candidate with the smallest spin, kind ``kind``);
    at each FPS point the spin-0 candidate is taken if it conflicts with
    nothing chosen so far, otherwise the next FPS point is tried.  When the
    spin-0 pass ends short of ``n_tiles`` (dense packings), a second pass
    walks the same FPS order allowing the other spins at each anchor
    (deterministic, smallest conflicting-free spin first); the number of
    such fallback picks is available as ``uniform_heuristic.last_fallback``.
    Raises ``Skip`` when still fewer than ``n_tiles``.
    """
    kinds = np.asarray(candidates.kinds, dtype=object)
    elig = np.asarray(candidates.eligible, dtype=bool) & (kinds == kind)
    ids = np.flatnonzero(elig)
    if ids.size == 0:
        raise Skip("uniform: no eligible %s candidates" % kind)
    # one representative per anchor: the smallest spin
    rep: Dict[int, int] = {}
    for c in ids:
        a = int(candidates.anchor_ids[c])
        if a not in rep or candidates.spins_deg[c] < candidates.spins_deg[rep[a]]:
            rep[a] = int(c)
    rep_ids = np.asarray(sorted(rep.values()), dtype=int)
    order = farthest_point_order(candidates.anchors[rep_ids])
    chosen: List[int] = []
    for k in order:
        c = int(rep_ids[k])
        if conflicts.is_feasible(chosen + [c]):
            chosen.append(c)
            if len(chosen) == int(n_tiles):
                break
    n_fallback = 0
    if len(chosen) < int(n_tiles):
        by_anchor: Dict[int, List[int]] = {}
        for c in ids:
            by_anchor.setdefault(int(candidates.anchor_ids[c]), []).append(int(c))
        used_anchors = {int(candidates.anchor_ids[c]) for c in chosen}
        for k in order:
            a = int(candidates.anchor_ids[rep_ids[k]])
            if a in used_anchors:
                continue
            for c in sorted(by_anchor.get(a, []), key=lambda i: candidates.spins_deg[i]):
                if conflicts.is_feasible(chosen + [c]):
                    chosen.append(c)
                    used_anchors.add(a)
                    n_fallback += 1
                    break
            if len(chosen) == int(n_tiles):
                break
    uniform_heuristic.last_fallback = n_fallback
    if len(chosen) < int(n_tiles):
        raise Skip("uniform: only %d of %d non-conflicting anchors (spin fallback included)"
                   % (len(chosen), n_tiles))
    return np.asarray(sorted(chosen), dtype=int)


uniform_heuristic.last_fallback = 0


def random_feasible(candidates, conflicts, n_tiles: int, rng, max_tries: int = 500
                    ) -> np.ndarray:
    """One random feasible selection: a random candidate order, taking each
    candidate that conflicts with nothing taken, until ``n_tiles``."""
    elig = np.flatnonzero(np.asarray(candidates.eligible, dtype=bool))
    if elig.size < int(n_tiles):
        raise Skip("random: %d eligible candidates < N=%d" % (elig.size, n_tiles))
    for _ in range(int(max_tries)):
        perm = rng.permutation(elig)
        chosen: List[int] = []
        compat = np.ones(len(candidates), dtype=bool)
        for c in perm:
            c = int(c)
            if not compat[c]:
                continue
            chosen.append(c)
            if len(chosen) == int(n_tiles):
                return np.asarray(sorted(chosen), dtype=int)
            compat &= conflicts.compatible_mask(chosen)
    raise Skip("random: no feasible %d-tile draw in %d tries" % (n_tiles, max_tries))


# ------------------------------------------------------ overlap proxy
PROXY_OVERLAP_MM = 18.0
# Two full tiles whose anchors are closer than 20 mm overlap for ANY spins
# (each contains the 10 mm disc about its anchor); 18 mm leaves 2 mm for
# chord-vs-surface effects.  Independent of ``find_overlapping_tiles``, whose
# footprint reconstruction was found to fail on symmetric placements on
# smooth spheres (reported 2026-10-07).


PROXY_SEED_MM = 9.0
# Every seed sits 5 mm inside its tile's edge, so two seeds of different
# tiles closer than 10 mm (chord <= surface distance) lie in overlapping
# 5 mm discs of the two sheets; 9 mm leaves 1 mm for the planner threshold.


def _proxy_pair(a: PlacedTile, b: PlacedTile, d_mm: float = PROXY_OVERLAP_MM,
                seed_mm: float = PROXY_SEED_MM) -> bool:
    if float(a.normal_ras @ b.normal_ras) <= 0.5:
        return False
    lim = d_mm if (a.kind == "full" and b.kind == "full") else d_mm / 2.0
    if np.linalg.norm(a.anchor_ras - b.anchor_ras) < lim:
        return True
    ds = np.linalg.norm(a.seed_centers[:, None, :] - b.seed_centers[None, :, :], axis=2)
    return bool(ds.min() < seed_mm)


def proxy_overlaps(tiles: Sequence[PlacedTile], d_mm: float = PROXY_OVERLAP_MM) -> List[Tuple[int, int]]:
    """Pairs of tiles that certainly overlap by geometry alone: anchors closer
    than ``d_mm`` (full/full; half the distance otherwise) or any inter-tile
    seed pair closer than ``PROXY_SEED_MM``, with agreeing normals (dot > 0.5).
    A conservative subset of true overlaps, independent of
    ``find_overlapping_tiles``."""
    tiles = list(tiles)
    return [(i, j) for i in range(len(tiles)) for j in range(i + 1, len(tiles))
            if _proxy_pair(tiles[i], tiles[j], d_mm)]


def augment_conflicts(candidates, conflicts) -> Tuple[Any, int]:
    """ConflictGraph = planner conflicts + the geometric proxy pairs (vectorized
    over candidates).  Returns ``(graph, n_added)``; cliques are kept (adding
    pairs cannot invalidate a clique)."""
    import scipy.sparse as sp
    from gtcore.plan import ConflictGraph
    c = len(candidates)
    A = np.asarray(candidates.anchors, dtype=float)
    Nn = np.array([t.normal_ras for t in candidates.tiles], dtype=float)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=1), 1e-12)[:, None]
    full = np.asarray(candidates.kinds, dtype=object) == "full"
    dots = Nn @ Nn.T
    D = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    lim = np.where(full[:, None] & full[None, :], PROXY_OVERLAP_MM, PROXY_OVERLAP_MM / 2.0)
    close = D < lim
    # seed-seed proximity (C, 4, 3) with NaN padding for half tiles
    S = np.asarray(candidates.seed_centers, dtype=float)
    S = np.where(np.isfinite(S), S, 1e6)
    flat = S.reshape(c * 4, 3)
    from scipy.spatial import cKDTree
    tree = cKDTree(flat)
    pairs = tree.query_pairs(PROXY_SEED_MM, output_type="ndarray")
    seed_close = np.zeros((c, c), dtype=bool)
    if pairs.size:
        ci, cj = pairs[:, 0] // 4, pairs[:, 1] // 4
        keep = ci != cj
        seed_close[ci[keep], cj[keep]] = True
        seed_close[cj[keep], ci[keep]] = True
    proxy = (close | seed_close) & (dots > 0.5)
    np.fill_diagonal(proxy, False)
    old = conflicts.pairs.toarray().astype(bool)
    new = old | proxy
    n_added = int((new & ~old).sum() // 2)
    graph = ConflictGraph(n=c, pairs=sp.csr_matrix(new), cliques=list(conflicts.cliques),
                          gap_mm=conflicts.gap_mm)
    return graph, n_added


# ------------------------------------------------------------- perturbation
def shift_seed_plane(tiles: Sequence[PlacedTile], offset_mm: float,
                     nominal_mm: float = 3.0) -> List[PlacedTile]:
    """Copies of ``tiles`` with every seed moved by ``offset_mm - nominal_mm``
    along the tile's inward normal (``normal_ras``), i.e. the seed plane at
    ``offset_mm`` off the wall instead of the nominal 3 mm.  The tile's
    anchor normal stands in for the per-seed local normal (not stored)."""
    out = []
    d = float(offset_mm) - float(nominal_mm)
    for t in tiles:
        n = np.asarray(t.normal_ras, dtype=float)
        n = n / max(np.linalg.norm(n), 1e-12)
        out.append(PlacedTile(kind=t.kind, center_ras=t.center_ras + d * n,
                              normal_ras=t.normal_ras.copy(), axis_ras=t.axis_ras.copy(),
                              seed_centers=t.seed_centers + d * n,
                              seed_axes=t.seed_axes.copy(), corners_ras=t.corners_ras.copy(),
                              anchor_ras=t.anchor_ras.copy()))
    return out


# -------------------------------------------------------------- truth tiles
def truth_tiles_conformed(mesh, truth) -> List[PlacedTile]:
    """The phantom's truth tiles re-draped by ``conform_tile`` at their own
    anchor (seed centroid pushed 3 mm back to the wall along the OUTWARD
    normal) with the first seed axis as the spin hint."""
    out = []
    for t in truth.tiles:
        outward = np.asarray(t.normal_ras, dtype=float)
        anchor = np.asarray(t.center_ras, dtype=float) + 3.0 * outward
        surf, n_in = snap_to_wall(mesh, anchor)
        hint = np.asarray(truth.seeds[int(t.seed_ids[0])].axis_ras, dtype=float)
        out.append(conform_tile(mesh, surf, n_in, hint, kind=t.kind))
    return out


def truth_tiles_raw(truth) -> List[PlacedTile]:
    """PlacedTiles carrying the RAW truth seed poses (no conformer); the
    footprint corners are a flat 20 x 20 (10 x 20) square in the tile's
    tangent frame, so the overlap flag on these is approximate."""
    out = []
    for t in truth.tiles:
        ids = [int(i) for i in t.seed_ids]
        seeds = np.array([truth.seeds[i].center_ras for i in ids], dtype=float)
        axes = np.array([truth.seeds[i].axis_ras for i in ids], dtype=float)
        n_in = -np.asarray(t.normal_ras, dtype=float)
        n_in = n_in / max(np.linalg.norm(n_in), 1e-12)
        t1 = axes[0] - float(axes[0] @ n_in) * n_in
        t1 = t1 / max(np.linalg.norm(t1), 1e-12)
        t2 = np.cross(n_in, t1)
        cu = 10.0 if t.kind == "full" else 5.0
        c = seeds.mean(axis=0)
        corners = np.array([c + su * cu * t1 + sv * 10.0 * t2 - 3.0 * n_in
                            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        out.append(PlacedTile(kind=t.kind, center_ras=c, normal_ras=n_in, axis_ras=t1,
                              seed_centers=seeds, seed_axes=axes, corners_ras=corners,
                              anchor_ras=c - 3.0 * n_in))
    return out


# ------------------------------------------------------------------ targets
def target_from_faces(mesh, face_mask, offset_mm: float = 5.0, name: str = "shell+5mm(eligible)"
                      ) -> TargetSet:
    """+offset shell restricted to the vertices of ``face_mask`` faces, with
    one third of the adjacent eligible SHELL face areas as weights."""
    from gtcore.dose.dvh import shell_points
    pts = np.asarray(shell_points(mesh, float(offset_mm)), dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)[np.asarray(face_mask, dtype=bool)]
    if faces.shape[0] == 0:
        raise Skip("target_from_faces: no eligible faces")
    shell = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
    w = np.zeros(pts.shape[0], dtype=float)
    third = np.asarray(shell.area_faces, dtype=float) / 3.0
    for k in range(3):
        np.add.at(w, faces[:, k], third)
    keep = np.flatnonzero(w > 0)
    return TargetSet(points=pts[keep], weights=w[keep], name=name)


def visible_faces_local(mesh, center_ras) -> np.ndarray:
    """Fallback for ``plan.visible_faces``: faces whose centroid is the first
    hit of the ray from ``center_ras`` toward it."""
    cen = np.asarray(mesh.triangles_center, dtype=float)
    origin = np.asarray(center_ras, dtype=float).reshape(1, 3)
    dirs = cen - origin
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1), 1e-12)[:, None]
    origins = np.repeat(origin, cen.shape[0], axis=0)
    hit = mesh.ray.intersects_first(ray_origins=origins, ray_directions=dirs)
    return np.asarray(hit) == np.arange(cen.shape[0])


def visible_faces_any(mesh, center_ras) -> Tuple[np.ndarray, str]:
    try:
        return np.asarray(plan.visible_faces(mesh, center_ras), dtype=bool), "plan.visible_faces"
    except NotImplementedError:
        return visible_faces_local(mesh, center_ras), "local ray test (plan.visible_faces is a stub)"


def tile_count_rule(area_mm2: float) -> int:
    return int(np.ceil(float(area_mm2) / (plan.TILE_AREA_CM2 * 100.0)))


# ---------------------------------------------------------------- cavities
def cavity_key(seed: int, scale: float) -> str:
    return "s%d_x%.2f" % (int(seed), float(scale))


def get_cavity(seed: int, scale: float, cache_dir: str, use_cache: bool = True) -> Dict[str, Any]:
    """Phantom cavity ``seed`` at radius scale ``scale``: mesh, truth, mask."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "cavity_%s.pkl" % cavity_key(seed, scale))
    if use_cache and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import gtcore.phantom.generate as gen
    from gtcore.segment.surface import mask_to_mesh
    base = gen.CAVITY_RADII
    try:
        gen.CAVITY_RADII = tuple(float(scale) * r for r in base)
        vol, truth = gen.make_head_phantom(spacing=PHANTOM_SPACING_MM, n_tiles=N_TRUTH_TILES,
                                           rng_seed=int(seed))
    finally:
        gen.CAVITY_RADII = base
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    cav = {
        "key": cavity_key(seed, scale), "seed": int(seed), "scale": float(scale),
        "mesh": mesh, "truth": truth, "cavity_mask": truth.masks["cavity"],
        "affine": vol.affine,
        "volume_ml": float(truth.masks["cavity"].sum()) * PHANTOM_SPACING_MM ** 3 / 1000.0,
        "area_mm2": float(mesh.area), "n_truth_tiles": len(truth.tiles),
        "radii_mm": tuple(float(scale) * r for r in base),
    }
    with open(path, "wb") as fh:
        pickle.dump(cav, fh)
    return cav


# ---------------------------------------------------------------- instance
def build_instance(mesh, target: TargetSet, h_mm: float, n_spins: int, m_opt: int,
                   rng_seed: int, eligible_faces=None, cache_path: Optional[str] = None,
                   use_cache: bool = True) -> Dict[str, Any]:
    """Candidates + conflicts + influence + objective with stage timings.
    Raises NotImplementedError while A1/A2 are stubs."""
    if cache_path and use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            inst = pickle.load(fh)
        inst["objective"] = plan.make_objective(inst["influence"], inst["conflicts"],
                                                rx_cgy=RX_CGY)
        inst["from_cache"] = True
        return inst
    t = {}
    t0 = time.perf_counter()
    cand = plan.build_candidates(mesh, h_mm=h_mm, n_spins=n_spins, kinds=("full",),
                                 eligible_faces=eligible_faces, rng_seed=rng_seed)
    t["candidates"] = time.perf_counter() - t0
    if len(cand) == 0:
        raise Skip("build_candidates returned no candidates")
    t0 = time.perf_counter()
    conf0 = plan.build_conflicts(cand)
    t["conflicts"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    n_pairs_planner = conf0.count_pairs()
    conf, n_added = augment_conflicts(cand, conf0)
    t["conflicts_augment"] = time.perf_counter() - t0
    _log("  conflicts: %d planner pairs + %d proxy pairs added (%.1f %%)"
         % (n_pairs_planner, n_added, 100.0 * n_added / max(n_pairs_planner, 1)))
    t0 = time.perf_counter()
    infl = plan.build_influence(cand, target, rx_cgy=RX_CGY, m_opt=m_opt, rng_seed=rng_seed)
    t["influence"] = time.perf_counter() - t0
    obj = plan.make_objective(infl, conf, rx_cgy=RX_CGY)
    inst = {"candidates": cand, "conflicts": conf, "influence": infl, "objective": obj,
            "timings": t, "h_mm": h_mm, "n_spins": n_spins, "m_opt": m_opt,
            "from_cache": False, "n_pairs_planner": n_pairs_planner, "n_pairs_added": n_added}
    if cache_path:
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump({k: v for k, v in inst.items() if k != "objective"}, fh)
        except Exception as exc:                                      # noqa: BLE001
            _log("instance cache write failed: %s" % exc)
    return inst


def evaluate_tiles(mesh, tiles: Sequence[PlacedTile], target: TargetSet,
                   margin_mm: float = REPORT_MARGIN_MM, grid_mm: float = GRID_MM,
                   interference: bool = False, sk_per_seed_u: Optional[float] = None,
                   cavity_mask=None, cavity_affine=None, solver_result=None,
                   shadowing: bool = False, extra_params: Optional[Dict] = None,
                   use_target: bool = False) -> Tuple[Dict[str, Any], Any]:
    """``final_report`` -> flat endpoint row (+5 mm shell stats, weighted V100).

    ``use_target=True`` makes V100 / D90 / V150 / V200 the WEIGHTED stats of
    the full target (needed when the mesh carries faces outside the target,
    e.g. the printed phantom's outer shell); the shell-vertex numbers are
    then kept under ``*_shellverts``."""
    params = {"shadowing": shadowing}
    if sk_per_seed_u is not None:
        params["sk_per_seed_u"] = float(sk_per_seed_u)
    if extra_params:
        params.update(extra_params)
    rep = final_report(mesh, tiles, rx_cgy=RX_CGY, target=target, cavity_mask=cavity_mask,
                       cavity_affine=cavity_affine, interference=interference,
                       grid_mm=grid_mm, margin_mm=margin_mm, solver_result=solver_result,
                       parameters=params)
    src = rep.metrics_grid_interference if (interference and rep.metrics_grid_interference) \
        else rep.metrics_grid
    s5 = src[5.0]
    row = {"V100": s5["V100"], "D90": s5["D90"], "V150": s5["V150"], "V200": s5["V200"],
           "V100w": src["target"]["V100"], "D90w": src["target"]["D90"],
           "V100_wall": src[0.0]["V100"], "V100_10mm": src[10.0]["V100"],
           "n_overlaps": len(rep.overlaps), "n_proxy_overlaps": len(proxy_overlaps(tiles)),
           "report_s": rep.runtime.get("total", float("nan"))}
    if "rind" in src:
        row["rind_V100"] = src["rind"]["V100"]
        row["rind_D90"] = src["rind"]["D90"]
    if use_target:
        tg = src["target"]
        for k in ("V100", "D90", "V150", "V200"):
            row[k + "_shellverts"] = row[k]
            row[k] = tg[k]
    return row, rep


# ------------------------------------------------------------------- arms
def run_arm(arm: str, inst: Optional[Dict[str, Any]], n: int, sa_seed: int,
            mesh=None, target=None, milp_inst=None, time_budget_s: Optional[float] = None,
            start=None) -> Dict[str, Any]:
    """Run one solver arm; returns ``{"selection", "tiles", "solver", "solve_s",
    "n_placed"}``.  Raises NotImplementedError (stub), Skip (cannot run) or
    ArmFailed (ran but fewer than N tiles / conflicting / bad status)."""
    t0 = time.perf_counter()
    if arm in ("uniform", "greedy", "greedy+local", "sa", "continuous") and inst is None:
        raise Skip("no instance (candidates/influence are stubs)")
    cand = inst["candidates"] if inst is not None else None
    conf = inst["conflicts"] if inst is not None else None
    res = None
    if arm == "uniform":
        sel = uniform_heuristic(cand, conf, n)
        n_fb = uniform_heuristic.last_fallback
    elif arm == "greedy":
        res = plan.solve_greedy(inst["objective"], n, candidates=cand)
        sel = res.selection
    elif arm == "greedy+local":
        g = plan.solve_greedy(inst["objective"], n, candidates=cand)
        if g.status != "ok":
            raise ArmFailed("greedy %s: %s" % (g.status, g.reason), g.selection.size)
        res = plan.solve_local(inst["objective"], n, g.selection, candidates=cand)
        sel = res.selection
    elif arm == "sa":
        res = plan.solve_sa(inst["objective"], n, seed=int(sa_seed), candidates=cand,
                            **({"start": np.asarray(start, dtype=int)} if start is not None else {}))
        sel = res.selection
    elif arm == "milp":
        if milp_inst is None:
            raise Skip("no reduced instance")
        cand, conf = milp_inst["candidates"], milp_inst["conflicts"]
        if int(n) not in V3_N:
            raise Skip("milp arm restricted to N in %s" % (V3_N,))
        res = _solve_milp(milp_inst["objective"], n, time_limit_s=MILP_TIME_LIMIT_S)
        sel = res.selection
    elif arm == "continuous":
        fn = _solve_continuous()
        if fn is None:
            raise NotImplementedError("solve_continuous: not in gtcore.plan / plan.solvers yet (A3)")
        if mesh is None or target is None:
            raise Skip("continuous: mesh/target not supplied")
        kw = {"seed": int(sa_seed), "objective": inst["objective"], "conflicts": conf,
              "m_opt": int(inst.get("m_opt", M_OPT))}
        if time_budget_s is not None and np.isfinite(time_budget_s) and time_budget_s > 0:
            kw["time_budget_s"] = float(time_budget_s)
        tiles, res = fn(mesh, cand, target, RX_CGY, n, **kw)
        tiles = list(tiles)
        if res is not None and res.status not in ("ok", "optimal", "time_limit"):
            raise ArmFailed("continuous status %s: %s" % (res.status, res.reason or "(no reason)"),
                            len(tiles))
        if len(tiles) != int(n):
            raise ArmFailed("continuous returned %d tiles for N=%d" % (len(tiles), n), len(tiles))
        ov = find_overlapping_tiles(tiles) or proxy_overlaps(tiles)
        if ov:
            raise ArmFailed("continuous tiles overlap (planner rule / anchor proxy): %r" % (ov,), len(tiles))
        return {"selection": None, "tiles": tiles, "solver": res,
                "solve_s": time.perf_counter() - t0, "n_placed": len(tiles),
                "time_budget_s": time_budget_s}
    else:
        raise ValueError("unknown arm %r" % arm)
    sel = np.asarray(sel, dtype=int).reshape(-1)
    if res is not None and res.status not in ("ok", "optimal", "time_limit"):
        raise ArmFailed("%s status %s: %s" % (arm, res.status, res.reason or "(no reason)"), sel.size)
    if sel.size != int(n):
        raise ArmFailed("%s returned %d tiles for N=%d" % (arm, sel.size, n), sel.size)
    if not conf.is_feasible(sel):
        raise ArmFailed("%s selection violates a conflict (silent infeasible plan)" % arm, sel.size)
    px = proxy_overlaps(cand.tiles_of(sel))
    if px:
        raise ArmFailed("%s selection has certain footprint overlaps missed by the conflict graph "
                        "(anchors < %g mm): %r" % (arm, PROXY_OVERLAP_MM, px), sel.size)
    out = {"selection": sel, "tiles": cand.tiles_of(sel), "solver": res,
           "solve_s": time.perf_counter() - t0, "n_placed": int(sel.size)}
    if arm == "uniform":
        out["uniform_spin_fallback"] = int(n_fb)
    return out


# ------------------------------------------------------------------- I/O
def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def _fmt(v):
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, (np.floating,)):
        return "%.6g" % float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(v).tolist())
    return v


def md_table(rows: List[Dict[str, Any]], cols: List[str], fmt: Optional[Dict[str, str]] = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("—")
            elif c in fmt and isinstance(v, (int, float, np.floating, np.integer)):
                cells.append(fmt[c] % v)
            elif isinstance(v, float):
                cells.append("%.3f" % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def append_run(out_dir: str, section: str, argv: List[str], seeds, wall_s: float, note: str = "") -> None:
    path = os.path.join(out_dir, "runs.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "section", "command", "seeds", "commit", "wall_s", "note"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), section,
                    "python " + " ".join(argv), json.dumps(list(seeds)), git_commit(),
                    "%.1f" % wall_s, note])


# ------------------------------------------------------------------- V2
def section_v2(cfg) -> Dict[str, Any]:
    out, md = cfg.out, []
    rows: List[Dict[str, Any]] = []
    configs: Dict[Tuple[str, int, str], List[PlacedTile]] = {}
    skipped: Dict[str, str] = {}
    cavities = [get_cavity(s, x, cfg.cache_dir, cfg.use_cache)
                for x in cfg.scales for s in cfg.seeds]
    md.append("### V2 cavity grid\n")
    md.append(md_table([{"cavity": c["key"], "seed": c["seed"], "scale": c["scale"],
                         "radii_mm": "%.1f/%.1f/%.1f" % c["radii_mm"],
                         "volume_mL": c["volume_ml"], "area_mm2": c["area_mm2"],
                         "truth_tiles": c["n_truth_tiles"],
                         "rule_tiles": tile_count_rule(c["area_mm2"])} for c in cavities],
                       ["cavity", "seed", "scale", "radii_mm", "volume_mL", "area_mm2",
                        "truth_tiles", "rule_tiles"],
                       {"volume_mL": "%.1f", "area_mm2": "%.0f"}))

    for cav in cavities:
        mesh, truth = cav["mesh"], cav["truth"]
        target = TargetSet.from_shell(mesh, plan.TARGET_SHELL_OFFSET_MM)
        base = {"cavity": cav["key"], "seed": cav["seed"], "scale": cav["scale"],
                "volume_ml": cav["volume_ml"], "area_mm2": cav["area_mm2"]}
        _log("V2 cavity %s (%.1f mL, %.0f mm2)" % (cav["key"], cav["volume_ml"], cav["area_mm2"]))

        # truth arms (N fixed by the phantom)
        for arm, maker in (("truth", lambda: truth_tiles_conformed(mesh, truth)),
                           ("truth_raw", lambda: truth_tiles_raw(truth))):
            tiles, reason = _try(arm, maker)
            if tiles is None:
                rows.append(dict(base, N=len(truth.tiles), arm=arm, status="skipped", reason=reason))
                continue
            ev, _ = evaluate_tiles(mesh, tiles, target)
            rows.append(dict(base, N=len(tiles), arm=arm, status="ok", reason="", solve_s=0.0,
                             n_placed=len(tiles), **ev))
            configs[(cav["key"], len(tiles), arm)] = tiles

        inst, reason = _try("build_instance", lambda: build_instance(
            mesh, target, H_MM, N_SPINS, M_OPT, cav["seed"],
            cache_path=os.path.join(cfg.cache_dir, "inst_%s_h%g_s%d_m%d_cg%s.pkl"
                                    % (cav["key"], H_MM, N_SPINS, M_OPT, git_commit())),
            use_cache=cfg.use_cache))
        if inst is None:
            skipped["instance"] = reason
        milp_inst = None
        if "milp" in cfg.arms and inst is not None and (H_MM, N_SPINS, M_OPT) == (MILP_H_MM, MILP_N_SPINS, MILP_M_OPT):
            milp_inst = inst                      # same reduced grid: reuse
        elif "milp" in cfg.arms:
            milp_inst, reason = _try("build_instance(milp)", lambda: build_instance(
                mesh, target, MILP_H_MM, MILP_N_SPINS, MILP_M_OPT, cav["seed"],
                cache_path=os.path.join(cfg.cache_dir, "inst_%s_h%g_s%d_m%d_cg%s.pkl"
                                        % (cav["key"], MILP_H_MM, MILP_N_SPINS, MILP_M_OPT,
                                           git_commit())),
                use_cache=cfg.use_cache))
            if milp_inst is None:
                skipped["milp_instance"] = reason

        for n in cfg.n_list:
            # random feasible draws
            if "random" in cfg.arms:
                def _random():
                    if inst is None:
                        raise Skip("no instance (candidates/influence are stubs)")
                    rng = np.random.default_rng(10_000 * cav["seed"] + n)
                    vals = []
                    for k in range(cfg.n_random):
                        sel = random_feasible(inst["candidates"], inst["conflicts"], n, rng)
                        ev, _ = evaluate_tiles(mesh, inst["candidates"].tiles_of(sel), target)
                        vals.append(ev)
                    return vals
                vals, reason = _try("random N=%d" % n, _random)
                if vals is None:
                    st, n_placed = row_status(reason)
                    rows.append(dict(base, N=n, arm="random", status=st, reason=reason, n_placed=n_placed))
                else:
                    v100 = np.array([v["V100"] for v in vals])
                    d90 = np.array([v["D90"] for v in vals])
                    rows.append(dict(base, N=n, arm="random", status="ok", reason="",
                                     V100=float(np.median(v100)), V100_p95=float(np.percentile(v100, 95)),
                                     V100_p05=float(np.percentile(v100, 5)),
                                     D90=float(np.median(d90)), D90_p95=float(np.percentile(d90, 95)),
                                     V150=float(np.median([v["V150"] for v in vals])),
                                     V200=float(np.median([v["V200"] for v in vals])),
                                     V100w=float(np.median([v["V100w"] for v in vals])),
                                     n_draws=len(vals), solve_s=0.0, n_placed=n,
                                     report_s=float(np.sum([v["report_s"] for v in vals]))))
            budget = {}                       # greedy + SA wall time -> continuous budget
            for arm in [a for a in SOLVER_ARMS if a in cfg.arms]:
                tb = (budget.get("greedy", 0.0) + budget.get("sa", 0.0)) if arm == "continuous" else None
                if arm == "continuous" and tb == 0.0:
                    tb = None
                res, reason = _try("%s N=%d" % (arm, n), lambda: run_arm(
                    arm, inst, n, SA_SEED_BASE + cav["seed"], mesh=mesh, target=target,
                    milp_inst=milp_inst, time_budget_s=tb))
                if res is None:
                    st, n_placed = row_status(reason)
                    rows.append(dict(base, N=n, arm=arm, status=st, reason=reason, n_placed=n_placed))
                    continue
                budget[arm] = res["solve_s"]
                _log("  %s N=%d: %.1f s, n_placed %d" % (arm, n, res["solve_s"], res["n_placed"]))
                ev, _ = evaluate_tiles(mesh, res["tiles"], target, solver_result=res["solver"])
                extra = {"time_budget_s": res.get("time_budget_s"),
                         "uniform_spin_fallback": res.get("uniform_spin_fallback"),
                         "n_pairs_planner": inst.get("n_pairs_planner") if inst else None,
                         "n_pairs_added": inst.get("n_pairs_added") if inst else None}
                if res["solver"] is not None:
                    extra.update({"objective_influence": res["solver"].objective,
                                  "V100_influence": res["solver"].metrics.get("V100", float("nan")),
                                  "solver_status": res["solver"].status,
                                  "bound": res["solver"].bound, "mip_gap": res["solver"].mip_gap})
                rows.append(dict(base, N=n, arm=arm, status="ok", reason="", n_placed=res["n_placed"],
                                 solve_s=res["solve_s"], **ev, **extra))
                configs[(cav["key"], n, arm)] = res["tiles"]

    write_csv(os.path.join(out, "v2_rows.csv"), rows)
    with open(os.path.join(out, "v2_configs.pkl"), "wb") as fh:
        pickle.dump(configs, fh)

    # ---- summary: mean +- SD per (N, arm)
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = []
    for n in sorted({r["N"] for r in ok}):
        for arm in ARM_ORDER:
            sub = [r for r in ok if r["N"] == n and r["arm"] == arm]
            if not sub:
                continue
            row = {"N": n, "arm": arm, "n_cav": len(sub)}
            for key in ("V100", "D90", "V150", "V200", "V100w", "solve_s"):
                m, sd, _ = mean_sd([r.get(key, float("nan")) for r in sub])
                row[key + "_mean"] = m
                row[key + "_sd"] = sd
            summary.append(row)
    write_csv(os.path.join(out, "v2_summary.csv"), summary)

    # ---- paired comparisons per N: SA vs greedy, SA vs uniform (+ greedy vs uniform)
    paired = []
    cav_keys = [c["key"] for c in cavities]

    def _vals(arm, n, key="V100"):
        d = {r["cavity"]: r.get(key, float("nan")) for r in ok if r["arm"] == arm and r["N"] == n}
        return np.array([d.get(k, float("nan")) for k in cav_keys], dtype=float)

    for n in sorted({r["N"] for r in ok}):
        for a, b in (("sa", "greedy"), ("sa", "uniform"), ("greedy", "uniform"),
                     ("greedy+local", "greedy"), ("sa", "random"), ("continuous", "sa"),
                     ("continuous", "uniform")):
            va, vb = _vals(a, n), _vals(b, n)
            if np.isfinite(va).sum() == 0 or np.isfinite(vb).sum() == 0:
                continue
            ci = paired_ci(va, vb)
            paired.append({"N": n, "comparison": "%s - %s" % (a, b), "endpoint": "V100 (+5 mm)",
                           "mean_pp": 100 * ci["mean"], "ci_lo_pp": 100 * ci["ci_lo"],
                           "ci_hi_pp": 100 * ci["ci_hi"], "n": ci["n"],
                           "wilcoxon_p": wilcoxon_p(va, vb)})
    write_csv(os.path.join(out, "v2_paired.csv"), paired)

    # ---- PRIMARY endpoint: SA vs uniform at N* (uniform first reaches D90 >= rx)
    primary_rows = []
    for key in cav_keys:
        uni = {r["N"]: r for r in ok if r["cavity"] == key and r["arm"] == "uniform"}
        sa = {r["N"]: r for r in ok if r["cavity"] == key and r["arm"] == "sa"}
        if not uni or not sa:
            continue
        n_star = None
        for n in sorted(uni):
            if uni[n]["D90"] >= RX_CGY:
                n_star = n
                break
        flagged = n_star is None
        if flagged:
            n_star = max(uni)
        both = sorted(set(uni) & set(sa))
        if n_star not in sa:
            # declared N* has no SA row (packing-limited FAILED/skipped arm): fall
            # back to the largest N where both arms succeeded, flagged as such
            if not both:
                primary_rows.append({"cavity": key, "N_star": n_star, "uniform_never_reached_rx": flagged,
                                     "N_used": None, "fallback": True})
                continue
            n_used, fallback = both[-1], True
        else:
            n_used, fallback = n_star, False
        primary_rows.append({"cavity": key, "N_star": n_star, "uniform_never_reached_rx": flagged,
                             "N_used": n_used, "fallback": fallback,
                             "V100_uniform": uni[n_used]["V100"], "V100_sa": sa[n_used]["V100"],
                             "diff_pp": 100 * (sa[n_used]["V100"] - uni[n_used]["V100"])})
    primary = {}
    primary_rows_ok = [r for r in primary_rows if r.get("N_used") is not None]
    if primary_rows_ok:
        va = np.array([r["V100_sa"] for r in primary_rows_ok])
        vb = np.array([r["V100_uniform"] for r in primary_rows_ok])
        ci = paired_ci(va, vb)
        primary = {"comparison": "SA - uniform at N*", "mean_pp": 100 * ci["mean"],
                   "ci_lo_pp": 100 * ci["ci_lo"], "ci_hi_pp": 100 * ci["ci_hi"], "n": ci["n"],
                   "wilcoxon_p": wilcoxon_p(va, vb),
                   "n_flagged": int(sum(r["uniform_never_reached_rx"] for r in primary_rows)),
                   "n_fallback_N": int(sum(bool(r.get("fallback")) for r in primary_rows_ok)),
                   "n_no_pair": int(len(primary_rows) - len(primary_rows_ok))}
        write_csv(os.path.join(out, "v2_primary.csv"), primary_rows + [primary])

    # ---- figure: coverage vs N, mean +- SD band per arm
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in ((axes[0], "V100", "V100 of +5 mm shell"), (axes[1], "D90", "D90 of +5 mm shell [cGy]")):
        for arm in ARM_ORDER:
            sub = [r for r in summary if r["arm"] == arm]
            if not sub:
                continue
            ns = np.array([r["N"] for r in sub], dtype=float)
            m = np.array([r[key + "_mean"] for r in sub])
            sd = np.array([r[key + "_sd"] for r in sub])
            if arm.startswith("truth"):
                ax.errorbar(ns, m, yerr=sd, fmt="D" if arm == "truth" else "s", ms=6,
                            label=arm + " (N fixed)", capsize=3)
            else:
                ax.plot(ns, m, "o-", label=arm)
                ax.fill_between(ns, m - sd, m + sd, alpha=0.15)
        if key == "D90":
            ax.axhline(RX_CGY, color="k", lw=0.8, ls="--", label="rx")
        ax.set_xlabel("N tiles")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    fig.suptitle("V2: coverage vs N (mean ± SD over %d cavities; grid-based endpoints)" % len(cavities))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "v2_coverage_vs_n.png"), dpi=140)
    plt.close(fig)

    # ---- markdown
    md.append("\n### V2 results (mean ± SD across cavities; +5 mm shell, grid based) %s\n" % CAPTION)
    md.append(md_table([{"N": r["N"], "arm": r["arm"], "n_cav": r["n_cav"],
                         "V100": "%.3f ± %.3f" % (r["V100_mean"], r["V100_sd"]),
                         "D90 [cGy]": "%.0f ± %.0f" % (r["D90_mean"], r["D90_sd"]),
                         "V150": "%.3f ± %.3f" % (r["V150_mean"], r["V150_sd"]),
                         "V200": "%.3f ± %.3f" % (r["V200_mean"], r["V200_sd"]),
                         "solve s": "%.1f" % r["solve_s_mean"]} for r in summary],
                       ["N", "arm", "n_cav", "V100", "D90 [cGy]", "V150", "V200", "solve s"]))
    if paired:
        md.append("\nPaired differences in V100 (pp), t-based 95 % CI, Wilcoxon signed-rank:\n")
        md.append(md_table(paired, ["N", "comparison", "mean_pp", "ci_lo_pp", "ci_hi_pp", "n", "wilcoxon_p"],
                           {"mean_pp": "%.2f", "ci_lo_pp": "%.2f", "ci_hi_pp": "%.2f", "wilcoxon_p": "%.3f"}))
    if primary:
        md.append("\n**Primary endpoint** (SA - uniform, V100 at N*): %.2f pp [%.2f, %.2f], "
                  "n = %d, Wilcoxon p = %.3f; %d cavities where uniform never reached D90 >= rx "
                  "(N* := max N); %d cavities used the largest N with both arms OK instead of N* "
                  "(packing-limited arms); %d cavities with no N where both arms succeeded."
                  % (primary["mean_pp"], primary["ci_lo_pp"], primary["ci_hi_pp"], primary["n"],
                     primary["wilcoxon_p"], primary["n_flagged"], primary["n_fallback_N"],
                     primary["n_no_pair"]))
        md.append("\n" + md_table(primary_rows, ["cavity", "N_star", "N_used", "fallback",
                                                  "uniform_never_reached_rx", "V100_uniform", "V100_sa",
                                                  "diff_pp"], {"diff_pp": "%.2f"}))
    failed_rows = [r for r in rows if r.get("status") == "failed"]
    if failed_rows:
        md.append("\nFAILED arm rows (fewer than N tiles / conflict / bad status; excluded from all statistics):\n")
        md.append(md_table(failed_rows, ["cavity", "N", "arm", "n_placed", "reason"]))
    skipped_rows = sorted({(r["arm"], r["reason"]) for r in rows if r.get("status") == "skipped"})
    if skipped_rows:
        md.append("\nSkipped arms:\n")
        md.append(md_table([{"arm": a, "reason": b} for a, b in skipped_rows], ["arm", "reason"]))
    return {"rows": rows, "summary": summary, "paired": paired, "primary": primary,
            "md": "\n".join(md), "cavities": cavities, "configs": configs}


# ------------------------------------------------------------------- V3
def section_v3(cfg) -> Dict[str, Any]:
    out, md, rows = cfg.out, [], []
    cavities = [get_cavity(s, 1.0, cfg.cache_dir, cfg.use_cache) for s in cfg.seeds]
    for cav in cavities:
        mesh = cav["mesh"]
        target = TargetSet.from_shell(mesh, plan.TARGET_SHELL_OFFSET_MM)
        inst, reason = _try("build_instance(milp)", lambda: build_instance(
            mesh, target, MILP_H_MM, MILP_N_SPINS, MILP_M_OPT, cav["seed"],
            cache_path=os.path.join(cfg.cache_dir, "inst_%s_h%g_s%d_m%d_cg%s.pkl"
                                    % (cav["key"], MILP_H_MM, MILP_N_SPINS, MILP_M_OPT, git_commit())),
            use_cache=cfg.use_cache))
        for n in cfg.v3_n:
            base = {"cavity": cav["key"], "volume_ml": cav["volume_ml"], "N": n,
                    "h_mm": MILP_H_MM, "n_spins": MILP_N_SPINS, "m_opt": MILP_M_OPT}
            if inst is None:
                rows.append(dict(base, status="skipped", reason=reason))
                continue
            total_w = inst["influence"].target.total_weight
            mi, r_mi = _try("milp N=%d" % n, lambda: run_arm("milp", inst, n, 0, milp_inst=inst))
            sa, r_sa = _try("sa N=%d" % n, lambda: run_arm("sa", inst, n, SA_SEED_BASE + cav["seed"]))
            sa_start = "greedy"
            if sa is None and mi is not None and r_sa and "FAILED" in r_sa:
                # coordinator 2026-10-07: when SA's greedy start cannot pack N on the
                # reduced instance, start SA from the enumeration reference instead
                sa, r_sa2 = _try("sa(start=milp) N=%d" % n, lambda: run_arm(
                    "sa", inst, n, SA_SEED_BASE + cav["seed"], start=mi["selection"]))
                sa_start = "milp"
                r_sa = (r_sa + " | restart from milp: " + r_sa2) if sa is None else r_sa
            if sa is None or mi is None:
                st = "failed" if any(x and "FAILED" in x for x in (r_sa, r_mi)) else "skipped"
                rows.append(dict(base, status=st, reason="; ".join(x for x in (r_sa, r_mi) if x),
                                 milp_status=None if mi is None else mi["solver"].status,
                                 V100_milp_influence=None if mi is None else mi["solver"].metrics.get("V100")))
                continue
            v_sa = float(sa["solver"].metrics.get("V100", float("nan")))
            v_mi = float(mi["solver"].metrics.get("V100", float("nan")))
            bound = mi["solver"].bound
            if bound is not None and bound > 1.0 + 1e-9:
                bound = float(bound) / float(total_w)       # unnormalized coverage -> fraction
            ref = bound if (mi["solver"].status == "time_limit" and bound is not None) else v_mi
            ev_sa, _ = evaluate_tiles(mesh, sa["tiles"], target)
            ev_mi, _ = evaluate_tiles(mesh, mi["tiles"], target)
            row = dict(base, status="ok", reason="", V100_sa_influence=v_sa, sa_start=sa_start,
                       V100_milp_influence=v_mi, milp_status=mi["solver"].status,
                       milp_bound=bound, mip_gap=mi["solver"].mip_gap,
                       reference=ref, gap=(v_sa - ref) / ref if ref else float("nan"),
                       V100_sa_grid=ev_sa["V100"], V100_milp_grid=ev_mi["V100"],
                       sa_s=sa["solve_s"], milp_s=mi["solve_s"])
            if mi["solver"] is not None:
                row["milp_method"] = mi["solver"].extra.get("method") if isinstance(mi["solver"].extra, dict) else None
            gr, r_g = _try("greedy N=%d" % n, lambda: run_arm("greedy", inst, n, 0))
            tb = (gr["solve_s"] if gr else 0.0) + sa["solve_s"]
            co, r_c = _try("continuous N=%d" % n, lambda: run_arm(
                "continuous", inst, n, SA_SEED_BASE + cav["seed"], mesh=mesh, target=target,
                time_budget_s=tb))
            if co is not None:
                ev_co, _ = evaluate_tiles(mesh, co["tiles"], target)
                row.update(V100_continuous_grid=ev_co["V100"], continuous_s=co["solve_s"],
                           continuous_budget_s=tb,
                           gap_continuous_grid=((ev_co["V100"] - ev_mi["V100"]) / ev_mi["V100"]
                                                if ev_mi["V100"] else float("nan")))
            else:
                row["continuous_reason"] = r_c
            rows.append(row)
    write_csv(os.path.join(out, "v3_gap.csv"), rows)
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for n in sorted({r["N"] for r in ok}):
            g = [100 * r["gap"] for r in ok if r["N"] == n and np.isfinite(r["gap"])]
            ax.scatter([n] * len(g), g, label="N=%d" % n)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("N tiles")
        ax.set_ylabel("(SA − MILP ref) / ref on V100 [%]")
        ax.set_title("V3 optimality gap (reduced instances h=%g mm, %d spins, M=%d)"
                     % (MILP_H_MM, MILP_N_SPINS, MILP_M_OPT))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "v3_gap.png"), dpi=140)
        plt.close(fig)
        gaps = np.array([r["gap"] for r in ok if np.isfinite(r["gap"])])
        md.append("### V3 optimality gap %s\n" % CAPTION)
        md.append("Reference = MILP incumbent V100 (influence matrix, coverage only), or the "
                  "MILP bound when the time limit (%g s) was hit. Gap over %d instances: "
                  "mean %.2f %%, min %.2f %%, max %.2f %%, time-limit hits: %d.\n"
                  % (MILP_TIME_LIMIT_S, gaps.size, 100 * gaps.mean(), 100 * gaps.min(),
                     100 * gaps.max(), sum(r["milp_status"] == "time_limit" for r in ok)))
        md.append(md_table(ok, ["cavity", "N", "sa_start", "V100_sa_influence", "V100_milp_influence", "milp_status",
                                "milp_method", "milp_bound", "mip_gap", "gap", "V100_sa_grid", "V100_milp_grid",
                                "V100_continuous_grid",
                                "gap_continuous_grid", "milp_s", "continuous_s"],
                           {"gap": "%.4f", "gap_continuous_grid": "%.4f", "milp_s": "%.1f",
                            "continuous_s": "%.1f"}))
        md.append("\n`gap_continuous_grid` compares grid V100 of the continuous solver against the MILP "
                  "incumbent's grid V100 (the continuous solver has no influence-matrix objective; "
                  "it may exceed the discrete MILP because it is not restricted to the candidate grid).")
    else:
        md.append("### V3 optimality gap\n\nNot run: " + "; ".join(sorted({r["reason"] for r in rows})))
    bad = [r for r in rows if r["status"] in ("failed", "skipped")]
    if bad:
        md.append("\nV3 rows not evaluated (FAILED / skipped, never dropped):\n")
        md.append(md_table(bad, ["cavity", "N", "status", "milp_status", "reason"]))
    return {"rows": rows, "md": "\n".join(md)}


# ------------------------------------------------------------------- V4
def section_v4(cfg) -> Dict[str, Any]:
    out, md, rows = cfg.out, [], []
    cavities = [get_cavity(s, 1.0, cfg.cache_dir, cfg.use_cache) for s in cfg.seeds[:2]]
    # two cavities: the h = 2 mm / 12-spin cells dominate the overnight budget
    for cav in cavities:
        mesh = cav["mesh"]
        target = TargetSet.from_shell(mesh, plan.TARGET_SHELL_OFFSET_MM)
        for h in cfg.v4_h:
            for ns in cfg.v4_spins:
                if not cfg.quick and ((ns >= 12 and h < 4.0) or (ns >= 6 and h <= 2.5)):
                    continue        # overnight budget: 12 spins only at h = 4; (2.5, 6) dropped (notes)
                base = {"cavity": cav["key"], "h_mm": h, "n_spins": ns, "N": V4_N}
                inst, reason = _try("instance h=%g spins=%d" % (h, ns), lambda: build_instance(
                    mesh, target, h, ns, M_OPT, cav["seed"], use_cache=False))
                if inst is None:
                    rows.append(dict(base, status="skipped", reason=reason))
                    continue
                row = dict(base, status="ok", reason="", n_candidates=len(inst["candidates"]),
                           n_pairs=inst["conflicts"].count_pairs(), **{
                               "t_" + k: v for k, v in inst["timings"].items()})
                for arm in ("greedy", "sa", "continuous"):
                    tb = (row.get("greedy_s", 0.0) + row.get("sa_s", 0.0)) or None
                    res, reason = _try("%s h=%g spins=%d" % (arm, h, ns),
                                       lambda: run_arm(arm, inst, V4_N, SA_SEED_BASE + cav["seed"],
                                                       mesh=mesh, target=target,
                                                       time_budget_s=tb if arm == "continuous" else None))
                    if res is None:
                        row[arm + "_reason"] = reason
                        continue
                    ev, _ = evaluate_tiles(mesh, res["tiles"], target)
                    row[arm + "_objective"] = res["solver"].objective if res["solver"] is not None else None
                    row[arm + "_V100"] = ev["V100"]
                    row[arm + "_s"] = res["solve_s"]
                    row[arm + "_n_placed"] = res["n_placed"]
                    if arm == "sa":
                        ref, reason = _try("refine_continuous", lambda: plan.refine_continuous(
                            mesh, inst["candidates"], res["selection"], target, rx_cgy=RX_CGY))
                        if ref is not None:
                            tiles_ref, info = ref
                            ev2, _ = evaluate_tiles(mesh, tiles_ref, target)
                            row["e5_V100"] = ev2["V100"]
                            row["e5_gain_pp"] = 100 * (ev2["V100"] - ev["V100"])
                            row["e5_info"] = json.dumps({k: v for k, v in info.items()
                                                         if isinstance(v, (int, float, str))})
                        else:
                            row["e5_reason"] = reason
                rows.append(row)
    write_csv(os.path.join(out, "v4_discretization.csv"), rows)
    ok = [r for r in rows if r["status"] == "ok"]
    md.append("### V4 discretization (N = %d, M = %d) %s\n" % (V4_N, M_OPT, CAPTION))
    if ok:
        md.append(md_table(ok, ["cavity", "h_mm", "n_spins", "n_candidates", "t_candidates", "t_influence",
                                "greedy_V100", "greedy_s", "sa_V100", "sa_s", "e5_V100", "e5_gain_pp",
                                "continuous_V100", "continuous_s"],
                           {"t_candidates": "%.1f", "t_influence": "%.1f", "greedy_s": "%.1f", "sa_s": "%.1f",
                            "e5_gain_pp": "%.2f", "continuous_s": "%.1f"}))
    else:
        md.append("Not run: " + "; ".join(sorted({r["reason"] for r in rows})))
    return {"rows": rows, "md": "\n".join(md)}


# ------------------------------------------------------------------- V5
def section_v5(cfg) -> Dict[str, Any]:
    out, md, rows = cfg.out, [], []
    cavities = [get_cavity(s, 1.0, cfg.cache_dir, cfg.use_cache) for s in cfg.seeds[:3]]
    sk0 = None
    try:
        from gtcore.dose.engine import TG43Engine
        sk0 = float(TG43Engine.DEFAULT_SK_U)
    except Exception:                                                 # pragma: no cover
        sk0 = 3.5
    tau_rows = []
    for cav in cavities:
        mesh, truth = cav["mesh"], cav["truth"]
        target = TargetSet.from_shell(mesh, plan.TARGET_SHELL_OFFSET_MM)
        configs: Dict[str, List[PlacedTile]] = {}
        tiles, reason = _try("truth", lambda: truth_tiles_conformed(mesh, truth))
        if tiles is not None:
            configs["truth"] = tiles
        inst, reason = _try("instance", lambda: build_instance(
            mesh, target, H_MM, N_SPINS, M_OPT, cav["seed"],
            cache_path=os.path.join(cfg.cache_dir, "inst_%s_h%g_s%d_m%d_cg%s.pkl"
                                    % (cav["key"], H_MM, N_SPINS, M_OPT, git_commit())),
            use_cache=cfg.use_cache))
        for arm in ("uniform", "greedy", "sa"):
            res, reason = _try(arm, lambda: run_arm(arm, inst, V5_N, SA_SEED_BASE + cav["seed"]))
            if res is not None:
                configs[arm] = res["tiles"]
        if not configs:
            rows.append({"cavity": cav["key"], "status": "skipped", "reason": reason})
            continue
        # perturbations applied at evaluation time
        perts: List[Tuple[str, Dict[str, Any]]] = [("nominal", {})]
        for off in V5_OFFSETS_MM:
            if abs(off - 3.0) > 1e-9:
                perts.append(("seed_plane_%.2fmm" % off, {"offset": off}))
        perts += [("sk_-5%", {"sk": 0.95 * sk0}), ("sk_+5%", {"sk": 1.05 * sk0}),
                  ("interference_on", {"interference": True})]
        table: Dict[str, Dict[str, float]] = {}
        for pname, p in perts:
            for arm, tiles in configs.items():
                t_eval = shift_seed_plane(tiles, p["offset"]) if "offset" in p else tiles
                ev, _ = evaluate_tiles(mesh, t_eval, target, interference=p.get("interference", False),
                                       sk_per_seed_u=p.get("sk"))
                rows.append({"cavity": cav["key"], "perturbation": pname, "arm": arm, "status": "ok",
                             "V100": ev["V100"], "D90": ev["D90"], "V150": ev["V150"], "V200": ev["V200"]})
                table.setdefault(pname, {})[arm] = ev["V100"]
        # target subsample density at SOLVE time: M = 4000 instead of 1000
        inst4, reason = _try("instance M=4000", lambda: build_instance(
            mesh, target, H_MM, N_SPINS, plan.M_OPT_MAX, cav["seed"], use_cache=False))
        for arm in ("greedy", "sa"):
            res, reason = _try(arm + " M=4000", lambda: run_arm(arm, inst4, V5_N, SA_SEED_BASE + cav["seed"]))
            if res is None:
                rows.append({"cavity": cav["key"], "perturbation": "m_opt_4000", "arm": arm,
                             "status": "skipped", "reason": reason})
                continue
            ev, _ = evaluate_tiles(mesh, res["tiles"], target)
            rows.append({"cavity": cav["key"], "perturbation": "m_opt_4000", "arm": arm, "status": "ok",
                         "V100": ev["V100"], "D90": ev["D90"], "V150": ev["V150"], "V200": ev["V200"]})
            table.setdefault("m_opt_4000", {})[arm] = ev["V100"]
        arms = sorted(table["nominal"])
        nominal = [table["nominal"][a] for a in arms]
        for pname in table:
            if pname == "nominal":
                continue
            common = [a for a in arms if a in table[pname]]
            if len(common) < 2:
                continue
            tau = kendall_tau([table["nominal"][a] for a in common], [table[pname][a] for a in common])
            tau_rows.append({"cavity": cav["key"], "perturbation": pname, "arms": ",".join(common),
                             "kendall_tau": tau})
        _ = nominal
    write_csv(os.path.join(out, "v5_sensitivity.csv"), rows)
    write_csv(os.path.join(out, "v5_rank_stability.csv"), tau_rows)
    ok = [r for r in rows if r["status"] == "ok"]
    md.append("### V5 sensitivity (N = %d; V100 of +5 mm shell) %s\n" % (V5_N, CAPTION))
    if ok:
        # spread per arm: max - min over perturbations (nominal included), averaged over cavities
        spread = []
        for arm in sorted({r["arm"] for r in ok}):
            per_cav = []
            for key in sorted({r["cavity"] for r in ok}):
                v = [r["V100"] for r in ok if r["arm"] == arm and r["cavity"] == key]
                if len(v) > 1:
                    per_cav.append(max(v) - min(v))
            m, sd, n = mean_sd(per_cav)
            spread.append({"arm": arm, "spread_pp_mean": 100 * m, "spread_pp_sd": 100 * sd, "n_cav": n})
        md.append(md_table(spread, ["arm", "spread_pp_mean", "spread_pp_sd", "n_cav"],
                           {"spread_pp_mean": "%.2f", "spread_pp_sd": "%.2f"}))
        piv = []
        for key in sorted({r["cavity"] for r in ok}):
            for pname in sorted({r["perturbation"] for r in ok}):
                row = {"cavity": key, "perturbation": pname}
                for r in ok:
                    if r["cavity"] == key and r["perturbation"] == pname:
                        row[r["arm"]] = r["V100"]
                piv.append(row)
        md.append("\n" + md_table(piv, ["cavity", "perturbation", "truth", "uniform", "greedy", "sa"]))
        if tau_rows:
            md.append("\nArm-ranking stability (Kendall tau vs nominal):\n")
            md.append(md_table(tau_rows, ["cavity", "perturbation", "arms", "kendall_tau"], {"kendall_tau": "%.2f"}))
    else:
        md.append("Not run: " + "; ".join(sorted({r.get("reason", "") for r in rows})))
    return {"rows": rows, "md": "\n".join(md)}


# ------------------------------------------------------------------- V6
def load_printed_phantom(cache_dir: str, use_cache: bool = True, path: str = V6_PHANTOM_DIR) -> Dict[str, Any]:
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, "printed_phantom_%s.pkl" % os.path.basename(path.rstrip("\\/"))[:24])
    if use_cache and os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            return pickle.load(fh)
    if not os.path.isdir(path):
        raise Skip("printed phantom directory not found: %s" % path)
    from gtcore.io import load_volume
    from gtcore.pipeline import reconstruct
    from gtcore.tiles.auto import to_placed_tiles
    t0 = time.perf_counter()
    vol = load_volume(path)
    res = reconstruct(vol, verbose=False, n_full_tiles="auto", n_half_tiles=0)
    t_rec = time.perf_counter() - t0
    if "cavity" in res.meshes and np.asarray(res.cavity_mask).any():
        wall = res.meshes["cavity"]
        wall_name = "cavity"
    elif "body" in res.meshes:
        wall = res.meshes["body"]
        wall_name = "body"
    else:
        raise Skip("no cavity or body mesh in the pipeline result")
    tiles = to_placed_tiles(res.tiles, res.seeds.centers_ras, res.seeds.axes_ras) if res.tiles else []
    data = {"mesh": trimesh.Trimesh(vertices=np.asarray(wall.vertices), faces=np.asarray(wall.faces),
                                    process=False),
            "wall_name": wall_name, "seed_centers": np.asarray(res.seeds.centers_ras, dtype=float),
            "seed_axes": np.asarray(res.seeds.axes_ras, dtype=float), "tiles": tiles,
            "reconstruct_s": t_rec, "spacing": tuple(float(s) for s in vol.spacing),
            "shape": tuple(int(s) for s in vol.array.shape), "path": path,
            "has_cavity_mask": bool(np.asarray(res.cavity_mask).any())}
    with open(cpath, "wb") as fh:
        pickle.dump(data, fh)
    return data


def section_v6(cfg) -> Dict[str, Any]:
    out, md, rows = cfg.out, [], []
    ph, reason = _try("printed phantom", lambda: load_printed_phantom(cfg.cache_dir, cfg.use_cache))
    info: Dict[str, Any] = {}
    if ph is None:
        md.append("### V6 printed phantom\n\nNot run: %s" % reason)
        return {"rows": rows, "md": "\n".join(md), "info": info}
    mesh = ph["mesh"]
    centroid = ph["seed_centers"].mean(axis=0)
    vis_cache = os.path.join(cfg.cache_dir, "printed_visible_%d.npy" % len(mesh.faces))
    if cfg.use_cache and os.path.exists(vis_cache):
        vis = np.load(vis_cache)
        vis_src = "cached mask (%s)" % vis_cache
    else:
        vis, vis_src = visible_faces_any(mesh, centroid)
        np.save(vis_cache, vis)
    cen = np.asarray(mesh.triangles_center, dtype=float)
    from scipy.spatial import cKDTree
    d_seed, _ = cKDTree(ph["seed_centers"]).query(cen)
    near = d_seed <= V6_SEED_RADIUS_MM
    eligible = vis & near
    area_f = np.asarray(mesh.area_faces, dtype=float)
    info.update({"wall": ph["wall_name"], "mesh_area_mm2": float(mesh.area),
                 "mesh_volume_ml": float(mesh.volume) / 1000.0 if mesh.is_watertight else float("nan"),
                 "n_faces": int(len(mesh.faces)), "n_seeds": int(len(ph["seed_centers"])),
                 "n_tiles_fitted": len(ph["tiles"]), "visible_faces_source": vis_src,
                 "visible_area_mm2": float(area_f[vis].sum()),
                 "eligible_area_mm2": float(area_f[eligible].sum()),
                 "eligible_faces": int(eligible.sum()), "seed_radius_mm": V6_SEED_RADIUS_MM,
                 "reconstruct_s": ph["reconstruct_s"], "has_cavity_mask": ph["has_cavity_mask"]})
    rec, reason = _try("recommend_tile_count", lambda: plan.recommend_tile_count(mesh, eligible_faces=eligible))
    if rec is not None:
        info["recommended_tiles"] = int(rec.n_tiles)
        info["recommended_source"] = "plan.recommend_tile_count"
        info["recommended_ellipsoid"] = int(rec.n_tiles_ellipsoid)
    else:
        info["recommended_tiles"] = tile_count_rule(info["eligible_area_mm2"])
        info["recommended_source"] = "local rule ceil(area / 4 cm^2): " + str(reason)
    target = target_from_faces(mesh, eligible, plan.TARGET_SHELL_OFFSET_MM)
    info["target_points"] = len(target)
    info["target_area_mm2"] = target.total_weight
    _log("V6: %s mesh %.0f mm2, visible %.0f mm2, eligible %.0f mm2 (%d faces), %d seeds, "
         "%d fitted tiles, recommended %d (%s)"
         % (ph["wall_name"], info["mesh_area_mm2"], info["visible_area_mm2"], info["eligible_area_mm2"],
            info["eligible_faces"], info["n_seeds"], info["n_tiles_fitted"], info["recommended_tiles"],
            info["recommended_source"]))

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _record(arm, tiles, n, solver=None, extra=None):
        ev, rep = evaluate_tiles(mesh, tiles, target, margin_mm=REPORT_MARGIN_MM, solver_result=solver,
                                 use_target=True)
        row = {"arm": arm, "N": n, "status": "ok", "reason": "", **ev}
        if extra:
            row.update(extra)
        rows.append(row)
        tgt = rep.metrics_grid["target"]
        curves[arm] = (np.asarray(tgt["curve_x"]), np.asarray(tgt["curve_y"]))
        rep.to_json(os.path.join(out, "v6_%s.json" % arm.replace("+", "_").replace(" ", "_")))
        return ev

    # (a) as implanted, localized seeds
    tiles_a = ph["tiles"]
    if tiles_a:
        all_seeds = np.vstack([t.seed_centers for t in tiles_a])
        _cp, seed_dist, _tid = trimesh.proximity.closest_point(mesh, all_seeds)
        ev_a = _record("as_implanted", tiles_a, len(tiles_a),
                       extra={"mean_seed_wall_mm": float(np.mean(seed_dist)),
                              "max_seed_wall_mm": float(np.max(seed_dist))})
        info["mean_seed_wall_mm"] = float(np.mean(seed_dist))

        # (a') through the conformer: snap the tile's seed centroid (~1 mm off
        # the inner wall) rather than its anchor, and orient the normal toward
        # that centroid -- on a closed shell mesh the anchor can lie inside
        # the plastic, where "nearest point" may be the outer surface.
        def _conform():
            outl = []
            for t in tiles_a:
                surf, n_in = snap_to_wall(mesh, t.center_ras)
                if float(n_in @ (t.center_ras - surf)) < 0.0:
                    n_in = -n_in
                outl.append(conform_tile(mesh, surf, n_in, t.axis_ras, kind=t.kind))
            return outl
        tiles_c, reason = _try("conform as-implanted", _conform)
        if tiles_c is not None:
            ev_c = _record("as_implanted_conformed", tiles_c, len(tiles_c))
            info["conformer_delta_V100_pp"] = 100 * (ev_c["V100"] - ev_a["V100"])
            info["conformer_delta_D90_cgy"] = ev_c["D90"] - ev_a["D90"]
            # order-free: each conformed seed to its nearest localized seed
            ds = [float(np.linalg.norm(c.seed_centers[:, None] - a.seed_centers[None], axis=2)
                        .min(axis=1).mean()) for a, c in zip(tiles_a, tiles_c)]
            info["conformer_seed_shift_mm"] = float(np.mean(ds))
    else:
        rows.append({"arm": "as_implanted", "N": 0, "status": "skipped", "reason": "no fitted tiles"})

    # (b) optimized with the same N
    inst, reason = _try("instance (printed)", lambda: build_instance(
        mesh, target, H_MM, N_SPINS, M_OPT, 0, eligible_faces=eligible,
        cache_path=os.path.join(cfg.cache_dir, "inst_printed_h%g_s%d_m%d_cg%s.pkl"
                                % (H_MM, N_SPINS, M_OPT, git_commit())), use_cache=cfg.use_cache))
    milp_inst, r2 = _try("instance (printed, milp)", lambda: build_instance(
        mesh, target, MILP_H_MM, MILP_N_SPINS, MILP_M_OPT, 0, eligible_faces=eligible, use_cache=False))
    n_opt = len(tiles_a) if tiles_a else V6_N
    budget = {}
    for arm in ("uniform", "greedy", "sa", "milp", "continuous"):
        tb = (budget.get("greedy", 0.0) + budget.get("sa", 0.0)) or None
        res, reason = _try("%s N=%d" % (arm, n_opt), lambda: run_arm(
            arm, inst, n_opt, SA_SEED_BASE, mesh=mesh, target=target, milp_inst=milp_inst,
            time_budget_s=tb if arm == "continuous" else None))
        if res is None:
            st, n_placed = row_status(reason)
            rows.append({"arm": arm, "N": n_opt, "status": st, "reason": reason, "n_placed": n_placed})
            continue
        budget[arm] = res["solve_s"]
        _record(arm, res["tiles"], n_opt, solver=res["solver"],
                extra={"solve_s": res["solve_s"], "n_placed": res["n_placed"]})

    # (c) minimum N by P2
    sw, reason = _try("sweep_n", lambda: plan.sweep_n(
        mesh, target, cfg.v6_n_max, rx_cgy=RX_CGY, solver="greedy", seed=0,
        candidates=None if inst is None else inst["candidates"],
        influence=None if inst is None else inst["influence"],
        conflicts=None if inst is None else inst["conflicts"]))
    if sw is not None:
        info["min_n"] = {k: v for k, v in sw.min_n.items()}
        write_csv(os.path.join(out, "v6_sweep.csv"), sw.rows)
    else:
        info["min_n_reason"] = reason
    if cfg.clinical:
        info["clinical"] = "hook present; path given: %s -- NOT RUN (no cavity surface / RTSTRUCT handling yet)" % cfg.clinical
    else:
        info["clinical"] = "clinical case 2 is not on this machine: not run"
    write_csv(os.path.join(out, "v6_phantom.csv"), rows)
    with open(os.path.join(out, "v6_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)

    if curves:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for arm, (x, y) in curves.items():
            ax.plot(100 * x, 100 * y, label=arm)
        ax.axvline(100, color="k", lw=0.8, ls="--")
        ax.set_xlabel("dose [% rx]")
        ax.set_ylabel("eligible +5 mm shell at ≥ dose [% area]")
        ax.set_title("V6 printed phantom: as-implanted vs optimized (weighted shell DVH)")
        ax.set_xlim(0, 300)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "v6_dvh.png"), dpi=140)
        plt.close(fig)

    md.append("### V6 printed phantom %s\n" % CAPTION)
    md.append("Endpoints below are the WEIGHTED stats of the eligible +5 mm target (the mesh's own "
              "shell vertices include the outer surface of the printed shell; those are kept as "
              "`*_shellverts` in `v6_phantom.csv`).\n")
    md.append("Wall = `meshes[\"%s\"]` (%.0f mm², %.1f mL enclosed, %d faces); eligible = faces visible "
              "from the implant centroid (%s) AND within %g mm of a detected seed: %.0f mm² (%d faces); "
              "target = +5 mm shell of the eligible faces (%d points, %.0f mm²). %d localized seeds, "
              "%d fitted tiles. Recommended tiles: %d (%s)."
              % (info["wall"], info["mesh_area_mm2"], info["mesh_volume_ml"], info["n_faces"],
                 info["visible_faces_source"], V6_SEED_RADIUS_MM, info["eligible_area_mm2"],
                 info["eligible_faces"], info["target_points"], info["target_area_mm2"], info["n_seeds"],
                 info["n_tiles_fitted"], info["recommended_tiles"], info["recommended_source"]))
    if "conformer_delta_V100_pp" in info:
        md.append("\nConformer substitution alone: localized seeds sit %.2f mm off this mesh on average, "
                  "the conformer puts them at 3 mm (mean seed shift %.2f mm): V100 %+.2f pp, D90 %+.0f cGy."
                  % (info.get("mean_seed_wall_mm", float("nan")), info["conformer_seed_shift_mm"],
                     info["conformer_delta_V100_pp"], info["conformer_delta_D90_cgy"]))
    md.append("\n" + md_table(rows, ["arm", "N", "n_placed", "status", "V100", "D90", "V150", "V200", "V100w",
                                     "n_overlaps", "solve_s", "reason"], {"D90": "%.0f", "solve_s": "%.1f"}))
    md.append("\nMinimum N (P2): %s" % json.dumps(info.get("min_n", info.get("min_n_reason"))))
    md.append("\nClinical case: %s" % info["clinical"])
    return {"rows": rows, "md": "\n".join(md), "info": info}


# ------------------------------------------------------------------- V7
def section_v7(cfg) -> Dict[str, Any]:
    out, md, rows = cfg.out, [], []
    hw = hardware()
    cavities = [get_cavity(cfg.seeds[0], x, cfg.cache_dir, cfg.use_cache) for x in cfg.scales]
    for cav in cavities:
        mesh = cav["mesh"]
        target = TargetSet.from_shell(mesh, plan.TARGET_SHELL_OFFSET_MM)
        row = {"cavity": cav["key"], "volume_ml": cav["volume_ml"], "area_mm2": cav["area_mm2"], "N": V4_N}
        inst, reason = _try("instance", lambda: build_instance(mesh, target, H_MM, N_SPINS, plan.M_OPT_MAX,
                                                                 cav["seed"], use_cache=False))
        if inst is not None:
            row["n_candidates"] = len(inst["candidates"])
            row["n_pairs_planner"] = inst.get("n_pairs_planner")
            row["n_pairs_added"] = inst.get("n_pairs_added")
            for k, v in inst["timings"].items():
                row["t_" + k] = v
            for arm in ("greedy", "greedy+local", "sa", "milp", "continuous"):
                milp_inst = None
                tb = ((row.get("t_greedy") or 0.0) + (row.get("t_sa") or 0.0)) or None
                if arm == "milp":
                    milp_inst, _r = _try("instance(milp)", lambda: build_instance(
                        mesh, target, MILP_H_MM, MILP_N_SPINS, MILP_M_OPT, cav["seed"], use_cache=False))
                    if milp_inst is not None:
                        row["t_milp_instance"] = sum(milp_inst["timings"].values())
                res, reason = _try(arm, lambda: run_arm(arm, inst, V4_N, SA_SEED_BASE + cav["seed"],
                                                       mesh=mesh, target=target, milp_inst=milp_inst,
                                                       time_budget_s=tb if arm == "continuous" else None))
                row["t_" + arm] = res["solve_s"] if res is not None else None
                if arm == "continuous":
                    row["continuous_budget_s"] = tb
                if res is None:
                    row["reason_" + arm] = reason
            sel, _r = _try("uniform", lambda: uniform_heuristic(inst["candidates"], inst["conflicts"], V4_N))
            tiles = None if sel is None else inst["candidates"].tiles_of(sel)
        else:
            row["reason_instance"] = reason
            tiles = None
        if tiles is None:
            tiles, _r = _try("truth", lambda: truth_tiles_conformed(mesh, cav["truth"]))
        if tiles is not None:
            t0 = time.perf_counter()
            evaluate_tiles(mesh, tiles, target, margin_mm=FULL_MARGIN_MM, cavity_mask=cav["cavity_mask"],
                           cavity_affine=cav["affine"], shadowing=True)
            row["t_final_report_full_50mm"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            evaluate_tiles(mesh, tiles, target, margin_mm=REPORT_MARGIN_MM)
            row["t_final_report_15mm"] = time.perf_counter() - t0
        rows.append(row)
    write_csv(os.path.join(out, "v7_runtime.csv"), rows)
    with open(os.path.join(out, "v7_hardware.json"), "w", encoding="utf-8") as fh:
        json.dump(hw, fh, indent=2)
    md.append("### V7 runtime (s; N = %d, h = %g mm, %d spins, M = %d) %s\n"
              % (V4_N, H_MM, N_SPINS, plan.M_OPT_MAX, CAPTION))
    md.append("Hardware: %s; %s; %s cores; Python %s; numpy %s\n"
              % (hw["platform"], hw["processor"], hw["cores"], hw["python"], hw["numpy"]))
    cols = ["cavity", "volume_ml", "n_candidates", "t_candidates", "t_conflicts", "t_influence", "t_greedy",
            "t_greedy+local", "t_sa", "t_milp", "t_continuous", "t_final_report_15mm",
            "t_final_report_full_50mm"]
    md.append(md_table(rows, cols, {c: "%.1f" for c in cols if c.startswith("t_")}))
    return {"rows": rows, "md": "\n".join(md)}


# ------------------------------------------------------------------- V8
def _disc_faces(mesh, center, radius_mm):
    cen = np.asarray(mesh.triangles_center, dtype=float)
    return np.linalg.norm(cen - np.asarray(center, dtype=float), axis=1) <= radius_mm


def section_v8(cfg) -> Dict[str, Any]:
    out, md, rows = cfg.out, [], []
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import plan_fixtures as pf

    tiny = trimesh.creation.icosphere(subdivisions=3, radius=12.0)
    flat = pf.flat_wall_mesh(size_mm=60.0, step_mm=2.0)
    cen = np.asarray(flat.triangles_center)
    disc = pf.flat_wall_top_faces(flat) & (np.linalg.norm(cen[:, :2], axis=1) <= 5.0)
    narrow = pf.flat_wall_mesh(size_mm=12.0, step_mm=2.0)       # wall patch narrower than a tile
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=25.0)
    tiny_mask = np.asarray(sphere.triangles_center)[:, 2] > 24.3          # ~ < 5 % of the wall
    degenerate = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float),
                                 faces=np.array([[0, 1, 2]]), process=False)
    empty = trimesh.Trimesh()

    cases = [
        ("N larger than any packing (sphere r=12 mm, N=12)", "optimize",
         lambda: plan.optimize(tiny, n_full=12, solver="greedy", report=False)),
        ("N larger than any packing (sphere r=12 mm, N=12)", "build_candidates+solve_greedy",
         lambda: _greedy_pipeline(tiny, 12)),
        ("tile wider than the wall patch (12 mm wall mesh)", "build_candidates",
         lambda: _require_candidates(narrow)),
        ("tile wider than the wall patch (12 mm wall mesh)", "optimize",
         lambda: plan.optimize(narrow, n_full=1, report=False)),
        ("eligibility disc (10 mm) smaller than a tile on a 60 mm wall", "optimize",
         lambda: _checked_plan(plan.optimize(flat, n_full=1, eligible_faces=disc, report=False),
                               flat, disc, 1)),
        ("eligibility disc (10 mm) smaller than a tile, N=2 (cannot fit)", "optimize",
         lambda: _checked_plan(plan.optimize(flat, n_full=2, eligible_faces=disc, report=False),
                               flat, disc, 2)),
        ("eligibility mask excluding > 95 % of the wall (sphere cap, N=4)", "optimize",
         lambda: _checked_plan(plan.optimize(sphere, n_full=4, eligible_faces=tiny_mask, report=False),
                               sphere, tiny_mask, 4)),
        ("degenerate mesh (3 vertices)", "build_candidates", lambda: _require_candidates(degenerate)),
        ("degenerate mesh (3 vertices)", "optimize", lambda: plan.optimize(degenerate, n_full=2, report=False)),
        ("empty mesh", "build_candidates", lambda: _require_candidates(empty)),
        ("empty mesh", "optimize", lambda: plan.optimize(empty, n_full=2, report=False)),
        ("empty mesh", "final_report", lambda: final_report(empty, [], grid_mm=2.0)),
    ]
    for case, fn_name, fn in cases:
        t0 = time.perf_counter()
        try:
            res = fn()
        except NotImplementedError as exc:
            rows.append({"case": case, "function": fn_name, "outcome": "stub", "verdict": "not run",
                         "message": str(exc)[:160]})
            continue
        except Exception as exc:                                      # noqa: BLE001
            msg = str(exc).strip()
            rows.append({"case": case, "function": fn_name, "outcome": "raised " + type(exc).__name__,
                         "verdict": "PASS" if msg else "FAIL (empty reason)", "message": msg[:160],
                         "s": time.perf_counter() - t0})
            continue
        # returned: loud only if a status/reason says so, or the plan is valid
        verdict, outcome, msg = "FAIL (silent return)", "returned", ""
        if isinstance(res, dict) and "verdict" in res:
            verdict, outcome, msg = res["verdict"], res["outcome"], res["message"]
        elif isinstance(res, tuple) and len(res) == 2 and hasattr(res[1], "solver"):
            tiles, rep = res
            sr = rep.solver
            if sr is not None and sr.status != "ok" and sr.reason:
                verdict, outcome, msg = "PASS", "status=%s" % sr.status, sr.reason
            elif len(tiles) == 0:
                verdict, outcome = "FAIL (empty plan, no reason)", "returned []"
        elif hasattr(res, "status"):
            if res.status != "ok" and res.reason:
                verdict, outcome, msg = "PASS", "status=%s" % res.status, res.reason
        rows.append({"case": case, "function": fn_name, "outcome": outcome, "verdict": verdict,
                     "message": str(msg)[:160], "s": time.perf_counter() - t0})
    write_csv(os.path.join(out, "v8_failure_modes.csv"), rows)
    md.append("### V8 failure modes %s\n" % CAPTION)
    md.append(md_table(rows, ["case", "function", "outcome", "verdict", "message"]))
    return {"rows": rows, "md": "\n".join(md)}


def _checked_plan(res, mesh, eligible, n):
    """Classify an ``optimize`` return on an eligibility case: a valid plan
    (N tiles, no overlap, every anchor on an eligible face) is a PASS that is
    NOT a failure mode; a short / overlapping / off-mask plan without a
    reason is a FAIL."""
    tiles, rep = res
    sr = rep.solver
    if sr is not None and sr.status != "ok" and sr.reason:
        return {"verdict": "PASS", "outcome": "status=%s" % sr.status, "message": sr.reason}
    ov = find_overlapping_tiles(tiles)
    px = proxy_overlaps(tiles)
    _cp, _d, tid = trimesh.proximity.closest_point(mesh, np.array([t.anchor_ras for t in tiles]))
    on_e = bool(np.all(np.asarray(eligible, dtype=bool)[np.asarray(tid)]))
    msg = ("%d tiles; planner rule: %d overlaps; geometric proxy: %d overlaps; anchors on eligible "
           "faces: %s" % (len(tiles), len(ov), len(px), on_e))
    if len(tiles) == n and not ov and not px and on_e:
        return {"verdict": "PASS (valid plan; not a failure mode)", "outcome": "returned", "message": msg}
    if len(tiles) == n and not ov and px:
        return {"verdict": "FAIL under the planner rule alone (silent overlapping plan); the proxy "
                           "makes it fail loudly", "outcome": "returned", "message": msg}
    return {"verdict": "FAIL (silent: " + msg + ")", "outcome": "returned", "message": msg}


def _require_candidates(mesh, eligible=None):
    cand = plan.build_candidates(mesh, h_mm=H_MM, n_spins=3, eligible_faces=eligible)
    if len(cand) == 0:
        raise ValueError("build_candidates: zero candidates (rejections: %r)" % (cand.n_rejected,))
    return cand


def _greedy_pipeline(mesh, n):
    target = TargetSet.from_shell(mesh, plan.TARGET_SHELL_OFFSET_MM)
    inst = build_instance(mesh, target, 3.0, 3, M_OPT, 0, use_cache=False)
    res = plan.solve_greedy(inst["objective"], n)
    if res.status == "ok" and res.selection.size < n:
        raise AssertionError("greedy returned %d < %d tiles with status ok (silent)" % (res.selection.size, n))
    return res


# ------------------------------------------------------------------- main
class Config:
    pass


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--section", default="all",
                   help="v2|v3|v4|v5|v6|v7|v8|all (comma-separated allowed)")
    p.add_argument("--quick", action="store_true", help="2 cavities, N in (4, 8), n_random 10")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--scales", type=float, nargs="*", default=None)
    p.add_argument("--n-list", type=int, nargs="*", default=None)
    p.add_argument("--n-random", type=int, default=None)
    p.add_argument("--arms", default=None, help="comma-separated subset of the V2 arms")
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--clinical", default=None, help="clinical case DICOM directory (hook; not run)")
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args(argv)
    cfg = Config()
    cfg.sections = [s.strip() for s in a.section.split(",")]
    if "all" in cfg.sections:
        cfg.sections = ["v2", "v3", "v4", "v5", "v6", "v7", "v8"]
    cfg.quick = a.quick
    cfg.seeds = tuple(a.seeds) if a.seeds else ((1, 2) if a.quick else SEEDS)
    cfg.scales = tuple(a.scales) if a.scales else ((1.0,) if a.quick else SCALES)
    cfg.n_list = tuple(a.n_list) if a.n_list else ((4, 8) if a.quick else N_LIST)
    cfg.n_random = a.n_random if a.n_random is not None else (10 if a.quick else N_RANDOM)
    cfg.arms = tuple(a.arms.split(",")) if a.arms else ARM_ORDER
    cfg.v4_h = (4.0, 2.5) if a.quick else V4_H
    cfg.v4_spins = (3, 6) if a.quick else V4_SPINS
    cfg.v6_n_max = 10 if a.quick else V6_N_MAX
    cfg.v3_n = (4, 8) if a.quick else V3_N
    cfg.out = a.out
    cfg.cache_dir = os.path.join(a.out, "cache")
    cfg.use_cache = not a.no_cache
    cfg.clinical = a.clinical
    cfg.argv = list(argv if argv is not None else sys.argv[1:])
    return cfg


SECTIONS = {"v2": section_v2, "v3": section_v3, "v4": section_v4, "v5": section_v5,
            "v6": section_v6, "v7": section_v7, "v8": section_v8}


def main(argv=None) -> Dict[str, Any]:
    try:                                   # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                 # pragma: no cover
        pass
    cfg = parse_args(argv)
    os.makedirs(cfg.out, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    commit = git_commit()
    blocks = ["## validation_optimize run %s — commit %s — `python scripts/validation_optimize.py %s`\n"
              % (time.strftime("%Y-%m-%d %H:%M"), commit, " ".join(cfg.argv)),
              "Seeds %s × scales %s; N %s; n_random %d; rx %.0f cGy; grid %g mm; report margin %g mm; "
              "h %g mm; %d spins; M_opt %d; MILP reduced h %g mm / %d spins / M %d / %g s, N in %s.\n"
              % (cfg.seeds, cfg.scales, cfg.n_list, cfg.n_random, RX_CGY, GRID_MM, REPORT_MARGIN_MM,
                 H_MM, N_SPINS, M_OPT, MILP_H_MM, MILP_N_SPINS, MILP_M_OPT, MILP_TIME_LIMIT_S, cfg.v3_n)]
    results = {}
    for sec in cfg.sections:
        fn = SECTIONS.get(sec)
        if fn is None:
            _log("unknown section %r" % sec)
            continue
        _log("=== %s ===" % sec.upper())
        t0 = time.perf_counter()
        res, reason = _try("section " + sec, lambda: fn(cfg))
        wall = time.perf_counter() - t0
        if res is None:
            res = {"md": "### %s\n\nSection failed: %s" % (sec.upper(), reason)}
        results[sec] = res
        blocks.append(res["md"])
        blocks.append("\n_%s: %.0f s wall._\n" % (sec, wall))
        append_run(cfg.out, sec, ["scripts/validation_optimize.py"] + cfg.argv, cfg.seeds, wall,
                   note="quick" if cfg.quick else "")
    md = "\n".join(blocks)
    with open(os.path.join(cfg.out, "notes_block_%s.md" % time.strftime("%Y%m%d_%H%M%S")),
              "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + "=" * 78 + "\nMARKDOWN BLOCK (paste into docs/optimize-notes.md)\n" + "=" * 78 + "\n")
    print(md)
    return results


if __name__ == "__main__":
    main()
