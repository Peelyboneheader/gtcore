"""Scout for plan-tile-optimize section 7.3: does the exact MILP (section 3 E4)
scale on reduced instances, and how tight are the cheap bounds when it does
not?

Evidence-only script (never merged into gtcore).  Builds its own candidate
sets on synthetic cavities, assembles the E4 MILP in several formulations and
solves it with scipy.optimize.milp (HiGHS) under a time limit, then compares
the solver's dual bound with the LP relaxation and a pointwise "N-best" bound
against a greedy (+ 1-swap local search) incumbent.

    python scripts/scout_milp_bound.py            # declared default grid
    python scripts/scout_milp_bound.py --probe    # one small instance

Outputs: output/scout_milp/results.csv (one row per solve / bound),
         output/scout_milp/instances.csv (one row per instance build),
         output/scout_milp/cache/*.pkl (meshes, candidate sets, influence).
Fixed seeds throughout; identical inputs give identical outputs.  Rows
already present in results.csv are skipped, so the grid can be extended by
re-running with more --N / --forms / --limits.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import trimesh                                                     # noqa: E402
from scipy.optimize import milp, LinearConstraint, Bounds            # noqa: E402
from scipy.sparse import csr_matrix, hstack as sp_hstack, identity   # noqa: E402
from trimesh.proximity import closest_point                          # noqa: E402

from gtcore.interact import (snap_to_wall, conform_tile,             # noqa: E402
                             find_overlapping_tiles, _tangent_frame,
                             SEED_WALL_OFFSET_MM)
from gtcore.dose.engine import dose_at_points                        # noqa: E402
from gtcore.dose.dvh import shell_points                             # noqa: E402
from gtcore.phantom.generate import make_head_phantom                # noqa: E402
from gtcore.segment.surface import mask_to_mesh                      # noqa: E402

OUT = os.path.join(ROOT, "output", "scout_milp")
CACHE = os.path.join(OUT, "cache")
RX_CGY = 6000.0
DETACHED_MM = 1.5
SHELL_MM = 5.0
FPS_SEED = 0          # farthest-point sampling start vertex rng
SUB_SEED = 0          # target subsample rng
SPIN_SETS = {1: (0,), 2: (0, 45), 3: (0, 30, 60)}


# ----------------------------------------------------------------- geometry
def cavity_mesh(k):
    path = os.path.join(CACHE, f"mesh_seed{k}.pkl")
    if os.path.exists(path):
        d = pickle.load(open(path, "rb"))
    else:
        vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=k)
        m = mask_to_mesh(truth.masks["cavity"], vol.affine)
        d = {"vertices": np.asarray(m.vertices), "faces": np.asarray(m.faces)}
        pickle.dump(d, open(path, "wb"))
    return trimesh.Trimesh(d["vertices"], d["faces"], process=False)


def farthest_point_sample(pts, h, rng):
    """Greedy FPS until the farthest remaining point is closer than h."""
    sel = [int(rng.integers(len(pts)))]
    dmin = np.linalg.norm(pts - pts[sel[0]], axis=1)
    while True:
        i = int(np.argmax(dmin))
        if dmin[i] < h:
            break
        sel.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(sel)


def vertex_area_weights(mesh):
    fa = np.asarray(mesh.area_faces)
    w = np.zeros(len(mesh.vertices))
    for c in range(3):
        np.add.at(w, np.asarray(mesh.faces)[:, c], fa / 3.0)
    return w


def build_candidates(mesh, h, spins, rng):
    """(anchor, spin) grid -> conformed full tiles; detached ones rejected."""
    V = np.asarray(mesh.vertices)
    idx = farthest_point_sample(V, h, rng)
    tiles, meta, n_rej = [], [], 0
    for ai, a in enumerate(V[idx]):
        sp, n = snap_to_wall(mesh, a)
        _, t1r, t2r = _tangent_frame(n, np.array([0.0, 0.0, 1.0]))
        for th in spins:
            hint = np.cos(np.radians(th)) * t1r + np.sin(np.radians(th)) * t2r
            tile = conform_tile(mesh, sp, n, hint, "full")
            _, dist, _ = closest_point(mesh, tile.seed_centers)
            if np.abs(dist - SEED_WALL_OFFSET_MM).max() > DETACHED_MM:
                n_rej += 1
                continue
            tiles.append(tile)
            meta.append((ai, th))
    return tiles, np.array(meta), n_rej, len(idx)


def conflict_pairs(tiles):
    """Pairwise conflicts exactly as the planner defines them.

    find_overlapping_tiles already carries a bounding-sphere prefilter; on
    these cavities (equivalent radius ~18 mm) nearly every pair survives it,
    so the O(C^2) directed-hit loop is the cost (~0.15 ms per pair)."""
    return find_overlapping_tiles(tiles)


def influence(tiles, pts, exact):
    D = np.empty((len(tiles), len(pts)), dtype=np.float32)
    for c, t in enumerate(tiles):
        D[c] = dose_at_points(t.seed_centers, t.seed_axes, pts, exact=exact)
    return D


def stratified_subsample(pts, w, M, rng):
    """Area-weighted stratified subsample: FPS picks M well-spread shell
    points, every shell vertex is assigned to its nearest pick (Voronoi
    stratum) and the pick carries the stratum's total area."""
    from scipy.spatial import cKDTree
    if M >= len(pts):
        return np.arange(len(pts)), w.copy()
    sel = [int(rng.integers(len(pts)))]
    dmin = np.linalg.norm(pts - pts[sel[0]], axis=1)
    while len(sel) < M:
        i = int(np.argmax(dmin))
        sel.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[i], axis=1))
    sel = np.array(sel)
    _, owner = cKDTree(pts[sel]).query(pts)
    ws = np.zeros(M)
    np.add.at(ws, owner, w)
    return sel, ws


def instance(k, h, spins, exact_kernel=True):
    """Cached candidate set + conflicts + full-shell influence."""
    tag = f"seed{k}_h{h:g}_s{len(spins)}"
    path = os.path.join(CACHE, f"inst_{tag}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    mesh = cavity_mesh(k)
    rng = np.random.default_rng(FPS_SEED)
    t0 = time.time()
    tiles, meta, n_rej, n_anch = build_candidates(mesh, h, spins, rng)
    t1 = time.time()
    pairs = conflict_pairs(tiles)
    t2 = time.time()
    shell = shell_points(mesh, SHELL_MM)
    w = vertex_area_weights(mesh)
    D = influence(tiles, shell, exact_kernel)
    t3 = time.time()
    inst = dict(tag=tag, seed=k, h=h, spins=list(spins), C=len(tiles),
                n_anchors=n_anch, n_rejected=n_rej, pairs=np.array(pairs),
                D_full=D, w_full=w, shell=shell, meta=meta,
                area_mm2=float(mesh.area), volume_mm3=float(mesh.volume),
                t_cand=t1 - t0, t_conf=t2 - t1, t_infl=t3 - t2,
                dmax_single=float(D.max()), exact_kernel=exact_kernel)
    pickle.dump(inst, open(path, "wb"))
    return inst


# ---------------------------------------------------------- conflict cliques
def greedy_clique_cover(C, pairs):
    """Edge clique cover of the conflict graph: every conflict edge lies in
    at least one returned clique, so the clique rows alone are a complete
    (and tighter) replacement for the pairwise rows."""
    adj = [set() for _ in range(C)]
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    covered = set()
    cliques = []
    deg = np.array([len(a) for a in adj])
    order = sorted(map(tuple, pairs), key=lambda e: -(deg[e[0]] + deg[e[1]]))
    for i, j in order:
        if (i, j) in covered:
            continue
        q = [i, j]
        cand = adj[i] & adj[j]
        while cand:
            # extend by the common neighbour with most uncovered edges into q
            best, bs = None, -1
            for v in cand:
                s = sum((min(u, v), max(u, v)) not in covered for u in q)
                if s > bs or (s == bs and v < best):
                    best, bs = v, s
            q.append(best)
            cand = cand & adj[best]
        for a in range(len(q)):
            for b in range(a + 1, len(q)):
                covered.add((min(q[a], q[b]), max(q[a], q[b])))
        cliques.append(sorted(q))
    return cliques


# ------------------------------------------------------------------- model
def tightening_rows(D, N, rx):
    """Valid 'k of the strong' rows.  Let S_j be the sum of the j largest
    single-candidate doses at point m (S_0 = 0).  If a selection of N tiles
    covers m with only j < k of them above delta then
    rx <= S_j + (N - j) * delta; choosing
        delta_k = min_{j < k} (rx - S_j) / (N - j)   (minus a hair)
    makes that impossible whenever S_{k-1} < rx, so every covering selection
    has at least k candidates with D[c,m] > delta_k:
        k * y_m <= sum_{c : D[c,m] > delta_k} x_c.
    k = 1 is the usual support row y_m <= sum_{c : D[c,m] > rx/N} x_c."""
    Cn, M = D.shape
    rows, cols, vals = [], [], []
    srt = -np.sort(-D, axis=0)                      # descending per column
    cum = np.vstack([np.zeros((1, M)), np.cumsum(srt, axis=0)])
    r = 0
    for m in range(M):
        for k in range(1, N + 1):
            S = cum[k - 1, m]
            if S >= rx:
                break
            js = np.arange(k)
            delta = float(((rx - cum[js, m]) / (N - js)).min()) * (1.0 - 1e-6)
            cs = np.nonzero(D[:, m] > delta)[0]
            if len(cs) < k:       # point can never be covered: y_m = 0
                cs = np.array([], dtype=int)
            rows += [r] * len(cs)
            cols += list(cs)
            vals += [1.0] * len(cs)
            rows.append(r)
            cols.append(Cn + m)
            vals.append(-float(k))
            r += 1
            if len(cs) == 0:
                break
    return csr_matrix((vals, (rows, cols)), shape=(r, Cn + M))


def build_milp(D, w, N, pairs, cliques, form, rx):
    """Return (c, constraints, n_vars) for scipy.optimize.milp.
    Variables: x (C) then y (M).  Objective is minimized: -sum w y."""
    Cn, M = D.shape
    nv = Cn + M
    cons = []
    # coverage: sum_c D[c,m] x_c - rx y_m >= 0  (linear with binary y; no big-M)
    Dcsr = csr_matrix(D.T.astype(np.float64))                # (M, C)
    Acov = sp_hstack([Dcsr, -rx * identity(M, format="csr")]).tocsr()
    cons.append(LinearConstraint(Acov, 0.0, np.inf))
    # cardinality
    Acard = csr_matrix((np.ones(Cn), (np.zeros(Cn, int), np.arange(Cn))),
                       shape=(1, nv))
    cons.append(LinearConstraint(Acard, N, N))
    # conflicts
    if form in ("pair", "both"):
        P = np.asarray(pairs)
        if len(P):
            r = np.repeat(np.arange(len(P)), 2)
            A = csr_matrix((np.ones(2 * len(P)), (r, P.reshape(-1))),
                           shape=(len(P), nv))
            cons.append(LinearConstraint(A, -np.inf, 1.0))
    if form in ("clique", "both", "clique_tight"):
        if cliques:
            r = np.concatenate([[i] * len(q) for i, q in enumerate(cliques)])
            cidx = np.concatenate(cliques)
            A = csr_matrix((np.ones(len(cidx)), (r, cidx)),
                           shape=(len(cliques), nv))
            cons.append(LinearConstraint(A, -np.inf, 1.0))
    if form == "clique_tight":
        At = tightening_rows(D, N, rx)          # sum_strong x - k y >= 0
        if At.shape[0]:
            cons.append(LinearConstraint(At, 0.0, np.inf))
    c = np.concatenate([np.zeros(Cn), -w])
    return c, cons, nv


def solve(D, w, N, pairs, cliques, form, rx, time_limit, relax=False):
    c, cons, nv = build_milp(D, w, N, pairs, cliques, form, rx)
    integ = np.zeros(nv) if relax else np.ones(nv)
    t0 = time.time()
    res = milp(c, constraints=cons, integrality=integ,
               bounds=Bounds(0.0, 1.0),
               options={"time_limit": float(time_limit), "disp": False,
                        "mip_rel_gap": 1e-6})
    dt = time.time() - t0
    W = float(w.sum())
    out = dict(time=dt, status=int(res.status), message=str(res.message))
    out["obj"] = (-float(res.fun) / W) if res.x is not None else np.nan
    bound = getattr(res, "mip_dual_bound", None)
    if relax:
        out["bound"] = out["obj"]
    else:
        out["bound"] = (-float(bound) / W) if (bound is not None and
                                                np.isfinite(bound)) else np.nan
    gap = getattr(res, "mip_gap", None)
    out["gap"] = float(gap) if gap is not None else np.nan
    out["sel"] = (np.nonzero(res.x[:D.shape[0]] > 0.5)[0].tolist()
                  if (res.x is not None and not relax) else [])
    return out


# ------------------------------------------------------------------ bounds
def pointwise_bound(D, w, N, rx):
    """Cheap upper bound ignoring conflicts and consistency: point m can at
    best receive its N largest single-candidate doses.  (The 'N largest
    single-candidate coverages' bound of the brief is 0 here because no
    single tile reaches rx on the +5 mm shell; this is its usable form.)"""
    top = -np.sort(-D, axis=0)[:N].sum(axis=0)
    return float(w[top >= rx].sum() / w.sum())


def greedy(D, w, N, pairs, rx):
    """Forward selection by hard V100 gain; ties (common while no point is
    yet covered, since no single tile reaches rx) broken by the largest
    reduction in weighted dose shortfall, then lowest id."""
    Cn, M = D.shape
    adj = [set() for _ in range(Cn)]
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    sel, blocked = [], np.zeros(Cn, bool)
    dose = np.zeros(M)
    for _ in range(N):
        best, key = None, None
        for c in range(Cn):
            if blocked[c]:
                continue
            d = dose + D[c]
            hard = float(w[d >= rx].sum())
            soft = -float((w * np.maximum(0.0, rx - d)).sum())
            k = (hard, soft, -c)
            if key is None or k > key:
                best, key = c, k
        if best is None:
            break
        sel.append(best)
        dose = dose + D[best]
        blocked[best] = True
        for j in adj[best]:
            blocked[j] = True
    return sel, float(w[dose >= rx].sum() / w.sum())


def local_search(D, w, sel, pairs, rx, max_rounds=50):
    """First-improvement 1-swap on hard V100 (soft tie-break) from the
    greedy start -- a slightly better incumbent to measure bounds against."""
    Cn, M = D.shape
    adj = [set() for _ in range(Cn)]
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    sel = list(sel)

    def score(d):
        return (float(w[d >= rx].sum()),
                -float((w * np.maximum(0.0, rx - d)).sum()))
    cur = score(D[sel].sum(axis=0)) if sel else (0.0, 0.0)
    for _ in range(max_rounds):
        improved = False
        for pos in range(len(sel)):
            others = sel[:pos] + sel[pos + 1:]
            forbidden = set(others)
            for o in others:
                forbidden |= adj[o]
            base = D[others].sum(axis=0) if others else np.zeros(M)
            for c in range(Cn):
                if c in forbidden:
                    continue
                sc = score(base + D[c])
                if sc > cur:
                    sel[pos] = c
                    cur = sc
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return sel, cur[0] / float(w.sum())


# --------------------------------------------- alternative: enumeration B&B
def exact_bb(D, w, N, pairs, rx, time_limit, start_sel=None):
    """Exact depth-first branch-and-bound over conflict-free N-subsets.

    Alternative reference to the HiGHS MILP.  Candidates are visited in a
    fixed order (decreasing weighted dose potential) and a node may only add
    candidates later in that order than its last pick, so every subset is
    enumerated once.  At a node with partial dose d and r tiles still to
    place, every newly covered point m must get at least (rx - d_m)/r from
    its strongest new tile, so
        gain <= sum of the r largest  w({m uncovered : D[c,m] >= (rx-d_m)/r})
    over the still-allowed candidates c -- the bound used for pruning.  With
    one tile left the best completion is read off directly.  Returns
    (selection, V100, proven_optimal, n_nodes, seconds); the dual bound on
    early exit is the max over open nodes of their bound."""
    Cn, M = D.shape
    pot = (np.minimum(D, rx) * w).sum(axis=1)
    order = np.argsort(-pot, kind="stable")
    Dp = D[order]
    conf = np.zeros((Cn, Cn), bool)
    if len(pairs):
        inv = np.empty(Cn, int)
        inv[order] = np.arange(Cn)
        P = inv[np.asarray(pairs)]
        conf[P[:, 0], P[:, 1]] = True
        conf[P[:, 1], P[:, 0]] = True
    W = float(w.sum())
    best_val, best_sel = -1.0, []
    if start_sel:
        inv = np.empty(Cn, int)
        inv[order] = np.arange(Cn)
        ss = [int(inv[c]) for c in start_sel]
        best_val = float(w[Dp[ss].sum(axis=0) >= rx].sum())
        best_sel = ss
    t0 = time.time()
    nodes = 0
    open_bound = 0.0
    timed_out = False
    # stack entries: (chosen list, allowed bool mask over ids > last)
    stack = [([], np.ones(Cn, bool))]
    while stack:
        if time.time() - t0 > time_limit:
            timed_out = True
            break
        chosen, allowed = stack.pop()
        nodes += 1
        r = N - len(chosen)
        A = np.nonzero(allowed)[0]
        if len(A) < r:
            continue
        d = Dp[chosen].sum(axis=0) if chosen else np.zeros(M)
        need = rx - d
        u = need > 0.0
        base = float(w[~u].sum())
        DA = Dp[A][:, u]
        wu = w[u]
        if r == 1:
            gains = (DA >= need[u]) @ wu
            i = int(np.argmax(gains))
            if base + gains[i] > best_val:
                best_val = base + float(gains[i])
                best_sel = chosen + [int(A[i])]
            continue
        cov = (DA >= need[u] / r) @ wu
        ub = min(base + float(np.sort(cov)[-r:].sum()), W)
        if ub <= best_val:
            continue
        hard = (DA >= need[u]) @ wu
        soft = np.minimum(DA, need[u]) @ wu
        kids = A[np.lexsort((-soft, -hard))]
        # push in reverse so the most promising child is expanded first
        for c in kids[::-1]:
            na = allowed & ~conf[c]
            na[:c + 1] = False
            stack.append((chosen + [int(c)], na))
    if timed_out:
        # bound = best of (incumbent, bounds of the open nodes)
        for chosen, allowed in stack:
            r = N - len(chosen)
            A = np.nonzero(allowed)[0]
            if len(A) < r:
                continue
            d = Dp[chosen].sum(axis=0) if chosen else np.zeros(M)
            need = rx - d
            u = need > 0.0
            base = float(w[~u].sum())
            DA = Dp[A][:, u]
            if r == 1:
                ub = base + float(((DA >= need[u]) @ w[u]).max())
            else:
                cov = (DA >= need[u] / r) @ w[u]
                ub = base + float(np.sort(cov)[-r:].sum())
            open_bound = max(open_bound, min(ub, W))
    bound = max(best_val, open_bound) if timed_out else best_val
    sel = [int(order[i]) for i in best_sel]
    return (sel, best_val / W, not timed_out, nodes, time.time() - t0,
            bound / W)


# -------------------------------------------------------------------- main
FIELDS = ["seed", "h", "n_spins", "C", "M", "N", "n_pairs", "n_cliques",
          "kind", "form", "time_limit", "time_s", "status", "incumbent",
          "bound", "gap", "greedy", "greedy_ls", "pointwise_ub",
          "n_selected", "selection"]


def write_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        wr.writerow({k: row.get(k, "") for k in FIELDS})


def run_instance(inst, M, N_list, forms, limits, relax, results_path, seen,
                 bb_limit=None):
    rng = np.random.default_rng(SUB_SEED)
    sub, ws = stratified_subsample(inst["shell"], inst["w_full"], M, rng)
    D = inst["D_full"][:, sub].astype(np.float64)
    pairs = inst["pairs"]
    cliques = greedy_clique_cover(inst["C"], pairs)
    base = dict(seed=inst["seed"], h=inst["h"], n_spins=len(inst["spins"]),
                C=inst["C"], M=len(sub), n_pairs=len(pairs),
                n_cliques=len(cliques))
    for N in N_list:
        g_sel, g_v = greedy(D, ws, N, pairs, RX_CGY)
        ls_sel, ls_v = local_search(D, ws, g_sel, pairs, RX_CGY)
        pw = pointwise_bound(D, ws, N, RX_CGY)
        b = dict(base, N=N, greedy=g_v, greedy_ls=ls_v, pointwise_ub=pw)
        key = (inst["tag"], len(sub), N)
        if ("pointwise", key) not in seen:
            write_row(results_path, dict(
                b, kind="bound", form="pointwise", time_limit="", time_s=0.0,
                status="", incumbent=ls_v, bound=pw,
                gap=(pw - ls_v) / max(ls_v, 1e-9), n_selected=len(ls_sel),
                selection=" ".join(map(str, ls_sel))))
        print(f"  [{inst['tag']} M={len(sub)} N={N}] greedy={g_v:.4f} "
              f"(n={len(g_sel)}) ls={ls_v:.4f} pointwise_ub={pw:.4f}",
              flush=True)
        if bb_limit and ("bb", key, int(bb_limit)) not in seen:
            sel, v, proven, nodes, t, bnd = exact_bb(
                D, ws, N, pairs, RX_CGY, bb_limit, ls_sel)
            write_row(results_path, dict(
                b, kind="bb", form="enum_bb", time_limit=int(bb_limit),
                time_s=t, status=0 if proven else 1, incumbent=v, bound=bnd,
                gap=(bnd - v) / max(v, 1e-9), n_selected=len(sel),
                selection=" ".join(map(str, sel))))
            print(f"    BB[tl={int(bb_limit)}] proven={proven} inc={v:.4f} "
                  f"bound={bnd:.4f} nodes={nodes} {t:.1f}s", flush=True)
        for form in (relax or []):
            if ("lp", key, form) in seen:
                continue
            r = solve(D, ws, N, pairs, cliques, form, RX_CGY, 600, relax=True)
            write_row(results_path, dict(
                b, kind="lp", form=form, time_limit=600, time_s=r["time"],
                status=r["status"], incumbent=ls_v, bound=r["bound"],
                gap=(r["bound"] - ls_v) / max(ls_v, 1e-9), n_selected="",
                selection=""))
            print(f"    LP[{form}] bound={r['bound']:.4f} {r['time']:.1f}s",
                  flush=True)
        for form, tl in itertools.product(forms, limits):
            if ("milp", key, form, int(tl)) in seen:
                continue
            r = solve(D, ws, N, pairs, cliques, form, RX_CGY, tl)
            write_row(results_path, dict(
                b, kind="milp", form=form, time_limit=int(tl), time_s=r["time"],
                status=r["status"], incumbent=r["obj"], bound=r["bound"],
                gap=r["gap"], n_selected=len(r["sel"]),
                selection=" ".join(map(str, r["sel"]))))
            print(f"    MILP[{form} tl={int(tl)}] status={r['status']} "
                  f"inc={r['obj']:.4f} bound={r['bound']:.4f} "
                  f"gap={r['gap']:.4f} {r['time']:.1f}s "
                  f"({r['message'][:70]})", flush=True)


def load_seen(path):
    seen = set()
    if not os.path.exists(path):
        return seen
    for row in csv.DictReader(open(path)):
        key = (f"seed{row['seed']}_h{float(row['h']):g}_s{row['n_spins']}",
               int(row["M"]), int(row["N"]))
        if row["kind"] == "bound":
            seen.add(("pointwise", key))
        elif row["kind"] == "lp":
            seen.add(("lp", key, row["form"]))
        elif row["kind"] == "bb":
            seen.add(("bb", key, int(float(row["time_limit"]))))
        else:
            seen.add(("milp", key, row["form"], int(float(row["time_limit"]))))
    return seen


def merge_results(out_dir, dest):
    """Concatenate the per-run res_*.csv files written by parallel
    invocations (--results) into one results.csv, dropping duplicates."""
    import glob
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(out_dir, "res_*.csv"))):
        for row in csv.DictReader(open(f)):
            key = (row["seed"], row["h"], row["n_spins"], row["M"], row["N"],
                   row["kind"], row["form"], row["time_limit"])
            if key in seen:
                continue
            seen.add(key)
            if row["kind"] == "bb" and row["bound"]:
                # V100 units: an open-node bound above 1 carries no information
                row["bound"] = f"{min(float(row['bound']), 1.0):.6f}"
                row["gap"] = f"{(float(row['bound']) - float(row['incumbent'])) / max(float(row['incumbent']), 1e-9):.6f}"
            rows.append(row)
    rows.sort(key=lambda r: (int(r["seed"]), -float(r["h"]), int(r["n_spins"]),
                             int(r["M"]), int(r["N"]), r["kind"], r["form"],
                             float(r["time_limit"] or 0)))
    with open(dest, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def summarize(rows):
    """Markdown table: one line per (instance, N, formulation)."""
    def f(x, nd=3):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "-"
        return "-" if not np.isfinite(v) else f"{v:.{nd}f}"
    print("| seed | h | spins | C | M | N | kind | form | limit | time s | "
          "status | incumbent | bound | gap | greedy+LS |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        st = {"0": "optimal", "1": "limit", "2": "infeasible", "": ""}.get(
            r["status"], r["status"])
        print(f"| {r['seed']} | {float(r['h']):g} | {r['n_spins']} | {r['C']} | "
              f"{r['M']} | {r['N']} | {r['kind']} | {r['form']} | "
              f"{r['time_limit']} | {f(r['time_s'], 1)} | {st} | "
              f"{f(r['incumbent'])} | {f(r['bound'])} | {f(r['gap'], 2)} | "
              f"{f(r['greedy_ls'])} |")


def condensed(rows):
    """One line per (instance, M, N): incumbents and bounds side by side."""
    from collections import defaultdict
    g = defaultdict(dict)
    for r in rows:
        k = (int(r["seed"]), float(r["h"]), int(r["n_spins"]), int(r["M"]),
             int(r["N"]))
        g[k]["C"] = r["C"]
        g[k]["ls"] = r["greedy_ls"]
        if r["kind"] == "lp" and r["form"] == "clique":
            g[k]["lp"] = r["bound"]
        if r["kind"] == "milp" and r["form"] == "clique" and                 r["time_limit"] == "60":
            g[k]["milp"] = r
        if r["kind"] == "bb":
            g[k]["bb"] = r

    def f(x, nd=3):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "-"
        return "-" if not np.isfinite(v) else f"{v:.{nd}f}"
    st = {"0": "opt", "1": "limit", "2": "infeas"}
    print("| seed | h | spins | C | M | N | greedy+LS | LP(clique) | "
          "MILP clique 60 s: status / inc / bound / gap | "
          "enum B&B 300 s: status / inc / time |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(g, key=lambda k: (k[0], -k[1], k[2], k[3], k[4])):
        d = g[k]
        m = d.get("milp")
        ms = (f"{st.get(m['status'], m['status'])} / {f(m['incumbent'])} / "
              f"{f(m['bound'])} / {f(m['gap'], 2)}") if m else "-"
        b = d.get("bb")
        bs = (f"{st.get(b['status'], b['status'])} / {f(b['incumbent'])} / "
              f"{f(b['time_s'], 0)} s") if b else "-"
        print(f"| {k[0]} | {k[1]:g} | {k[2]} | {d['C']} | {k[3]} | {k[4]} | "
              f"{f(d['ls'])} | {f(d.get('lp'))} | {ms} | {bs} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--h", type=float, nargs="+", default=[4.0])
    ap.add_argument("--spins", type=int, default=3,
                    help="2 -> {0,45}; 3 -> {0,30,60}")
    ap.add_argument("--M", type=int, nargs="+", default=[300, 1000])
    ap.add_argument("--N", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--forms", nargs="*", default=["pair", "clique", "both"])
    ap.add_argument("--bb", type=float, default=None,
                    help="also run the enumeration B&B with this time limit")
    ap.add_argument("--limits", type=float, nargs="+", default=[60])
    ap.add_argument("--relax", nargs="*", default=None,
                    help="LP-relaxation forms to evaluate (default: none)")
    ap.add_argument("--results", default=os.path.join(OUT, "results.csv"))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="merge output/scout_milp/res_*.csv into results.csv "
                         "and print the markdown table")
    ap.add_argument("--build-only", action="store_true",
                    help="build/cache the instances and stop")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if a.merge:
        rows = merge_results(OUT, os.path.join(OUT, "results.csv"))
        condensed(rows)
        print()
        summarize(rows)
        return
    spins = SPIN_SETS[a.spins]
    if a.probe:
        a.seeds, a.h, a.M, a.N = [1], [6.0], [300], [4]
    seen = load_seen(a.results)
    inst_path = os.path.join(OUT, "instances.csv")
    for k, h in itertools.product(a.seeds, a.h):
        inst = instance(k, h, spins)
        row = {kk: inst[kk] for kk in ("tag", "seed", "h", "C", "n_anchors",
                                       "n_rejected", "area_mm2", "volume_mm3",
                                       "t_cand", "t_conf", "t_infl",
                                       "dmax_single")}
        row["n_spins"] = len(spins)
        row["n_pairs"] = len(inst["pairs"])
        row["M_full"] = len(inst["shell"])
        new = not os.path.exists(inst_path)
        with open(inst_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        print(f"instance {inst['tag']}: C={inst['C']} anchors={inst['n_anchors']} "
              f"rejected={inst['n_rejected']} pairs={len(inst['pairs'])} "
              f"dmax_single={inst['dmax_single']:.0f} cGy "
              f"(cand {inst['t_cand']:.1f}s conf {inst['t_conf']:.1f}s "
              f"infl {inst['t_infl']:.1f}s)", flush=True)
        if a.build_only:
            continue
        for M in a.M:
            run_instance(inst, M, a.N, a.forms, a.limits, a.relax,
                         a.results, seen, bb_limit=a.bb)


if __name__ == "__main__":
    main()
