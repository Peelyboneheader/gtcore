"""Scout for plan alternative 7.4: does discretization error dominate?

Question
--------
How large is the discretization error of the candidate grid (anchor spacing
``h`` and spin count) in the discrete tile-placement optimizer, and does a
direct continuous optimization over each tile's (u, v, theta) beat
"discrete greedy + E5 Nelder-Mead polish" at equal wall time?

Arms (per synthetic cavity, N = 6 full tiles, rx = 6000 cGy, target = +5 mm
shell vertices area-weighted subsampled to <= 1500 points)
  (1) discrete forward greedy at h in {4, 3, 2} mm x spins in {2, 3, 6};
  (2) E5 polish of the h = 3 mm / 6-spin greedy result (coordinate descent,
      Nelder-Mead on (u, v, theta) per tile through ``conform_tile``, 2 passes);
  (3) direct continuous multi-start (3 random feasible starts + the same
      coordinate-descent Nelder-Mead) time-capped to wall(1)+wall(2) at
      h = 3 mm / 6 spins;
  (4) finite-difference projected gradient ascent on the soft objective,
      same starts, same time cap.

The hard objective is V100 (fraction of target weight receiving >= rx);
D90 is logged.  Inner continuous optimizations use a smooth surrogate (mean
sigmoid coverage, tau = 0.05 rx) but a tile step is accepted only when hard
V100 does not decrease and the soft objective improves.

Everything is deterministic from the fixed seeds below.  Output:
``output/scout_continuous/results.csv`` (+ cached cavity meshes), and a
markdown table on stdout.  Run from the repo root::

    python scripts/scout_continuous.py            # all 3 cavities
    python scripts/scout_continuous.py --seeds 1  # quick look
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import trimesh
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtcore.dose.dvh import shell_points  # noqa: E402
from gtcore.dose.engine import TG43Engine, dose_at_points  # noqa: E402
from gtcore.interact import (  # noqa: E402
    SEED_WALL_OFFSET_MM,
    _rodrigues,
    _tangent_frame,
    conform_tile,
    find_overlapping_tiles,
    snap_to_wall,
)
from gtcore.phantom.generate import make_head_phantom  # noqa: E402
from gtcore.segment.surface import mask_to_mesh  # noqa: E402
from gtcore.tiles.surface import DETACHED_MM  # noqa: E402

OUT_DIR = os.path.join("output", "scout_continuous")

# ------------------------------------------------------------ parameters
RX_CGY = 6000.0
N_TILES = 6
M_TARGET = 1500
SHELL_MM = 5.0
TAU = 0.05 * RX_CGY
H_LIST = [4.0, 3.0, 2.0]
SPIN_SETS = {2: [0.0, 45.0], 3: [0.0, 30.0, 60.0],
             6: [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]}
REF_H, REF_SPINS = 3.0, 6
E5_PASSES = 2
N_STARTS = 3
GLOBAL_HINT = np.array([1.0, 0.0, 0.0])
# Nelder-Mead initial simplex (mm, mm, rad): polish uses fit_on_surface's
# scale; the from-random continuous phase starts with a wider simplex for
# its first pass, then the polish scale.
SIMPLEX_POLISH = (1.5, np.deg2rad(10.0))
SIMPLEX_GLOBAL = (4.0, np.deg2rad(20.0))
NM_MAXFEV = 100
# finite-difference gradient steps and scaling (theta scaled so 1 rad of
# spin ~ 10 mm of corner travel, comparable to a 1 mm anchor move)
FD_STEP = np.array([0.5, 0.5, np.deg2rad(2.0)])
THETA_SCALE_MM_PER_RAD = 10.0
FD_STEP0_MM = 2.0
SPHERE_MARGIN_MM = 2.0
MATCH_TOL_PP = 0.25     # "matches" = within this many percentage points


# --------------------------------------------------------------- metrics
def v100(d, w):
    return float(w @ (d >= RX_CGY))


def d90(d, w):
    order = np.argsort(d)
    cum = np.cumsum(w[order])
    return float(d[order][np.searchsorted(cum, 0.10)])


def soft(d, w):
    return float(w @ (1.0 / (1.0 + np.exp(-(d - RX_CGY) / TAU))))


# ------------------------------------------------------------ cavities
def load_cavity(seed):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "cavity_seed%d.ply" % seed)
    if os.path.exists(path):
        return trimesh.load(path, process=False)
    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=seed)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    mesh.export(path)
    return mesh


def make_target(mesh, rng):
    """+5 mm shell vertices, subsampled with probability ~ vertex area.

    Sampling proportional to the vertex area (one third of the adjacent face
    areas) and then weighting equally is the unbiased estimator of the
    area-weighted metrics on the full shell.
    """
    faces = np.asarray(mesh.faces)
    fa = np.asarray(mesh.area_faces) / 3.0
    va = np.zeros(len(mesh.vertices))
    for c in range(3):
        np.add.at(va, faces[:, c], fa)
    pts = shell_points(mesh, SHELL_MM)
    if len(pts) > M_TARGET:
        idx = rng.choice(len(pts), size=M_TARGET, replace=False, p=va / va.sum())
        pts = pts[idx]
    w = np.full(len(pts), 1.0 / len(pts))
    return pts, w


def fps_anchors(mesh, h):
    """Farthest-point sampling of the vertex set until spacing < h."""
    V = np.asarray(mesh.vertices, dtype=float)
    start = int(np.argmin(V[:, 0]))
    chosen = [start]
    dmin = np.linalg.norm(V - V[start], axis=1)
    while True:
        j = int(np.argmax(dmin))
        if dmin[j] < h:
            break
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(V - V[j], axis=1))
    return V[chosen]


# ------------------------------------------------------------ placement
class Placer:
    """Conformer + dose rows + feasibility on one cavity / target."""

    def __init__(self, mesh, T, w):
        self.mesh, self.T, self.w = mesh, T, w
        self.eng = TG43Engine()
        self.n_conform = 0

    def place(self, surf, n_in, hint):
        self.n_conform += 1
        return conform_tile(self.mesh, surf, n_in, hint, kind="full")

    def row(self, tile):
        return dose_at_points(tile.seed_centers, tile.seed_axes, self.T,
                              exact=False, engine=self.eng)

    def detached(self, tile):
        _s, dist, _t = trimesh.proximity.closest_point(self.mesh, tile.seed_centers)
        return bool(np.any(np.abs(np.asarray(dist) - SEED_WALL_OFFSET_MM) > DETACHED_MM))

    @staticmethod
    def sphere(tile):
        p = np.vstack([tile.corners_ras, tile.seed_centers])
        c = p.mean(axis=0)
        return c, float(np.linalg.norm(p - c, axis=1).max())

    def overlaps(self, tile, others):
        c, r = self.sphere(tile)
        for o in others:
            co, ro = self.sphere(o)
            if np.linalg.norm(c - co) > r + ro + 1.0 + SPHERE_MARGIN_MM:
                continue
            if find_overlapping_tiles([tile, o]):
                return True
        return False

    # (u, v, theta) parameterization over the conformer, exactly as
    # gtcore.tiles.surface.fit_on_surface: anchor = surf0 + u t1 + v t2,
    # re-snapped to the wall, axis hint = t1 rotated by theta about n0.
    def frame(self, tile):
        n0, t1, t2 = _tangent_frame(tile.normal_ras, tile.axis_ras)
        return tile.anchor_ras.copy(), n0, t1, t2

    def place_uvt(self, frame, x):
        surf0, n0, t1, t2 = frame
        anchor = surf0 + x[0] * t1 + x[1] * t2
        surf, n_in = snap_to_wall(self.mesh, anchor)
        hint = _rodrigues(t1, n0, float(x[2]))
        return self.place(surf, n_in, hint)


# ----------------------------------------------------- (1) discrete greedy
def build_candidates(pl, h):
    """All (anchor, spin) candidates at spacing h for the 6-spin set.

    Returns dict with tiles, rows, spin (deg), anchor id, per-candidate
    build seconds (so a spin subset's build time is its own share), and the
    FPS time.
    """
    t0 = time.perf_counter()
    anchors = fps_anchors(pl.mesh, h)
    t_fps = time.perf_counter() - t0
    tiles, rows, spins, aid, tcand, n_rej = [], [], [], [], [], 0
    for a_i, a in enumerate(anchors):
        surf, n_in = snap_to_wall(pl.mesh, a)
        n0, t1, _t2 = _tangent_frame(n_in, GLOBAL_HINT)
        for s in SPIN_SETS[6]:
            tc = time.perf_counter()
            hint = _rodrigues(t1, n0, np.deg2rad(s))
            tile = pl.place(surf, n_in, hint)
            if pl.detached(tile):
                n_rej += 1
                tcand_rej = time.perf_counter() - tc
                tcand.append(-tcand_rej)      # negative = rejected, still paid
                continue
            r = pl.row(tile).astype(np.float32)
            tiles.append(tile)
            rows.append(r)
            spins.append(s)
            aid.append(a_i)
            tcand.append(time.perf_counter() - tc)
    return dict(tiles=tiles, rows=np.vstack(rows), spin=np.array(spins),
                aid=np.array(aid), tcand=np.array(tcand), t_fps=t_fps,
                n_anchor=len(anchors), n_rej=n_rej)


def greedy(pl, cand, spin_deg, n_tiles):
    """Forward greedy, lexicographic (hard V100 gain, soft gain, lowest id).

    Conflicts are evaluated lazily: candidates are walked in gain order and
    the first one that does not overlap any chosen tile (pairwise
    find_overlapping_tiles with a bounding-sphere prefilter) is taken.
    """
    keep = np.isin(cand["spin"], spin_deg)
    ids = np.flatnonzero(keep)
    R = cand["rows"][ids]
    w = pl.w.astype(np.float32)
    d = np.zeros(R.shape[1], dtype=np.float32)
    chosen, chosen_tiles = [], []
    base_h, base_s = v100(d, w), soft(d, w)
    for _ in range(n_tiles):
        tot = d[None, :] + R
        gh = (tot >= RX_CGY).astype(np.float32) @ w - base_h
        gs = (1.0 / (1.0 + np.exp(-(tot - RX_CGY) / TAU))) @ w - base_s
        gh = np.round(gh, 6)
        order = np.lexsort((np.arange(len(ids)), -gs, -gh))
        picked = None
        for j in order:
            if j in chosen:
                continue
            t = cand["tiles"][ids[j]]
            if not pl.overlaps(t, chosen_tiles):
                picked = j
                break
        if picked is None:
            raise RuntimeError("no feasible candidate for tile %d" % (len(chosen) + 1))
        chosen.append(picked)
        chosen_tiles.append(cand["tiles"][ids[picked]])
        d = d + R[picked]
        base_h, base_s = v100(d, w), soft(d, w)
    rows = [cand["rows"][ids[j]].astype(float) for j in chosen]
    return chosen_tiles, rows


# --------------------------------------------- continuous: NM coordinate descent
class _Deadline(Exception):
    pass


def nm_tile(pl, tiles, rows, i, simplex, maxfev=NM_MAXFEV, deadline=None):
    """Nelder-Mead on tile i's (u, v, theta); returns True if accepted.

    With a ``deadline`` (perf_counter seconds) the inner optimization is
    aborted at the first evaluation past it and the tile is left unchanged,
    so a time cap is honoured to within one conformer evaluation."""
    others = [t for k, t in enumerate(tiles) if k != i]
    d_others = np.sum([r for k, r in enumerate(rows) if k != i], axis=0)
    frame = pl.frame(tiles[i])
    w = pl.w
    cache = {}

    def evaluate(x):
        key = tuple(np.round(x, 9))
        if key in cache:
            return cache[key]
        if deadline is not None and time.perf_counter() > deadline:
            raise _Deadline()
        try:
            t = pl.place_uvt(frame, x)
            if pl.detached(t) or pl.overlaps(t, others):
                res = (None, None, 1e6)
            else:
                r = pl.row(t)
                res = (t, r, -soft(d_others + r, w))
        except Exception:
            res = (None, None, 1e6)
        cache[key] = res
        return res

    h_old = v100(d_others + rows[i], w)
    s_old = soft(d_others + rows[i], w)
    sm, sa = simplex
    try:
        sol = minimize(lambda x: evaluate(x)[2], np.zeros(3), method="Nelder-Mead",
                       options=dict(maxfev=maxfev, xatol=0.05, fatol=1e-7,
                                    initial_simplex=np.array(
                                        [[0, 0, 0], [sm, 0, 0], [0, sm, 0], [0, 0, sa]], float)))
    except _Deadline:
        return False
    t_new, r_new, c_new = evaluate(sol.x)
    if t_new is None:
        return False
    h_new = v100(d_others + r_new, w)
    if h_new >= h_old and -c_new > s_old + 1e-12:
        tiles[i], rows[i] = t_new, r_new
        return True
    return False


def coordinate_descent(pl, tiles, rows, passes=None, budget_s=None,
                       simplex_first=SIMPLEX_POLISH, simplex_rest=SIMPLEX_POLISH):
    """Coordinate descent over tiles; stops after `passes` or when the
    budget is exhausted (checked before each tile).  Returns passes done."""
    t0 = time.perf_counter()
    deadline = None if budget_s is None else t0 + budget_s
    done = 0
    p = 0
    while True:
        if passes is not None and p >= passes:
            break
        if budget_s is not None and time.perf_counter() - t0 >= budget_s:
            break
        simplex = simplex_first if p == 0 else simplex_rest
        n_acc = 0
        for i in range(len(tiles)):
            if budget_s is not None and time.perf_counter() - t0 >= budget_s:
                break
            n_acc += int(nm_tile(pl, tiles, rows, i, simplex, deadline=deadline))
        p += 1
        done = p
        if passes is None and n_acc == 0 and p >= 2:
            # converged on all tiles twice in a row at the polish scale
            break
    return done


# --------------------------------------------- continuous: FD gradient ascent
def fd_tile(pl, tiles, rows, i):
    """One projected-gradient ascent step on tile i (soft objective).

    Central differences on (u, v, theta); backtracking line search along the
    gradient (scaled coordinates); a step is accepted only when feasible,
    hard V100 does not decrease and the soft objective improves."""
    others = [t for k, t in enumerate(tiles) if k != i]
    d_others = np.sum([r for k, r in enumerate(rows) if k != i], axis=0)
    frame = pl.frame(tiles[i])
    w = pl.w

    def fsoft(x):
        try:
            t = pl.place_uvt(frame, x)
        except Exception:
            return None, None, None
        return t, pl.row(t), None

    s0 = soft(d_others + rows[i], w)
    h0 = v100(d_others + rows[i], w)
    g = np.zeros(3)
    for k in range(3):
        e = np.zeros(3)
        e[k] = FD_STEP[k]
        _tp, rp, _ = fsoft(e)
        _tm, rm, _ = fsoft(-e)
        if rp is None or rm is None:
            return False
        g[k] = (soft(d_others + rp, w) - soft(d_others + rm, w)) / (2 * FD_STEP[k])
    # scaled coordinates z = (u, v, theta * 10 mm/rad): dS/dz3 = g3 / 10
    scale = np.array([1.0, 1.0, 1.0 / THETA_SCALE_MM_PER_RAD])
    gs = g * scale
    gnorm = np.linalg.norm(gs)
    if gnorm < 1e-12:
        return False
    direction = gs / gnorm * scale      # back to (mm, mm, rad)
    step = FD_STEP0_MM
    for _ in range(5):
        x = step * direction
        t, r, _ = fsoft(x)
        if t is not None and not pl.detached(t) and not pl.overlaps(t, others):
            s1, h1 = soft(d_others + r, w), v100(d_others + r, w)
            if s1 > s0 + 1e-12 and h1 >= h0:
                tiles[i], rows[i] = t, r
                return True
        step *= 0.5
    return False


def fd_ascent(pl, tiles, rows, budget_s):
    t0 = time.perf_counter()
    it = 0
    while time.perf_counter() - t0 < budget_s:
        n_acc = 0
        for i in range(len(tiles)):
            if time.perf_counter() - t0 >= budget_s:
                break
            n_acc += int(fd_tile(pl, tiles, rows, i))
        it += 1
        if n_acc == 0:
            break
    return it


# ------------------------------------------------------- random starts
def random_start(pl, rng, n_tiles, max_tries=2000):
    V = np.asarray(pl.mesh.vertices)
    tiles, rows = [], []
    for _ in range(max_tries):
        a = V[rng.integers(len(V))]
        surf, n_in = snap_to_wall(pl.mesh, a)
        n0, t1, _t2 = _tangent_frame(n_in, GLOBAL_HINT)
        hint = _rodrigues(t1, n0, rng.uniform(0.0, np.pi / 2))
        t = pl.place(surf, n_in, hint)
        if pl.detached(t) or pl.overlaps(t, tiles):
            continue
        tiles.append(t)
        rows.append(pl.row(t))
        if len(tiles) == n_tiles:
            return tiles, rows
    raise RuntimeError("could not draw a random feasible start")


# ------------------------------------------------------------------ main
def total(rows):
    return np.sum(rows, axis=0)


def run_cavity(seed, writer, log):
    rng = np.random.default_rng(12345 + seed)
    mesh = load_cavity(seed)
    T, w = make_target(mesh, rng)
    pl = Placer(mesh, T, w)
    log("cavity seed %d: area %.0f mm^2, %d verts, target M=%d"
        % (seed, mesh.area, len(mesh.vertices), len(T)))

    def rec(method, h, spins, tiles_rows, t_build, t_solve, extra="", n_cand=""):
        d = total(tiles_rows[1])
        row = dict(cavity=seed, method=method, h=h, spins=spins, n_cand=n_cand,
                   V100=round(100 * v100(d, w), 3), D90=round(d90(d, w), 1),
                   t_build_s=round(t_build, 2), t_solve_s=round(t_solve, 2),
                   t_total_s=round(t_build + t_solve, 2), extra=extra)
        writer.writerow(row)
        log("  %-16s h=%-4s spins=%-3s V100=%6.2f D90=%7.1f build=%6.1fs solve=%6.1fs %s"
            % (method, h, spins, row["V100"], row["D90"], t_build, t_solve, extra))
        return row

    results = {}
    ref = None
    for h in H_LIST:
        cand = build_candidates(pl, h)
        log("  h=%g: %d anchors, %d candidates, %d rejected (detached), build %.1fs"
            % (h, cand["n_anchor"], len(cand["tiles"]), cand["n_rej"],
               cand["t_fps"] + np.abs(cand["tcand"]).sum()))
        for ns, spins in SPIN_SETS.items():
            # build time of this spin subset = FPS + its own candidates' time;
            # rejected candidates' time is charged to every subset that
            # would have tried them (their spins are unknown -> all).
            keep = np.isin(cand["spin"], spins)
            tc = cand["tcand"]
            t_build = cand["t_fps"] + tc[tc > 0][keep].sum() + (-tc[tc < 0]).sum() * (ns / 6.0)
            t0 = time.perf_counter()
            tiles, rows = greedy(pl, cand, spins, N_TILES)
            t_solve = time.perf_counter() - t0
            r = rec("greedy", h, ns, (tiles, rows), t_build, t_solve,
                    n_cand=int(keep.sum()))
            results[(h, ns)] = r
            if h == REF_H and ns == REF_SPINS:
                ref = (tiles, rows, t_build, t_solve)

    # (2) E5 polish of the reference greedy
    tiles, rows, t_build, t_greedy = ref
    tiles, rows = list(tiles), list(rows)
    t0 = time.perf_counter()
    coordinate_descent(pl, tiles, rows, passes=E5_PASSES)
    t_e5 = time.perf_counter() - t0
    r_e5 = rec("greedy+E5", REF_H, REF_SPINS, (tiles, rows), t_build,
               t_greedy + t_e5, extra="E5 %.1fs, %d passes" % (t_e5, E5_PASSES))
    t_ref = t_build + t_greedy + t_e5
    log("  reference wall time (build+greedy+E5) = %.1fs -> continuous cap" % t_ref)

    # (3) direct continuous multi-start NM, (4) FD gradient, same starts
    starts = []
    for k in range(N_STARTS):
        srng = np.random.default_rng(777 + 100 * seed + k)
        starts.append(random_start(pl, srng, N_TILES))
    best_nm, best_fd = None, None
    for k, (tiles0, rows0) in enumerate(starts):
        rec("random_start", "", "", (tiles0, rows0), 0.0, 0.0, extra="start %d" % k)
        tiles, rows = list(tiles0), list(rows0)
        t0 = time.perf_counter()
        passes = coordinate_descent(pl, tiles, rows, budget_s=t_ref / N_STARTS,
                                    simplex_first=SIMPLEX_GLOBAL,
                                    simplex_rest=SIMPLEX_POLISH)
        t_s = time.perf_counter() - t0
        r = rec("cont_nm", "", "", (tiles, rows), 0.0, t_s,
                extra="start %d, %d passes" % (k, passes))
        if best_nm is None or r["V100"] > best_nm["V100"]:
            best_nm = r
        tiles, rows = list(tiles0), list(rows0)
        t0 = time.perf_counter()
        its = fd_ascent(pl, tiles, rows, budget_s=t_ref / N_STARTS)
        t_s = time.perf_counter() - t0
        r = rec("cont_fd", "", "", (tiles, rows), 0.0, t_s,
                extra="start %d, %d sweeps" % (k, its))
        if best_fd is None or r["V100"] > best_fd["V100"]:
            best_fd = r
    for name, b in (("cont_nm_best", best_nm), ("cont_fd_best", best_fd)):
        row = dict(b)
        row["method"] = name
        row["extra"] = "best of %d starts, total %.1fs cap %.1fs" % (
            N_STARTS, N_STARTS * b["t_solve_s"], t_ref)
        writer.writerow(row)
        log("  %-16s V100=%6.2f D90=%7.1f (%s)" % (name, row["V100"], row["D90"], row["extra"]))

    return dict(greedy=results, e5=r_e5, nm=best_nm, fd=best_fd, t_ref=t_ref)


def verdict(per_cavity, log):
    log("")
    log("## Verdict (pre-declared criteria, plan section 7.4)")
    n_disc_dom = 0
    n_nm_match = n_fd_match = 0
    for seed, r in per_cavity.items():
        g3 = r["greedy"][(3.0, 6)]["V100"]
        g2 = r["greedy"][(2.0, 6)]["V100"]
        e5 = r["e5"]["V100"]
        gain = e5 - g3
        dom = gain > (g2 - g3) and gain > 1.0
        n_disc_dom += int(dom)
        nm_ok = r["nm"]["V100"] >= e5 - MATCH_TOL_PP
        fd_ok = r["fd"]["V100"] >= e5 - MATCH_TOL_PP
        n_nm_match += int(nm_ok)
        n_fd_match += int(fd_ok)
        log("cavity %d: greedy h3/6 %.2f, h2/6 %.2f (dh = %+.2f pp), E5 gain %+.2f pp -> "
            "discretization dominates: %s; cont NM %.2f (%s), cont FD %.2f (%s)"
            % (seed, g3, g2, g2 - g3, gain, dom, r["nm"]["V100"],
               "match/beat" if nm_ok else "worse", r["fd"]["V100"],
               "match/beat" if fd_ok else "worse"))
    n = len(per_cavity)
    log("discretization error dominates on %d/%d cavities" % (n_disc_dom, n))
    log("continuous NM matches/beats discrete+E5 at equal wall time on %d/%d cavities "
        "(promote if >= 2/3): %s" % (n_nm_match, n, "YES" if n_nm_match >= 2 else "NO"))
    log("continuous FD matches/beats discrete+E5 at equal wall time on %d/%d cavities "
        "(promote if >= 2/3): %s" % (n_fd_match, n, "YES" if n_fd_match >= 2 else "NO"))


def markdown_table(csv_path, log):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    log("")
    log("| cavity | method | h | spins | n_cand | V100 % | D90 cGy | build s | solve s | total s | notes |")
    log("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        log("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["cavity"], r["method"], r["h"], r["spins"], r["n_cand"], r["V100"],
            r["D90"], r["t_build_s"], r["t_solve_s"], r["t_total_s"], r["extra"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "results.csv")
    log_path = os.path.join(OUT_DIR, "log.txt")
    logf = open(log_path, "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t_all = time.perf_counter()
    per_cavity = {}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cavity", "method", "h", "spins", "n_cand", "V100", "D90",
            "t_build_s", "t_solve_s", "t_total_s", "extra"])
        writer.writeheader()
        for seed in args.seeds:
            per_cavity[seed] = run_cavity(seed, writer, log)
            f.flush()
    verdict(per_cavity, log)
    markdown_table(csv_path, log)
    log("total wall %.1f s" % (time.perf_counter() - t_all))
    logf.close()


if __name__ == "__main__":
    main()
