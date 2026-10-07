"""Scout for plan-tile-optimize.md section 7.1: is the hard V100 objective
flat or misleading as a tile-selection objective?

Question
--------
Does hard V100 on the +5 mm shell (area-weighted vertices) discriminate
between placement arms, or does it saturate / go flat so that a different
objective (soft sigmoid coverage, D90, conformity-style) is needed?

Method (smallest experiment that answers it)
--------------------------------------------
On 4 synthetic cavities (``make_head_phantom`` rng_seed 1..4, cavity mesh
only; truth tiles ignored):

1. Candidates: farthest-point-sampled anchors at ~4 mm spacing on the mesh
   vertices, spins 0 and 45 deg, each conformed with ``conform_tile``
   (kind "full"); drop candidates with any seed farther than 1.5 mm from
   the nominal 3 mm wall offset (mesh proximity query).
2. Influence rows: ``dose_at_points`` (tabulated kernel, total decay,
   default S_K) at every +5 mm shell vertex; the optimizer sees a
   weight-proportional subsample of <= 2000 vertices (equal weights after
   sampling); final numbers are also reported on the full area-weighted
   shell.
3. Conflicts: ``find_overlapping_tiles`` over the whole candidate list in
   one call (its own bounding-sphere pruning handles the O(C^2) pairs).
4. For N in (4, 6, 8): (i) 200 random feasible selections, (ii) greedy
   forward selection (ties -> lowest id) under each objective:
   hard V100; soft sigmoid coverage with tau = 0.05, 0.02, 0.10 rx; D90;
   conformity-style V100 - 0.5 * max(0, V200 - 0.10); plus (added after
   the first run) hard V100 with ties broken by soft coverage.
   rx = 6000 cGy.

Reported per (cavity, N, objective): hard V100 and D90 of the greedy
result (subsample and full shell), number of greedy steps with ties in the
hard-V100 marginal gain (>= 2 feasible candidates share the best hard gain)
and with a flat best hard gain (== 0), and the Spearman correlation between
the objective and hard V100 over the 200 random selections.

Decision criterion (written before running)
-------------------------------------------
An alternative objective is worth promoting only if, on at least 3 of 4
cavities, greedy under it reaches a hard V100 >= 1 percentage point higher
than greedy under hard V100 itself at the same N, or if hard-V100 greedy
shows ties at >= 30 % of its steps.

Outputs: ``output/scout_objective/results.csv`` (one row per cavity, N,
arm), ``output/scout_objective/random.csv`` (random-selection summary),
``output/scout_objective/summary.md`` (markdown tables) and cached
instances ``output/scout_objective/instance_seed{k}.pkl``.

Run from the repo root::

    python scripts/scout_objective.py
"""
from __future__ import annotations

import csv
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import trimesh
from scipy.special import expit
from scipy.stats import spearmanr

from gtcore.dose.dvh import shell_points
from gtcore.dose.engine import dose_at_points
from gtcore.interact import conform_tile, find_overlapping_tiles, snap_to_wall
from gtcore.phantom.generate import make_head_phantom
from gtcore.segment.surface import mask_to_mesh

# ------------------------------------------------------------------ parameters
SEEDS = (1, 2, 3, 4)          # phantom rng seeds (cavity shapes)
SPACING_MM = 1.0              # phantom voxel size; 1 mm keeps builds fast
H_ANCHOR_MM = 4.0             # farthest-point anchor spacing on the wall
SPINS_DEG = (0.0, 45.0)       # in-plane spins per anchor (2x2 symmetry -> [0, 90))
DETACHED_MM = 1.5             # reject if any seed is > this from its 3 mm offset
SEED_WALL_OFFSET_MM = 3.0     # nominal seed-to-wall offset
SHELL_MM = 5.0                # target shell offset (clinical HR-CTV proxy)
M_OPT = 2000                  # optimizer-side target subsample size
RX_CGY = 6000.0               # prescription
N_VALUES = (4, 6, 8)          # tile counts
N_RANDOM = 200                # random feasible selections per (cavity, N)
TAUS = (0.05, 0.02, 0.10)     # soft-coverage widths as fractions of rx
LAMBDA_HOT = 0.5              # conformity-style hot-spot penalty weight
V200_TOL = 0.10               # conformity-style hot-spot tolerance
RANDOM_SEED = 20261007        # base seed for all rng use in this script
TIE_EPS = 1e-12

OUT_DIR = os.path.join("output", "scout_objective")
ARMS = ["hard_v100", "hard_lex_soft0.05", "soft_tau0.05", "soft_tau0.02",
        "soft_tau0.10", "d90", "conformity"]
LEX_EPS = 1e-6                # < 1/M_OPT so the tie-break never outranks hard V100


# ------------------------------------------------------------------ helpers
def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def vertex_area_weights(mesh):
    """One third of the adjacent face areas per vertex."""
    faces = np.asarray(mesh.faces)
    fa = np.asarray(mesh.area_faces, dtype=float)
    w = np.zeros(len(mesh.vertices), dtype=float)
    for k in range(3):
        np.add.at(w, faces[:, k], fa / 3.0)
    return w


def farthest_point_sample(points, spacing):
    """Greedy FPS on a point set until the farthest point is < spacing away.

    Deterministic: starts from the point nearest the centroid.
    """
    pts = np.asarray(points, dtype=float)
    start = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    chosen = [start]
    dmin = np.linalg.norm(pts - pts[start], axis=1)
    while True:
        nxt = int(np.argmax(dmin))
        if dmin[nxt] < spacing:
            break
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.asarray(chosen, dtype=int)


def spin_hint(n, spin_deg):
    """In-plane axis hint: a fixed global reference projected to the tangent
    plane, rotated by ``spin_deg`` about the inward normal ``n``."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ n)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    t1 = ref - float(ref @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    a = np.deg2rad(spin_deg)
    return np.cos(a) * t1 + np.sin(a) * t2


def build_instance(seed):
    """Cavity mesh, candidates, influence rows (full shell), conflicts."""
    t0 = time.time()
    vol, truth = make_head_phantom(spacing=SPACING_MM, n_tiles=1, rng_seed=seed)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    t_phantom = time.time() - t0

    verts = np.asarray(mesh.vertices, dtype=float)
    anchor_idx = farthest_point_sample(verts, H_ANCHOR_MM)

    t0 = time.time()
    tiles, anchors, spins = [], [], []
    n_rejected = 0
    for ai in anchor_idx:
        sp, n_in = snap_to_wall(mesh, verts[ai])
        for spin in SPINS_DEG:
            tile = conform_tile(mesh, sp, n_in, spin_hint(n_in, spin), "full")
            _s, dist, _t = trimesh.proximity.closest_point(mesh, tile.seed_centers)
            if np.any(np.abs(dist - SEED_WALL_OFFSET_MM) > DETACHED_MM):
                n_rejected += 1
                continue
            tiles.append(tile)
            anchors.append(sp)
            spins.append(spin)
    t_cand = time.time() - t0

    t0 = time.time()
    shell = shell_points(mesh, SHELL_MM)
    w_full = vertex_area_weights(mesh)
    D_full = np.empty((len(tiles), len(shell)), dtype=np.float32)
    for c, tile in enumerate(tiles):
        D_full[c] = dose_at_points(tile.seed_centers, tile.seed_axes, shell,
                                   exact=False)
    t_infl = time.time() - t0

    t0 = time.time()
    pairs = find_overlapping_tiles(tiles)
    C = len(tiles)
    conflict = np.zeros((C, C), dtype=bool)
    for i, j in pairs:
        conflict[i, j] = conflict[j, i] = True
    t_conf = time.time() - t0

    return {
        "seed": seed,
        "mesh": mesh,
        "mesh_area_mm2": float(mesh.area),
        "n_vertices": int(len(verts)),
        "n_anchors": int(len(anchor_idx)),
        "n_candidates": C,
        "n_rejected": int(n_rejected),
        "anchors": np.asarray(anchors),
        "spins": np.asarray(spins),
        "tiles": tiles,
        "shell": shell,
        "w_full": w_full,
        "D_full": D_full,
        "conflict": conflict,
        "n_conflict_pairs": int(len(pairs)),
        "timing": {"phantom+mesh": t_phantom, "candidates": t_cand,
                   "influence": t_infl, "conflicts": t_conf},
    }


def load_instance(seed):
    """Cached instance (phantom + mesh + candidates + rows + conflicts).

    Returns ``(instance, built)``; ``built`` is False when it came from the
    pickle cache, so the summary can say which runtime it reports.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "instance_seed%d.pkl" % seed)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), False
    inst = build_instance(seed)
    with open(path, "wb") as f:
        pickle.dump(inst, f)
    return inst, True


# ------------------------------------------------------------------ objectives
def weighted_quantile(values, weights, q):
    """Weighted q-quantile (q in [0, 1]) of a 1-D sample."""
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, q * cw[-1], side="left")])


def metrics_weighted(dose, w):
    """Hard metrics of one dose vector with arbitrary weights."""
    wt = w / w.sum()
    return {
        "V100": float(wt[dose >= RX_CGY].sum()),
        "V150": float(wt[dose >= 1.5 * RX_CGY].sum()),
        "V200": float(wt[dose >= 2.0 * RX_CGY].sum()),
        "D90": weighted_quantile(dose, w, 0.10),
    }


# Vectorized objectives over rows of a (F, M) dose matrix with equal weights.
def obj_hard(d):
    return (d >= RX_CGY).mean(axis=1)


def obj_soft(tau):
    def f(d):
        return expit((d - RX_CGY) / (tau * RX_CGY)).mean(axis=1)
    return f


def obj_d90(d):
    return np.percentile(d, 10.0, axis=1)


def obj_conformity(d):
    v100 = (d >= RX_CGY).mean(axis=1)
    v200 = (d >= 2.0 * RX_CGY).mean(axis=1)
    return v100 - LAMBDA_HOT * np.maximum(0.0, v200 - V200_TOL)


def obj_hard_lex_soft(d):
    """Hard V100, ties broken by soft coverage (tau = 0.05 rx).  Added after
    the first run showed that the only tie step is step 1 (no single tile
    reaches rx on the +5 mm shell): isolates 'flat at step 1' from
    'misleading later'."""
    return obj_hard(d) + LEX_EPS * obj_soft(0.05)(d)


OBJECTIVES = {
    "hard_v100": obj_hard,
    "hard_lex_soft0.05": obj_hard_lex_soft,
    "soft_tau0.05": obj_soft(0.05),
    "soft_tau0.02": obj_soft(0.02),
    "soft_tau0.10": obj_soft(0.10),
    "d90": obj_d90,
    "conformity": obj_conformity,
}


# ------------------------------------------------------------------ solvers
def greedy(D, conflict, n, objective):
    """Greedy forward selection under ``objective``; ties -> lowest id.

    Returns (selection, per-step tie records) where each record is
    ``(n_tied_at_best_hard_gain, best_hard_gain, chosen_hard_gain)``.
    """
    C = D.shape[0]
    feasible = np.ones(C, dtype=bool)
    dose = np.zeros(D.shape[1], dtype=np.float64)
    sel, steps = [], []
    for _ in range(n):
        cand = np.flatnonzero(feasible)
        if cand.size == 0:
            break
        trial = dose[None, :] + D[cand]
        vals = objective(trial)
        k = int(np.argmax(vals))                    # first max -> lowest id
        hard_now = float((dose >= RX_CGY).mean())
        hard_gain = obj_hard(trial) - hard_now
        best = float(hard_gain.max())
        n_tied = int((hard_gain >= best - TIE_EPS).sum())
        steps.append((n_tied, best, float(hard_gain[k])))
        c = int(cand[k])
        sel.append(c)
        dose += D[c]
        feasible[c] = False
        feasible &= ~conflict[c]
    return sel, steps


def random_feasible(conflict, n, rng, max_tries=1000):
    C = conflict.shape[0]
    for _ in range(max_tries):
        order = rng.permutation(C)
        feasible = np.ones(C, dtype=bool)
        sel = []
        for c in order:
            if not feasible[c]:
                continue
            sel.append(int(c))
            feasible[c] = False
            feasible &= ~conflict[c]
            if len(sel) == n:
                return sel
    raise RuntimeError("could not find a feasible random selection")


# ------------------------------------------------------------------ main
def run():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, rand_rows, inst_rows = [], [], []
    n_built = 0

    for seed in SEEDS:
        inst, built = load_instance(seed)
        n_built += int(built)
        D_full = inst["D_full"].astype(np.float64)
        w_full = inst["w_full"]
        conflict = inst["conflict"]
        C = D_full.shape[0]
        inst_rows.append({
            "cavity": seed, "area_mm2": round(inst["mesh_area_mm2"], 1),
            "n_vertices": inst["n_vertices"], "n_anchors": inst["n_anchors"],
            "n_candidates": C, "n_rejected": inst["n_rejected"],
            "n_conflict_pairs": inst["n_conflict_pairs"],
            "conflict_density": round(inst["n_conflict_pairs"]
                                      / (C * (C - 1) / 2.0), 3),
            **{"t_" + k: round(v, 1) for k, v in inst["timing"].items()},
        })

        # weight-proportional subsample, equal weights afterwards
        rng = np.random.default_rng(RANDOM_SEED + seed)
        M = D_full.shape[1]
        if M > M_OPT:
            idx = rng.choice(M, size=M_OPT, replace=False, p=w_full / w_full.sum())
            idx.sort()
        else:
            idx = np.arange(M)
        D = D_full[:, idx]

        for n in N_VALUES:
            # (i) random feasible selections
            rng_r = np.random.default_rng(RANDOM_SEED * 10 + seed * 100 + n)
            sels = [random_feasible(conflict, n, rng_r) for _ in range(N_RANDOM)]
            doses = np.stack([D[s].sum(axis=0) for s in sels])
            obj_vals = {name: f(doses) for name, f in OBJECTIVES.items()}
            hard_rand = obj_vals["hard_v100"]
            full_rand = np.array([metrics_weighted(D_full[s].sum(axis=0), w_full)["V100"]
                                  for s in sels])
            rand_rows.append({
                "cavity": seed, "N": n,
                "hard_v100_mean": round(float(hard_rand.mean()), 4),
                "hard_v100_sd": round(float(hard_rand.std()), 4),
                "hard_v100_max": round(float(hard_rand.max()), 4),
                "hard_v100_min": round(float(hard_rand.min()), 4),
                "hard_v100_n_unique": int(np.unique(hard_rand).size),
                "full_v100_mean": round(float(full_rand.mean()), 4),
                "full_v100_max": round(float(full_rand.max()), 4),
            })

            # (ii) greedy under each objective
            hard_ref = None
            for arm in ARMS:
                sel, steps = greedy(D, conflict, n, OBJECTIVES[arm])
                dose_sub = D[sel].sum(axis=0)
                m_sub = metrics_weighted(dose_sub, np.ones(dose_sub.size))
                m_full = metrics_weighted(D_full[sel].sum(axis=0), w_full)
                if arm == "hard_v100":
                    rho, p = 1.0, 0.0
                else:
                    if np.unique(hard_rand).size < 2 or np.unique(obj_vals[arm]).size < 2:
                        rho, p = float("nan"), float("nan")
                    else:
                        rho, p = spearmanr(obj_vals[arm], hard_rand)
                n_tie = sum(1 for s in steps if s[0] >= 2)
                n_flat = sum(1 for s in steps if s[1] <= TIE_EPS)
                n_zero_chosen = sum(1 for s in steps if s[2] <= TIE_EPS)
                row = {
                    "cavity": seed, "N": n, "arm": arm,
                    "n_selected": len(sel),
                    "v100_sub": round(m_sub["V100"], 4),
                    "d90_sub": round(m_sub["D90"], 1),
                    "v100_full": round(m_full["V100"], 4),
                    "d90_full": round(m_full["D90"], 1),
                    "v150_full": round(m_full["V150"], 4),
                    "v200_full": round(m_full["V200"], 4),
                    "n_steps": len(steps),
                    "n_tie_steps": n_tie,
                    "n_flat_steps": n_flat,
                    "n_zero_gain_chosen": n_zero_chosen,
                    "max_tied_at_best": max(s[0] for s in steps),
                    "spearman_vs_hard": round(float(rho), 3),
                    "spearman_p": float(p),
                    "random_mean_v100": round(float(hard_rand.mean()), 4),
                    "random_max_v100": round(float(hard_rand.max()), 4),
                    "selection": " ".join(str(c) for c in sel),
                }
                if arm == "hard_v100":
                    hard_ref = row
                    row["delta_v100_pp_vs_hard"] = 0.0
                    row["delta_v100_full_pp_vs_hard"] = 0.0
                else:
                    row["delta_v100_pp_vs_hard"] = round(
                        100.0 * (row["v100_sub"] - hard_ref["v100_sub"]), 2)
                    row["delta_v100_full_pp_vs_hard"] = round(
                        100.0 * (row["v100_full"] - hard_ref["v100_full"]), 2)
                rows.append(row)
            print("cavity %d N=%d done (%.0f s elapsed)" % (seed, n, time.time() - t_start))

    runtime = time.time() - t_start
    write_csv(os.path.join(OUT_DIR, "results.csv"), rows)
    write_csv(os.path.join(OUT_DIR, "random.csv"), rand_rows)
    write_csv(os.path.join(OUT_DIR, "instances.csv"), inst_rows)
    md = summarize(rows, rand_rows, inst_rows, runtime, n_built)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print("total runtime %.0f s" % runtime)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def summarize(rows, rand_rows, inst_rows, runtime, n_built):
    out = []
    out.append("commit %s, runtime %.0f s (%d of %d instances built this run, the rest "
               "from the pickle cache), command `python scripts/scout_objective.py`\n"
               % (git_hash(), runtime, n_built, len(SEEDS)))
    out.append("### Instances\n")
    out.append("| cavity | area mm2 | verts | anchors | candidates | rejected | conflict pairs | density |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in inst_rows:
        out.append("| %d | %.0f | %d | %d | %d | %d | %d | %.2f |" % (
            r["cavity"], r["area_mm2"], r["n_vertices"], r["n_anchors"],
            r["n_candidates"], r["n_rejected"], r["n_conflict_pairs"],
            r["conflict_density"]))
    out.append("")
    out.append("### Greedy result per objective (hard V100 on the optimizer subsample / full shell, D90 full, cGy)\n")
    for n in N_VALUES:
        out.append("N = %d\n" % n)
        out.append("| cavity | random mean / max V100 | arm | placed | V100 sub | V100 full | D90 full | V200 full | tie steps | flat steps | rho vs hard | dV100 pp |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["N"] != n:
                continue
            out.append("| %d | %.3f / %.3f | %s | %d | %.3f | %.3f | %.0f | %.3f | %d/%d | %d/%d | %s | %+.1f |" % (
                r["cavity"], r["random_mean_v100"], r["random_max_v100"], r["arm"],
                r["n_selected"], r["v100_sub"], r["v100_full"], r["d90_full"], r["v200_full"],
                r["n_tie_steps"], r["n_steps"], r["n_flat_steps"], r["n_steps"],
                ("%.2f" % r["spearman_vs_hard"]) if r["arm"] != "hard_v100" else "-",
                r["delta_v100_pp_vs_hard"]))
        out.append("")
    # criterion check
    out.append("### Criterion check\n")
    out.append("| arm | N | cavities with dV100 >= +1 pp (sub) | cavities with dV100 >= +1 pp (full) | mean dV100 pp (sub) | mean rho |")
    out.append("|---|---|---|---|---|---|")
    for arm in ARMS[1:]:
        for n in N_VALUES:
            rs = [r for r in rows if r["arm"] == arm and r["N"] == n]
            n_win = sum(1 for r in rs if r["delta_v100_pp_vs_hard"] >= 1.0)
            n_win_full = sum(1 for r in rs if r["delta_v100_full_pp_vs_hard"] >= 1.0)
            mean_d = np.mean([r["delta_v100_pp_vs_hard"] for r in rs])
            rhos = [r["spearman_vs_hard"] for r in rs if np.isfinite(r["spearman_vs_hard"])]
            out.append("| %s | %d | %d/4 | %d/4 | %+.2f | %.2f |" % (
                arm, n, n_win, n_win_full, mean_d, np.mean(rhos) if rhos else float("nan")))
    hard = [r for r in rows if r["arm"] == "hard_v100"]
    tie_frac = sum(r["n_tie_steps"] for r in hard) / float(sum(r["n_steps"] for r in hard))
    flat_frac = sum(r["n_flat_steps"] for r in hard) / float(sum(r["n_steps"] for r in hard))
    out.append("")
    out.append("hard-V100 greedy: tie steps %.0f %% of all steps, flat steps %.0f %% of all steps"
               % (100 * tie_frac, 100 * flat_frac))
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    run()
