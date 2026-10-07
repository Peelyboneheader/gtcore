"""§8 go/no-go experiment for the placement optimizer (docs/plan-tile-optimize.md).

On a small set of synthetic cavities: candidates at h = 3 mm, 6 spins; influence
on the +5 mm shell; greedy only; V100/D90 against the uniform heuristic
(farthest-point anchors, spin 0) and random feasible placements for
N = 4, 6, 8.  Metrics here are the influence-matrix metrics on the FULL +5 mm
shell target (no subsample); the validation campaign re-evaluates on a dose
grid.  One configuration per cavity is cross-checked against
``gtcore.plan.final_report`` when that function is implemented.

Usage (from the repo root)::

    python scripts/go_no_go_optimize.py --seeds 1 2 3 4 5 6 --out output/go_no_go

Decision rule (declared before running): gain = greedy V100 - uniform V100 in
percentage points, at each N.  >= 3 pp on most cavities -> proceed with §3 as
written; 1-3 -> proceed with §7.7 framing; < 1 -> stop and revisit §7.1.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gtcore import plan  # noqa: E402
from gtcore.interact import find_overlapping_tiles  # noqa: E402
from gtcore.phantom.generate import make_head_phantom  # noqa: E402
from gtcore.segment.surface import mask_to_mesh  # noqa: E402

H_MM = 3.0
N_SPINS = 6
N_LIST = (4, 6, 8)
N_RANDOM = 100
SPACING = 1.0


def cavity_mesh(seed, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "cavity_s%d_%.1fmm.pkl" % (seed, SPACING))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    vol, truth = make_head_phantom(spacing=SPACING, n_tiles=3, rng_seed=seed)
    mask = truth.masks["cavity"]
    mesh = mask_to_mesh(mask, vol.affine)
    vol_mm3 = float(mask.sum()) * float(np.prod(vol.spacing))
    out = {"mesh": mesh, "volume_mm3": vol_mm3, "area_mm2": float(mesh.area)}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def farthest_point_ids(points, n, start=0):
    pts = np.asarray(points, float)
    chosen = [int(start)]
    d = np.linalg.norm(pts - pts[start], axis=1)
    for _ in range(1, n):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(chosen)


def uniform_heuristic(cands, conflicts, n):
    """Farthest-point anchors, spin 0, skipping picks that conflict."""
    spin0 = np.where(np.isclose(cands.spins_deg, 0.0))[0]
    anchors = cands.anchors[spin0]
    # centroid-farthest start for determinism
    start = int(np.argmax(np.linalg.norm(anchors - anchors.mean(0), axis=1)))
    order = farthest_point_ids(anchors, len(spin0), start=start)
    sel = []
    for k in order:
        c = int(spin0[k])
        if all(not conflicts.conflicts(c, s) for s in sel):
            sel.append(c)
        if len(sel) == n:
            break
    return np.array(sel, int)


def random_feasible(conflicts, n, rng, n_cands):
    for _ in range(200):
        order = rng.permutation(n_cands)
        sel = []
        for c in order:
            if all(not conflicts.conflicts(int(c), s) for s in sel):
                sel.append(int(c))
            if len(sel) == n:
                return np.array(sel, int)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "go_no_go"))
    ap.add_argument("--rx", type=float, default=plan.DEFAULT_RX_CGY)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=ROOT).decode().strip()
    except Exception:
        commit = "unknown"
    rows = []
    t_all = time.time()
    for seed in args.seeds:
        cav = cavity_mesh(seed, os.path.join(args.out, "cache"))
        mesh = cav["mesh"]
        t0 = time.time()
        cands = plan.build_candidates(mesh, h_mm=H_MM, n_spins=N_SPINS, kinds=("full",),
                                      rng_seed=0)
        t_c = time.time() - t0
        target = plan.TargetSet.from_shell(mesh, plan.TARGET_SHELL_OFFSET_MM)
        t0 = time.time()
        infl = plan.build_influence(cands, target, rx_cgy=args.rx, m_opt=10 ** 9,
                                    rng_seed=0)
        t_i = time.time() - t0
        t0 = time.time()
        confl = plan.build_conflicts(cands)
        t_k = time.time() - t0
        obj = plan.make_objective(infl, confl)
        print("seed %d: cavity %.1f mL, wall %.0f mm2, target M=%d, C=%d cands "
              "(rejected %s), %d conflict pairs; build %.1f+%.1f+%.1f s" % (
                  seed, cav["volume_mm3"] / 1000.0, cav["area_mm2"], len(target.points),
                  len(cands), cands.n_rejected, confl.count_pairs(), t_c, t_i, t_k))
        rng = np.random.default_rng(seed)
        for n in N_LIST:
            t0 = time.time()
            g = plan.solve_greedy(obj, n)
            t_g = time.time() - t0
            mg = obj.metrics(g.selection) if g.feasible else {}
            u = uniform_heuristic(cands, confl, n)
            mu = obj.metrics(u) if len(u) == n else {}
            rv = []
            for _ in range(N_RANDOM):
                r = random_feasible(confl, n, rng, len(cands))
                if r is not None:
                    rv.append(obj.metrics(r))
            v100_r = np.array([m["V100"] for m in rv]) if rv else np.array([np.nan])
            d90_r = np.array([m["D90"] for m in rv]) if rv else np.array([np.nan])
            ok = (g.feasible and len(find_overlapping_tiles(cands.tiles_of(g.selection))) == 0)
            row = dict(seed=seed, N=n, volume_ml=cav["volume_mm3"] / 1000.0,
                       n_candidates=len(cands),
                       greedy_V100=mg.get("V100", np.nan), greedy_D90=mg.get("D90", np.nan),
                       uniform_V100=mu.get("V100", np.nan), uniform_D90=mu.get("D90", np.nan),
                       random_V100_median=float(np.nanmedian(v100_r)),
                       random_V100_p95=float(np.nanpercentile(v100_r, 95)),
                       random_D90_median=float(np.nanmedian(d90_r)),
                       n_random_feasible=len(rv),
                       gain_vs_uniform_pp=100.0 * (mg.get("V100", np.nan) - mu.get("V100", np.nan)),
                       gain_vs_random_median_pp=100.0 * (mg.get("V100", np.nan) - float(np.nanmedian(v100_r))),
                       greedy_seconds=t_g, greedy_no_overlap=ok, commit=commit)
            rows.append(row)
            print("  N=%d greedy V100 %.3f D90 %.0f | uniform V100 %.3f D90 %.0f | random "
                  "median %.3f p95 %.3f | gain %.1f pp (vs uniform) %.1f pp (vs random)" % (
                      n, row["greedy_V100"], row["greedy_D90"], row["uniform_V100"],
                      row["uniform_D90"], row["random_V100_median"], row["random_V100_p95"],
                      row["gain_vs_uniform_pp"], row["gain_vs_random_median_pp"]))
        # grid cross-check of one configuration when final_report exists
        try:
            rep = plan.final_report(mesh, cands.tiles_of(g.selection), rx_cgy=args.rx,
                                    target=target, grid_mm=1.0)
            gm = rep.metrics_grid.get(5.0, {})
            print("  grid check (N=%d greedy): shell+5 V100 %.3f D90 %.0f vs influence "
                  "V100 %.3f D90 %.0f" % (n, gm.get("V100", np.nan), gm.get("D90", np.nan),
                                          mg.get("V100", np.nan), mg.get("D90", np.nan)))
        except NotImplementedError:
            print("  (final_report not implemented yet: no grid cross-check)")
        except Exception as e:  # pragma: no cover
            print("  grid check failed:", e)

    path = os.path.join(args.out, "go_no_go.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", path, "(%.0f s total)" % (time.time() - t_all))
    # summary table (markdown)
    print("\n| N | greedy V100 mean±SD | uniform V100 mean±SD | random median mean | "
          "gain vs uniform pp (per cavity) | cavities with gain>=3 / >=1 |")
    print("|---|---|---|---|---|---|")
    for n in N_LIST:
        sub = [r for r in rows if r["N"] == n]
        gv = np.array([r["greedy_V100"] for r in sub])
        uv = np.array([r["uniform_V100"] for r in sub])
        rv = np.array([r["random_V100_median"] for r in sub])
        gain = np.array([r["gain_vs_uniform_pp"] for r in sub])
        print("| %d | %.3f±%.3f | %.3f±%.3f | %.3f | %s | %d/%d / %d/%d |" % (
            n, np.nanmean(gv), np.nanstd(gv), np.nanmean(uv), np.nanstd(uv), np.nanmean(rv),
            ", ".join("%.1f" % x for x in gain), int(np.sum(gain >= 3)), len(gain),
            int(np.sum(gain >= 1)), len(gain)))
    print("\ncommand: python scripts/go_no_go_optimize.py --seeds %s   commit %s" % (
        " ".join(str(s) for s in args.seeds), commit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
