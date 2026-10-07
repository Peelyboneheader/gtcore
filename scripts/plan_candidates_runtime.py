"""V7 runtime measurement for ``gtcore.plan`` candidates / conflicts (A1).

Runs ``build_candidates`` + ``build_conflicts`` on the synthetic cavity
``make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)`` (cavity mesh from
``truth.masks["cavity"]`` via ``segment.surface.mask_to_mesh``) at h = 3 mm /
6 spins and h = 2.5 mm / 6 spins, and ``visible_faces`` + candidates on the
hollow-sphere test mesh.  Prints a Markdown table for
``docs/optimize-notes.md`` ("V7 Runtime", "A1 candidates/conflicts").

Usage (from the repo root)::

    python scripts/plan_candidates_runtime.py [--quick]

``--quick`` skips the h = 2.5 mm run.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time

import numpy as np
import trimesh


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _row(name, mesh, cs, build_s, g, conf_s):
    deg = np.asarray(g.pairs.sum(axis=1)).ravel()
    sizes = [q.size for q in g.cliques]
    n_enum = len(cs) + sum(cs.n_rejected.values())
    return ("| %s | %d / %.0f | %d | %d | %s | %.1f | %.2f | %d | %.1f (%.3f) | %d (max %d) | %.1f |"
            % (name, len(mesh.faces), mesh.area, n_enum, len(cs),
               ", ".join("%s %d" % kv for kv in cs.n_rejected.items() if kv[1]) or "none",
               build_s, 100.0 * build_s / max(n_enum, 1), g.count_pairs(),
               deg.mean(), deg.mean() / max(len(cs) - 1, 1), len(g.cliques),
               max(sizes) if sizes else 0, conf_s))


def main(argv):
    quick = "--quick" in argv
    from gtcore.phantom import make_head_phantom
    from gtcore.plan import build_candidates, build_conflicts, visible_faces
    from gtcore.segment import mask_to_mesh

    print("commit %s; python %s; numpy %s; trimesh %s; %s"
          % (_commit(), platform.python_version(), np.__version__,
             trimesh.__version__, platform.processor() or platform.machine()))
    header = ("| run | faces / area mm^2 | enumerated | C | rejected | build s | s per 100 "
              "| conflict pairs | mean degree (density) | cliques | conflicts s |")
    print(header)
    print("|" + "---|" * 11)

    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=1)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    runs = [("cavity seed 1, h=3, 6 spins", mesh, dict(h_mm=3.0, n_spins=6, rng_seed=0))]
    if not quick:
        runs.append(("cavity seed 1, h=2.5, 6 spins", mesh,
                     dict(h_mm=2.5, n_spins=6, rng_seed=0)))

    inner = trimesh.creation.icosphere(subdivisions=3, radius=25.0)
    inner.invert()
    hollow = trimesh.util.concatenate([inner, trimesh.creation.icosphere(subdivisions=3,
                                                                          radius=35.0)])
    t0 = time.perf_counter()
    vis = visible_faces(hollow, [0.0, 0.0, 0.0])
    vis_s = time.perf_counter() - t0
    r = np.linalg.norm(hollow.triangles_center, axis=1)
    print("hollow sphere (r 25 inner, 35 outer, %d faces): visible_faces %.2f s; "
          "inner visible %d/%d, outer visible %d/%d"
          % (len(hollow.faces), vis_s, vis[r < 30].sum(), (r < 30).sum(),
             vis[r > 30].sum(), (r > 30).sum()))
    runs.append(("hollow sphere inner wall, h=3, 6 spins", hollow,
                 dict(h_mm=3.0, n_spins=6, rng_seed=0, eligible_faces=vis)))

    for name, m, kw in runs:
        t0 = time.perf_counter()
        cs = build_candidates(m, **kw)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        g = build_conflicts(cs)
        conf_s = time.perf_counter() - t0
        print(_row(name, m, cs, build_s, g, conf_s))
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
