"""The ``gt`` command — one simple entry point for the whole project.

    gt view <dicom-folder-or-file>   load a CT, run the pipeline, open 3D view
    gt view                          same, on the synthetic phantom
    gt plan <dicom-folder-or-file>   pipeline + interactive tile planner
    gt plan                          same, on the synthetic phantom
    gt optimize <scan> --tiles N     pipeline + opt-in placement optimizer -> JSON/CSV
    gt demo                          full phantom demo (writes output/ files)
    gt test                          run the test suite

Installed as a console script (see pyproject.toml); the repo root also has a
``gt.bat`` so it works without activating the venv.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


# planner / `gt plan --optimizer` default solver (SA after the V2 campaign)
DEFAULT_OPTIMIZER = "sa"

def _int_or_auto(text):
    if str(text).lower() == "auto":
        return "auto"
    return int(text)


def _load(path, spacing):
    if path:
        from .io import load_volume

        print("loading", path)
        vol = load_volume(path)
        title = os.path.basename(str(path).rstrip("\\/"))[:60]
        return vol, title
    from .phantom import make_head_phantom

    vol, _truth = make_head_phantom(spacing=spacing)
    return vol, "synthetic phantom (%.1f mm)" % spacing


def cmd_view(args):
    from .pipeline import reconstruct
    from .viz import show_scene

    vol, title = _load(args.path, args.spacing)
    print("volume %s @ %s mm" % (vol.array.shape, tuple(round(s, 2) for s in vol.spacing)))
    result = reconstruct(vol, n_full_tiles=args.tiles, n_half_tiles=args.half)
    out = show_scene(result, title="IntraOp GammaTile — " + title,
                     screenshot=args.snapshot)
    if out:
        print("snapshot written to", out)
    return 0


def cmd_plan(args):
    from .pipeline import reconstruct
    from .planner import run_planner
    from .tiles import ImplantPrior

    vol, title = _load(args.path, args.spacing)
    print("volume %s @ %s mm" % (vol.array.shape, tuple(round(s, 2) for s in vol.spacing)))
    prior = ImplantPrior(n_full=args.tiles, n_half=args.half, n_seeds=args.seeds)
    print("implant prior:", prior.describe())
    result = reconstruct(vol, n_seeds_expected=prior.n_seeds)
    # additive: the historical call is untouched unless --optimizer is set
    # to something other than the planner's own default (DEFAULT_OPTIMIZER)
    extra = {}
    if getattr(args, "optimizer", DEFAULT_OPTIMIZER) != DEFAULT_OPTIMIZER:
        extra["solver"] = args.optimizer
    if getattr(args, "hrctv", False):
        extra["hrctv"] = True
    run_planner(result, rx_cgy=args.rx, suggest=bool(args.suggest), prior=prior, **extra)
    return 0


def _implant_center(result, mesh):
    """Detected-seed mean when seeds exist, else the mesh centroid (RAS)."""
    import numpy as np

    seeds = getattr(result, "seeds", None)
    if seeds is not None and len(seeds):
        return np.asarray(seeds.centers_ras, dtype=float).mean(axis=0)
    return np.asarray(mesh.vertices, dtype=float).mean(axis=0)


def cmd_optimize(args):
    """``gt optimize``: pipeline -> wall -> tile-count recommendation ->
    gtcore.plan.optimize (or sweep_n with --min-n) -> JSON + CSV + seed CSV."""
    import json
    import time

    import numpy as np

    import gtcore.plan as plan
    from .interact import export_plan_csv
    from .pipeline import reconstruct
    from .planner import wall_mesh_for

    vol, title = _load(args.path, args.spacing)
    print("volume %s @ %s mm" % (vol.array.shape, tuple(round(s, 2) for s in vol.spacing)))
    result = reconstruct(vol)
    mesh, label = wall_mesh_for(result)
    if mesh is None or not len(getattr(mesh, "vertices", ())):
        print("no cavity or body surface in this scan -- nothing to optimize on")
        return 2
    print("wall: %s (%d faces)" % (label, len(mesh.faces)))

    try:
        eligible = None
        if label != "cavity wall":
            # a closed printed shell has an inner and an outer surface: only
            # the faces seen from the implant are eligible (plan section 10)
            eligible = np.asarray(plan.visible_faces(mesh, _implant_center(result, mesh)),
                                  dtype=bool)
            print("eligible faces (visible from the implant): %d of %d"
                  % (int(eligible.sum()), eligible.size))
        rec = plan.recommend_tile_count(mesh, eligible_faces=eligible)
        print(rec.describe())
        n_tiles = args.tiles
        cap = None
        if n_tiles is None or args.min_n:
            # the rule has no packing loss: also measure what fits at this grid
            # (with --min-n the sweep runs up to this capacity, not --tiles)
            from .plan.api import packing_capacity
            cap, cap_info = packing_capacity(
                mesh, rx_cgy=args.rx, kind="full", h_mm=args.h, n_spins=args.spins,
                eligible_faces=eligible)
            print("recommended %d (%g cm^2 rule) / fits at this grid (h %g mm, %s spins): "
                  "%s%d  [capacity greedy %.1f s, %d candidates]"
                  % (rec.n_tiles, plan.TILE_AREA_CM2, args.h,
                     args.spins if args.spins is not None else "default",
                     "at least " if cap_info.get("at_least") else "", cap,
                     cap_info.get("seconds", 0.0), cap_info.get("n_candidates", 0)))
        if n_tiles is None:
            n_tiles = min(int(rec.n_tiles), int(cap)) if cap > 0 else int(rec.n_tiles)
            which = ("the packing capacity" if cap < rec.n_tiles
                     else "the recommended count")
            print("--tiles not given: using %s, %d" % (which, n_tiles))
        if n_tiles < 1:
            print("tile count must be >= 1 (got %d)" % n_tiles)
            return 1

        out_dir = args.out or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            time.strftime("optimize_%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)

        kw = dict(rx_cgy=args.rx, solver=args.solver, seed=args.seed, h_mm=args.h,
                  n_spins=args.spins, eligible_faces=eligible,
                  time_budget_s=float(args.budget))
        n_full = n_tiles
        if args.min_n:
            from .plan.api import default_target
            target = default_target(mesh, eligible)   # +5 mm shell of the eligible wall
            # sweep_n forwards **kw to the discrete solvers, which take no budget
            kw_sweep = {k: v for k, v in kw.items() if k != "time_budget_s"}
            # P2 asks for the smallest N that meets the criterion, so sweep up
            # to what fits at this grid (the sweep stops by itself at the
            # first N that cannot be packed), not just up to --tiles
            n_sweep = max(int(n_tiles), int(cap)) if cap else int(n_tiles)
            sweep = plan.sweep_n(mesh, target, n_sweep, **kw_sweep)
            print("coverage vs N (%s):" % args.solver)
            for row in sweep.rows:
                print("  N %2d  V100 %.3f  D90 %.0f cGy  V150 %.3f  V200 %.3f  %.1f s"
                      % (row.get("N", 0), row.get("V100", float("nan")),
                         row.get("D90", float("nan")), row.get("V150", float("nan")),
                         row.get("V200", float("nan")), row.get("runtime_s", 0.0)))
            print("minimum N:", ", ".join("%s -> %s" % (k, v) for k, v in sweep.min_n.items()))
            with open(os.path.join(out_dir, "sweep.json"), "w", encoding="utf-8") as fh:
                json.dump(plan._jsonable({"rows": sweep.rows, "min_n": sweep.min_n}),
                          fh, indent=2)
            chosen = sweep.min_n.get("D90>=rx")
            if chosen is None:
                n_placed = max([int(r.get("N", 0)) for r in sweep.rows] or [n_tiles])
                print("no N <= %d reaches D90 >= rx (largest N that packed: %d); "
                      "placing %d tiles" % (n_sweep, n_placed, n_tiles))
                chosen = n_tiles
            n_full = int(chosen)
        tiles, rep = plan.optimize(mesh, n_full, args.half, refine=bool(args.refine),
                                   report=not args.no_report, verbose=True, **kw)
    except NotImplementedError as exc:
        print("optimizer not available yet:", exc)
        return 2
    except (ValueError, RuntimeError) as exc:
        print("optimize failed:", exc)
        return 1

    print(rep.summary())
    rep.to_json(os.path.join(out_dir, "report.json"))
    rep.to_csv(os.path.join(out_dir, "report.csv"))
    seeds_csv = os.path.join(out_dir, "plan_seeds.csv")
    n_seeds = export_plan_csv(seeds_csv, tiles, result.seeds.centers_ras,
                              result.seeds.axes_ras, rx_cgy=args.rx)
    print("%d tiles, %d seeds written to %s" % (len(tiles), n_seeds, out_dir))
    return 0


def cmd_demo(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "run_phantom_demo.py")
    cmd = [sys.executable, script, "--spacing", str(args.spacing),
           "--tiles", str(args.tiles)]
    if args.streaks:
        cmd.append("--streaks")
    return subprocess.call(cmd)


def cmd_test(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.call([sys.executable, "-m", "pytest",
                            os.path.join(root, "tests"), "-q"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="gt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("view", help="run the pipeline on a CT and open the 3D viewer")
    v.add_argument("path", nargs="?", default=None,
                   help="DICOM folder or .nrrd/.nii/.mha file (default: phantom)")
    v.add_argument("--spacing", type=float, default=0.7,
                   help="phantom voxel size in mm (phantom mode only)")
    v.add_argument("--snapshot", default=None,
                   help="render off-screen to this PNG instead of opening a window")
    v.add_argument("--tiles", type=_int_or_auto, default=None,
                   help="implanted FULL tile count, or 'auto' to infer the "
                        "configuration from the seeds alone: runs tile "
                        "fitting and colours/labels seeds per recovered tile")
    v.add_argument("--half", type=int, default=0,
                   help="implanted HALF (2x1) tile count (with --tiles); with "
                        "--tiles auto any non-zero value allows half tiles")
    v.set_defaults(fn=cmd_view)

    pln = sub.add_parser("plan", help="run the pipeline and open the interactive tile planner")
    pln.add_argument("path", nargs="?", default=None,
                     help="DICOM folder or .nrrd/.nii/.mha file (default: phantom)")
    pln.add_argument("--spacing", type=float, default=0.7,
                     help="phantom voxel size in mm (phantom mode only)")
    pln.add_argument("--rx", type=float, default=6000.0,
                     help="prescription dose in cGy for the isodose levels")
    pln.add_argument("--suggest", action="store_true",
                     help="start with the tile configuration inferred from "
                          "the detected seeds (same as pressing 'T')")
    pln.add_argument("--tiles", type=int, default=None,
                     help="implanted FULL tile count when the OR team knows "
                          "it (trusted input for 'T'); omit if unknown")
    pln.add_argument("--half", type=int, default=0,
                     help="implanted HALF (2x1) tile count (default 0: with "
                          "no OR confirmation, assume no half tiles)")
    pln.add_argument("--seeds", type=int, default=None,
                     help="implanted seed count (4 per full, 2 per half): "
                          "detection is checked against it and, on coarse "
                          "scans, the HU threshold is lowered stepwise "
                          "until that many seeds are found near the implant")
    pln.add_argument("--optimizer", choices=("greedy", "sa", "continuous"),
                     default=DEFAULT_OPTIMIZER,
                     help="placement optimizer solver for the planner's 'O' key "
                          "(Shift+O cycles it in the window)")
    pln.add_argument("--hrctv", action="store_true",
                     help="start with the HR-CTV (5 mm tissue rind outside the "
                          "wall) shown and scored as a volume (the 'V' key)")
    pln.set_defaults(fn=cmd_plan)

    o = sub.add_parser("optimize", help="run the pipeline and the opt-in placement "
                                        "optimizer (gtcore.plan); writes JSON + CSV")
    o.add_argument("path", nargs="?", default=None,
                   help="DICOM folder or .nrrd/.nii/.mha file (default: phantom)")
    o.add_argument("--tiles", type=int, default=None,
                   help="FULL tiles to place (default: the manufacturer-rule "
                        "recommendation from the wall area, printed first)")
    o.add_argument("--half", type=int, default=0, help="HALF (2x1) tiles to place")
    o.add_argument("--min-n", action="store_true", dest="min_n",
                   help="sweep N = 1..--tiles and place the smallest N with D90 >= rx")
    o.add_argument("--solver", choices=("greedy", "local", "sa", "milp", "continuous"),
                   default="sa")
    o.add_argument("--budget", type=float, default=60.0,
                   help="wall-time budget in s for --solver continuous (default 60)")
    o.add_argument("--seed", type=int, default=0, help="RNG seed (stochastic solvers)")
    o.add_argument("--rx", type=float, default=6000.0, help="prescription dose in cGy")
    o.add_argument("--h", type=float, default=2.5, help="candidate anchor spacing in mm")
    o.add_argument("--spins", type=int, default=None,
                   help="spins per anchor (default 6 full / 12 half)")
    o.add_argument("--out", default=None,
                   help="output folder (default output/optimize_<timestamp>/)")
    o.add_argument("--refine", action="store_true",
                   help="continuous (u, v, theta) polish after the discrete solve")
    o.add_argument("--no-report", action="store_true", dest="no_report",
                   help="skip the dose-grid final report (influence metrics only)")
    o.add_argument("--spacing", type=float, default=0.7,
                   help="phantom voxel size in mm (phantom mode only)")
    o.set_defaults(fn=cmd_optimize)

    d = sub.add_parser("demo", help="full phantom demo, writes output/ files")
    d.add_argument("--spacing", type=float, default=0.7)
    d.add_argument("--tiles", type=int, default=3)
    d.add_argument("--streaks", action="store_true")
    d.set_defaults(fn=cmd_demo)

    t = sub.add_parser("test", help="run the test suite")
    t.set_defaults(fn=cmd_test)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
