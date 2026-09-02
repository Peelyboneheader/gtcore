"""The ``gt`` command — one simple entry point for the whole project.

    gt view <dicom-folder-or-file>   load a CT, run the pipeline, open 3D view
    gt view                          same, on the synthetic phantom
    gt plan <dicom-folder-or-file>   pipeline + interactive tile planner
    gt plan                          same, on the synthetic phantom
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

    vol, title = _load(args.path, args.spacing)
    print("volume %s @ %s mm" % (vol.array.shape, tuple(round(s, 2) for s in vol.spacing)))
    result = reconstruct(vol)
    run_planner(result, rx_cgy=args.rx)
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
    v.add_argument("--tiles", type=int, default=None,
                   help="implanted FULL tile count: runs tile fitting and "
                        "colours/labels seeds per recovered tile")
    v.add_argument("--half", type=int, default=0,
                   help="implanted HALF (2x1) tile count (with --tiles)")
    v.set_defaults(fn=cmd_view)

    pln = sub.add_parser("plan", help="run the pipeline and open the interactive tile planner")
    pln.add_argument("path", nargs="?", default=None,
                     help="DICOM folder or .nrrd/.nii/.mha file (default: phantom)")
    pln.add_argument("--spacing", type=float, default=0.7,
                     help="phantom voxel size in mm (phantom mode only)")
    pln.add_argument("--rx", type=float, default=6000.0,
                     help="prescription dose in cGy for the isodose levels")
    pln.set_defaults(fn=cmd_plan)

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
