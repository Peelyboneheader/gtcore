"""Sensitivity sweep of the seed-fragment merge distance (localization stage 3).

    python scripts/localization_mergesplit_sweep.py [--singles 200] [--pairs 60]

For each grid (G1 printed-phantom 0.59x0.59x1.0; G2s PostOp geometry, 1 mm
slices every 2 mm; G3 2 mm slabs with every other slice interpolated) and
each MERGE_MAX_SEP_MM in {3.0, 3.5, 4.0, 4.5, 5.0}: how many random lone
seeds fragment without the merge and stay fragmented with it, the merged
centre error, and how many distinct close seed pairs (collinear 4.6-5.6 mm,
parallel 2-4 mm, random 2.5-5.5 mm; never overlapping) get fused.
Scenes: tests/merge_split_scenes.py.  Prints a markdown table.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "tests"))
import merge_split_scenes as M  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    rows = M.sweep(n_single=args.singles, n_pair=args.pairs, seed=args.seed)
    print(M.format_sweep(rows))
    print("\n%.1f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
