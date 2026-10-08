"""Seed-localization harness (docs/plan-localization.md, stage 0).

Renders isolated and crowded Cs-131 seeds with the physically integrated
renderer (:mod:`gtcore.phantom.seed_render`: exact 4.5 x 0.8 mm capsule,
Gaussian PSF, slab integration over the true voxel footprint, noise, 3071 HU
clip) on four acquisition grids and scores seed localization against the
exact truth.

Grids
-----
G1  0.59 x 0.59 x 1.0 mm   printed-phantom-like (contrast METAL_HU_PRINTED:
                           31/32 seeds clip at 3071 HU, as measured)
G2  0.5 x 0.5 x 2.0 mm     PostOp-like contrast (METAL_HU_POSTOP: median peak
                           1700 HU)
G3  PostOp-like gap volume (PostOp contrast): rendered at 0.5 x 0.5 x 1.0 mm
    (the PostOp series really is 1 mm slices, SliceThickness 1, present at
    irregular 1 mm multiples), slices kept with irregular steps of 1-3 mm
    (median 2), then rebuilt by the DICOM loader's own rule onto its grid
    (median kept spacing = 2.0 mm; ``drop_and_interpolate(grid="loader")``):
    a 0.5 x 0.5 x 2.0 mm volume in which every grid slice without a kept
    slice at its position is interpolated, like the PostOp export.
G3L the literal reading (not run by default): 0.5 x 0.5 x 1.0 mm with every
    other slice dropped and re-interpolated on the 1 mm grid.  The pipeline
    then applies its thin-slice 2000 HU threshold, and at PostOp contrast
    only 5-10 % of the seeds are detected, so it cannot measure
    localization; kept for completeness.
G4  0.7 x 0.7 x 2.8 mm     coarse (PostOp contrast)

Layouts (fixed seed list): ``sparse`` = 40 seeds per volume, dart throwing
with every pair >= 8 mm apart; ``crowded`` = 40 seeds grown so that every
seed has a neighbour 7-8 mm away (the minimum on a real tile).  Layout rng r
(``--seeds``, default 0-4) gives the same truth on every grid; 5 x 40 = 200
seeds per grid and layout.  Positions are continuous in a 44 mm box, so the
sub-voxel offsets are uniform; axes are uniform on the sphere.

Methods
-------
baseline  ``detect_seed_candidates`` at the threshold
          ``pipeline.seed_detection_params(spacing)`` gives, then
          ``filter_seed_shaped`` (exactly the pipeline's detection).
merge     baseline detection with ``merge_fragments=True`` (stage 3 fragment
          merge forced on, whatever the pipeline default would decide).
legacy    the pre-stage-3 split (all-blob median, no halves guard,
          k-means++ on positions), merge off -- the detector of 3cf35af.
centroid, model
          ``gtcore.seeds.refine.refine_seed_candidates(vol, cands,
          method=...)`` on the baseline candidates; looked up at run time and
          SKIPPED with a logged reason when absent.

Metrics per grid x layout x method: detections are matched to truth by a
Hungarian assignment within 3 mm; recall, false positives, PPV; per voxel
axis (i, j, k) bias and RMS; 3-D error mean +/- SD, P95, max; seed-axis error
(deg); threshold sensitivity = median centre shift of seeds matched at both
1200 and 2000 HU; NEES = mean e^T Sigma^-1 e / 3 and the fraction inside the
95 % chi-square(3) ellipsoid when the candidates carry ``cov_ras``; ms per
seed of the method's own step.

Outputs (``--out``, default ``output/validation_seed_localization/``):
``seeds_<tag>.csv`` (one row per truth seed and method), ``summary_<tag>.csv``,
``summary_<tag>.md`` (the markdown block for docs/localization-notes.md) and
an appended ``runs.csv`` with the git commit.  ``--calibrate`` instead
prints the rendered peak distribution behind the seed_render calibration.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from gtcore.phantom.seed_render import (
    METAL_HU_POSTOP,
    METAL_HU_PRINTED,
    SATURATE_HU,
    drop_and_interpolate,
    random_seed_layout,
    render_seeds,
)
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import detect_seed_candidates

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GRIDS = {
    "G1": dict(spacing=(0.59, 0.59, 1.0), metal_hu=METAL_HU_PRINTED,
               label="0.59x0.59x1.0"),
    "G2": dict(spacing=(0.5, 0.5, 2.0), metal_hu=METAL_HU_POSTOP,
               label="0.5x0.5x2.0"),
    "G3": dict(spacing=(0.5, 0.5, 1.0), metal_hu=METAL_HU_POSTOP,
               keep="irregular", label="1 mm slabs, irregular gaps, "
               "loader grid 0.5x0.5x2.0"),
    "G3L": dict(spacing=(0.5, 0.5, 1.0), metal_hu=METAL_HU_POSTOP,
                keep="every_other", label="0.5x0.5x1.0, odd slices "
                "interpolated"),
    "G4": dict(spacing=(0.7, 0.7, 2.8), metal_hu=METAL_HU_POSTOP,
               label="0.7x0.7x2.8"),
}
LAYOUTS = {
    "sparse": dict(min_sep_mm=8.0, max_nn_mm=None, rng_offset=0),
    "crowded": dict(min_sep_mm=7.0, max_nn_mm=8.0, rng_offset=100),
}
METHODS = ("baseline", "merge", "legacy", "centroid", "model")
N_PER_VOLUME = 40
BOX_MM = 44.0            # seed centres live in [-22, 22] mm
MARGIN_MM = 8.0          # volume = box + margin on every side
MATCH_MM = 3.0
SENS_THRESHOLDS = (1200.0, 2000.0)
BACKGROUND_HU = 35.0
NOISE_HU = 10.0
PSF_SIGMA_MM = 0.45
CHI2_3_95 = float(chi2.ppf(0.95, 3))


# ----------------------------------------------------------------- utilities
def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"],
                                    cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO, text=True).strip()
        return h, bool(dirty)
    except Exception:
        return "unknown", True


def parse_seeds(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def grid_affine(spacing, fov_mm):
    n_ijk = [int(np.ceil(fov_mm / s)) for s in spacing]
    aff = np.eye(4)
    for a in range(3):
        aff[a, a] = spacing[a]
        aff[a, 3] = -(n_ijk[a] - 1) * spacing[a] / 2.0
    return tuple(n_ijk[::-1]), aff


def make_volume(grid_name, centers, axes, noise_seed, metal_hu=None):
    """Render one harness volume on grid ``grid_name``."""
    g = GRIDS[grid_name]
    shape, aff = grid_affine(g["spacing"], BOX_MM + 2 * MARGIN_MM)
    vol = render_seeds(shape, aff, centers, axes,
                       metal_hu=g["metal_hu"] if metal_hu is None else metal_hu,
                       background_hu=BACKGROUND_HU, psf_sigma_mm=PSF_SIGMA_MM,
                       noise_hu=NOISE_HU, saturate_hu=SATURATE_HU,
                       rng_seed=noise_seed)
    return degrade(vol, g, noise_seed)


def irregular_keep(nk, rng):
    """Kept slice indices with steps of 1, 2 or 3 slices (P = 1/4, 1/2, 1/4),
    redrawn until the median step is exactly 2 (so the loader grid is 2x)."""
    while True:
        steps = rng.choice([1, 2, 3], size=nk, p=[0.25, 0.5, 0.25])
        k = np.concatenate([[0], np.cumsum(steps)])
        k = k[k < nk]
        if len(k) > 3 and np.median(np.diff(k)) == 2:
            return k


def degrade(vol, g, seed):
    """Apply the grid's slice-drop / gap-fill step (G3, G3L)."""
    keep = g.get("keep")
    if keep == "every_other":
        return drop_and_interpolate(vol, np.arange(0, vol.array.shape[0], 2))
    if keep == "irregular":
        k = irregular_keep(vol.array.shape[0], np.random.default_rng(seed))
        return drop_and_interpolate(vol, k, grid="loader")
    return vol


# detection variants that are not refinements (stage 3 merge/split repair)
DETECT_VARIANTS = {
    "baseline": {},
    "merge": dict(merge_fragments=True),
    "legacy": dict(split_window=False, split_guard=False,
                   split_weighted=False),
}


def detect_baseline(vol, hu_threshold=None, **detect_kw):
    p = seed_detection_params(vol.spacing)
    thr = p["hu_threshold"] if hu_threshold is None else float(hu_threshold)
    raw = detect_seed_candidates(vol, hu_threshold=thr, min_mm3=p["min_mm3"],
                                 max_mm3=p["max_mm3"], **detect_kw)
    return filter_seed_shaped(raw, min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
                              min_elong=p["min_elong"],
                              max_elong=p["max_elong"])


def get_refiner():
    """(callable, None) or (None, reason) for gtcore.seeds.refine."""
    try:
        import gtcore.seeds.refine as refine_mod
    except Exception as exc:  # module not built yet
        return None, "gtcore.seeds.refine not importable (%s)" % exc
    fn = getattr(refine_mod, "refine_seed_candidates", None)
    if fn is None:
        return None, "gtcore.seeds.refine has no refine_seed_candidates"
    return fn, None


def match(truth, est, gate=MATCH_MM):
    """Hungarian assignment of truth rows to estimates within ``gate`` mm."""
    truth = np.asarray(truth, float).reshape(-1, 3)
    est = np.asarray(est, float).reshape(-1, 3)
    if len(truth) == 0 or len(est) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = cdist(truth, est)
    cost = np.where(d <= gate, d, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = d[r, c] <= gate
    return r[ok], c[ok]


def peak_near(vol, center, half_mm=1.5):
    ijk = vol.ras_to_index(center)
    sp = vol.spacing
    lo = np.maximum(np.floor(ijk - half_mm / sp).astype(int), 0)
    hi = np.minimum(np.ceil(ijk + half_mm / sp).astype(int) + 1,
                    np.asarray(vol.shape_ijk))
    sub = vol.array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    return float(sub.max()) if sub.size else float("nan")


# ------------------------------------------------------------------- methods
class Method:
    """Runs one localization method; ``reason`` is set when it is skipped."""

    def __init__(self, name):
        self.name = name
        self.reason = None
        self.refiner = None
        if name not in DETECT_VARIANTS:
            self.refiner, self.reason = get_refiner()

    def run(self, vol, hu_threshold=None):
        """(candidates, seconds of the method's own step)."""
        t0 = time.perf_counter()
        cands = detect_baseline(vol, hu_threshold,
                                **DETECT_VARIANTS.get(self.name, {}))
        t_det = time.perf_counter() - t0
        if self.name in DETECT_VARIANTS:
            return cands, t_det
        t0 = time.perf_counter()
        try:
            out = self.refiner(vol, cands, method=self.name)
        except (NotImplementedError, ValueError, TypeError) as exc:
            self.reason = "refine_seed_candidates(method=%r) failed: %s" % (
                self.name, exc)
            return None, 0.0
        return out, time.perf_counter() - t0


# ------------------------------------------------------------------- scoring
def score_volume(vol, centers, axes, method, meta):
    """Per-truth-seed rows for one volume and one method (None = skipped)."""
    cands, secs = method.run(vol)
    if cands is None:
        return None
    est = np.asarray(cands.centers_ras, float).reshape(-1, 3)
    r, c = match(centers, est)
    det_of = {int(a): int(b) for a, b in zip(r, c)}
    cov = getattr(cands, "cov_ras", None)

    # threshold sensitivity: same method at 1200 and 2000 HU
    shifts = {}
    runs = []
    for thr in SENS_THRESHOLDS:
        cs, _ = method.run(vol, hu_threshold=thr)
        e = np.zeros((0, 3)) if cs is None else np.asarray(cs.centers_ras,
                                                          float)
        rr, cc = match(centers, e)
        runs.append({int(a): e[int(b)] for a, b in zip(rr, cc)})
    for ti in set(runs[0]) & set(runs[1]):
        shifts[ti] = float(np.linalg.norm(runs[0][ti] - runs[1][ti]))

    direction = vol.direction
    rows = []
    for ti, (tc, ta) in enumerate(zip(centers, axes)):
        row = dict(meta, seed=ti, method=method.name,
                   truth_x=tc[0], truth_y=tc[1], truth_z=tc[2],
                   k_phase=float(vol.ras_to_index(tc)[2] % 1.0),
                   peak_hu=peak_near(vol, tc),
                   matched=ti in det_of, n_det=len(est),
                   ms_per_seed=1000.0 * secs / max(1, len(est)),
                   thr_shift_mm=shifts.get(ti, np.nan))
        if ti in det_of:
            j = det_of[ti]
            e = est[j] - tc
            ev = direction.T @ e                  # voxel-axis components
            a = np.asarray(cands.axes_ras[j], float)
            cosang = abs(float(a @ ta)) / max(1e-12, np.linalg.norm(a))
            row.update(err_i=ev[0], err_j=ev[1], err_k=ev[2],
                       err_3d=float(np.linalg.norm(e)),
                       axis_err_deg=float(np.degrees(np.arccos(min(1.0,
                                                                   cosang)))))
            if cov is not None:
                S = np.asarray(cov[j], float)
                try:
                    q = float(e @ np.linalg.solve(S, e))
                except np.linalg.LinAlgError:
                    q = np.nan
                row.update(nees=q / 3.0, in95=bool(q <= CHI2_3_95))
        rows.append(row)
    # false positives are per volume: record on the first row only
    rows[0]["n_fp_volume"] = len(est) - len(det_of)
    return rows


def summarize(rows):
    """Aggregate per-seed rows of one grid x layout x method."""
    m = [r for r in rows if r["matched"]]
    n_truth = len(rows)
    n_fp = int(sum(r.get("n_fp_volume", 0) for r in rows))
    n_match = len(m)
    out = dict(n_truth=n_truth, n_match=n_match,
               recall=n_match / max(1, n_truth), fp=n_fp,
               ppv=n_match / max(1, n_match + n_fp))
    if m:
        E = np.array([[r["err_i"], r["err_j"], r["err_k"]] for r in m])
        e3 = np.array([r["err_3d"] for r in m])
        ax = np.array([r["axis_err_deg"] for r in m])
        for a, name in enumerate("ijk"):
            out["bias_" + name] = float(E[:, a].mean())
            out["rms_" + name] = float(np.sqrt((E[:, a] ** 2).mean()))
        out.update(mean_3d=float(e3.mean()), sd_3d=float(e3.std(ddof=1))
                   if len(e3) > 1 else 0.0,
                   p95_3d=float(np.percentile(e3, 95)), max_3d=float(e3.max()),
                   axis_med_deg=float(np.median(ax)),
                   axis_p95_deg=float(np.percentile(ax, 95)))
    sh = np.array([r["thr_shift_mm"] for r in rows
                   if np.isfinite(r.get("thr_shift_mm", np.nan))])
    out["thr_pairs"] = int(sh.size)
    out["thr_shift_med"] = float(np.median(sh)) if sh.size else np.nan
    nees = np.array([r["nees"] for r in m if "nees" in r and
                     np.isfinite(r["nees"])])
    out["nees_mean"] = float(nees.mean()) if nees.size else np.nan
    out["nees_in95"] = float(np.mean([r["in95"] for r in m if "in95" in r])) \
        if nees.size else np.nan
    pk = np.array([r["peak_hu"] for r in rows])
    out["peak_med"] = float(np.nanmedian(pk))
    out["sat_frac"] = float(np.mean(pk >= SATURATE_HU - 0.5))
    vols = {(r["rng"]) for r in rows}
    ms = [r["ms_per_seed"] for r in rows if r["seed"] == 0]
    out["ms_per_seed"] = float(np.mean(ms)) if ms else np.nan
    out["n_volumes"] = len(vols)
    return out


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return ("%%.%df" % nd) % x


def markdown(summary, header):
    lines = [header, "",
             "| grid | layout | method | recall | FP | bias i / j / k (mm) "
             "| RMS i / j / k (mm) | 3D mean ± SD | P95 | max | axis° med "
             "| thr shift (n) | NEES (in 95 %) | ms/seed | peak HU (sat) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(
            "| %s | %s | %s | %s | %d | %s / %s / %s | %s / %s / %s | "
            "%s ± %s | %s | %s | %s | %s (%d) | %s | %s | %.0f (%.0f %%) |" % (
                s["grid"], s["layout"], s["method"], fmt(s["recall"]),
                s["fp"], fmt(s.get("bias_i"), 3), fmt(s.get("bias_j"), 3),
                fmt(s.get("bias_k"), 3), fmt(s.get("rms_i")),
                fmt(s.get("rms_j")), fmt(s.get("rms_k")),
                fmt(s.get("mean_3d")), fmt(s.get("sd_3d")),
                fmt(s.get("p95_3d")), fmt(s.get("max_3d")),
                fmt(s.get("axis_med_deg"), 1), fmt(s["thr_shift_med"]),
                s["thr_pairs"],
                ("%s (%s)" % (fmt(s["nees_mean"]), fmt(s["nees_in95"])))
                if np.isfinite(s["nees_mean"]) else "–",
                fmt(s["ms_per_seed"], 1), s["peak_med"],
                100.0 * s["sat_frac"]))
    return "\n".join(lines)


# --------------------------------------------------------------- calibration
def calibrate(n=400, out_dir=None):
    """Rendered peak distribution per grid and contrast (seed_render doc)."""
    rows = []
    for gname, g in GRIDS.items():
        sp = g["spacing"]
        shape, aff = grid_affine(sp, 16.0)
        for label, M in (("postop", METAL_HU_POSTOP),
                         ("printed", METAL_HU_PRINTED)):
            rng = np.random.default_rng(0)
            pk = []
            for t in range(n):
                c = rng.uniform(-0.5, 0.5, 3) * np.asarray(sp)
                u = rng.standard_normal(3)
                vol = render_seeds(shape, aff, [c], [u], metal_hu=M,
                                   background_hu=BACKGROUND_HU,
                                   psf_sigma_mm=PSF_SIGMA_MM,
                                   noise_hu=NOISE_HU, saturate_hu=None,
                                   rng_seed=t)
                vol = degrade(vol, g, t)
                pk.append(float(vol.array.max()))
            pk = np.array(pk)
            rows.append((gname, g["label"], label, M, np.median(pk),
                         *np.percentile(pk, [25, 75, 5, 95]),
                         float(np.mean(pk >= SATURATE_HU))))
    lines = ["| grid | contrast | metal_hu | median peak | IQR | P5–P95 "
             "| ≥ 3071 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s (%s) | %s | %.0f | %.0f | %.0f–%.0f | %.0f–%.0f "
                     "| %.0f %% |" % (r[0], r[1], r[2], r[3], r[4], r[5],
                                      r[6], r[7], r[8], 100 * r[9]))
    text = "\n".join(lines)
    print(text)
    if out_dir:
        with open(os.path.join(out_dir, "calibration.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(text + "\n")
    return rows


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grids", default="G1,G2,G3,G4")
    ap.add_argument("--methods", default="baseline,centroid,model")
    ap.add_argument("--layouts", default="sparse,crowded")
    ap.add_argument("--seeds", default="0-4",
                    help="layout rng list, e.g. 0-4 or 0,2,5")
    ap.add_argument("--n-per-volume", type=int, default=N_PER_VOLUME)
    ap.add_argument("--metal-hu", type=float, default=None,
                    help="override the per-grid capsule contrast")
    ap.add_argument("--tag", default=None,
                    help="output file tag (default: the method list)")
    ap.add_argument("--out", default=os.path.join(
        REPO, "output", "validation_seed_localization"))
    ap.add_argument("--calibrate", action="store_true",
                    help="print the seed_render peak calibration and exit")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)
    if args.calibrate:
        calibrate(out_dir=args.out)
        return 0

    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    seeds = parse_seeds(args.seeds)
    for g in grids:
        if g not in GRIDS:
            ap.error("unknown grid %s" % g)
    for m in methods:
        if m not in METHODS:
            ap.error("unknown method %s" % m)
    tag = args.tag or "-".join(methods)
    commit, dirty = git_commit()
    t_start = time.time()

    method_objs = [Method(m) for m in methods]
    skipped = {}
    all_rows, summary = [], []
    for gname in grids:
        for lname in layouts:
            L = LAYOUTS[lname]
            per_method = {m.name: [] for m in method_objs}
            for r in seeds:
                centers, axes = random_seed_layout(
                    args.n_per_volume, np.random.default_rng(r + L["rng_offset"]),
                    BOX_MM, min_sep_mm=L["min_sep_mm"],
                    max_nn_mm=L["max_nn_mm"])
                vol = make_volume(gname, centers, axes,
                                  noise_seed=1000 * (r + L["rng_offset"])
                                  + list(GRIDS).index(gname),
                                  metal_hu=args.metal_hu)
                meta = dict(grid=gname, layout=lname, rng=r)
                for m in method_objs:
                    if m.reason:
                        skipped[m.name] = m.reason
                        continue
                    rows = score_volume(vol, centers, axes, m, meta)
                    if rows is None:
                        skipped[m.name] = m.reason
                        continue
                    per_method[m.name].extend(rows)
            for m in method_objs:
                rows = per_method[m.name]
                if not rows:
                    continue
                all_rows.extend(rows)
                s = dict(grid=gname, layout=lname, method=m.name,
                         metal_hu=args.metal_hu or GRIDS[gname]["metal_hu"])
                s.update(summarize(rows))
                summary.append(s)
                print("%s %-7s %-8s recall %.3f FP %3d  3D %.3f ± %.3f "
                      "P95 %.3f max %.3f  RMS k %.3f  thr %.3f mm"
                      % (gname, lname, m.name, s["recall"], s["fp"],
                         s.get("mean_3d", np.nan), s.get("sd_3d", np.nan),
                         s.get("p95_3d", np.nan), s.get("max_3d", np.nan),
                         s.get("rms_k", np.nan), s["thr_shift_med"]))
    for name, why in skipped.items():
        print("SKIPPED method %s: %s" % (name, why))
    if skipped and not args.tag:
        # never let a run with a missing method overwrite a complete result
        tag += "-skipped"

    wall = time.time() - t_start
    cmd = "python scripts/validation_seed_localization.py " + " ".join(
        sys.argv[1:] if argv is None else argv)
    if all_rows:
        keys = []
        for row in all_rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out, "seeds_%s.csv" % tag), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    if summary:
        keys = []
        for row in summary:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out, "summary_%s.csv" % tag), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)
    header = ("`%s` — commit %s%s, %s; layouts rng %s × %d seeds per volume "
              "(%d per grid and layout); match: Hungarian ≤ %.0f mm; "
              "contrast G1 %.0f HU, G2–G4 %.0f HU above 35 HU background; "
              "PSF σ %.2f mm, noise %.0f HU, clip %.0f HU; wall %.0f s"
              % (cmd, commit, " (dirty)" if dirty else "",
                 datetime.date.today().isoformat(), args.seeds,
                 args.n_per_volume, args.n_per_volume * len(seeds),
                 MATCH_MM,
                 args.metal_hu or METAL_HU_PRINTED,
                 args.metal_hu or METAL_HU_POSTOP, PSF_SIGMA_MM, NOISE_HU,
                 SATURATE_HU, wall))
    if skipped:
        header += "\n\n" + "\n".join("Skipped `%s`: %s" % kv
                                     for kv in skipped.items())
    md = markdown(summary, header)
    with open(os.path.join(args.out, "summary_%s.md" % tag), "w",
              encoding="utf-8") as fh:
        fh.write(md + "\n")
    print()
    print(md)

    runs = os.path.join(args.out, "runs.csv")
    new = not os.path.exists(runs)
    with open(runs, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "command", "grids", "methods", "layouts",
                        "seeds", "commit", "dirty", "wall_s", "skipped"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    cmd, ",".join(grids), ",".join(methods),
                    ",".join(layouts), args.seeds, commit, int(dirty),
                    "%.1f" % wall, "; ".join("%s: %s" % kv
                                             for kv in skipped.items())])
    return 0


if __name__ == "__main__":
    sys.exit(main())
