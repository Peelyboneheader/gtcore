"""Synthetic seed scenes for the merge/split repair (localization stage 3).

Not a test module; import from test files as ``import merge_split_scenes``
(``tests/`` is on ``sys.path`` when pytest runs from the repo root) and from
``scripts/localization_mergesplit_sweep.py``.

Rendering model (deliberately simple and independent of the detector): a
4.5 mm line segment blurred by an anisotropic Gaussian PSF (closed-form erf
profile along the segment), averaged over each voxel's footprint by
supersampling (<= 0.25 mm sub-steps; the footprint along k is the slice
THICKNESS, which may be thinner than the slice spacing), plus a background,
Gaussian noise and the scanner's 3071 HU clip.

Grids (``GRIDS``):

* ``G1``  0.59 x 0.59 x 1.0 mm, contiguous slices: the printed-phantom scan.
* ``G2s`` 0.5 x 0.5 x 2.0 mm with 1.0 mm thick slices: the PostOp export
  geometry (DICOM SliceThickness 1.0 at 2.0 mm spacing), i.e. a 1 mm
  unimaged gap between slabs -- where a capsule crossing that gap leaves
  two non-touching traces in adjacent slices.
* ``G3``  0.5 x 0.5 x 2.0 mm, 2 mm slabs with every other slice replaced by
  linear interpolation of its neighbours (the DICOM gap filler).
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from gtcore.pipeline import seed_detection_params
from gtcore.seeds import detect_seed_candidates
from gtcore.volume import Volume

SEED_LEN_MM = 4.5
SEED_DIA_MM = 0.8

GRIDS = {
    "G1": dict(spacing=(0.59, 0.59, 1.0), thickness=None, sigma=0.6,
               sigma_z=None, amp=5000.0, interpolate=False),
    "G2s": dict(spacing=(0.5, 0.5, 2.0), thickness=1.0, sigma=0.6,
                sigma_z=0.3, amp=3500.0, interpolate=False),
    "G3": dict(spacing=(0.5, 0.5, 2.0), thickness=None, sigma=0.6,
               sigma_z=None, amp=4200.0, interpolate=True),
}


def line_source(X, c, u, sig, length=SEED_LEN_MM):
    """Line segment ``c +- length/2 * u`` blurred by an anisotropic Gaussian
    (per-axis sigma ``sig``), normalised so an in-plane segment peaks at 1."""
    sig = np.asarray(sig, float)
    up = np.asarray(u, float) / sig
    nu = float(np.linalg.norm(up))
    uh = up / nu
    d = (X - c) / sig
    t = d @ uh
    perp2 = np.einsum("...i,...i->...", d, d) - t * t
    h = 0.5 * length * nu
    along = 0.5 * (erf((h - t) / np.sqrt(2.0)) + erf((h + t) / np.sqrt(2.0)))
    return (1.0 / (nu * sig[0])) * np.exp(-0.5 * np.clip(perp2, 0.0, None)) * along


def render(shape, spacing, centers, axes, amp, origin=(0.0, 0.0, 0.0),
           bg=35.0, sigma=0.6, sigma_z=None, noise=20.0, clip=3071.0, rng=0,
           thickness=None, sub_mm=0.25, specks=None):
    """Render seeds into a ``Volume`` of ``shape`` ([k, j, i]) on an axis-
    aligned grid.  ``specks``: optional list of ((k, j, i), HU) voxels set
    before the noise (tiny dense specks)."""
    spacing = np.asarray(spacing, float)
    origin = np.asarray(origin, float)
    sig = np.array([sigma, sigma, sigma if sigma_z is None else sigma_z])
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = spacing
    aff[:3, 3] = origin
    acc = np.zeros(shape, float)
    foot = spacing.copy()
    if thickness is not None:
        foot[2] = float(thickness)
    nsub = np.maximum(1, np.ceil(foot / sub_mm).astype(int))
    offs = [((np.arange(n) + 0.5) / n - 0.5) * s for n, s in zip(nsub, foot)]
    O = np.stack(np.meshgrid(*offs, indexing="ij"), axis=-1).reshape(-1, 3)
    for c, u in zip(np.atleast_2d(centers), np.atleast_2d(axes)):
        u = np.asarray(u, float) / np.linalg.norm(u)
        c = np.asarray(c, float)
        margin = SEED_LEN_MM / 2 + 4 * sig.max() + spacing
        lo = np.maximum(np.floor((c - margin - origin) / spacing).astype(int), 0)
        hi = np.minimum(np.ceil((c + margin - origin) / spacing).astype(int) + 1,
                        np.array(shape[::-1]))
        if np.any(hi <= lo):
            continue
        ii, jj, kk = [np.arange(a, b) for a, b in zip(lo, hi)]
        G = np.stack(np.meshgrid(ii, jj, kk, indexing="ij"), axis=-1)
        P = origin + G * spacing
        vals = np.zeros(P.shape[:3])
        for o in O:
            vals += line_source(P + o, c, u, sig)
        vals /= len(O)
        # seeds add (two blooms overlap like real attenuation)
        acc[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] += vals.transpose(2, 1, 0)
    out = bg + amp * acc
    if specks is not None:
        for kji, hu in specks:
            out[tuple(kji)] = hu
    if noise:
        out = out + np.random.default_rng(rng).normal(0.0, noise, out.shape)
    if clip is not None:
        out = np.minimum(out, clip)
    return Volume(out.astype(np.float32), aff)


def interpolate_alternate(vol, start=1):
    """Replace every other slice by the mean of its neighbours (the DICOM
    gap filler's linear interpolation at the midpoint)."""
    a = np.array(vol.array, dtype=np.float32)
    for k in range(start, a.shape[0] - 1, 2):
        a[k] = 0.5 * (a[k - 1] + a[k + 1])
    return Volume(a, vol.affine)


def render_grid(grid, shape, centers, axes, rng=0, **kw):
    g = dict(GRIDS[grid])
    interp = g.pop("interpolate")
    g.update(kw)
    vol = render(shape, g.pop("spacing"), centers, axes, g.pop("amp"), rng=rng, **g)
    return interpolate_alternate(vol) if interp else vol


def detect(vol, **kw):
    """Detection with the pipeline's spacing-aware parameters."""
    p = seed_detection_params(vol.spacing)
    return detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                                  min_mm3=p["min_mm3"], max_mm3=p["max_mm3"], **kw)


def random_axis(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def segment_distance(c1, u1, c2, u2, length=SEED_LEN_MM, n=41):
    """Minimum distance between two capsule axes (sampled; adequate here)."""
    t = np.linspace(-length / 2, length / 2, n)
    p = c1[None, :] + t[:, None] * u1[None, :]
    q = c2[None, :] + t[:, None] * u2[None, :]
    return float(np.min(np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)))


def scene_box(grid, extent_mm=(20.0, 20.0, 28.0)):
    sp = np.asarray(GRIDS[grid]["spacing"], float)
    shape_ijk = np.ceil(np.asarray(extent_mm) / sp).astype(int)
    return tuple(shape_ijk[::-1]), np.asarray(extent_mm) / 2.0


def single_scene(grid, rng):
    """One random seed near the middle of a small volume (random axis,
    random sub-voxel and sub-slice position)."""
    shape, mid = scene_box(grid)
    sp = np.asarray(GRIDS[grid]["spacing"], float)
    c = mid + rng.uniform(0.0, 1.0, 3) * sp * np.array([1, 1, 2])
    u = random_axis(rng)
    return shape, np.array([c]), np.array([u])


def pair_scene(grid, rng, kind):
    """Two DISTINCT, non-overlapping seeds close together.

    ``kind``: ``collinear`` (end to end, centres 4.6-5.6 mm apart),
    ``parallel`` (side by side, 2.0-4.0 mm apart), ``random`` (random axes,
    centres 2.5-5.5 mm apart).  Capsules never intersect (axis distance >=
    the 0.8 mm diameter plus 0.1 mm).
    """
    shape, mid = scene_box(grid)
    sp = np.asarray(GRIDS[grid]["spacing"], float)
    base = mid + rng.uniform(0.0, 1.0, 3) * sp
    for _ in range(1000):
        u1 = random_axis(rng)
        if kind == "collinear":
            d = rng.uniform(4.6, 5.6)
            u2 = u1.copy()
            off = d * u1
        elif kind == "parallel":
            d = rng.uniform(2.0, 4.0)
            v = np.cross(u1, random_axis(rng))
            v /= np.linalg.norm(v)
            u2 = u1.copy()
            off = d * v
        else:
            d = rng.uniform(2.5, 5.5)
            u2 = random_axis(rng)
            off = d * random_axis(rng)
        c1 = base - off / 2
        c2 = base + off / 2
        if segment_distance(c1, u1, c2, u2) >= SEED_DIA_MM + 0.1:
            return shape, np.array([c1, c2]), np.array([u1, u2])
    raise RuntimeError("could not place a non-overlapping pair")


def count_fragments(cands, centers, radius_mm=SEED_LEN_MM):
    """Candidates per true seed (nearest seed within ``radius_mm``)."""
    out = np.zeros(len(centers), int)
    for c in cands.centers_ras:
        d = np.linalg.norm(centers - c, axis=1)
        i = int(np.argmin(d))
        if d[i] <= radius_mm:
            out[i] += 1
    return out


def point_segment_distance(p, c, u, length=SEED_LEN_MM):
    t = float(np.clip((p - c) @ u, -length / 2, length / 2))
    return float(np.linalg.norm(p - (c + t * u)))


def owner(p, centers, axes):
    """Index of the true capsule whose axis SEGMENT is nearest to ``p``."""
    d = [point_segment_distance(p, c, u / np.linalg.norm(u))
         for c, u in zip(centers, axes)]
    return int(np.argmin(d))


def false_merges(cands, centers, axes):
    """Merges whose two fragments belong to DIFFERENT true capsules
    (fragment owner = nearest capsule axis segment)."""
    n = 0
    for m in (cands.info or {}).get("merge_log", ()):
        a, b = (np.asarray(x, float) for x in m["centers"])
        n += int(owner(a, centers, axes) != owner(b, centers, axes))
    return n


def sweep(caps=(3.0, 3.5, 4.0, 4.5, 5.0), grids=("G1", "G2s", "G3"),
          n_single=100, n_pair=40, seed=2026):
    """Sensitivity of the fragment merge to ``MERGE_MAX_SEP_MM``.

    Per grid: ``n_single`` random lone seeds (how many fragment without the
    merge, and how many stay fragmented after it) and ``n_pair`` distinct
    close pairs of each kind (collinear / parallel / random; how many merges
    join two different seeds).  Returns a list of row dicts.
    """
    rows = []
    for grid in grids:
        rng = np.random.default_rng(seed)
        singles = [single_scene(grid, rng) for _ in range(n_single)]
        pairs = [(kind,) + pair_scene(grid, rng, kind)
                 for kind in ("collinear", "parallel", "random")
                 for _ in range(n_pair)]
        vols_s = [render_grid(grid, sh, c, u, rng=i)
                  for i, (sh, c, u) in enumerate(singles)]
        vols_p = [render_grid(grid, sh, c, u, rng=1000 + i)
                  for i, (_k, sh, c, u) in enumerate(pairs)]
        base_frag = []
        for v, (_sh, c, _u) in zip(vols_s, singles):
            base_frag.append(int(count_fragments(detect(v), c)[0] > 1))
        base_lost = 0
        for v, (_k, _sh, c, _u) in zip(vols_p, pairs):
            base_lost += int((count_fragments(detect(v), c) == 0).sum())
        n_frag = int(sum(base_frag))
        for cap in caps:
            still, err = 0, []
            for v, (_sh, c, _u), fr in zip(vols_s, singles, base_frag):
                if not fr:
                    continue
                cands = detect(v, merge_fragments=True, merge_max_sep_mm=cap)
                k = count_fragments(cands, c)[0]
                still += int(k > 1)
                if k == 1:
                    d = np.linalg.norm(cands.centers_ras - c[0], axis=1)
                    err.append(float(d.min()))
            fm = {"collinear": 0, "parallel": 0, "random": 0}
            for v, (kind, _sh, c, _u) in zip(vols_p, pairs):
                cands = detect(v, merge_fragments=True, merge_max_sep_mm=cap)
                fm[kind] += false_merges(cands, c, _u)
            rows.append(dict(grid=grid, cap=float(cap), n_single=n_single,
                             fragmented=n_frag, still_fragmented=still,
                             merged_err_mean=float(np.mean(err)) if err else float("nan"),
                             merged_err_max=float(np.max(err)) if err else float("nan"),
                             n_pairs=len(pairs), pairs_lost_by_detection=base_lost,
                             false_collinear=fm["collinear"],
                             false_parallel=fm["parallel"],
                             false_random=fm["random"]))
    return rows


def format_sweep(rows):
    head = ("| grid | cap (mm) | lone seeds fragmented (no merge) | still fragmented "
            "after merge | merged centre error mean / max (mm) | false merges: "
            "collinear / parallel / random (of %d pairs each) |" % (
                rows[0]["n_pairs"] // 3 if rows else 0))
    lines = [head, "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %.1f | %d / %d | %d | %.2f / %.2f | %d / %d / %d |" % (
            r["grid"], r["cap"], r["fragmented"], r["n_single"],
            r["still_fragmented"], r["merged_err_mean"], r["merged_err_max"],
            r["false_collinear"], r["false_parallel"], r["false_random"]))
    return "\n".join(lines)
