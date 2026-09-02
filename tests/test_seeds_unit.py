"""Unit tests for gtcore.seeds.detect on a hand-painted capsule phantom.

The phantom is built here rather than imported from gtcore.phantom so these
tests stand alone.

Painting model: a 0.8 mm diameter capsule is thinner than a 1 mm voxel, so
binary painting would quantise the centre onto the grid and make sub-voxel
accuracy untestable, and a plain partial-volume paint would produce a spiky
field that fragments into several components once thresholded.  Neither
resembles the CT anyway.  Instead each capsule is rendered as its *bloom*: HU
falls off as a Gaussian of the distance to the capsule's axis segment, which
puts the 2000 HU iso-surface about 3 mm across and 4.5 mm long -- the
appearance the domain description gives for a real Cs-131 seed.  The field is
exactly symmetric about the true centre and about the true axis, so a
detector that is right gets the right answer to well under a voxel.
"""
import numpy as np
import pytest
from scipy import ndimage

from gtcore.volume import Volume
from gtcore.seeds import SeedCandidates, detect_seed_candidates


SEED_LEN_MM = 4.5
SEED_DIA_MM = 0.8
BACKGROUND_HU = 35.0
SEED_HU = 8000.0

SHAPE = (34, 38, 42)                      # [k, j, i] -- deliberately non-cubic
SPACING = (1.0, 1.0, 1.0)
ORIGIN = (-15.0, -20.0, -25.0)


def make_affine(spacing=SPACING, origin=ORIGIN):
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = spacing
    aff[:3, 3] = origin
    return aff


BLOOM_SIGMA_MM = 0.9  # puts the 2000 HU iso-surface ~3 mm across


def paint_capsule(arr, affine, center_ras, axis, length=SEED_LEN_MM,
                  bloom_sigma=BLOOM_SIGMA_MM):
    """Add one bloomed capsule to ``arr`` (in place); core reaches SEED_HU."""
    affine = np.asarray(affine, dtype=float)
    spacing = np.array([affine[0, 0], affine[1, 1], affine[2, 2]], dtype=float)
    origin = affine[:3, 3]
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    c = np.asarray(center_ras, dtype=float)
    half = length / 2.0

    # bounding box in voxel index space, with margin for the bloom tail
    margin = half + 4.0 * bloom_sigma
    ijk_c = (c - origin) / spacing
    nk, nj, ni = arr.shape
    lo = np.maximum(np.floor(ijk_c - margin / spacing).astype(int), 0)
    hi = np.minimum(np.ceil(ijk_c + margin / spacing).astype(int) + 1,
                    np.array([ni, nj, nk]))

    ii = np.arange(lo[0], hi[0])
    jj = np.arange(lo[1], hi[1])
    kk = np.arange(lo[2], hi[2])

    X = origin[0] + ii * spacing[0] - c[0]
    Y = origin[1] + jj * spacing[1] - c[1]
    Z = origin[2] + kk * spacing[2] - c[2]
    gx = X[None, None, :]
    gy = Y[None, :, None]
    gz = Z[:, None, None]

    t = gx * u[0] + gy * u[1] + gz * u[2]
    t_clamped = np.clip(t, -half, half)          # distance to the axis *segment*
    d2 = ((gx - t_clamped * u[0]) ** 2
          + (gy - t_clamped * u[1]) ** 2
          + (gz - t_clamped * u[2]) ** 2)
    field = np.exp(-0.5 * d2 / (bloom_sigma ** 2))

    sub = arr[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    arr[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = np.maximum(
        sub, BACKGROUND_HU + field * (SEED_HU - BACKGROUND_HU)
    )


TRUE_AXES = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
])
TRUE_AXES = TRUE_AXES / np.linalg.norm(TRUE_AXES, axis=1, keepdims=True)

# voxel-index placements >= 8 mm apart, with deliberate sub-voxel offsets
TRUE_IJK = np.array([
    [10.0, 10.3, 17.0],
    [21.4, 10.0, 17.2],
    [32.0, 10.0, 16.5],
    [10.0, 28.0, 17.0],
    [21.0, 27.6, 17.0],
    [32.2, 28.0, 17.4],
])


def build_seed_phantom(sigma_vox=0.6, extra_ijk=None, extra_axes=None):
    affine = make_affine()
    spacing = np.array(SPACING)
    origin = np.array(ORIGIN)
    ijk = TRUE_IJK
    axes = TRUE_AXES
    if extra_ijk is not None:
        ijk = np.vstack([ijk, np.atleast_2d(extra_ijk)])
        axes = np.vstack([axes, np.atleast_2d(extra_axes)])
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    centers = origin + ijk * spacing

    arr = np.full(SHAPE, BACKGROUND_HU, dtype=np.float64)
    for c, u in zip(centers, axes):
        paint_capsule(arr, affine, c, u)
    arr = ndimage.gaussian_filter(arr, sigma=sigma_vox)
    return Volume(arr.astype(np.float32), affine), centers, axes


@pytest.fixture(scope="module")
def phantom():
    return build_seed_phantom()


def match_by_nearest(found, truth):
    """Greedy nearest-neighbour pairing of found -> true centres."""
    d = np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2)
    pairs = {}
    used = set()
    for fi in np.argsort(d.min(axis=1)):
        order = np.argsort(d[fi])
        for ti in order:
            if ti not in used:
                pairs[int(fi)] = int(ti)
                used.add(int(ti))
                break
    return pairs


def test_detect_finds_exactly_six(phantom):
    vol, centers, axes = phantom
    cands = detect_seed_candidates(vol)
    assert isinstance(cands, SeedCandidates)
    assert len(cands) == 6, "found %d candidates" % len(cands)
    assert cands.centers_ras.shape == (6, 3)
    assert cands.axes_ras.shape == (6, 3)
    assert cands.volumes_mm3.shape == (6,)
    assert cands.elongations.shape == (6,)
    assert cands.mask.shape == vol.array.shape
    assert cands.mask.dtype == bool


def test_detect_centroid_accuracy(phantom):
    vol, centers, axes = phantom
    cands = detect_seed_candidates(vol)
    pairs = match_by_nearest(cands.centers_ras, centers)
    assert len(pairs) == 6
    errs = []
    for fi, ti in pairs.items():
        errs.append(np.linalg.norm(cands.centers_ras[fi] - centers[ti]))
    errs = np.array(errs)
    print("\nseed centroid errors (mm): %s  max=%.3f" % (np.round(errs, 3), errs.max()))
    assert errs.max() < 0.7, "max centroid error %.3f mm" % errs.max()


def test_detect_axis_accuracy(phantom):
    vol, centers, axes = phantom
    cands = detect_seed_candidates(vol)
    pairs = match_by_nearest(cands.centers_ras, centers)
    dots = []
    for fi, ti in pairs.items():
        a = cands.axes_ras[fi]
        assert abs(np.linalg.norm(a) - 1.0) < 1e-6, "axis not unit length"
        dots.append(abs(float(np.dot(a, axes[ti]))))
    dots = np.array(dots)
    print("\nseed |axis dots|: %s  min=%.3f" % (np.round(dots, 3), dots.min()))
    assert dots.min() > 0.8, "worst |dot| %.3f" % dots.min()


def test_detect_elongation_and_volume(phantom):
    vol, centers, axes = phantom
    cands = detect_seed_candidates(vol)
    assert (cands.elongations > 1.2).all(), cands.elongations
    assert (cands.volumes_mm3 > 0.2).all()
    assert (cands.volumes_mm3 < 120.0).all()
    # ordered by descending volume
    assert np.all(np.diff(cands.volumes_mm3) <= 1e-9)


def test_detect_mask_covers_every_seed(phantom):
    vol, centers, axes = phantom
    cands = detect_seed_candidates(vol)
    assert cands.mask.sum() > 0
    labels, n = ndimage.label(cands.mask)
    assert n == 6
    assert (vol.array[cands.mask] > 2000.0).all()


def test_detect_volume_filter_rejects_large_and_tiny():
    vol, centers, axes = build_seed_phantom()
    arr = np.array(vol.array)
    # a big metal plate, and an isolated single hot voxel
    arr[5:9, 5:20, 5:20] = 9000.0
    arr[30, 34, 38] = 9000.0
    v2 = Volume(arr, vol.affine)

    cands = detect_seed_candidates(v2, min_mm3=2.0, max_mm3=120.0)
    assert len(cands) == 6, "filters let %d through" % len(cands)
    assert not cands.mask[6, 10, 10]
    assert not cands.mask[30, 34, 38]

    loose = detect_seed_candidates(v2, min_mm3=0.2, max_mm3=1e9)
    assert len(loose) == 8


def test_detect_splits_two_touching_seeds():
    """Two seeds end to end merge into one blob; the volume/median rule and
    the k-means split have to recover both."""
    # 7 mm along +k from seed 0, same axis -- their blooms just touch
    vol, centers, axes = build_seed_phantom(
        extra_ijk=[10.0, 10.3, 24.0], extra_axes=[0.0, 0.0, 1.0]
    )
    assert len(centers) == 7

    merged = detect_seed_candidates(vol, split_merged=False)
    assert len(merged) == 6, "phantom pair is not actually merged"
    assert merged.volumes_mm3[0] > 1.6 * np.median(merged.volumes_mm3)

    cands = detect_seed_candidates(vol)
    assert len(cands) == 7, "split produced %d candidates" % len(cands)

    pairs = match_by_nearest(cands.centers_ras, centers)
    errs = np.array([np.linalg.norm(cands.centers_ras[f] - centers[t])
                     for f, t in pairs.items()])
    print("\nsplit-pair centroid errors (mm): %s" % np.round(errs, 3))
    assert errs.max() < 1.0, "max centroid error after split %.3f mm" % errs.max()


def test_detect_does_not_split_large_foreign_metal():
    """A plate is many times the median blob, which means it is not seeds."""
    vol, centers, axes = build_seed_phantom()
    arr = np.array(vol.array)
    arr[5:9, 5:20, 5:20] = 9000.0
    cands = detect_seed_candidates(Volume(arr, vol.affine), max_mm3=1e9)
    assert len(cands) == 7, "plate was split into pieces (%d)" % len(cands)
    assert cands.volumes_mm3.max() > 500.0


def test_detect_empty_volume():
    affine = make_affine()
    vol = Volume(np.full(SHAPE, BACKGROUND_HU, dtype=np.float32), affine)
    cands = detect_seed_candidates(vol)
    assert len(cands) == 0
    assert cands.centers_ras.shape == (0, 3)
    assert not cands.mask.any()


def test_detect_degenerate_blob():
    """A 1-voxel blob gets the documented fallback axis, not a NaN."""
    affine = make_affine()
    arr = np.full(SHAPE, BACKGROUND_HU, dtype=np.float32)
    arr[20, 20, 20] = 9000.0
    cands = detect_seed_candidates(Volume(arr, affine), min_mm3=0.2)
    assert len(cands) == 1
    assert np.allclose(cands.axes_ras[0], [0.0, 0.0, 1.0])
    assert cands.elongations[0] == pytest.approx(1.0)
    assert np.isfinite(cands.centers_ras).all()


def test_detect_respects_affine_offset():
    """Centres are RAS, not voxel indices."""
    vol_a, centers_a, _ = build_seed_phantom()
    arr = np.array(vol_a.array)
    aff_b = make_affine(origin=(100.0, 200.0, 300.0))
    cands_b = detect_seed_candidates(Volume(arr, aff_b))
    shift = np.array([100.0, 200.0, 300.0]) - np.array(ORIGIN)
    cands_a = detect_seed_candidates(vol_a)
    assert np.allclose(np.sort(cands_b.centers_ras, axis=0),
                       np.sort(cands_a.centers_ras + shift, axis=0), atol=1e-4)
