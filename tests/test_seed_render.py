"""Seed-localization harness renderer (gtcore.phantom.seed_render).

The renderer is the truth generator for every localization gate in
docs/plan-localization.md, so its physics is pinned here:
- mass: noise-free and unclipped, sum(v - bg) * voxel volume equals the
  capsule contrast times the capsule volume;
- position: the background-subtracted grey centroid of a rendered seed is the
  true centre on an anisotropic grid fine enough to sample the PSF;
- slab integration: a 0.5 mm rendering block-averaged to 2.0 mm slices is the
  direct 2.0 mm rendering;
- saturation clips at the ceiling; drop_and_interpolate marks the invented
  slices like the DICOM loader; the head phantom default is bit-identical.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gtcore.phantom import make_head_phantom
from gtcore.phantom.seed_render import (
    DEFAULT_METAL_HU,
    SATURATE_HU,
    capsule_volume_mm3,
    drop_and_interpolate,
    random_seed_layout,
    render_seeds,
    thick_slices,
)
from gtcore.volume import Volume


def _grid(spacing, side_mm=16.0, signs=(1, 1, 1)):
    """Odd-sized grid centred on the RAS origin; returns (shape_kji, affine)."""
    n_ijk = [int(np.ceil(side_mm / s)) | 1 for s in spacing]
    aff = np.eye(4)
    for a in range(3):
        aff[a, a] = signs[a] * spacing[a]
        aff[a, 3] = -signs[a] * (n_ijk[a] - 1) * spacing[a] / 2.0
    return tuple(n_ijk[::-1]), aff


def _grey_centroid(vol, background):
    a = vol.array.astype(float) - background
    kk, jj, ii = np.indices(a.shape)
    ras = vol.index_to_ras(np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1))
    return (ras * a.ravel()[:, None]).sum(axis=0) / a.sum()


def test_capsule_volume():
    # 0.8 mm capsule, 4.5 mm overall: 3.7 mm cylinder + one sphere
    r = 0.4
    assert np.isclose(capsule_volume_mm3(4.5, 0.8),
                      np.pi * r * r * 3.7 + 4.0 / 3.0 * np.pi * r ** 3)


def test_mass_conservation_without_saturation():
    rng = np.random.default_rng(3)
    shape, aff = _grid((0.5, 0.6, 1.3), side_mm=40.0, signs=(-1, 1, 1))
    centers = rng.uniform(-8.0, 8.0, (3, 3))
    centers[1] += 12.0            # keep the three seeds apart (patches may sum)
    axes = rng.standard_normal((3, 3))
    vol = render_seeds(shape, aff, centers, axes, metal_hu=DEFAULT_METAL_HU,
                       background_hu=35.0, psf_sigma_mm=(0.45, 0.6),
                       noise_hu=0.0, saturate_hu=None)
    voxel = float(np.prod(vol.spacing))
    mass = float((vol.array.astype(float) - 35.0).sum()) * voxel
    expected = 3 * DEFAULT_METAL_HU * capsule_volume_mm3()
    assert abs(mass / expected - 1.0) < 0.02


def test_grey_centroid_is_truth_on_anisotropic_grid():
    rng = np.random.default_rng(11)
    shape, aff = _grid((0.4, 0.5, 0.7))
    for _ in range(6):
        c = rng.uniform(-0.5, 0.5, 3)          # random sub-voxel offset
        u = rng.standard_normal(3)
        vol = render_seeds(shape, aff, [c], [u], background_hu=35.0,
                           noise_hu=0.0, saturate_hu=None)
        err = _grey_centroid(vol, 35.0) - c
        assert np.all(np.abs(err) < 0.02), err


def test_block_average_equals_direct_thick_rendering():
    rng = np.random.default_rng(7)
    c = rng.uniform(-0.6, 0.6, 3)
    u = rng.standard_normal(3)
    # fine 0.5 mm slices; 4 of them make one 2.0 mm slab
    shape_f, aff_f = _grid((0.5, 0.5, 0.5), side_mm=16.0)
    nk = (shape_f[0] // 4) * 4
    shape_f = (nk,) + shape_f[1:]
    fine = render_seeds(shape_f, aff_f, [c], [u], background_hu=0.0,
                        noise_hu=0.0, saturate_hu=None)
    thick = thick_slices(fine, 4)
    direct = render_seeds(thick.array.shape, thick.affine, [c], [u],
                          background_hu=0.0, noise_hu=0.0, saturate_hu=None)
    assert np.allclose(direct.spacing, (0.5, 0.5, 2.0))
    peak = float(direct.array.max())
    assert float(np.abs(thick.array - direct.array).max()) < 0.01 * peak


def test_saturation_clips_at_ceiling():
    shape, aff = _grid((0.59, 0.59, 1.0))
    vol = render_seeds(shape, aff, [[0.1, -0.2, 0.3]], [[1.0, 0.2, 0.1]],
                       metal_hu=4.0 * DEFAULT_METAL_HU, noise_hu=10.0,
                       saturate_hu=SATURATE_HU)
    assert float(vol.array.max()) == SATURATE_HU
    assert int((vol.array == SATURATE_HU).sum()) >= 2
    free = render_seeds(shape, aff, [[0.1, -0.2, 0.3]], [[1.0, 0.2, 0.1]],
                        metal_hu=4.0 * DEFAULT_METAL_HU, noise_hu=10.0,
                        saturate_hu=None)
    assert float(free.array.max()) > SATURATE_HU
    below = free.array < SATURATE_HU
    assert np.array_equal(vol.array[below], free.array[below])


def test_rejects_rotated_affine():
    shape, aff = _grid((0.5, 0.5, 1.0))
    rot = aff.copy()
    rot[0, 1] = 0.1
    with pytest.raises(ValueError):
        render_seeds(shape, rot, [[0, 0, 0]], [[1, 0, 0]])


def test_drop_and_interpolate_marks_the_invented_slices():
    nk = 10
    arr = np.broadcast_to(np.arange(nk, dtype=np.float32)[:, None, None] * 10.0,
                          (nk, 3, 4)).copy()
    aff = np.diag([0.5, 0.5, 1.0, 1.0])
    aff[:3, 3] = (1.0, 2.0, 3.0)
    vol = Volume(arr, aff, {"modality": "CT"})

    out = drop_and_interpolate(vol, [0, 2, 4, 5, 8, 9])
    assert out.meta["interpolated_k"] == [1, 3, 6, 7]
    assert out.meta["slices_interpolated"] == 4
    assert out.meta["slices_present"] == 6
    # linear in k by construction, so interpolation reproduces it exactly
    assert np.allclose(out.array[:, 0, 0], 10.0 * np.arange(nk))
    assert np.allclose(out.affine, vol.affine)

    # outside the kept range the loader has no grid: cropped, origin moves
    out2 = drop_and_interpolate(vol, [2, 4, 6])
    assert out2.array.shape[0] == 5
    assert out2.meta["interpolated_k"] == [1, 3]
    assert np.allclose(out2.index_to_ras([0, 0, 0]), vol.index_to_ras([0, 0, 2]))

    # the loader's own grid step (median kept spacing): regular keep -> no gaps
    out3 = drop_and_interpolate(vol, [0, 2, 4, 6, 8], grid="loader")
    assert out3.array.shape[0] == 5 and out3.meta["interpolated_k"] == []
    assert np.isclose(out3.spacing[2], 2.0)


def test_random_seed_layout_separations_and_axes():
    sparse, ax = random_seed_layout(40, np.random.default_rng(0), 44.0,
                                    min_sep_mm=8.0)
    d = np.linalg.norm(sparse[:, None] - sparse[None], axis=2)
    np.fill_diagonal(d, np.inf)
    assert sparse.shape == (40, 3) and d.min() >= 8.0
    assert np.all(np.abs(sparse) <= 22.0)
    assert np.allclose(np.linalg.norm(ax, axis=1), 1.0)

    crowded, _ = random_seed_layout(40, 1, 44.0, min_sep_mm=7.0, max_nn_mm=8.0)
    d = np.linalg.norm(crowded[:, None] - crowded[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    assert nn.min() >= 7.0 and nn.max() <= 8.0 + 1e-9


# sha256 of make_head_phantom(spacing=1.0, rng_seed=0).array recorded at
# commit 3cf35af (before seed_render existed) with numpy 2.5.2: the default
# phantom must not move by a single bit.  A different numpy build may round
# float32 transcendentals differently, so the hash is only asserted there.
_HEAD_HASH = "6ca876cab0c749699bc172721b7b7497efb6b751157c0184e4cceff16fa980e5"
_HEAD_HASH_NUMPY = "2.5.2"


def test_head_phantom_default_is_bit_identical():
    vol, truth = make_head_phantom(spacing=1.0, rng_seed=0)
    explicit, _ = make_head_phantom(spacing=1.0, rng_seed=0,
                                    seed_render="binary", saturate_hu=None)
    assert np.array_equal(vol.array, explicit.array)
    if np.__version__ == _HEAD_HASH_NUMPY:
        digest = hashlib.sha256(np.ascontiguousarray(vol.array).tobytes())
        assert digest.hexdigest() == _HEAD_HASH

    analytic, truth_a = make_head_phantom(spacing=1.0, rng_seed=0,
                                          seed_render="analytic",
                                          saturate_hu=SATURATE_HU)
    # same anatomy draw and truth; seeds now exact capsules, clipped
    assert np.allclose([s.center_ras for s in truth_a.seeds],
                       [s.center_ras for s in truth.seeds])
    assert float(analytic.array.max()) <= SATURATE_HU
    far = ~truth.masks["seeds"]
    far &= np.abs(vol.array - analytic.array) < 1e-3
    assert far.mean() > 0.99
