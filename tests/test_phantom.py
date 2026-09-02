"""Ground-truth checks for the synthetic head phantom.

Everything here runs at 1.0 mm for speed; the geometry is spacing-independent
so the invariants hold at the 0.7 mm default too.
"""
import numpy as np
import pytest

from gtcore.phantom import (
    BRAIN_RADII,
    PhantomTruth,
    make_head_phantom,
)
from gtcore.volume import Volume

SPACING = 1.0
N_TILES = 3
NOISE_HU = 4.0
FOV_MM = 200.0


@pytest.fixture(scope="module")
def phantom():
    return make_head_phantom(
        spacing=SPACING, n_tiles=N_TILES, noise_hu=NOISE_HU, rng_seed=0,
        fov_mm=FOV_MM,
    )


def _nearest_voxel(vol, mask, point):
    """Value of ``mask`` at the voxel nearest a RAS point (pragmatic lookup)."""
    idx = np.round(vol.ras_to_index(np.asarray(point, dtype=float))).astype(int)
    ni, nj, nk = vol.shape_ijk
    i = int(np.clip(idx[0], 0, ni - 1))
    j = int(np.clip(idx[1], 0, nj - 1))
    k = int(np.clip(idx[2], 0, nk - 1))
    return bool(mask[k, j, i])


# ------------------------------------------------------------------- geometry
def test_returns_volume_with_expected_geometry(phantom):
    vol, truth = phantom
    assert isinstance(vol, Volume)
    assert isinstance(truth, PhantomTruth)

    n = int(round(FOV_MM / SPACING))
    assert vol.array.shape == (n, n, n)
    assert vol.array.dtype == np.float32

    expected = np.eye(4)
    expected[0, 0] = expected[1, 1] = expected[2, 2] = SPACING
    expected[:3, 3] = -(n - 1) * SPACING / 2.0
    assert np.allclose(vol.affine, expected)

    # Origin-centred cube: the FOV centre is the RAS origin.
    assert np.allclose(vol.bounds_ras().mean(axis=0), 0.0, atol=1e-9)
    assert vol.meta == {"phantom": True, "modality": "CT", "n_tiles": N_TILES}


def test_hu_range_is_sane(phantom):
    vol, _ = phantom
    lo = float(vol.array.min())
    hi = float(vol.array.max())
    # Air in/around the head, jittered by noise only.
    assert -1000.0 - 12.0 * NOISE_HU < lo < -900.0
    # Metal seeds survive the PSF blur.
    assert hi > 3000.0


# ---------------------------------------------------------------------- seeds
def test_seed_count_and_tile_membership(phantom):
    _, truth = phantom
    assert len(truth.seeds) == 4 * N_TILES
    assert len(truth.tiles) == N_TILES
    for tile in truth.tiles:
        assert len(tile.seed_ids) == 4
    all_ids = sorted(i for t in truth.tiles for i in t.seed_ids)
    assert all_ids == sorted(s.seed_id for s in truth.seeds)


def test_seed_centers_are_metal_bright(phantom):
    vol, truth = phantom
    for s in truth.seeds:
        assert vol.sample_ras(s.center_ras) > 1500.0, s.seed_id


def test_seed_axes_are_unit_and_tangential(phantom):
    _, truth = phantom
    for s in truth.seeds:
        assert abs(np.linalg.norm(s.axis_ras) - 1.0) < 1e-9
        radial = s.center_ras - truth.cavity_center_ras
        radial = radial / np.linalg.norm(radial)
        # Seeds lie flat on the wall, so the long axis is nearly tangential.
        assert abs(float(s.axis_ras @ radial)) < 0.35, s.seed_id


def test_seeds_sit_just_inside_the_cavity_wall(phantom):
    """Seed centres are inset ~2 mm from the wall, embedded in the metal mask.

    ``masks['cavity']`` is the fluid/air lumen with the seed voxels removed, so
    a seed centre must *not* be cavity, while a point 3 mm further toward the
    cavity centre lands in open lumen.
    """
    vol, truth = phantom
    cavity = truth.masks["cavity"]
    seed_mask = truth.masks["seeds"]
    for s in truth.seeds:
        radial = s.center_ras - truth.cavity_center_ras
        radial = radial / np.linalg.norm(radial)
        assert _nearest_voxel(vol, seed_mask, s.center_ras), s.seed_id
        assert not _nearest_voxel(vol, cavity, s.center_ras), s.seed_id
        inward = s.center_ras - 3.0 * radial
        assert _nearest_voxel(vol, cavity, inward), s.seed_id


# ---------------------------------------------------------------------- masks
def test_masks_present_and_shaped(phantom):
    vol, truth = phantom
    for name in ("brain", "skull", "cavity", "seeds"):
        m = truth.masks[name]
        assert m.dtype == np.bool_
        assert m.shape == vol.array.shape
        assert m.any(), name


def test_cavity_lies_inside_the_preop_brain_ellipsoid(phantom):
    vol, truth = phantom
    cavity = truth.masks["cavity"]
    assert cavity.sum() > 0
    kk, jj, ii = np.nonzero(cavity)
    pts = vol.index_to_ras(np.stack([ii, jj, kk], axis=1))
    q = ((pts / np.asarray(BRAIN_RADII)) ** 2).sum(axis=1)
    assert q.max() < 1.0


def test_seed_mask_voxel_count_is_plausible(phantom):
    _, truth = phantom
    count = int(truth.masks["seeds"].sum())
    assert 4 * N_TILES * 2 <= count <= 4 * N_TILES * 80


def test_brain_and_cavity_are_disjoint(phantom):
    _, truth = phantom
    assert not np.any(truth.masks["brain"] & truth.masks["cavity"])
    assert not np.any(truth.masks["brain"] & truth.masks["seeds"])


def test_craniotomy_removed_bone_from_the_skull_mask(phantom):
    vol, truth = phantom
    from gtcore.phantom.generate import ENTRY_DIR_U0, SKULL_OUTER_RADII

    # A point on the outer skull surface straight along the entry direction
    # must no longer be bone.
    radii = np.asarray(SKULL_OUTER_RADII)
    p = ENTRY_DIR_U0 * radii
    p = p / np.sqrt(((p / radii) ** 2).sum()) * 0.98
    assert not _nearest_voxel(vol, truth.masks["skull"], p)
    # ...while the opposite pole still is.
    assert _nearest_voxel(vol, truth.masks["skull"], -p)


def test_streaks_flag_perturbs_seed_slices_only():
    """Qualitative streaks must run, change seed slices, and stay bounded."""
    plain, truth = make_head_phantom(
        spacing=2.0, n_tiles=2, noise_hu=0.0, rng_seed=3, fov_mm=160.0
    )
    streaky, _ = make_head_phantom(
        spacing=2.0, n_tiles=2, noise_hu=0.0, rng_seed=3, fov_mm=160.0,
        streaks=True,
    )
    assert streaky.array.shape == plain.array.shape
    diff = np.abs(streaky.array - plain.array)
    seed_slices = truth.masks["seeds"].any(axis=(1, 2))
    assert diff[~seed_slices].max() == 0.0
    assert diff[seed_slices].max() > 1.0
    assert diff.max() < 200.0


# ------------------------------------------------------------- reproducibility
def test_same_seed_gives_identical_volumes():
    a, ta = make_head_phantom(spacing=2.0, n_tiles=2, rng_seed=7, fov_mm=160.0)
    b, tb = make_head_phantom(spacing=2.0, n_tiles=2, rng_seed=7, fov_mm=160.0)
    assert np.array_equal(a.array, b.array)
    assert np.array_equal(ta.masks["seeds"], tb.masks["seeds"])
    for sa, sb in zip(ta.seeds, tb.seeds):
        assert np.array_equal(sa.center_ras, sb.center_ras)
        assert np.array_equal(sa.axis_ras, sb.axis_ras)


def test_different_seed_gives_different_volume():
    a, _ = make_head_phantom(spacing=2.0, n_tiles=2, rng_seed=7, fov_mm=160.0)
    b, _ = make_head_phantom(spacing=2.0, n_tiles=2, rng_seed=8, fov_mm=160.0)
    assert not np.array_equal(a.array, b.array)
