"""Unit tests for gtcore.segment on small hand-built synthetic volumes.

Nothing here imports gtcore.phantom -- every array is built in this file so the
tests stay independent of the phantom generator.
"""
import numpy as np
import pytest
from scipy import ndimage

from gtcore.volume import Volume
from gtcore.segment import mask_to_mesh, segment_cavity, segment_head
from gtcore.segment._morph import close_mm, largest_cc, match_shape


# --------------------------------------------------------------- helpers
def make_affine(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Axis-aligned RAS affine: voxel (i, j, k) -> origin + spacing * (i, j, k)."""
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = spacing
    aff[:3, 3] = origin
    return aff


def ras_grid(shape_kji, affine):
    """(X, Y, Z) RAS coordinate grids shaped like a [k, j, i] array."""
    nk, nj, ni = shape_kji
    kk, jj, ii = np.meshgrid(
        np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij"
    )
    x = affine[0, 0] * ii + affine[0, 3]
    y = affine[1, 1] * jj + affine[1, 3]
    z = affine[2, 2] * kk + affine[2, 3]
    return x, y, z


def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * float((a & b).sum()) / float(denom)


def shell_with_cap_hole(shape_kji, affine, center, r_mm, thick_mm, cap_full_deg,
                        cap_dir=(0.0, 0.0, 1.0)):
    """Hollow spherical shell with a conical cap removed -- a toy craniotomy.

    ``cap_full_deg`` is the *full* cone angle, so a 20 degree cap on a 30 mm
    shell is a defect about 10 mm across.  That is deliberately matched to the
    14 mm closing radius the vault segmentation uses: a rolling ball only
    bridges a defect appreciably narrower than its own diameter, so a defect
    much wider than this would be a test of the wrong thing (it would be
    asserting that a 14 mm closing does something a 14 mm closing cannot do).
    """
    x, y, z = ras_grid(shape_kji, affine)
    dx, dy, dz = x - center[0], y - center[1], z - center[2]
    rad = np.sqrt(dx * dx + dy * dy + dz * dz)
    shell = (rad >= r_mm - thick_mm / 2.0) & (rad <= r_mm + thick_mm / 2.0)

    u = np.asarray(cap_dir, dtype=float)
    u = u / np.linalg.norm(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (dx * u[0] + dy * u[1] + dz * u[2]) / np.maximum(rad, 1e-9)
    cap = cosang >= np.cos(np.deg2rad(cap_full_deg / 2.0))
    return shell & ~cap


# ------------------------------------------------------------- _morph
def test_match_shape_crops_and_pads():
    m = np.ones((3, 4, 5), dtype=bool)
    assert match_shape(m, (2, 4, 7)).shape == (2, 4, 7)
    out = match_shape(m, (2, 4, 7))
    assert out[:, :, :5].all()
    assert not out[:, :, 5:].any()
    assert match_shape(m, (3, 4, 5)).shape == (3, 4, 5)


def test_largest_cc_empty_and_multi():
    assert not largest_cc(np.zeros((4, 4, 4), dtype=bool)).any()
    m = np.zeros((10, 10, 10), dtype=bool)
    m[1:3, 1:3, 1:3] = True     # 8 voxels
    m[6:9, 6:9, 6:9] = True     # 27 voxels
    out = largest_cc(m)
    assert out.sum() == 27
    assert out[7, 7, 7] and not out[2, 2, 2]


def test_close_mm_seals_craniotomy_and_is_extensive():
    shape = (90, 90, 90)
    affine = make_affine((1.0, 1.0, 1.0), (-45.0, -45.0, -45.0))
    spacing = (1.0, 1.0, 1.0)
    shell = shell_with_cap_hole(shape, affine, (0.0, 0.0, 0.0),
                                r_mm=30.0, thick_mm=4.0, cap_full_deg=20.0)

    # The unsealed shell leaks: filling it does not create a big cavity.
    leaked = ndimage.binary_fill_holes(shell) & ~shell
    true_interior_vox = (4.0 / 3.0) * np.pi * 28.0 ** 3

    closed = close_mm(shell, spacing, radius_mm=14.0)

    # closing is extensive
    assert (closed | shell == closed).all(), "close_mm must be a superset of its input"

    sealed_interior = largest_cc(ndimage.binary_fill_holes(closed) & ~closed)
    assert sealed_interior.sum() > 0.6 * true_interior_vox, (
        "craniotomy not sealed: interior only %d voxels" % sealed_interior.sum()
    )
    assert sealed_interior.sum() > 5.0 * (largest_cc(leaked).sum() + 1)


def test_close_mm_handles_empty_mask():
    m = np.zeros((8, 8, 8), dtype=bool)
    assert not close_mm(m, (1.0, 1.0, 1.0), 5.0).any()


def test_close_mm_anisotropic_returns_input_shape():
    shape = (30, 60, 60)
    affine = make_affine((0.5, 0.5, 2.0), (-15.0, -15.0, -30.0))
    m = shell_with_cap_hole(shape, affine, (0.0, 0.0, 0.0),
                            r_mm=10.0, thick_mm=3.0, cap_full_deg=20.0)
    out = close_mm(m, (0.5, 0.5, 2.0), radius_mm=6.0)
    assert out.shape == shape
    assert (out | m == out).all()


# -------------------------------------------------------------- head CT
HEAD_SHAPE = (96, 96, 96)
HEAD_AFFINE = make_affine((1.0, 1.0, 1.0), (-48.0, -48.0, -48.0))
HEAD_CENTER = (0.0, 0.0, 0.0)
SCALP_R = 36.0
SKULL_R = 32.0
SKULL_T = 4.0
BRAIN_R = 30.0


def build_head_ct(cavity=None, air_pocket=None, cap_full_deg=20.0):
    """Toy head: air / scalp / skull-with-craniotomy / brain, optional cavity."""
    x, y, z = ras_grid(HEAD_SHAPE, HEAD_AFFINE)
    rad = np.sqrt(x * x + y * y + z * z)

    arr = np.full(HEAD_SHAPE, -1000.0, dtype=np.float32)
    arr[rad <= SCALP_R] = 40.0                      # scalp / soft tissue
    skull = shell_with_cap_hole(HEAD_SHAPE, HEAD_AFFINE, HEAD_CENTER,
                                r_mm=SKULL_R, thick_mm=SKULL_T,
                                cap_full_deg=cap_full_deg)
    arr[skull] = 900.0
    interior = rad <= BRAIN_R
    arr[interior] = 35.0

    cav_mask = np.zeros(HEAD_SHAPE, dtype=bool)
    if cavity is not None:
        c, r = cavity
        cr = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
        cav_mask = cr <= r
        arr[cav_mask] = 15.0
        if air_pocket is not None:
            ac, ar = air_pocket
            arad = np.sqrt((x - ac[0]) ** 2 + (y - ac[1]) ** 2 + (z - ac[2]) ** 2)
            arr[arad <= ar] = -1000.0

    vol = Volume(arr, HEAD_AFFINE)
    return vol, interior, cav_mask


@pytest.fixture(scope="module")
def head_result():
    vol, interior, _ = build_head_ct()
    return vol, interior, segment_head(vol)


def test_segment_head_finds_brain(head_result):
    vol, true_interior, seg = head_result
    d = dice(seg["brain"], true_interior)
    assert d > 0.8, "brain dice %.3f" % d


def test_segment_head_seals_craniotomy(head_result):
    vol, true_interior, seg = head_result
    interior = seg["cranial_interior"]
    assert interior.sum() < 1.5 * true_interior.sum(), (
        "interior leaked through the craniotomy: %d vs true %d"
        % (interior.sum(), true_interior.sum())
    )
    # and it must not escape the skull at all
    x, y, z = ras_grid(HEAD_SHAPE, HEAD_AFFINE)
    rad = np.sqrt(x * x + y * y + z * z)
    outside = interior & (rad > SKULL_R + SKULL_T)
    assert outside.sum() < 0.02 * interior.sum()


def test_segment_head_body_and_skull(head_result):
    vol, true_interior, seg = head_result
    assert seg["body"].sum() > seg["cranial_interior"].sum()
    assert seg["skull"].any()
    assert not (seg["skull"] & seg["brain"]).any()


def test_segment_head_escalates_for_a_wide_defect():
    """A defect the default 14 mm closing cannot bridge must not silently
    produce an empty brain -- the closing radius escalates instead."""
    vol, true_interior, _ = build_head_ct(cap_full_deg=40.0)

    # the default radius on its own genuinely fails on this one
    naive_vault = close_mm((np.asarray(vol.array) > 300.0), vol.spacing, 14.0)
    naive = largest_cc(ndimage.binary_fill_holes(naive_vault) & ~naive_vault)
    assert naive.sum() == 0, "phantom defect is not wide enough to test escalation"

    seg = segment_head(vol)
    assert seg["cranial_interior"].sum() > 0
    d = dice(seg["brain"], true_interior)
    assert d > 0.8, "brain dice on the wide-defect head %.3f" % d


def test_segment_head_metal_mask_excluded_from_bone():
    """A bright metal blob inside the brain must not end up in the skull."""
    vol, true_interior, _ = build_head_ct()
    arr = np.array(vol.array)
    metal = np.zeros(HEAD_SHAPE, dtype=bool)
    metal[48, 48, 40:43] = True   # a few voxels well inside the brain
    arr[metal] = 8000.0
    vol2 = Volume(arr, HEAD_AFFINE)

    seg = segment_head(vol2, metal_mask=metal)
    assert not (seg["skull"] & metal).any()
    d = dice(seg["brain"], true_interior)
    assert d > 0.8, "brain dice with metal present %.3f" % d


# --------------------------------------------------------------- cavity
def test_segment_cavity_with_seed_prior():
    cav_c = np.array([10.0, 0.0, 0.0])
    cav_r = 12.0
    vol, true_interior, true_cav = build_head_ct(
        cavity=(cav_c, cav_r), air_pocket=(cav_c + np.array([0.0, 0.0, 5.0]), 3.0)
    )
    seg = segment_head(vol)

    # four fake tiles sitting on the cavity wall
    dirs = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
    seeds = cav_c + dirs * (cav_r - 1.0)

    cav = segment_cavity(vol, seg["cranial_interior"], seg["brain"],
                         seed_centers_ras=seeds, seed_radius_mm=14.0)
    d = dice(cav, true_cav)
    assert d > 0.7, "cavity dice %.3f" % d


def test_segment_cavity_prior_rejects_the_decoy():
    """Two dark pockets; only the one near the seeds may be returned."""
    cav_c = np.array([12.0, 0.0, 0.0])
    cav_r = 10.0
    decoy_c = np.array([-14.0, 0.0, 0.0])
    decoy_r = 11.0

    x, y, z = ras_grid(HEAD_SHAPE, HEAD_AFFINE)
    vol, true_interior, true_cav = build_head_ct(cavity=(cav_c, cav_r))
    arr = np.array(vol.array)
    drad = np.sqrt((x - decoy_c[0]) ** 2 + (y - decoy_c[1]) ** 2
                   + (z - decoy_c[2]) ** 2)
    decoy = drad <= decoy_r
    arr[decoy] = 10.0
    vol = Volume(arr, HEAD_AFFINE)
    seg = segment_head(vol)

    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    seeds = cav_c + dirs * (cav_r - 1.0)

    cav = segment_cavity(vol, seg["cranial_interior"], seg["brain"],
                         seed_centers_ras=seeds, seed_radius_mm=12.0)
    assert dice(cav, true_cav) > 0.7
    assert (cav & decoy).sum() < 0.2 * decoy.sum(), "seed prior picked up the decoy"

    # without the prior the (larger) decoy wins -- which is exactly the failure
    # mode the prior exists to prevent.
    no_prior = segment_cavity(vol, seg["cranial_interior"], seg["brain"])
    assert (no_prior & decoy).sum() > (no_prior & true_cav).sum()


def test_segment_cavity_empty_when_nothing_dark():
    vol, true_interior, _ = build_head_ct()
    seg = segment_head(vol)
    cav = segment_cavity(vol, seg["cranial_interior"], seg["brain"])
    assert cav.sum() < 0.02 * true_interior.sum()


# ---------------------------------------------------------------- mesh
def test_mask_to_mesh_sphere():
    shape = (60, 60, 60)
    affine = make_affine((1.0, 1.0, 1.0), (-3.0, 7.0, 11.0))
    center_ras = np.array([-3.0 + 30.0, 7.0 + 30.0, 11.0 + 30.0])
    r = 20.0
    x, y, z = ras_grid(shape, affine)
    mask = np.sqrt((x - center_ras[0]) ** 2 + (y - center_ras[1]) ** 2
                   + (z - center_ras[2]) ** 2) <= r

    mesh = mask_to_mesh(mask, affine, smooth_iterations=10, largest_only=True)

    assert mesh.is_watertight, "sphere mesh should be watertight"

    expected_area = 4.0 * np.pi * r ** 2
    rel = abs(mesh.area - expected_area) / expected_area
    assert rel < 0.15, "area %.1f vs expected %.1f (%.1f%%)" % (
        mesh.area, expected_area, 100 * rel
    )

    centroid_err = np.linalg.norm(np.asarray(mesh.centroid) - center_ras)
    assert centroid_err < 1.0, "centroid error %.3f mm" % centroid_err

    v = np.asarray(mesh.vertices) - center_ras
    n = np.asarray(mesh.vertex_normals)
    outward = (np.einsum("ij,ij->i", v, n) > 0).mean()
    assert outward > 0.9, "only %.1f%% of normals point outward" % (100 * outward)


def test_mask_to_mesh_empty_and_largest_only():
    affine = make_affine()
    assert len(mask_to_mesh(np.zeros((5, 5, 5), dtype=bool), affine).faces) == 0

    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[5:15, 5:15, 5:15] = True    # big cube
    mask[30:34, 30:34, 30:34] = True  # speck
    big = mask_to_mesh(mask, affine, smooth_iterations=0, largest_only=True)
    both = mask_to_mesh(mask, affine, smooth_iterations=0, largest_only=False)
    assert len(big.split(only_watertight=False)) == 1
    assert len(both.split(only_watertight=False)) == 2


def test_mask_to_mesh_uses_affine_axis_order():
    """A slab thin along i must be thin along RAS x -- catches a k/j/i flip."""
    shape = (40, 40, 40)
    affine = make_affine((1.0, 1.0, 1.0))
    mask = np.zeros(shape, dtype=bool)
    mask[8:32, 8:32, 18:22] = True   # [k, j, i] -> thin in i
    mesh = mask_to_mesh(mask, affine, smooth_iterations=0)
    extents = mesh.bounds[1] - mesh.bounds[0]
    assert extents[0] < extents[1] and extents[0] < extents[2]
