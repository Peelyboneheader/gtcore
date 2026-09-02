"""Round-trip tests for gtcore.io: RAS <-> LPS must survive both directions."""
from __future__ import annotations

import numpy as np
import pytest

from gtcore import Volume
from gtcore.io import (
    load_volume,
    save_volume,
    sitk_from_volume,
    volume_from_sitk,
)


def _nontrivial_volume():
    """Anisotropic spacing, axis-flipping direction cosines, offset origin."""
    rng = np.random.default_rng(20240901)
    array = rng.normal(size=(7, 9, 11)).astype(np.float32) * 100.0

    spacing = np.array([0.6, 0.9, 2.0])
    # Non-identity, right-handed-ish direction with two axis flips and a swap.
    direction = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = [-37.5, 12.25, 88.125]
    return Volume(array, affine, {"note": "synthetic"})


def test_sitk_roundtrip_preserves_affine_and_array():
    vol = _nontrivial_volume()
    back = volume_from_sitk(sitk_from_volume(vol))

    assert back.array.shape == vol.array.shape
    np.testing.assert_allclose(back.array, vol.array, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(back.affine, vol.affine, rtol=1e-9, atol=1e-9)


def test_sitk_roundtrip_preserves_geometry_accessors():
    vol = _nontrivial_volume()
    back = volume_from_sitk(sitk_from_volume(vol))

    np.testing.assert_allclose(back.spacing, vol.spacing, atol=1e-9)
    np.testing.assert_allclose(back.direction, vol.direction, atol=1e-9)
    np.testing.assert_allclose(back.origin_ras, vol.origin_ras, atol=1e-9)
    assert back.shape_ijk == vol.shape_ijk


def test_sitk_image_is_lps():
    """The RAS origin must appear negated in x/y on the SimpleITK (LPS) side."""
    vol = _nontrivial_volume()
    img = sitk_from_volume(vol)
    origin_lps = np.array(img.GetOrigin())
    expected = vol.origin_ras * np.array([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(origin_lps, expected, atol=1e-9)


@pytest.mark.parametrize("suffix", [".nrrd", ".nii.gz"])
def test_save_load_roundtrip(tmp_path, suffix):
    vol = _nontrivial_volume()
    path = tmp_path / ("roundtrip" + suffix)

    save_volume(vol, str(path))
    assert path.exists()

    back = load_volume(str(path))
    np.testing.assert_allclose(back.array, vol.array, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(back.affine, vol.affine, rtol=1e-5, atol=1e-5)
    assert back.meta["source"] == str(path)


def test_load_volume_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_volume(str(tmp_path / "does_not_exist.nrrd"))
