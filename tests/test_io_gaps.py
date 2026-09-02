"""Loader behaviour for DICOM series with missing / irregular slices.

SimpleITK spreads N slices uniformly across the scanned extent; with missing
slices that distorts z coordinates by up to the largest gap. The loader must
instead place each present slice at its true ImagePositionPatient and
interpolate the gaps (seen in practice with partially synced OneDrive
exports of the phantom and post-op scans).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from gtcore.io import load_dicom_series


def _write_slice(directory, z, value, series_uid, instance):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = series_uid  # good enough for a synthetic series
    ds.FrameOfReferenceUID = series_uid
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [-10.0, -20.0, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    ds.Rows = 16
    ds.Columns = 16
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1000.0
    arr = np.full((16, 16), int(value), dtype=np.int16)
    ds.PixelData = arr.tobytes()
    path = os.path.join(directory, "ct_%03d.dcm" % instance)
    ds.save_as(path, enforce_file_format=True)
    return path


def _make_series(directory, zs):
    from pydicom.uid import generate_uid

    uid = generate_uid()
    for i, z in enumerate(zs):
        # slice value encodes z so interpolation is checkable: HU = 10*z
        _write_slice(directory, z, 1000 + 10 * z, uid, i + 1)


def test_uniform_series_untouched(tmp_path):
    _make_series(str(tmp_path), [0, 1, 2, 3, 4])
    vol = load_dicom_series(str(tmp_path))
    assert vol.array.shape[0] == 5
    assert not vol.meta.get("z_gap_interpolated", False)
    assert np.isclose(vol.spacing[2], 1.0)


def test_gapped_series_rebuilt_on_true_grid(tmp_path):
    # dz = 1 mm with slices 3 and 6 missing over 0..8
    zs = [0, 1, 2, 4, 5, 7, 8]
    _make_series(str(tmp_path), zs)
    with pytest.warns(UserWarning, match="non-uniform slice positions"):
        vol = load_dicom_series(str(tmp_path))

    assert vol.meta["z_gap_interpolated"] is True
    assert vol.meta["slices_present"] == 7
    assert vol.array.shape[0] == 9                       # full 0..8 grid
    assert np.isclose(vol.spacing[2], 1.0, atol=0.01)    # true dz, not extent/N

    # z coordinate of voxel (0,0,k) must be the TRUE position k*1mm
    for k in (0, 3, 6, 8):
        ras = vol.index_to_ras([0.0, 0.0, float(k)])
        assert np.isclose(ras[2], k * 1.0, atol=0.05), "slice %d at z=%.2f" % (k, ras[2])

    # slice values are HU = 10*z: present slices exact, gaps interpolated
    for k in range(9):
        expected = 10.0 * k
        got = float(vol.array[k].mean())
        assert abs(got - expected) < 0.5, "slice %d: %.2f vs %.2f" % (k, got, expected)


def test_in_plane_geometry_preserved_when_rebuilt(tmp_path):
    _make_series(str(tmp_path), [0, 1, 2, 4, 5])
    with pytest.warns(UserWarning):
        vol = load_dicom_series(str(tmp_path))
    # LPS IPP (-10, -20, z) -> RAS (+10, +20, z)
    ras0 = vol.index_to_ras([0.0, 0.0, 0.0])
    assert np.allclose(ras0, [10.0, 20.0, 0.0], atol=0.01)
    assert np.allclose(vol.spacing[:2], [0.5, 0.5], atol=0.001)
