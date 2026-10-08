"""With-truth hook of scripts/validation_realdata_proxies.py: a truth seed
list in an arbitrary rigid frame is registered onto the detections (rigid
ICP, Hungarian correspondences) and the per-seed errors come back at the
noise level, globally and per tile; misses and false positives are tolerated.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def proxies():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                        "validation_realdata_proxies.py")
    spec = importlib.util.spec_from_file_location("validation_realdata_proxies",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rotation(rng):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _layout(rng, n_tiles=8):
    """n_tiles 10 mm seed squares scattered on a 25 mm sphere."""
    pts, tid = [], []
    for t in range(n_tiles):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        e1 = np.cross(d, [0.3, 0.5, 0.8])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        c = 25.0 * d
        for a in (-5.0, 5.0):
            for b in (-5.0, 5.0):
                pts.append(c + a * e1 + b * e2)
                tid.append(t)
    return np.array(pts), np.array(tid)


def test_truth_registration_recovers_noise_level(proxies, tmp_path):
    rng = np.random.default_rng(4)
    det, tid = _layout(rng)
    noisy = det + rng.normal(0.0, 0.2, det.shape)
    # truth in another rigid frame, one seed missing from detection and two
    # false positives among the detections
    R, t = _rotation(rng), rng.uniform(-50, 50, 3)
    truth = (det - t) @ R            # inverse transform: det = R truth + t
    detections = np.vstack([np.delete(noisy, 5, axis=0),
                            [[60.0, 0.0, 0.0], [0.0, -70.0, 10.0]]])

    csv_path = tmp_path / "truth.csv"
    with open(csv_path, "w") as fh:
        fh.write("x,y,z,tile_id,comment\n")
        for p, k in zip(truth, tid):
            fh.write("%.4f,%.4f,%.4f,%d,ok\n" % (p[0], p[1], p[2], k))
    tpts, ttiles = proxies.read_truth(str(csv_path))
    assert tpts.shape == (32, 3) and ttiles is not None

    out = proxies.truth_report(tpts, ttiles, detections, {})
    assert out["n_match"] == 31
    g = out["global_fit"]
    assert g["n"] == 31 and g["mean_3d"] < 0.5 and g["max_3d"] < 1.0
    pt = out["per_tile_fit"]
    assert out["n_groups"] == 8 and pt["n"] == 31
    # a per-tile fit absorbs part of the noise: never worse than global
    assert pt["mean_3d"] <= g["mean_3d"] + 1e-9


def test_truth_csv_needs_xyz(proxies, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        proxies.read_truth(str(bad))
