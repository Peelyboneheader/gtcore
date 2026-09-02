"""End-to-end pipeline on the ground-truth phantom.

phantom CT -> seed detection -> metal inpainting -> head segmentation
-> cavity segmentation (seed-cloud prior) -> surface meshes,
with quantitative checks against the phantom's exact ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from gtcore.phantom import make_head_phantom
from gtcore.preprocess import inpaint_metal
from gtcore.seeds import detect_seed_candidates
from gtcore.segment import mask_to_mesh, segment_cavity, segment_head


@pytest.fixture(scope="module")
def case():
    vol, truth = make_head_phantom(spacing=0.8, n_tiles=3, rng_seed=2)
    return vol, truth


def _dice(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    inter = np.logical_and(a, b).sum()
    return 2.0 * inter / max(a.sum() + b.sum(), 1)


def _greedy_match_errors(truth_pts, est_pts):
    """Per-truth-point distance to its greedily matched estimate (mm)."""
    from scipy.spatial.distance import cdist

    D = cdist(truth_pts, est_pts)
    used = set()
    errs = []
    for ti in range(len(truth_pts)):
        for j in np.argsort(D[ti]):
            if int(j) not in used:
                used.add(int(j))
                errs.append(D[ti, int(j)])
                break
    return np.array(errs)


def test_seed_detection_against_truth(case):
    vol, truth = case
    cands = detect_seed_candidates(vol)
    assert len(cands) == len(truth.seeds) == 12

    t_centers = np.array([s.center_ras for s in truth.seeds])
    errs = _greedy_match_errors(t_centers, cands.centers_ras)
    assert errs.mean() < 0.7, "mean seed centroid error %.2f mm" % errs.mean()
    assert errs.max() < 1.5, "max seed centroid error %.2f mm" % errs.max()


def test_full_pipeline(case):
    vol, truth = case
    cands = detect_seed_candidates(vol)

    clean = inpaint_metal(vol, cands.mask)
    assert clean.array[cands.mask].max() < 500.0, "metal bloom not removed"

    masks = segment_head(clean, metal_mask=cands.mask)
    assert _dice(masks["brain"], truth.masks["brain"]) > 0.75

    cavity = segment_cavity(
        clean, masks["cranial_interior"], masks["brain"], cands.centers_ras
    )
    d = _dice(cavity, truth.masks["cavity"])
    assert d > 0.6, "cavity dice %.3f" % d

    mesh = mask_to_mesh(cavity, vol.affine)
    assert len(mesh.vertices) > 200
    # every truth seed should sit close to the reconstructed cavity wall
    t_centers = np.array([s.center_ras for s in truth.seeds])
    closest = mesh.nearest.on_surface(t_centers)[1]
    assert np.mean(closest) < 4.0, "seeds far from cavity wall: %.2f mm" % np.mean(closest)
