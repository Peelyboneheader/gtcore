"""Stage 1 of docs/plan-localization (2026-10-08): plumbing and bug fixes.

* ``SeedCandidates.subset`` keeps every per-seed field (incl. ``cov_ras``);
* ``fit_tiles(..., "auto")`` forwards ``spacing_mm`` (was dropped, so the
  pipeline ran the cover pass at the 1 mm tolerance on coarse scans);
* the counted ``fit_tiles_prior`` path reports the pose of the bent-tile fit
  it draws, and placed tiles always face into the cavity.
"""
import numpy as np
import pytest

from gtcore.seeds.detect import SeedCandidates
from gtcore.tiles import fit_tiles_prior, to_placed_tiles, ImplantPrior
from gtcore.tiles.fit import fit_tiles
from gtcore.phantom.generate import make_head_phantom


def _cands(n=5, with_cov=True):
    rng = np.random.default_rng(0)
    cov = np.stack([np.eye(3) * (i + 1) for i in range(n)]) if with_cov else None
    return SeedCandidates(
        mask=np.zeros((2, 2, 2), bool),
        centers_ras=rng.normal(size=(n, 3)),
        axes_ras=np.tile([1.0, 0, 0], (n, 1)),
        volumes_mm3=np.arange(n, dtype=float),
        elongations=np.ones(n),
        cov_ras=cov,
        info={"status": np.array(["ok"] * n), "note": "run"},
    )


def test_subset_keeps_every_field_bool_and_index():
    c = _cands()
    for keep in (np.array([True, False, True, False, True]), np.array([0, 2, 4])):
        s = c.subset(keep)
        assert len(s) == 3
        assert np.allclose(s.centers_ras, c.centers_ras[[0, 2, 4]])
        assert np.allclose(s.volumes_mm3, [0, 2, 4])
        assert np.allclose(s.cov_ras[:, 0, 0], [1, 3, 5])
        assert list(s.info["status"]) == ["ok"] * 3 and s.info["note"] == "run"
        assert s.mask is c.mask


def test_subset_without_cov():
    s = _cands(with_cov=False).subset([1])
    assert s.cov_ras is None and len(s) == 1


@pytest.fixture(scope="module")
def phantom():
    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=3)
    c = np.array([s.center_ras for s in truth.seeds])
    a = np.array([s.axis_ras for s in truth.seeds])
    return c, a, truth


def test_auto_fit_receives_spacing(phantom):
    c, a, truth = phantom
    res = fit_tiles(c, a, "auto", cavity_center_ras=truth.cavity_center_ras,
                    spacing_mm=(0.5, 0.5, 2.0))
    assert res.spacing_tol == pytest.approx(2.0)
    res1 = fit_tiles(c, a, "auto", cavity_center_ras=truth.cavity_center_ras)
    assert res1.spacing_tol == pytest.approx(1.0)


def test_counted_prior_pose_matches_the_drawn_bent_tile(phantom):
    c, a, truth = phantom
    res = fit_tiles_prior(c, a, ImplantPrior(n_full=3),
                          cavity_center_ras=truth.cavity_center_ras)
    assert len(res.tiles) == 3
    cav = truth.cavity_center_ras
    for pose in res.tiles:
        assert pose.deform is not None
        d = abs(float(pose.normal_ras @ pose.deform.pose.normal))
        assert d > 1 - 1e-9
        # fit.py convention: pose normal points away from the cavity
        assert float(pose.normal_ras @ (pose.center_ras - cav)) > 0.0
        assert abs(float(pose.axis_ras @ pose.normal_ras)) < 1e-9
    placed = to_placed_tiles(res, c, a)
    for t, pose in zip(placed, res.tiles):
        # placed tiles face INTO the cavity
        assert float(t.normal_ras @ (cav - pose.center_ras)) > 0.0
