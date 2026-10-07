"""Tests for ``gtcore.plan.final_report`` (section 3 G, branch plan/validation)."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

import plan_fixtures as pf
from gtcore.dose.engine import dose_at_points
from gtcore.interact import conform_tile, snap_to_wall
from gtcore.plan import DEFAULT_RX_CGY, OptimizeReport, TargetSet, final_report
from gtcore.plan.report import grid_bounds, weighted_quantile_fallback, weighted_stats

GRID_MM = 2.0
RX = 3000.0   # two tiles 24 mm apart do not reach 6000 cGy at +5 mm; halve rx so V100 > 0
MARGIN_MM = 12.0


@pytest.fixture(scope="module")
def flat_two_tiles():
    mesh = pf.flat_wall_mesh(size_mm=60.0, step_mm=2.0, depth_mm=30.0)
    tiles = []
    for x in (-12.0, 12.0):
        surf, n_in = snap_to_wall(mesh, np.array([x, 0.0, 0.0]))
        tiles.append(conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0])))
    return mesh, tiles


@pytest.fixture(scope="module")
def report(flat_two_tiles):
    mesh, tiles = flat_two_tiles
    return final_report(mesh, tiles, rx_cgy=RX, grid_mm=GRID_MM,
                        margin_mm=MARGIN_MM, parameters={"seed": 7, "h_mm": 2.5})


# ------------------------------------------------------------------ helpers
def test_weighted_quantile_matches_numpy_for_unit_weights():
    rng = np.random.default_rng(0)
    v = rng.normal(size=2001)
    for q in (0.1, 0.5, 0.9):
        assert abs(weighted_quantile_fallback(v, np.ones_like(v), q)
                   - np.percentile(v, 100 * q)) < 0.01


def test_weighted_quantile_respects_weights():
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 100.0])
    assert weighted_quantile_fallback(v, w, 0.5) == pytest.approx(3.0, abs=0.05)
    assert weighted_quantile_fallback([], [], 0.5) == 0.0


def test_weighted_stats_fractions():
    d = np.array([5000.0, 7000.0, 9500.0, 13000.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    st = weighted_stats(d, w, 6000.0)
    assert st["V100"] == pytest.approx(0.75)
    assert st["V150"] == pytest.approx(0.5)
    assert st["V200"] == pytest.approx(0.25)
    assert st["Dmean"] == pytest.approx(d.mean())
    assert st["total_weight"] == 4.0
    w2 = np.array([10.0, 1.0, 1.0, 1.0])
    assert weighted_stats(d, w2, 6000.0)["V100"] == pytest.approx(3.0 / 13.0)


def test_grid_bounds_clips_margin():
    b = np.array([[0.0, 0.0, 0.0], [40.0, 40.0, 40.0]])
    bounds, used = grid_bounds(b, 50.0, 1.0, max_voxels=1_000_000)
    assert used < 50.0
    n = np.prod(np.floor((bounds[1] - bounds[0]) + 1e-9).astype(int) + 1)
    assert n <= 1_000_000
    bounds2, used2 = grid_bounds(b, 10.0, 1.0)
    assert used2 == 10.0 and np.allclose(bounds2, [[-10] * 3, [50] * 3])


# ------------------------------------------------------------------- report
def test_report_fields_populated(report, flat_two_tiles):
    mesh, tiles = flat_two_tiles
    assert isinstance(report, OptimizeReport)
    assert len(report.tiles) == 2
    assert report.overlaps == []
    assert report.gtcore_version
    assert report.seed == 7
    assert report.wall_clock_s > 0
    assert set(report.metrics_grid) >= {0.0, 5.0, 10.0, "target"}
    for off in (0.0, 5.0, 10.0):
        st = report.metrics_grid[off]
        for key in ("V100", "D90", "V150", "V200"):
            assert key in st
        assert 0.0 <= st["V100"] <= 1.0
        assert st["curve_x"].shape == st["curve_y"].shape
    assert 0.0 <= report.metrics_grid[5.0]["V100"] <= 1.0
    assert report.metrics_grid[5.0]["V100"] > 0.0              # seeds deliver dose
    assert report.metrics_grid["target"]["name"] == "shell+5mm"
    assert report.parameters["grid_mm"] == GRID_MM
    assert report.parameters["grid_margin_used_mm"] == MARGIN_MM
    assert report.parameters["n_seeds"] == 8
    assert report.parameters["h_mm"] == 2.5
    assert len(report.parameters["grid_shape"]) == 3
    assert "dose_grid" in report.runtime and "total" in report.runtime
    assert report.metrics_grid_interference is None
    assert isinstance(report.shadowing, list)
    assert "rind" not in report.metrics_grid           # no cavity mask given
    assert "summary" in dir(report) and "shell 5.0" in report.summary()


def test_report_json_and_csv_write(report, tmp_path):
    jp = tmp_path / "r.json"
    cp = tmp_path / "r.csv"
    report.to_json(jp)
    report.to_csv(cp)
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["n_tiles"] == 2
    assert "5.0" in data["metrics_grid"]
    assert 0.0 <= data["metrics_grid"]["5.0"]["V100"] <= 1.0
    assert data["parameters"]["grid_mm"] == GRID_MM
    with open(cp, newline="", encoding="utf-8") as fh:
        keys = {row["key"] for row in csv.DictReader(fh)}
    assert "metrics_grid.5.0.V100" in keys
    assert "metrics_grid.target.V100" in keys


def test_full_target_grid_vs_point_consistency(report, flat_two_tiles):
    """Grid-sampled weighted V100 on the full target agrees with the direct
    exact ``dose_at_points`` evaluation on the same points within 1 pp."""
    mesh, tiles = flat_two_tiles
    target = TargetSet.from_shell(mesh, 5.0)
    centers = np.vstack([t.seed_centers for t in tiles])
    axes = np.vstack([t.seed_axes for t in tiles])
    direct = dose_at_points(centers, axes, target.points, exact=True)
    ref = weighted_stats(direct, target.weights, RX)
    got = report.metrics_grid["target"]
    assert got["n"] == len(target)
    assert abs(got["V100"] - ref["V100"]) <= 0.01
    assert abs(got["D90"] - ref["D90"]) / RX < 0.03


def test_report_sk_scaling_and_interference(flat_two_tiles):
    mesh, tiles = flat_two_tiles
    base = final_report(mesh, tiles, grid_mm=GRID_MM, margin_mm=MARGIN_MM,
                        parameters={"shadowing": False})
    up = final_report(mesh, tiles, grid_mm=GRID_MM, margin_mm=MARGIN_MM,
                      parameters={"sk_per_seed_u": base.parameters["sk_per_seed_u"] * 1.05,
                                  "shadowing": False})
    assert up.metrics_grid[5.0]["D90"] == pytest.approx(
        1.05 * base.metrics_grid[5.0]["D90"], rel=1e-6)
    assert up.metrics_grid[5.0]["V100"] >= base.metrics_grid[5.0]["V100"]

    interf = final_report(mesh, tiles, grid_mm=GRID_MM, margin_mm=MARGIN_MM,
                          interference=True, parameters={"shadowing": False})
    assert interf.metrics_grid_interference is not None
    assert 5.0 in interf.metrics_grid_interference
    # capsules only attenuate: never more dose with the model on
    assert (interf.metrics_grid_interference[5.0]["Dmean"]
            <= interf.metrics_grid[5.0]["Dmean"] + 1e-9)
    assert any("interference" in n for n in interf.notes)


def test_report_rind_with_cavity_mask(flat_two_tiles):
    mesh, tiles = flat_two_tiles
    # a coarse cavity mask: the box interior (z < 0) on a 2 mm lattice
    sp = 2.0
    lo = np.array([-30.0, -30.0, -30.0])
    n = 31
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = sp
    affine[:3, 3] = lo
    zz = lo[2] + sp * np.arange(n)
    mask = np.zeros((n, n, n), dtype=bool)
    mask[zz < 0.0, :, :] = True
    rep = final_report(mesh, tiles, grid_mm=GRID_MM, margin_mm=MARGIN_MM,
                       cavity_mask=mask, cavity_affine=affine,
                       parameters={"shadowing": False})
    assert "rind" in rep.metrics_grid
    r = rep.metrics_grid["rind"]
    assert r["volume_cc"] > 0 and 0.0 <= r["V100"] <= 1.0


def test_report_empty_tiles_and_bad_mesh(flat_two_tiles):
    mesh, _ = flat_two_tiles
    rep = final_report(mesh, [], grid_mm=4.0, margin_mm=5.0)
    assert rep.metrics_grid[5.0]["V100"] == 0.0
    assert any("no tiles" in n for n in rep.notes)
    import trimesh
    with pytest.raises(ValueError, match="mesh is empty"):
        final_report(trimesh.Trimesh(), [], grid_mm=2.0)
