"""Smoke tests for the helpers in ``scripts/validation_optimize.py`` on the
stub ``toy_instance`` (no A1-A4 functions needed)."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

import plan_fixtures as pf
from gtcore.interact import find_overlapping_tiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def vo():
    path = os.path.join(ROOT, "scripts", "validation_optimize.py")
    spec = importlib.util.spec_from_file_location("validation_optimize", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def toy():
    return pf.toy_instance(n_candidates=25, n_targets=100, rng_seed=0)


# ------------------------------------------------------------ heuristics
def test_uniform_heuristic_is_feasible_and_spread(vo, toy):
    cand, conf = toy["candidates"], toy["conflicts"]
    sel = vo.uniform_heuristic(cand, conf, 4)
    assert sel.shape == (4,) and np.all(np.diff(sel) > 0)
    assert conf.is_feasible(sel)
    assert find_overlapping_tiles(cand.tiles_of(sel), threshold_mm=1.0) == []
    # FPS on a 5x5 grid of 10 mm pitch: the four picks are >= 20 mm apart
    a = cand.anchors[sel]
    d = np.linalg.norm(a[:, None] - a[None], axis=2)
    assert d[~np.eye(4, dtype=bool)].min() >= 20.0 - 1e-6
    # deterministic
    assert np.array_equal(sel, vo.uniform_heuristic(cand, conf, 4))


def test_uniform_heuristic_skips_when_infeasible(vo, toy):
    with pytest.raises(vo.Skip, match="uniform"):
        vo.uniform_heuristic(toy["candidates"], toy["conflicts"], 30)


def test_random_feasible_draws(vo, toy):
    cand, conf = toy["candidates"], toy["conflicts"]
    rng = np.random.default_rng(3)
    sels = [vo.random_feasible(cand, conf, 4, rng) for _ in range(5)]
    for s in sels:
        assert s.shape == (4,) and conf.is_feasible(s)
    assert len({tuple(s) for s in sels}) > 1
    with pytest.raises(vo.Skip):
        vo.random_feasible(cand, conf, 26, rng)


def test_farthest_point_order(vo):
    pts = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [5, 0, 0]], dtype=float)
    order = vo.farthest_point_order(pts)
    assert sorted(order.tolist()) == [0, 1, 2, 3]
    assert set(order[:2].tolist()) == {0, 2}          # the two extremes first
    assert vo.farthest_point_order(np.zeros((0, 3))).size == 0


# ------------------------------------------------------------ statistics
def test_paired_ci_and_wilcoxon(vo):
    a = np.array([0.80, 0.82, 0.85, 0.90, 0.88, 0.91])
    b = a - 0.03
    ci = vo.paired_ci(a, b)
    assert ci["n"] == 6 and ci["mean"] == pytest.approx(0.03)
    assert ci["sd"] == pytest.approx(0.0, abs=1e-12)
    assert ci["ci_lo"] == pytest.approx(0.03) and ci["ci_hi"] == pytest.approx(0.03)
    b2 = b + np.array([0.01, -0.01, 0.02, -0.02, 0.0, 0.01])
    ci2 = vo.paired_ci(a, b2)
    assert ci2["ci_lo"] < ci2["mean"] < ci2["ci_hi"]
    # scipy's t for n=6: half-width = 2.571 * sd / sqrt(6)
    assert ci2["ci_hi"] - ci2["mean"] == pytest.approx(2.5706 * ci2["sd"] / np.sqrt(6), rel=1e-3)
    assert vo.wilcoxon_p(a, a) == 1.0                         # all-zero differences
    assert vo.wilcoxon_p(a, b) < 0.05                          # consistent shift
    assert vo.paired_ci([np.nan], [1.0])["n"] == 0


def test_mean_sd_and_kendall(vo):
    m, sd, n = vo.mean_sd([1.0, 2.0, 3.0, None, float("nan")])
    assert (m, n) == (2.0, 3) and sd == pytest.approx(1.0)
    assert vo.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert vo.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(vo.kendall_tau([1.0], [2.0]))


# ---------------------------------------------------------- perturbations
def test_shift_seed_plane_moves_seeds_by_stated_amount(vo, toy):
    tiles = toy["candidates"].tiles_of([0, 7])
    for off in (2.25, 3.75):
        moved = vo.shift_seed_plane(tiles, off)
        for t0, t1 in zip(tiles, moved):
            delta = t1.seed_centers - t0.seed_centers
            assert np.allclose(np.linalg.norm(delta, axis=1), abs(off - 3.0))
            # along the inward normal, sign = sign(off - 3)
            assert np.allclose(delta @ t0.normal_ras, off - 3.0)
            assert np.allclose(t1.corners_ras, t0.corners_ras)     # footprint untouched
            assert np.allclose(t1.center_ras - t0.center_ras, (off - 3.0) * t0.normal_ras)
    same = vo.shift_seed_plane(tiles, 3.0)
    assert np.allclose(same[0].seed_centers, tiles[0].seed_centers)


# --------------------------------------------------------------- misc
def test_truth_tile_builders(vo):
    import gtcore.phantom.generate as gen
    from gtcore.segment.surface import mask_to_mesh
    vol, truth = gen.make_head_phantom(spacing=1.5, n_tiles=3, rng_seed=1)
    mesh = mask_to_mesh(truth.masks["cavity"], vol.affine)
    raw = vo.truth_tiles_raw(truth)
    conf = vo.truth_tiles_conformed(mesh, truth)
    assert len(raw) == len(conf) == 3
    for r, c, t in zip(raw, conf, truth.tiles):
        assert r.seed_centers.shape == (4, 3)
        assert np.allclose(r.seed_centers.mean(axis=0), t.center_ras)
        # the conformed tile lands near the truth tile (within a few mm)
        assert np.linalg.norm(c.center_ras - t.center_ras) < 4.0


def test_target_from_faces_and_tile_rule(vo):
    mesh = pf.flat_wall_mesh(size_mm=40.0, step_mm=2.0)
    top = pf.flat_wall_top_faces(mesh)
    t = vo.target_from_faces(mesh, top, 5.0)
    assert len(t) > 0 and np.all(t.points[:, 2] > 0)
    # the wall itself (offset 0) weighs exactly the top-face area; the +5 mm
    # shell is larger because edge vertices carry averaged (tilted) normals
    t0 = vo.target_from_faces(mesh, top, 0.0)
    assert t0.total_weight == pytest.approx(40.0 * 40.0, rel=1e-6)
    assert 40.0 ** 2 < t.total_weight < 50.0 ** 2
    assert vo.tile_count_rule(4178.0) == 11 and vo.tile_count_rule(400.0) == 1
    with pytest.raises(vo.Skip):
        vo.target_from_faces(mesh, np.zeros(len(mesh.faces), dtype=bool))


def test_visible_faces_local_isolates_inner_wall(vo):
    import trimesh
    outer = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    inner.invert()
    shell = trimesh.util.concatenate([outer, inner])
    vis = vo.visible_faces_local(shell, np.zeros(3))
    n_outer = len(outer.faces)
    assert vis[n_outer:].all() and not vis[:n_outer].any()


def test_md_table_and_csv(vo, tmp_path):
    rows = [{"a": 1, "b": 0.5, "c": float("nan")}, {"a": 2, "b": None, "c": "x"}]
    md = vo.md_table(rows, ["a", "b", "c"], {"b": "%.2f"})
    assert "| 1 | 0.50 |" in md and "—" in md
    p = tmp_path / "t.csv"
    vo.write_csv(str(p), rows)
    assert p.read_text(encoding="utf-8").splitlines()[0] == "a,b,c"


def test_parse_args_quick(vo):
    cfg = vo.parse_args(["--quick", "--section", "v8"])
    assert cfg.sections == ["v8"] and cfg.seeds == (1, 2) and cfg.n_list == (4, 8)
    assert cfg.n_random == 10 and cfg.scales == (1.0,)
    full = vo.parse_args([])
    assert full.sections == ["v2", "v3", "v4", "v5", "v6", "v7", "v8"]
    assert full.seeds == vo.SEEDS and full.n_random == vo.N_RANDOM
