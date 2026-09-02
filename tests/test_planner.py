"""Smoke test for the interactive planner's off-screen snapshot path.

Runs the full pipeline on a coarse phantom, drops two tiles programmatically
(one full, one half), attempts a dose update (a friendly on-screen message if
the dose engine is not on board yet), and renders to a PNG.  A second test
drives the same path with inter-seed attenuation switched on.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pyvista")

from gtcore.phantom import make_head_phantom
from gtcore.pipeline import reconstruct
from gtcore.planner import snapshot_planner


def test_snapshot_planner_smoke():
    vol, _truth = make_head_phantom(spacing=1.0)
    result = reconstruct(vol, verbose=False)
    if "cavity" not in result.meshes:
        pytest.skip("pipeline found no cavity on this phantom")
    mesh = result.meshes["cavity"]

    # two well-separated cavity wall points
    verts = np.asarray(mesh.vertices)
    p1 = verts[0]
    p2 = verts[int(np.argmax(np.linalg.norm(verts - p1, axis=1)))]

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "planner_smoke.png")
    if os.path.exists(out):
        os.remove(out)

    actions = [p1, {"point": p2, "kind": "half"}, "update"]
    try:
        snapshot_planner(result, actions, out)
    except Exception as exc:  # headless CI without OpenGL etc.
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))

    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000, "snapshot PNG is suspiciously small"


def test_snapshot_planner_with_interference(monkeypatch):
    """The 'i' toggle must actually reach the dose engine, and still render.

    The planner is the only place a clinician meets this correction, so the
    wiring gets a test of its own.  ``compute_dose_grid`` is wrapped to
    record what it was handed: a model must arrive, holding one capsule per
    seed on the board and -- because the carrier density is unmeasured --
    no carriers at all.
    """
    import gtcore.dose as dose_mod

    vol, _truth = make_head_phantom(spacing=1.0)
    result = reconstruct(vol, verbose=False)
    if "cavity" not in result.meshes:
        pytest.skip("pipeline found no cavity on this phantom")
    verts = np.asarray(result.meshes["cavity"].vertices)

    seen = {}
    real = dose_mod.compute_dose_grid

    def spy(centers, axes, bounds, **kw):
        seen["model"] = kw.get("interference")
        seen["n_seeds"] = len(centers)
        return real(centers, axes, bounds, **kw)

    monkeypatch.setattr(dose_mod, "compute_dose_grid", spy)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "planner_interference.png")
    if os.path.exists(out):
        os.remove(out)

    actions = [{"interference": True}, verts[0], "update"]
    try:
        snapshot_planner(result, actions, out)
    except Exception as exc:  # headless CI without OpenGL etc.
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))

    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

    model = seen.get("model")
    assert model is not None, "toggle did not reach the dose engine"
    assert len(model.capsules) == seen["n_seeds"]
    assert model.carriers == []


def test_planner_toggle_flips_the_flag():
    """The keypress handler itself, with no rendering involved."""
    from gtcore.planner import _PlannerApp

    app = _PlannerApp.__new__(_PlannerApp)   # no window, no pipeline
    app.interference = False
    app._update_status = lambda *a, **k: None
    app._toggle_interference()
    assert app.interference is True
    app._toggle_interference()
    assert app.interference is False


def _stacked_tiles():
    """Two tiles where the second sits between the first and its target."""
    from gtcore.interact import PlacedTile

    def make(z):
        seeds = np.array([[dx, dy, z] for dx in (-5.0, 5.0)
                          for dy in (-5.0, 5.0)])
        return PlacedTile(
            kind="full", center_ras=seeds.mean(axis=0),
            normal_ras=[0.0, 0.0, 1.0], axis_ras=[1.0, 0.0, 0.0],
            seed_centers=seeds,
            seed_axes=np.tile([1.0, 0.0, 0.0], (4, 1)),
            corners_ras=np.array([[-10.0, -10.0, z], [10.0, -10.0, z],
                                  [10.0, 10.0, z], [-10.0, 10.0, z]]),
        )

    return [make(0.0), make(-4.0)]


def _shadow_app(tiles=None, **overrides):
    """A bare planner carrying only the state `_refresh_shadowing` reads.

    Built with __new__ so no window, pipeline or GPU is needed. Every field
    the sweep and its guards consult is set here in one place, so a planner
    refactor that adds guard state fails with a clear name rather than
    scattering AttributeErrors across several tests.
    """
    from gtcore.planner import _PlannerApp

    app = _PlannerApp.__new__(_PlannerApp)
    app.tiles = _stacked_tiles() if tiles is None else tiles
    app._drag_idx = -1        # not dragging
    app._adopting = False     # not in startup adoption
    app._shadow_pairs = []
    for key, value in overrides.items():
        setattr(app, key, value)
    return app


def test_planner_flags_dosimetric_shadowing():
    """Tiles standing in each other's line of fire get flagged."""
    app = _shadow_app()
    app._refresh_shadowing()

    assert len(app._shadow_pairs) == 1
    i, j, pct = app._shadow_pairs[0]
    assert (i, j) == (0, 1)
    assert pct > 2.0


def test_planner_shadowing_reaches_the_status_text():
    """The flag is worthless unless it reaches the line a surgeon reads.

    Exercises the real `_update_status` on a real off-screen planner rather
    than a hand-built stub, so it stays honest about the merged status
    layout (header, WARNING lines, then NOTE lines).
    """
    vol, _truth = make_head_phantom(spacing=1.0)
    result = reconstruct(vol, verbose=False)
    if "cavity" not in result.meshes:
        pytest.skip("pipeline found no cavity on this phantom")

    from gtcore.planner import _PlannerApp

    try:
        app = _PlannerApp(result, off_screen=True)
    except Exception as exc:  # headless CI without OpenGL etc.
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    try:
        app.tiles = _stacked_tiles()
        app._overlap_pairs = [(0, 1)]
        app._shadow_pairs = [(0, 1, 5.7)]
        app._update_status()
        # Both checks report, and the overlap CAUTION (overlap is allowed --
        # collagen stacks -- but dosimetrically risky) stays ahead of the
        # shadowing NOTE -- other planner tests substring-match that order.
        assert "CAUTION: tiles 1 & 2 overlap" in app._last_status
        assert "shadow each other" in app._last_status
        assert "tiles 1 & 2 shadow" in app._last_status
        assert (app._last_status.index("CAUTION: tiles 1 & 2 overlap")
                < app._last_status.index("tiles 1 & 2 shadow"))
    finally:
        app.close()


def test_planner_shadowing_is_skipped_mid_drag():
    """The drag path budgets tens of milliseconds; the sweep costs more."""
    app = _shadow_app(_drag_idx=0, _shadow_pairs=[(0, 1, 9.9)])
    app._refresh_shadowing()
    assert app._shadow_pairs == []


def test_planner_shadowing_is_skipped_during_startup_adoption():
    """Adopting the fitted implant must not add half a second to launch.

    At startup the user has not asked a question yet, so the sweep waits for
    the first real edit rather than spending launch latency on an answer
    nobody requested.
    """
    app = _shadow_app(_adopting=True, _shadow_pairs=[(0, 1, 9.9)])
    app._refresh_shadowing()
    assert app._shadow_pairs == []


def test_planner_shadowing_survives_a_broken_check():
    """Advisory feature: a failure must never take the planner down."""
    import gtcore.dose.interference as mod

    app = _shadow_app()
    real = mod.find_shadowing_tiles
    mod.find_shadowing_tiles = lambda *a, **k: 1 / 0
    try:
        app._refresh_shadowing()
    finally:
        mod.find_shadowing_tiles = real
    assert app._shadow_pairs == []
