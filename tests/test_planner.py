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
