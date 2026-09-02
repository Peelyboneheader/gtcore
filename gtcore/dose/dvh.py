"""Shell dose-volume statistics for the planner's on-screen dose panel.

GammaTile's clinical target is the resection-cavity wall plus a few mm of
surrounding brain, so the natural "structures" to score are offset shells of
the cavity surface: the wall itself (offset 0) and copies pushed a fixed
distance OUTWARD into tissue.  Each shell is sampled at the mesh vertices
(trilinear from the dose grid; points outside the grid read 0 cGy), and the
usual DVH quantities are reported over those samples.

Pure numpy/trimesh: no rendering, unit-testable with synthetic grids.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..volume import Volume

DEFAULT_SHELL_OFFSETS_MM = (0.0, 5.0, 10.0)
DEFAULT_CURVE_FRACTIONS = tuple(np.linspace(0.0, 3.0, 61))  # 0..300% rx


def outward_normals(mesh):
    """Unit vertex normals oriented AWAY from the mesh centroid.

    Marching-cubes cavity meshes carry outward normals already, but the
    planner must not depend on that: a shell pushed the wrong way would
    silently score the cavity air instead of brain tissue.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float).copy()
    centroid = verts.mean(axis=0)
    flip = np.einsum("ij,ij->i", normals, verts - centroid) < 0.0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1)
    norms[norms < 1e-12] = 1.0
    return normals / norms[:, None]


def shell_points(mesh, offset_mm: float):
    """Mesh vertices pushed ``offset_mm`` outward (into tissue) along normals."""
    verts = np.asarray(mesh.vertices, dtype=float)
    if abs(float(offset_mm)) < 1e-12:
        return verts.copy()
    return verts + float(offset_mm) * outward_normals(mesh)


def sample_doses(dose_volume: Volume, points):
    """Trilinear dose [cGy] at RAS ``points``; 0 outside the grid."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros(0)
    return np.asarray(dose_volume.sample_ras(pts, order=1, fill=0.0),
                      dtype=float).reshape(-1)


def dvh_curve(doses, levels_cgy):
    """Cumulative DVH: fraction of samples receiving >= each level."""
    d = np.asarray(doses, dtype=float).reshape(-1)
    levels = np.asarray(levels_cgy, dtype=float).reshape(-1)
    if d.size == 0:
        return np.zeros(levels.shape)
    d_sorted = np.sort(d)
    # number of samples >= level = n - (index of first sample >= level)
    idx = np.searchsorted(d_sorted, levels, side="left")
    return (d_sorted.size - idx) / float(d_sorted.size)


def dvh_stats(doses, rx_cgy: float) -> Dict[str, float]:
    """D90/D50/Dmin/Dmax/Dmean [cGy] and V100/V150/V200 [fraction of samples].

    ``Dxx`` is the dose received by at least xx% of the samples (the
    (100-xx)th percentile); ``Vyyy`` is the fraction receiving at least yyy%
    of the prescription.  Empty input yields all-zero stats.
    """
    d = np.asarray(doses, dtype=float).reshape(-1)
    rx = float(rx_cgy)
    if d.size == 0:
        return {k: 0.0 for k in ("D90", "D50", "Dmin", "Dmax", "Dmean",
                                 "V100", "V150", "V200", "n")}
    v100, v150, v200 = dvh_curve(d, [1.0 * rx, 1.5 * rx, 2.0 * rx])
    return {
        "D90": float(np.percentile(d, 10.0)),
        "D50": float(np.percentile(d, 50.0)),
        "Dmin": float(d.min()),
        "Dmax": float(d.max()),
        "Dmean": float(d.mean()),
        "V100": float(v100),
        "V150": float(v150),
        "V200": float(v200),
        "n": float(d.size),
    }


def shell_report(dose_volume: Volume, mesh, rx_cgy: float,
                 offsets_mm: Sequence[float] = DEFAULT_SHELL_OFFSETS_MM,
                 curve_fractions: Sequence[float] = DEFAULT_CURVE_FRACTIONS):
    """DVH stats + cumulative curve for each cavity shell.

    Returns ``{offset_mm: {"stats": dvh_stats(...), "curve_x": fractions of
    rx, "curve_y": fraction of shell >= that dose}}`` in the order given.
    """
    rx = float(rx_cgy)
    fracs = np.asarray(curve_fractions, dtype=float).reshape(-1)
    out: Dict[float, Dict] = {}
    for off in offsets_mm:
        doses = sample_doses(dose_volume, shell_points(mesh, off))
        out[float(off)] = {
            "stats": dvh_stats(doses, rx),
            "curve_x": fracs.copy(),
            "curve_y": dvh_curve(doses, fracs * rx),
        }
    return out


def format_report(report: Dict[float, Dict], rx_cgy: float) -> str:
    """Fixed-width text table for the planner's dose panel."""
    lines = ["shell     D90    D50   Dmin   V100  V150"]
    for off, entry in report.items():
        s = entry["stats"]
        name = "wall" if abs(off) < 1e-9 else "+%g mm" % off
        lines.append("%-7s %6.0f %6.0f %6.0f  %4.0f%% %4.0f%%" % (
            name, s["D90"], s["D50"], s["Dmin"],
            100.0 * s["V100"], 100.0 * s["V150"]))
    return "\n".join(lines)
