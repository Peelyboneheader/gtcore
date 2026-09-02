"""Dosimetry: TG-43U1 formalism for Cs-131 GammaTile seeds.

``DoseInterpolator`` is the verbatim v1 port (regression reference, frozen);
``TG43Engine`` / ``compute_dose_grid`` / ``isodose_surfaces`` are the
corrected, vectorized v2 engine (see docs/tg43-port-notes.md).
"""
from .engine import TG43Engine, compute_dose_grid, isodose_surfaces
from .tg43 import DoseInterpolator

__all__ = [
    "DoseInterpolator",
    "TG43Engine",
    "compute_dose_grid",
    "isodose_surfaces",
]
