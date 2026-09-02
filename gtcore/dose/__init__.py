"""Dosimetry: TG-43U1 formalism for Cs-131 GammaTile seeds.

``DoseInterpolator`` is the verbatim v1 port (regression reference, frozen);
``TG43Engine`` / ``compute_dose_grid`` / ``isodose_surfaces`` are the
corrected, vectorized v2 engine (see docs/tg43-port-notes.md).

``InterferenceModel`` adds the effect TG-43 superposition cannot express:
seeds and collagen tile carriers attenuating each other's primary fluence
(see docs/interference-notes.md).  It is opt-in -- pass one as
``compute_dose_grid(..., interference=model)`` -- so the bare formalism stays
the default and stays regression-pinned.
"""
from .engine import TG43Engine, compute_dose_grid, isodose_surfaces
from .interference import (
    InterferenceModel,
    SeedCapsule,
    TileCarrier,
    interference_report,
)
from .tg43 import DoseInterpolator

__all__ = [
    "DoseInterpolator",
    "TG43Engine",
    "compute_dose_grid",
    "isodose_surfaces",
    "InterferenceModel",
    "SeedCapsule",
    "TileCarrier",
    "interference_report",
]
