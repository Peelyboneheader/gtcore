"""Dosimetry: TG-43U1 formalism for Cs-131 GammaTile seeds.

``DoseInterpolator`` is the verbatim v1 port (regression reference, frozen);
``TG43Engine`` / ``compute_dose_grid`` / ``dose_at_points`` /
``isodose_surfaces`` are the corrected, vectorized v2 engine (see
docs/tg43-port-notes.md); ``metrics`` adds DVH, cavity-rind and wall-at-depth
coverage statistics for the planner, and ``dvh`` the cheap cavity-shell
(wall / +5 / +10 mm) statistics behind the planner's on-screen dose panel.

``InterferenceModel`` adds the effect TG-43 superposition cannot express:
seeds and collagen tile carriers attenuating each other's primary fluence
(see docs/interference-notes.md).  It is opt-in -- pass one as
``compute_dose_grid(..., interference=model)`` -- so the bare formalism stays
the default and stays regression-pinned.
"""
from .dvh import dvh_curve, dvh_stats, format_report, shell_report
from .engine import TG43Engine, compute_dose_grid, dose_at_points, \
    isodose_surfaces
from .interference import (
    InterferenceModel,
    SeedCapsule,
    TileCarrier,
    interference_report,
)
from .metrics import DVH, dose_metrics, dvh, resample_mask_to, rind_mask, \
    surface_coverage, wall_dose
from .tg43 import DoseInterpolator

__all__ = [
    "DoseInterpolator",
    "TG43Engine",
    "compute_dose_grid",
    "dose_at_points",
    "isodose_surfaces",
    "InterferenceModel",
    "SeedCapsule",
    "TileCarrier",
    "interference_report",
    "DVH",
    "dvh",
    "dose_metrics",
    "resample_mask_to",
    "rind_mask",
    "wall_dose",
    "surface_coverage",
    "dvh_curve",
    "dvh_stats",
    "format_report",
    "shell_report",
]
