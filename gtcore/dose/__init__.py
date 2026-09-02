"""Dosimetry: TG-43U1 formalism for Cs-131 GammaTile seeds.

``DoseInterpolator`` is the verbatim v1 port (regression reference, frozen);
``TG43Engine`` / ``compute_dose_grid`` / ``dose_at_points`` /
``isodose_surfaces`` are the corrected, vectorized v2 engine (see
docs/tg43-port-notes.md); ``metrics`` adds DVH, cavity-rind and wall-at-depth
coverage statistics for the planner.
"""
from .engine import CLRP_V2, DATASETS, SeedDataset, TG43Engine, \
    TG43U1S2_CONSENSUS, compute_dose_grid, dose_at_points, isodose_surfaces
from .metrics import DVH, dose_metrics, dvh, resample_mask_to, rind_mask, \
    surface_coverage, wall_dose
from .tg43 import DoseInterpolator

__all__ = [
    "DoseInterpolator",
    "SeedDataset",
    "DATASETS",
    "TG43U1S2_CONSENSUS",
    "CLRP_V2",
    "TG43Engine",
    "compute_dose_grid",
    "dose_at_points",
    "isodose_surfaces",
    "DVH",
    "dvh",
    "dose_metrics",
    "resample_mask_to",
    "rind_mask",
    "wall_dose",
    "surface_coverage",
]
