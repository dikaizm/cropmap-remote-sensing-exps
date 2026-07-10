"""Named loss functions for Stage 3 training.

CLI key (``--loss``) → description
    wce                — WeightedCrossEntropy        (baseline; inverse-freq weights)
    phenology          — PhenologyAwareLoss          (NDVI dormancy weighting)
    focal_tversky      — FocalCE + FocalTversky      (median-freq weights;
                                                      spatial generalisation)
    dynamic_balanced   — DynamicEffectiveClassBalanced (per-batch Cui+2019 weights)
    recall             — RecallLoss                  (EMA per-class recall weighting)
    boundary_weighted  — BoundaryWeightedLoss        (dynamic_balanced + boundary
                                                      pixel upweighting, U-Net style)

Usage:
    from crop_mapping_pipeline.stages.losses import (
        build_wce, build_phenology, build_focal_tversky,
        build_dynamic_balanced, build_recall, build_boundary_weighted,
    )
"""

from crop_mapping_pipeline.stages.losses.wce               import build_wce
from crop_mapping_pipeline.stages.losses.phenology         import (
    PhenologyAwareLoss, build_phenology,
)
from crop_mapping_pipeline.stages.losses.focal_tversky     import (
    FocalTverskyLoss,
    effective_number_weights, build_focal_tversky,
)
from crop_mapping_pipeline.stages.losses.dynamic_balanced  import (
    DynamicEffectiveClassBalancedLoss, build_dynamic_balanced,
)
from crop_mapping_pipeline.stages.losses.recall            import (
    RecallLoss, build_recall,
)
from crop_mapping_pipeline.stages.losses.boundary_weighted import (
    BoundaryWeightedLoss, build_boundary_weighted,
)

__all__ = [
    "build_wce",
    "build_phenology",
    "build_focal_tversky",
    "build_dynamic_balanced",
    "build_recall",
    "build_boundary_weighted",
    "PhenologyAwareLoss",
    "FocalCEPlusFocalTversky",
    "FocalCELoss",
    "FocalTverskyLoss",
    "effective_number_weights",
    "DynamicEffectiveClassBalancedLoss",
    "RecallLoss",
    "BoundaryWeightedLoss",
]
