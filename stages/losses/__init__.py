"""Named loss functions for Stage 3 training.

CLI key (``--loss``) → description
    wce                — WeightedCrossEntropy        (baseline; inverse-freq weights)
    focal_tversky      — Focal Tversky               (median-freq weighted
                                                      class-mean; recall-biased)
    dynamic_balanced   — DynamicEffectiveClassBalanced (per-batch Cui+2019 weights;
                                                      thesis primary loss, DECB-CE)

Usage:
    from crop_mapping_pipeline.stages.losses import (
        build_wce, build_focal_tversky, build_dynamic_balanced,
    )
"""

from crop_mapping_pipeline.stages.losses.wce               import build_wce
from crop_mapping_pipeline.stages.losses.focal_tversky     import (
    FocalTverskyLoss, effective_number_weights, build_focal_tversky,
)
from crop_mapping_pipeline.stages.losses.dynamic_balanced  import (
    DynamicEffectiveClassBalancedLoss, build_dynamic_balanced,
)

__all__ = [
    "build_wce",
    "build_focal_tversky",
    "build_dynamic_balanced",
    "FocalTverskyLoss",
    "effective_number_weights",
    "DynamicEffectiveClassBalancedLoss",
]
