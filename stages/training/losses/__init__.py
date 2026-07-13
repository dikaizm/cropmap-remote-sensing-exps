"""Named loss functions for Stage 3 training.

CLI key (``--loss``) → description
    wce                — WeightedCrossEntropy          (baseline; inverse-freq weights)
    focal_tversky      — FocalCE + FocalTversky        (median-freq weights)
    dynamic_balanced   — DynamicEffectiveClassBalanced (per-batch Cui+2019 weights; DECB-CE, primary)

Usage:
    from cropmap_pipeline.stages.training.losses import (
        build_wce, build_focal_tversky, build_dynamic_balanced,
    )
"""

from cropmap_pipeline.stages.training.losses.wce              import build_wce
from cropmap_pipeline.stages.training.losses.focal_tversky    import (
    FocalTverskyLoss, effective_number_weights, build_focal_tversky,
)
from cropmap_pipeline.stages.training.losses.dynamic_balanced import (
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
