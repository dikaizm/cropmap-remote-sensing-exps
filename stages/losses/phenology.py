"""Phenology-Aware Loss (key: ``phenology``).

Extends WeightedCrossEntropy with a dynamic per-pixel weight modifier:
for labeled crop pixels whose NDVI falls below a dormancy threshold, the
loss contribution is reduced so the model is not penalised for predicting
background during off-season periods.

Reference: novel contribution; requires B4 (Red) and B8 (NIR) bands to be
present in the input tensor to compute NDVI at training time.
"""

import torch
import torch.nn as nn


class PhenologyAwareLoss(nn.Module):
    """CrossEntropyLoss with dynamic per-pixel phenology weighting.

    For pixels labeled as a crop class (label > 0), if the on-the-fly NDVI
    computed from the input image is below `threshold` (dormant season), the
    loss weight is reduced by `penalty_reduction` to avoid penalising the
    model for predicting background on dormant crop pixels.

    Args:
        base_weight:       Class-frequency inverse weights (NUM_CLASSES,).
        red_idx:           Channel index of the Red (B4) band in the input tensor.
        nir_idx:           Channel index of the NIR (B8) band in the input tensor.
        threshold:         NDVI value below which a crop pixel is considered dormant.
        penalty_reduction: Weight multiplier applied to dormant crop pixels.
    """

    def __init__(self, base_weight, red_idx, nir_idx, threshold=0.3, penalty_reduction=0.1):
        super().__init__()
        self.base_criterion   = nn.CrossEntropyLoss(weight=base_weight, reduction="none")
        self.red_idx          = red_idx
        self.nir_idx          = nir_idx
        self.threshold        = threshold
        self.reduction_factor = penalty_reduction

    def forward(self, logits, labels, images):
        """
        Args:
            logits: (B, C, H, W)
            labels: (B, H, W)  long
            images: (B, C, H, W)  — used to compute per-pixel NDVI
        """
        red  = images[:, self.red_idx,  :, :]
        nir  = images[:, self.nir_idx,  :, :]
        ndvi = (nir - red) / (nir + red + 1e-7)

        loss     = self.base_criterion(logits, labels)   # (B, H, W)
        is_crop  = labels > 0
        dormant  = is_crop & (ndvi < self.threshold)

        weights            = torch.ones_like(loss)
        weights[dormant]   = self.reduction_factor
        return (loss * weights).mean()


def build_phenology(class_weights_tensor, band_names_list):
    """Build a PhenologyAwareLoss by locating B4/B8 in band_names_list.

    Args:
        class_weights_tensor: 1-D float tensor (NUM_CLASSES,) on CPU.
        band_names_list:      List of band name strings for the current experiment
                              (e.g. ["B1_20220730", "B4_20220730", ...]).

    Returns:
        (PhenologyAwareLoss, red_idx, nir_idx)

    Raises:
        ValueError: if B4 or B8 cannot be found in band_names_list.
    """
    red_idx = next((i for i, n in enumerate(band_names_list) if n.startswith("B4")), None)
    nir_idx = next((i for i, n in enumerate(band_names_list) if n.startswith("B8")), None)

    if red_idx is None or nir_idx is None:
        missing = []
        if red_idx is None: missing.append("B4")
        if nir_idx is None: missing.append("B8")
        raise ValueError(
            f"PhenologyAwareLoss (v2) requires {missing} band(s) in the input, "
            f"but they were not found in band_names_list: {band_names_list}"
        )

    criterion = PhenologyAwareLoss(
        base_weight=class_weights_tensor,
        red_idx=red_idx,
        nir_idx=nir_idx,
        threshold=0.3,
        penalty_reduction=0.1,
    )
    return criterion, red_idx, nir_idx
