"""Focal Tversky loss (key: ``focal_tversky``).

Region-level IoU-style loss for class imbalance, biased toward recall (α>β) so
rare crop classes are not drowned out by abundant ones. Aligns with the mIoU
evaluation metric.

Components:
  - Focal Tversky (Abraham & Khan 2018): per-class Tversky index
    T_c = TP / (TP + α·FN + β·FP); loss = (1 - T_c)^γ. α weights false negatives,
    β weights false positives — α>β favours recall. γ<1 focuses gradients on the
    classes with the lowest Tversky index (hardest classes).
  - Optional per-class weighting of the class-mean (Median-Frequency Balancing —
    Eigen & Fergus 2015 — by default), so rare classes contribute more to the
    aggregate loss. Pass class_counts to enable; omit for an unweighted mean.

References:
  - Abraham & Khan 2018 — A Novel Focal Tversky Loss for Lesion Segmentation.
  - Eigen & Fergus 2015 — Predicting Depth, Surface Normals & Semantic Labels.
  - Cui et al. 2019 — Class-Balanced Loss Based on Effective Number of Samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def median_frequency_weights(class_counts):
    """Eigen & Fergus 2015. w_c = median(f) / f_c. Returns tensor (mean ≈ 1)."""
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    freq   = counts / (counts.sum() + 1e-12)
    med    = torch.median(freq)
    w      = med / (freq + 1e-12)
    return w.float()


def inverse_sqrt_freq_weights(class_counts):
    """w_c = 1/√f_c, normalised to mean=1. Less extreme than 1/f, more than uniform."""
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    freq   = counts / (counts.sum() + 1e-12)
    w      = 1.0 / torch.sqrt(freq + 1e-12)
    w      = w / w.mean()
    return w.float()


def effective_number_weights(class_counts, beta=None):
    """Cui et al. 2019. w_c = (1-β) / (1-β^n_c).

    If beta is None, auto-select β = 1 - 1/n_min so that β^n_min ≈ 1/e
    and weights span a meaningful range (avoids collapse at very large n).
    Returns tensor normalised to mean=1.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if beta is None:
        n_min = counts[counts > 0].min().item()
        beta  = max(0.0, 1.0 - 1.0 / max(n_min, 1.0))
    eff = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)
    w   = (1.0 - beta) / (eff + 1e-12)
    w   = w / w.mean()
    return w.float()


def build_class_weights(class_counts, mode="median_freq", beta=None):
    """Dispatch on mode: 'median_freq' (default), 'invsqrt', 'effnum'."""
    if mode == "median_freq":
        return median_frequency_weights(class_counts)
    if mode == "invsqrt":
        return inverse_sqrt_freq_weights(class_counts)
    if mode == "effnum":
        return effective_number_weights(class_counts, beta=beta)
    raise ValueError(f"Unknown class-weight mode: {mode}")


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss. Skips background by default.

    α weights FN, β weights FP — set α > β to favour recall on rare classes.
    γ < 1 focuses gradients on classes with low Tversky index.
    class_weights (optional, 1-D, length C): per-class weight applied to the
    class-mean so rare classes contribute more (registered as buffer → moves
    with .to(device)).
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75,
                 ignore_background=True, smooth=1e-6, class_weights=None):
        super().__init__()
        self.alpha             = alpha
        self.beta              = beta
        self.gamma             = gamma
        self.ignore_background = ignore_background
        self.smooth            = smooth
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, target):
        C = logits.shape[1]
        p  = F.softmax(logits, dim=1)                                # (B, C, H, W)
        oh = F.one_hot(target.clamp(min=0), num_classes=C)           # (B, H, W, C)
        oh = oh.permute(0, 3, 1, 2).float()                           # (B, C, H, W)

        classes = list(range(1, C) if self.ignore_background else range(C))
        losses, weights = [], []
        for c in classes:
            pc = p[:, c]
            gc = oh[:, c]
            tp = (pc * gc).sum()
            fn = ((1 - pc) * gc).sum()
            fp = (pc * (1 - gc)).sum()
            t  = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            losses.append((1.0 - t).clamp(min=1e-7) ** self.gamma)
            if self.class_weights is not None:
                weights.append(self.class_weights[c])

        loss = torch.stack(losses)
        if self.class_weights is not None:
            w = torch.stack(weights).to(loss)
            return (loss * w).sum() / (w.sum() + 1e-12)
        return loss.mean()


def build_focal_tversky(class_weights_tensor=None, class_counts=None,
                        weight_mode="median_freq", beta=None,
                        tv_alpha=0.7, tv_beta=0.3, tv_gamma=0.75):
    """Build the focal_tversky loss (pure Focal Tversky).

    Pass class_counts (preferred — derives per-class weights via build_class_weights
    with weight_mode) or a pre-computed class_weights_tensor (fallback). Omit both
    for an unweighted class-mean.
    """
    if class_counts is not None:
        class_weights = build_class_weights(class_counts, mode=weight_mode, beta=beta)
    else:
        class_weights = class_weights_tensor.float() if class_weights_tensor is not None else None
    return FocalTverskyLoss(
        alpha=tv_alpha, beta=tv_beta, gamma=tv_gamma, class_weights=class_weights,
    )
