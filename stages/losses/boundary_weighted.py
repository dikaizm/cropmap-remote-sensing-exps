"""Boundary-Weighted Dynamic Class-Balanced Loss (key: ``boundary_weighted``).

Combines two complementary weighting strategies:

1. **Dynamic effective-class-balance** (Cui et al. 2019 / Zhou et al. 2023):
   Per-batch pixel counts → effective-number weights → class-level CE weight.
   Handles pixel-count class imbalance (Rice vs. Walnuts).

2. **Boundary pixel upweighting** (after Ronneberger et al. 2015, U-Net):
   GT label transitions → 1-pixel boundary mask → dilated by `dilation_kernel`
   → boundary pixels get `boundary_weight × base loss`. Pulls predicted edges
   toward the hard GT boundary instead of letting them blur into field interiors.

Why boundary weighting matters here:
- DeepLab/SegFormer encoder stride-32 blurs spatial features → predicted class
  boundary drifts toward field interior where confidence is higher.
- Plain CE loss averages over all pixels: boundary pixels (~5%) contribute
  negligible gradient. Model ignores them without penalty.
- CDL labels have hard edges (nearest-neighbor resampling); predictions are
  smooth. Boundary weighting closes this gap by tripling gradient at edges.

References:
  Ronneberger et al. 2015 — "U-Net: Convolutional Networks for Biomedical
  Image Segmentation". MICCAI. https://arxiv.org/abs/1505.04597

  Cui et al. 2019 — "Class-Balanced Loss Based on Effective Number of Samples"
  CVPR. https://arxiv.org/abs/1901.05555

  Zhou et al. 2023 — "A Dynamic Effective Class Balanced Approach for
  Remote Sensing Imagery Semantic Segmentation of Imbalanced Data"
  Remote Sensing 15(7). https://doi.org/10.3390/rs15071768
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryWeightedLoss(nn.Module):
    """Dynamic class-balanced CE + boundary pixel upweighting.

    Algorithm:
      1. Count batch pixel counts per class → effective-number class weights w_c.
      2. Compute CE with w_c, reduction='none' → per-pixel loss map (B, H, W).
      3. Build boundary mask: GT horizontal/vertical class transitions → dilate.
      4. Pixel weight map: 1.0 interior, `boundary_weight` at boundary zone.
      5. Loss = mean(ce_map × pixel_weight_map).

    Args:
        num_classes:      Total classes including background.
        beta:             Effective-number smoothing factor (default 0.9999).
        fallback_weight:  Class weight for classes absent from batch (default 2.0).
        boundary_weight:  Multiplier applied to boundary-zone pixels (default 3.0).
        dilation_kernel:  Max-pool kernel size for boundary dilation (default 5 →
                          ~2px radius; use 7 for ~3px radius).
        ignore_index:     Label index ignored in CE (default -100).
    """

    def __init__(
        self,
        num_classes,
        beta             = 0.9999,
        fallback_weight  = 2.0,
        boundary_weight  = 3.0,
        dilation_kernel  = 5,
        ignore_index     = -100,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.beta            = beta
        self.fallback_weight = fallback_weight
        self.boundary_weight = boundary_weight
        self.dilation_kernel = dilation_kernel
        self.ignore_index    = ignore_index

    def _class_weights(self, target, device):
        """Compute per-batch effective-number class weights."""
        C    = self.num_classes
        beta = self.beta

        flat   = target.view(-1)
        counts = torch.zeros(C, dtype=torch.float32, device=device)
        for c in range(C):
            if c != self.ignore_index:
                counts[c] = (flat == c).sum().float()

        one_minus_beta = 1.0 - beta
        beta_pow       = torch.pow(torch.tensor(beta, device=device), counts)
        w              = one_minus_beta / (1.0 - beta_pow).clamp(min=1e-12)

        absent    = counts == 0
        w[absent] = self.fallback_weight

        present_w = w[~absent]
        if present_w.numel() > 0:
            w[~absent] = w[~absent] / present_w.mean()

        return w  # (C,)

    def _boundary_pixel_weights(self, target):
        """Build pixel-wise weight map: 1.0 interior, boundary_weight at edges.

        Steps:
          1. Detect horizontal and vertical class transitions in GT.
          2. Mark both pixels on each side of each transition.
          3. Dilate with max-pool (dilation_kernel × dilation_kernel).
          4. Return float map: 1.0 + (boundary_weight - 1.0) × boundary_mask.
        """
        B, H, W = target.shape

        # Detect class transitions along H and W
        diff_h = target[:, 1:, :] != target[:, :-1, :]   # (B, H-1, W)
        diff_w = target[:, :, 1:] != target[:, :, :-1]   # (B, H, W-1)

        # Mark both sides of each transition
        bnd = torch.zeros(B, H, W, dtype=torch.bool, device=target.device)
        bnd[:, 1:,  :] |= diff_h   # lower row of vertical transition
        bnd[:, :-1, :] |= diff_h   # upper row
        bnd[:, :,  1:] |= diff_w   # right col of horizontal transition
        bnd[:, :, :-1] |= diff_w   # left col

        # Dilate: max-pool expands True pixels outward
        pad    = self.dilation_kernel // 2
        dilated = F.max_pool2d(
            bnd.float().unsqueeze(1),
            kernel_size=self.dilation_kernel,
            stride=1,
            padding=pad,
        ).squeeze(1)  # (B, H, W)

        # Map to weight: interior=1.0, boundary=boundary_weight
        return 1.0 + (self.boundary_weight - 1.0) * dilated  # (B, H, W)

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) float
            target: (B, H, W)    long
        """
        class_w  = self._class_weights(target, logits.device)

        # Per-pixel CE with class weights, no reduction
        ce = F.cross_entropy(
            logits, target,
            weight=class_w,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)

        pixel_w = self._boundary_pixel_weights(target)  # (B, H, W)

        return (ce * pixel_w).mean()


def build_boundary_weighted(
    num_classes,
    beta            = 0.9999,
    fallback_weight = 2.0,
    boundary_weight = 3.0,
    dilation_kernel = 5,
    ignore_index    = -100,
):
    """Build BoundaryWeightedLoss.

    Args:
        num_classes:      Number of output classes.
        beta:             Effective-number beta (0.9999 for batch-pixel scale).
        fallback_weight:  Weight for classes absent from batch.
        boundary_weight:  CE multiplier at GT boundary pixels (default 3.0).
        dilation_kernel:  Boundary dilation kernel size (default 5 = ~2px radius).
        ignore_index:     Label to ignore.
    """
    return BoundaryWeightedLoss(
        num_classes=num_classes,
        beta=beta,
        fallback_weight=fallback_weight,
        boundary_weight=boundary_weight,
        dilation_kernel=dilation_kernel,
        ignore_index=ignore_index,
    )
