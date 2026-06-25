"""Dynamic Effective Class Balanced Cross-Entropy (DECB-CE) — key: ``dynamic_balanced``.

Faithful implementation of the DECB weighting method of Zhou et al. (2023),
"A Dynamic Effective Class Balanced Approach for Remote Sensing Imagery Semantic
Segmentation of Imbalanced Data", Remote Sensing 15(7):1768, eqs (11),(14),(15).

Per batch (a batch acts as a sample subspace):
  n_batch  = total valid pixels in the batch
  β_batch  = (1/(10^p + 1))^(1/n_batch)                          [eq (15), on n_batch]
  E_nbatch = (1 - 1/(10^p+1)) / (1 - β_batch)                    [eq (11)/(7)]
  for each class i with batch count n_i:
      if n_i < E_nbatch:                    # minority class
          β_i  = (1/(10^p+1))^(1/n_i)                            [eq (15)]
          E_ni = (1 - 1/(10^p+1)) / (1 - β_i)                    [eq (5)/(11)]
          W_i  = 1 - E_ni / n_batch                              [eq (14), minority]
      else:                                 # majority class
          W_i  = 1 - n_i / n_batch                               [eq (14), majority]
  weights W_i applied to standard cross-entropy.

The "dynamic" aspect is the per-class β derived from that class's batch count
(eq 15) — NOT a fixed β. p>=3 (default 3 → 10^3+1 = 1001), matching the paper.

Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"
(CVPR) is the origin of the effective-number concept that Zhou et al. build on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicEffectiveClassBalancedLoss(nn.Module):
    """DECB-CE (Zhou et al. 2023). Per-batch, per-class dynamic effective-number weights.

    Args:
        num_classes:     Total number of classes (including background).
        p:               Effective-space hyperparameter; 10^p + 1 controls the
                         dynamic β (paper uses p>=3; default 3 → 1001).
        fallback_weight: Weight for classes absent from the batch (n_i = 0).
        ignore_index:    Label index to ignore (default -100).
    """

    def __init__(self, num_classes, p=3, fallback_weight=1.0, ignore_index=-100):
        super().__init__()
        self.num_classes     = num_classes
        self.base            = 1.0 / (10 ** p + 1)   # 1/(10^p + 1), e.g. 1/1001
        self.fallback_weight = fallback_weight
        self.ignore_index    = ignore_index

    def forward(self, logits, target):
        """logits: (B, C, H, W) float ; target: (B, H, W) long."""
        C    = self.num_classes
        base = self.base
        dev  = logits.device

        flat    = target.view(-1)
        valid   = flat != self.ignore_index
        n_batch = valid.sum().float().clamp(min=1.0)

        counts = torch.zeros(C, dtype=torch.float32, device=dev)
        for c in range(C):
            counts[c] = (flat == c).sum().float()

        # Effective sample-subspace size of the batch (eq 11):
        #   β_batch = base^(1/n_batch);  E_nbatch = (1-base)/(1-β_batch)
        beta_batch = base ** (1.0 / n_batch)
        E_nbatch   = (1.0 - base) / (1.0 - beta_batch).clamp(min=1e-12)

        w = torch.empty(C, dtype=torch.float32, device=dev)
        for c in range(C):
            n_i = counts[c]
            if n_i <= 0:
                w[c] = self.fallback_weight
            elif n_i < E_nbatch:                       # minority (eq 14, top)
                beta_i = base ** (1.0 / n_i)
                E_ni   = (1.0 - base) / (1.0 - beta_i).clamp(min=1e-12)
                w[c]   = 1.0 - E_ni / n_batch
            else:                                      # majority (eq 14, bottom)
                w[c]   = 1.0 - n_i / n_batch

        w = w.clamp(min=1e-6)
        return F.cross_entropy(logits, target, weight=w,
                               ignore_index=self.ignore_index)


def build_dynamic_balanced(num_classes, p=3, fallback_weight=1.0, ignore_index=-100,
                           **_legacy):
    """Build DECB-CE loss (Zhou et al. 2023).

    Accepts and ignores legacy kwargs (e.g. ``beta``) for backward compatibility
    with older call sites; β is now dynamic per-class and not a fixed parameter.
    """
    return DynamicEffectiveClassBalancedLoss(
        num_classes=num_classes, p=p,
        fallback_weight=fallback_weight, ignore_index=ignore_index,
    )
