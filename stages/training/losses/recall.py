"""Recall Loss for Semantic Segmentation (key: ``recall``).

Weights cross-entropy loss per class by instantaneous recall w_c = 1 - R_c,
where R_c is the per-class recall computed from batch predictions. Classes the
model ignores (R_c → 0) receive maximum weight; well-recalled classes are
down-weighted. An EMA smooths recall estimates across batches for stability.

The loss "changes gradually between standard CE and inverse-frequency
weighted CE" as training progresses, avoiding the excessive false-positive
problem of always using the maximum minority-class weight.

Key advantage over wce (static inverse-freq CE):
- Responds to actual model behaviour during training, not train-set prior.
- If model stops predicting Walnuts in test_b, Walnuts weight spikes
  automatically in the next batch — without knowing test-set distribution.

Reference:
  Tian et al. 2021 — "Striking the Right Balance: Recall Loss for Semantic
  Segmentation". CVPR 2022. https://arxiv.org/abs/2106.14917
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallLoss(nn.Module):
    """Cross-entropy weighted by per-class (1 - recall), EMA-smoothed.

    Algorithm:
      1. Compute batch recall R_c = TP_c / (TP_c + FN_c) from argmax preds.
      2. EMA update: r̄_c ← m * r̄_c + (1 - m) * R_c  (only for classes
         present in the batch; absent classes keep previous estimate).
      3. Weight: w_c = (1 - r̄_c), normalised to mean = 1.
      4. Apply weights to CE loss.

    Args:
        num_classes:  Total classes including background.
        momentum:     EMA momentum for recall estimates (default 0.9).
        init_recall:  Initial recall estimate per class (default 0.0 —
                      all classes start maximally up-weighted).
        ignore_index: Label index ignored in CE (default -100).
    """

    def __init__(self, num_classes, momentum=0.9,
                 init_recall=0.0, ignore_index=-100):
        super().__init__()
        self.num_classes  = num_classes
        self.momentum     = momentum
        self.ignore_index = ignore_index
        # running recall estimate — kept as buffer so it moves with .to(device)
        self.register_buffer(
            "running_recall",
            torch.full((num_classes,), init_recall, dtype=torch.float32),
        )

    @torch.no_grad()
    def _update_recall(self, logits, target):
        """Compute batch recall per class and EMA-update running estimates."""
        preds = logits.argmax(dim=1)           # (B, H, W) — no grad
        flat_p = preds.view(-1)
        flat_t = target.view(-1)

        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            gt_mask = flat_t == c
            n_gt    = gt_mask.sum().item()
            if n_gt == 0:
                continue                        # class absent; keep old EMA
            tp      = ((flat_p == c) & gt_mask).sum().float()
            recall  = tp / (n_gt + 1e-12)
            self.running_recall[c] = (
                self.momentum * self.running_recall[c]
                + (1.0 - self.momentum) * recall
            )

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) float
            target: (B, H, W)    long
        """
        self._update_recall(logits, target)

        # w_c = 1 - r̄_c ; normalise to mean = 1
        w = 1.0 - self.running_recall           # (C,)
        w = w.clamp(min=0.0)
        mean_w = w.mean()
        if mean_w > 1e-8:
            w = w / mean_w
        else:
            w = torch.ones_like(w)             # fallback: uniform

        return F.cross_entropy(logits, target,
                               weight=w, ignore_index=self.ignore_index)


def build_recall(num_classes, momentum=0.9,
                 init_recall=0.0, ignore_index=-100):
    """Build RecallLoss.

    Args:
        num_classes:  Number of output classes.
        momentum:     EMA momentum for recall (0.9 = slow update = stable).
        init_recall:  Starting recall estimate; 0.0 = max weight initially.
        ignore_index: Label to ignore.
    """
    return RecallLoss(
        num_classes=num_classes, momentum=momentum,
        init_recall=init_recall, ignore_index=ignore_index,
    )
