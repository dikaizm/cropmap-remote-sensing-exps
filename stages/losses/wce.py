"""Weighted Cross-Entropy loss (key: ``wce``).

Standard nn.CrossEntropyLoss with inverse-frequency class weights.
Used as the baseline loss for all Exp A / B / C runs.
"""

import torch.nn as nn


def build_wce(class_weights_tensor):
    """Return WeightedCrossEntropy criterion.

    Args:
        class_weights_tensor: 1-D float tensor of shape (NUM_CLASSES,) on CPU.
            Will be moved to the correct device by the caller before training.

    Returns:
        nn.CrossEntropyLoss instance.
    """
    return nn.CrossEntropyLoss(weight=class_weights_tensor)
