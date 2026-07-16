"""Architecture factory: build a DeepLabV3+CBAM or SegFormer model."""

import logging

from config import ARCH_CFG
from models import DeepLabV3PlusCBAM, build_segformer
from stages.training.run_state import DEVICE

log = logging.getLogger(__name__)


def build_model(arch, in_channels, num_classes):
    cfg = ARCH_CFG[arch]
    if arch == "deeplabv3plus_cbam":
        model = DeepLabV3PlusCBAM(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif arch == "segformer":
        model = build_segformer(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    n = sum(p.numel() for p in model.parameters())
    log.info(f"  {arch} ({cfg['encoder']}): {n:,} params")
    model._n_params = n
    return model.to(DEVICE)
