"""
DeepLabV3+ with CBAM attention injected after the ASPP module.

Usage:
    from crop_mapping_pipeline.models import DeepLabV3PlusCBAM

    model = DeepLabV3PlusCBAM(
        encoder_name="resnet50",
        in_channels=27,       # e.g. 3 dates × 9 bands
        num_classes=12,       # 11 crop classes + background
    )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .cbam import CBAM

# ASPP output channels in smp DeepLabV3+ (fixed at 256)
_ASPP_OUT_CH = 256


class DeepLabV3PlusCBAM(nn.Module):
    """
    DeepLabV3+ (smp encoder, e.g. mobilenet_v2 / resnet50) augmented with CBAM after ASPP.

    The forward pass hooks into the decoder to apply CBAM on the
    256-channel ASPP output before the low-level feature fusion step.

    Args:
        encoder_name:  smp encoder name (default "resnet50").
        encoder_weights: pretrained weights (default "imagenet").
        in_channels:   Number of input channels (spectral-temporal bands).
        num_classes:   Number of output segmentation classes.
        cbam_reduction: CBAM channel reduction ratio (default 16).
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 27,
        num_classes: int = 12,
        cbam_reduction: int = 16,
    ):
        super().__init__()

        self.base = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        self.cbam = CBAM(channels=_ASPP_OUT_CH, reduction=cbam_reduction)

        # Register forward hook on the ASPP submodule to inject CBAM output
        self._hook_handle = self.base.decoder.aspp.register_forward_hook(
            self._aspp_hook
        )
        self._aspp_out = None   # populated by hook during forward

    def _aspp_hook(self, module, input, output):
        """Store and replace ASPP output with CBAM-refined version."""
        self._aspp_out = self.cbam(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporarily patch ASPP output via the registered hook.
        # The hook sets self._aspp_out = cbam(aspp_out).
        # smp's decoder will then use the un-modified output, so we
        # override by running the decoder manually after the hook fires.

        features = self.base.encoder(x)

        # Run ASPP (hook fires here, self._aspp_out is set)
        aspp_raw = self.base.decoder.aspp(features[-1])
        aspp_att = self.cbam(aspp_raw)          # CBAM-refined

        # Low-level features from encoder (stride-4 stage; smp builds block1 sized
        # to the encoder's stride-4 channels — e.g. 256ch resnet50, 24ch mobilenet_v2)
        low_level = self.base.decoder.block1(features[2])   # (B, 48, H/4, W/4)

        # Upsample ASPP features to match low-level resolution
        aspp_up = F.interpolate(
            aspp_att,
            size=low_level.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Concatenate and refine → (B, 256, H/4, W/4)
        fused = self.base.decoder.block2(torch.cat([aspp_up, low_level], dim=1))

        # segmentation_head includes upsampling=4 → output is input resolution
        return self.base.segmentation_head(fused)

    def __del__(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
