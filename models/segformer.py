"""
SegFormer wrapper using segmentation-models-pytorch.

Usage:
    from crop_mapping_pipeline.models import build_segformer

    model = build_segformer(
        encoder_name="mit_b2",
        in_channels=27,
        num_classes=12,
    )
"""
import torch
import segmentation_models_pytorch as smp


def _apply_contiguous_hooks(model: torch.nn.Module) -> None:
    """
    Register forward hooks that call .contiguous() on every module output.

    Required for Apple MPS backend: the MiT efficient self-attention uses
    transpose → view patterns that produce non-contiguous gradients during
    backpropagation on MPS, causing 'view size is not compatible' errors.
    Making all intermediate activations contiguous avoids the issue.
    """
    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            return out.contiguous()

    for module in model.modules():
        module.register_forward_hook(_hook)


def build_segformer(
    encoder_name: str = "mit_b2",
    encoder_weights: str = "imagenet",
    in_channels: int = 27,
    num_classes: int = 12,
) -> smp.Segformer:
    """
    Build a SegFormer model with Mix Transformer (MiT) encoder.

    SegFormer already contains Efficient Self-Attention in every encoder
    block — no additional attention module is required.

    Contiguous forward hooks are applied automatically to support Apple MPS.

    Args:
        encoder_name:     MiT variant: "mit_b0" … "mit_b5" (default "mit_b2").
        encoder_weights:  Pretrained weights (default "imagenet").
        in_channels:      Number of input channels (spectral-temporal bands).
        num_classes:      Number of output segmentation classes.

    Returns:
        smp.Segformer model instance.
    """
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    _apply_contiguous_hooks(model)
    return model
