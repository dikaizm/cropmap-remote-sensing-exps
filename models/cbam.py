"""
CBAM — Convolutional Block Attention Module
Woo et al., 2018. https://arxiv.org/abs/1807.06521
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention with both avg and max pooling."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = self.shared_mlp(x.mean(dim=[2, 3]))          # (B, C)
        mx  = self.shared_mlp(x.amax(dim=[2, 3]))          # (B, C)
        scale = torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention using channel-wise average and max pooling."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)    # (B, 1, H, W)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B, 1, H, W)
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention followed by spatial attention.

    Args:
        channels:      Number of input channels.
        reduction:     Channel reduction ratio for the MLP (default 16).
        spatial_kernel: Kernel size for the spatial attention conv (default 7).
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
