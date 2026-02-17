"""
Vision encoder: ResNet-style CNN built from scratch (no pretrained weights).

Takes the multi-channel image tensor and produces a spatial feature map
plus a global feature vector. The spatial features feed into the UNet
decoder for cursor prediction; the global features feed into the fusion bridge.

Architecture:
  Input: (B, C_in, H, W) where C_in = 3+3+L+1
  Output:
    - features: List of (B, C_i, H_i, W_i) at multiple scales (for skip connections)
    - global_feat: (B, fusion_dim)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class VisionEncoder(nn.Module):
    """
    Multi-scale vision encoder with skip connections for UNet.

    Produces feature maps at 4 scales:
      scale 0: H/1,  W/1   (stem output)
      scale 1: H/2,  W/2
      scale 2: H/4,  W/4
      scale 3: H/8,  W/8

    And a global feature vector via adaptive average pooling.
    """

    def __init__(self, in_channels: int, base_channels: int = 32,
                 fusion_dim: int = 256):
        super().__init__()
        C = base_channels

        # Stem: preserve spatial resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Encoder stages (each downsamples by 2x)
        self.stage1 = nn.Sequential(
            ResBlock(C, C * 2, stride=2),
            ResBlock(C * 2, C * 2),
        )
        self.stage2 = nn.Sequential(
            ResBlock(C * 2, C * 4, stride=2),
            ResBlock(C * 4, C * 4),
        )
        self.stage3 = nn.Sequential(
            ResBlock(C * 4, C * 8, stride=2),
            ResBlock(C * 8, C * 8),
        )

        # Global feature head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(C * 8, fusion_dim)

        # Store channel counts for UNet decoder
        self.skip_channels = [C, C * 2, C * 4, C * 8]

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            skips: list of feature maps at each scale
            global_feat: (B, fusion_dim)
        """
        s0 = self.stem(x)             # (B, C, H, W)
        s1 = self.stage1(s0)          # (B, 2C, H/2, W/2)
        s2 = self.stage2(s1)          # (B, 4C, H/4, W/4)
        s3 = self.stage3(s2)          # (B, 8C, H/8, W/8)

        g = self.global_pool(s3).flatten(1)  # (B, 8C)
        g = self.global_fc(g)                # (B, fusion_dim)

        return [s0, s1, s2, s3], g
