"""
Action Heads: Tool Head (MLP) and Cursor Head (UNet decoder).

Tool Head:
  - Takes fused feature vector
  - Outputs logits over the tool set + value estimate

Cursor Head:
  - Takes bottleneck features + skip connections from VisionEncoder
  - Outputs a spatial heatmap for cursor placement
  - Also used as selection mask when MULTISELECT is active
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToolHead(nn.Module):
    """
    MLP that outputs tool selection logits and value estimate.

    Input:  (B, fusion_dim)
    Output:
        tool_logits: (B, num_tools)
        value: (B, 1)
        param_pred: (B, 1) - optional numeric parameter prediction
    """

    def __init__(self, fusion_dim: int = 256, num_tools: int = 55,
                 hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.tool_logits = nn.Linear(hidden_dim, num_tools)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.param_head = nn.Linear(hidden_dim, 1)

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(fused)
        return {
            "tool_logits": self.tool_logits(h),  # (B, num_tools)
            "value": self.value_head(h),          # (B, 1)
            "param": torch.tanh(self.param_head(h)),  # (B, 1) in [-1, 1]
        }


class UNetUpBlock(nn.Module):
    """Single upsampling block with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from rounding
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CursorHead(nn.Module):
    """
    UNet-style decoder that produces a spatial heatmap for cursor placement.

    Takes the deepest encoder features (bottleneck) and skip connections
    from the VisionEncoder, and decodes back to full resolution.

    Input:
        bottleneck: (B, C_deep, H/8, W/8)
        skips: [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/4, W/4)]
        fused: (B, fusion_dim) - injected as spatial bias

    Output:
        cursor_heatmap: (B, 1, H, W)
    """

    def __init__(self, skip_channels: List[int], fusion_dim: int = 256):
        """
        Args:
            skip_channels: [C0, C1, C2, C3] channel counts from VisionEncoder
            fusion_dim: dimension of fused feature vector
        """
        super().__init__()
        # skip_channels = [C, 2C, 4C, 8C]
        C0, C1, C2, C3 = skip_channels

        # Inject fused features into bottleneck via spatial broadcast
        self.fuse_proj = nn.Linear(fusion_dim, C3)

        # Decoder path (reverse of encoder)
        self.up3 = UNetUpBlock(C3, C2, C2)     # H/8 -> H/4
        self.up2 = UNetUpBlock(C2, C1, C1)     # H/4 -> H/2
        self.up1 = UNetUpBlock(C1, C0, C0)     # H/2 -> H

        # Final 1x1 conv to heatmap
        self.head = nn.Conv2d(C0, 1, 1)

    def forward(self, skips: List[torch.Tensor],
                fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skips: [s0, s1, s2, s3] from VisionEncoder
            fused: (B, fusion_dim)
        Returns:
            heatmap: (B, 1, H, W)
        """
        s0, s1, s2, s3 = skips

        # Inject fused context into bottleneck
        ctx = self.fuse_proj(fused)  # (B, C3)
        ctx = ctx.unsqueeze(-1).unsqueeze(-1)  # (B, C3, 1, 1)
        bottleneck = s3 + ctx  # broadcast add

        x = self.up3(bottleneck, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)

        return self.head(x)  # (B, 1, H, W)
