"""
Fusion Bridge: combines vision, text, and state features.

Uses cross-attention between text and vision global features,
then fuses with the scalar state vector to produce a unified
representation for the Tool Head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBridge(nn.Module):
    """
    Multi-modal fusion via cross-attention + concatenation.

    Inputs:
        vision_global: (B, fusion_dim)  - from VisionEncoder
        text_feat:     (B, fusion_dim)  - from TextEncoder
        state_vec:     (B, state_dim)   - scalar state vector

    Output:
        fused: (B, fusion_dim)  - unified feature for Tool Head
    """

    def __init__(self, fusion_dim: int = 256, state_dim: int = 16,
                 num_heads: int = 4):
        super().__init__()

        # Project state vector to fusion_dim
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-attention: text attends to vision
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(fusion_dim)

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, vision_global: torch.Tensor, text_feat: torch.Tensor,
                state_vec: torch.Tensor) -> torch.Tensor:
        """Fuse all modalities into a single feature vector."""
        # Cross attention: text queries, vision as key/value
        # Need to add sequence dim for attention
        q = text_feat.unsqueeze(1)       # (B, 1, D)
        kv = vision_global.unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, D)
        attn_out = self.attn_norm(attn_out.squeeze(1) + text_feat)  # residual

        # Project state
        state_feat = self.state_proj(state_vec)  # (B, D)

        # Concatenate and fuse
        combined = torch.cat([vision_global, attn_out, state_feat], dim=1)  # (B, 3D)
        fused = self.fusion_mlp(combined)  # (B, D)

        return fused
