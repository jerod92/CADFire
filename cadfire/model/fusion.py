"""
Fusion Bridge: combines vision, text, spectral, and state features.

Uses cross-attention between text and vision global features,
then fuses with the spectral encoding and scalar state vector to produce
a unified representation for the Tool Head.

Modalities fused
────────────────
  vision_global  (B, fusion_dim) – spatial image encoding from VisionEncoder
  text_feat      (B, fusion_dim) – prompt encoding from TextEncoder
  spectral_feat  (B, fusion_dim) – frequency-domain encoding from SpectralEncoder
  state_vec      (B, state_dim)  – scalar tool/viewport state vector
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBridge(nn.Module):
    """
    Multi-modal fusion via cross-attention + concatenation.

    Inputs:
        vision_global  : (B, fusion_dim)  - from VisionEncoder
        text_feat      : (B, fusion_dim)  - from TextEncoder
        state_vec      : (B, state_dim)   - scalar state vector
        spectral_feat  : (B, fusion_dim)  - from SpectralEncoder

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

        # Final fusion MLP: 4 modalities concatenated
        # vision_global | attn_out (text+vision) | spectral_feat | state_feat
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, vision_global: torch.Tensor, text_feat: torch.Tensor,
                state_vec: torch.Tensor,
                spectral_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse all modalities into a single feature vector.

        Args:
            vision_global : (B, fusion_dim)
            text_feat     : (B, fusion_dim)
            state_vec     : (B, state_dim)
            spectral_feat : (B, fusion_dim)

        Returns:
            fused: (B, fusion_dim)
        """
        # Cross attention: text queries, vision as key/value
        q  = text_feat.unsqueeze(1)       # (B, 1, D)
        kv = vision_global.unsqueeze(1)   # (B, 1, D)
        attn_out, _ = self.cross_attn(q, kv, kv)          # (B, 1, D)
        attn_out = self.attn_norm(attn_out.squeeze(1) + text_feat)  # residual

        # Project state
        state_feat = self.state_proj(state_vec)  # (B, D)

        # Concatenate all four modalities and fuse
        combined = torch.cat(
            [vision_global, attn_out, spectral_feat, state_feat], dim=1
        )  # (B, 4D)
        fused = self.fusion_mlp(combined)  # (B, D)

        return fused
