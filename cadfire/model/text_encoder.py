"""
Text encoder: small GRU-based encoder for CAD prompts.

Takes tokenized text ids and produces a fixed-size feature vector
for the fusion bridge. Built from scratch - no pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Embedding + bidirectional GRU + projection.

    Input:  (B, seq_len) int64 token ids
    Output: (B, fusion_dim) text feature vector
    """

    def __init__(self, vocab_size: int = 4096, embed_dim: int = 128,
                 hidden_dim: int = 256, fusion_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        # bidirectional doubles hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, seq_len) int64
        Returns:
            text_feat: (B, fusion_dim)
        """
        embedded = self.embedding(token_ids)   # (B, seq_len, embed_dim)
        output, hidden = self.gru(embedded)    # output: (B, seq_len, 2*hidden_dim)

        # Use final hidden states from both directions
        # hidden: (num_layers*2, B, hidden_dim)
        fwd = hidden[-2]  # last layer forward
        bwd = hidden[-1]  # last layer backward
        combined = torch.cat([fwd, bwd], dim=1)  # (B, 2*hidden_dim)

        return self.projection(combined)  # (B, fusion_dim)
