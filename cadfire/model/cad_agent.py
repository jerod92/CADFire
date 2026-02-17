"""
CADAgent: the unified model that combines all components.

This is the single nn.Module used by the training loop. It takes
the observation dict and returns action distributions + value estimates.

Key design for extensibility:
  - Tool head size is read from config, not hard-coded
  - When adding new tools, create a new config with the extended tool list,
    call agent.extend_tools(new_num_tools), and it will grow the head
    while preserving all existing weights
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from cadfire.model.vision_encoder import VisionEncoder
from cadfire.model.text_encoder import TextEncoder
from cadfire.model.fusion import FusionBridge
from cadfire.model.action_heads import ToolHead, CursorHead
from cadfire.utils.config import load_config, tool_list, num_tools


class CADAgent(nn.Module):
    """
    Full RL agent model for CAD drafting.

    Observation -> Feature Extraction -> Fusion -> Action Heads
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        self.config = config or load_config()
        m = self.config["model"]
        layers_cfg = self.config["layers"]

        # Calculate total input channels to match renderer output:
        #   3 (viewport RGB) + 3 (reference) + L (layers) + 1 (selection)
        #   + 4 (coordinate grids: ground x/y tanh, window x/y linear)
        max_layers = layers_cfg["max_layers"]
        in_channels = 3 + 3 + max_layers + 1 + 4  # = 19 for L=8

        self.in_channels = in_channels
        n_tools = num_tools()

        # Store tool map for checkpoint compatibility
        self._tool_list = tool_list()

        # Sub-modules
        self.vision = VisionEncoder(
            in_channels=in_channels,
            base_channels=m["vision_base_channels"],
            fusion_dim=m["fusion_dim"],
        )
        self.text = TextEncoder(
            vocab_size=m["text_vocab_size"],
            embed_dim=m["text_embed_dim"],
            hidden_dim=m["text_hidden_dim"],
            fusion_dim=m["fusion_dim"],
        )
        self.fusion = FusionBridge(
            fusion_dim=m["fusion_dim"],
            state_dim=m["state_dim"],
        )
        self.tool_head = ToolHead(
            fusion_dim=m["fusion_dim"],
            num_tools=n_tools,
        )
        self.cursor_head = CursorHead(
            skip_channels=self.vision.skip_channels,
            fusion_dim=m["fusion_dim"],
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            obs: {
                "image": (B, H, W, C) float32
                "text_ids": (B, seq_len) int64
                "state_vec": (B, state_dim) float32
            }

        Returns: {
            "tool_logits": (B, num_tools)
            "cursor_heatmap": (B, 1, H, W)
            "value": (B, 1)
            "param": (B, 1)
        }
        """
        # Rearrange image from (B, H, W, C) to (B, C, H, W)
        image = obs["image"].permute(0, 3, 1, 2)
        text_ids = obs["text_ids"].long()
        state_vec = obs["state_vec"]

        # Encode
        skips, vision_global = self.vision(image)
        text_feat = self.text(text_ids)

        # Fuse
        fused = self.fusion(vision_global, text_feat, state_vec)

        # Action heads
        tool_out = self.tool_head(fused)
        cursor_heatmap = self.cursor_head(skips, fused)

        return {
            "tool_logits": tool_out["tool_logits"],
            "cursor_heatmap": cursor_heatmap,
            "value": tool_out["value"],
            "param": tool_out["param"],
        }

    def act(self, obs: Dict[str, torch.Tensor],
            deterministic: bool = False) -> Dict[str, Any]:
        """
        Sample an action from the policy.

        Returns: {
            "tool_id": int,
            "cursor": (H, W) ndarray,
            "param": float,
            "tool_log_prob": float,
            "cursor_log_prob": float,
            "value": float,
            "tool_entropy": float,
        }
        """
        with torch.no_grad() if deterministic else torch.enable_grad():
            out = self.forward(obs)

        # Tool selection
        tool_logits = out["tool_logits"]  # (B, num_tools)
        tool_dist = Categorical(logits=tool_logits)
        if deterministic:
            tool_id = tool_logits.argmax(dim=-1)
        else:
            tool_id = tool_dist.sample()
        tool_log_prob = tool_dist.log_prob(tool_id)
        tool_entropy = tool_dist.entropy()

        # Cursor: treat as spatial categorical
        heatmap = out["cursor_heatmap"].squeeze(1)  # (B, H, W)
        B, H, W = heatmap.shape
        flat_logits = heatmap.reshape(B, -1)  # (B, H*W)
        cursor_dist = Categorical(logits=flat_logits)
        if deterministic:
            cursor_flat = flat_logits.argmax(dim=-1)
        else:
            cursor_flat = cursor_dist.sample()
        cursor_log_prob = cursor_dist.log_prob(cursor_flat)

        # Convert flat index to 2D heatmap
        cursor_heatmap_out = torch.zeros(B, H, W, device=heatmap.device)
        for i in range(B):
            cy, cx = divmod(cursor_flat[i].item(), W)
            cursor_heatmap_out[i, cy, cx] = 1.0

        return {
            "tool_id": tool_id,
            "cursor": cursor_heatmap_out,
            "param": out["param"].squeeze(-1),
            "tool_log_prob": tool_log_prob,
            "cursor_log_prob": cursor_log_prob,
            "value": out["value"].squeeze(-1),
            "tool_entropy": tool_entropy,
        }

    def evaluate_actions(self, obs: Dict[str, torch.Tensor],
                         tool_ids: torch.Tensor,
                         cursor_flat_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        Used during PPO training to compute the ratio pi(a|s)/pi_old(a|s).
        """
        out = self.forward(obs)

        # Tool
        tool_dist = Categorical(logits=out["tool_logits"])
        tool_log_prob = tool_dist.log_prob(tool_ids)
        tool_entropy = tool_dist.entropy()

        # Cursor
        heatmap = out["cursor_heatmap"].squeeze(1)
        B, H, W = heatmap.shape
        flat_logits = heatmap.reshape(B, -1)
        cursor_dist = Categorical(logits=flat_logits)
        cursor_log_prob = cursor_dist.log_prob(cursor_flat_ids)
        cursor_entropy = cursor_dist.entropy()

        return {
            "tool_log_prob": tool_log_prob,
            "cursor_log_prob": cursor_log_prob,
            "value": out["value"].squeeze(-1),
            "tool_entropy": tool_entropy,
            "cursor_entropy": cursor_entropy,
        }

    # ─── Extensibility ──────────────────────────────────────────────────

    def extend_tools(self, new_num_tools: int):
        """
        Grow the tool head to accommodate new tools.
        Preserves existing weights for all existing tools.
        """
        old_num = self.tool_head.tool_logits.out_features
        if new_num_tools <= old_num:
            return

        old_weight = self.tool_head.tool_logits.weight.data
        old_bias = self.tool_head.tool_logits.bias.data

        self.tool_head.tool_logits = nn.Linear(
            self.tool_head.tool_logits.in_features, new_num_tools
        )

        # Copy old weights
        with torch.no_grad():
            self.tool_head.tool_logits.weight.data[:old_num] = old_weight
            self.tool_head.tool_logits.bias.data[:old_num] = old_bias
            # Initialize new weights with small random values
            nn.init.xavier_uniform_(self.tool_head.tool_logits.weight.data[old_num:])
            self.tool_head.tool_logits.bias.data[old_num:] = 0.0

    def save_checkpoint(self, path: str, extra_meta: Dict[str, Any] | None = None):
        """Save model with metadata for compatibility checking."""
        meta = {
            "tool_list": self._tool_list,
            "num_tools": len(self._tool_list),
            "config_model": self.config["model"],
            "in_channels": self.in_channels,
        }
        if extra_meta:
            meta.update(extra_meta)

        torch.save({
            "model_state_dict": self.state_dict(),
            "metadata": meta,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, config: Dict[str, Any] | None = None) -> "CADAgent":
        """
        Load model from checkpoint, handling tool list changes gracefully.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint.get("metadata", {})

        cfg = config or load_config()
        agent = cls(cfg)

        # Handle tool list extension
        saved_num_tools = meta.get("num_tools", agent.tool_head.tool_logits.out_features)
        current_num_tools = num_tools()

        if current_num_tools > saved_num_tools:
            # Load old weights first, then extend
            # Temporarily resize to match saved
            agent.tool_head.tool_logits = nn.Linear(
                agent.tool_head.tool_logits.in_features, saved_num_tools
            )
            agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
            agent.extend_tools(current_num_tools)
        else:
            agent.load_state_dict(checkpoint["model_state_dict"], strict=False)

        agent._tool_list = tool_list()
        return agent
