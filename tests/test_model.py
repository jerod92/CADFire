"""Tests for the model architecture."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cadfire.model.cad_agent import CADAgent
from cadfire.model.vision_encoder import VisionEncoder
from cadfire.model.text_encoder import TextEncoder
from cadfire.model.fusion import FusionBridge
from cadfire.model.action_heads import ToolHead, CursorHead
from cadfire.utils.config import load_config, num_tools

_cfg = load_config()
_RS = _cfg["canvas"]["render_width"]   # render size from config
_NC = 3 + 3 + _cfg["layers"]["max_layers"] + 1 + 4  # total image channels (incl. coord grids)


class TestVisionEncoder:
    def test_forward_shape(self):
        enc = VisionEncoder(in_channels=15, base_channels=16, fusion_dim=64)
        x = torch.randn(2, 15, _RS, _RS)
        skips, global_feat = enc(x)
        assert len(skips) == 4
        assert global_feat.shape == (2, 64)
        assert skips[0].shape[2:] == (_RS, _RS)
        assert skips[1].shape[2:] == (_RS // 2, _RS // 2)
        assert skips[2].shape[2:] == (_RS // 4, _RS // 4)
        assert skips[3].shape[2:] == (_RS // 8, _RS // 8)


class TestTextEncoder:
    def test_forward_shape(self):
        enc = TextEncoder(vocab_size=256, embed_dim=32, hidden_dim=64, fusion_dim=64)
        ids = torch.randint(0, 256, (2, 128))
        feat = enc(ids)
        assert feat.shape == (2, 64)


class TestFusionBridge:
    def test_forward_shape(self):
        fusion = FusionBridge(fusion_dim=64, state_dim=16)
        vision = torch.randn(2, 64)
        text = torch.randn(2, 64)
        state = torch.randn(2, 16)
        out = fusion(vision, text, state)
        assert out.shape == (2, 64)


class TestToolHead:
    def test_forward_shape(self):
        head = ToolHead(fusion_dim=64, num_tools=55)
        x = torch.randn(2, 64)
        out = head(x)
        assert out["tool_logits"].shape == (2, 55)
        assert out["value"].shape == (2, 1)
        assert out["param"].shape == (2, 1)


class TestCursorHead:
    def test_forward_shape(self):
        skip_ch = [16, 32, 64, 128]
        head = CursorHead(skip_channels=skip_ch, fusion_dim=64)
        skips = [
            torch.randn(2, 16, _RS, _RS),
            torch.randn(2, 32, _RS // 2, _RS // 2),
            torch.randn(2, 64, _RS // 4, _RS // 4),
            torch.randn(2, 128, _RS // 8, _RS // 8),
        ]
        fused = torch.randn(2, 64)
        heatmap = head(skips, fused)
        assert heatmap.shape == (2, 1, _RS, _RS)


class TestCADAgent:
    def test_forward(self):
        agent = CADAgent()
        obs = {
            "image": torch.randn(2, _RS, _RS, _NC),
            "text_ids": torch.randint(0, 256, (2, 128)),
            "state_vec": torch.randn(2, 16),
        }
        out = agent(obs)
        assert "tool_logits" in out
        assert "cursor_heatmap" in out
        assert "value" in out
        assert out["tool_logits"].shape[0] == 2
        assert out["cursor_heatmap"].shape == (2, 1, _RS, _RS)

    def test_act(self):
        agent = CADAgent()
        obs = {
            "image": torch.randn(1, _RS, _RS, _NC),
            "text_ids": torch.randint(0, 256, (1, 128)),
            "state_vec": torch.randn(1, 16),
        }
        action = agent.act(obs, deterministic=True)
        assert "tool_id" in action
        assert "cursor" in action
        assert "value" in action
        assert action["cursor"].shape == (1, _RS, _RS)

    def test_extend_tools(self):
        agent = CADAgent()
        old_num = agent.tool_head.tool_logits.out_features
        agent.extend_tools(old_num + 5)
        assert agent.tool_head.tool_logits.out_features == old_num + 5

    def test_evaluate_actions(self):
        agent = CADAgent()
        obs = {
            "image": torch.randn(4, _RS, _RS, _NC),
            "text_ids": torch.randint(0, 256, (4, 128)),
            "state_vec": torch.randn(4, 16),
        }
        tool_ids = torch.randint(0, num_tools(), (4,))
        cursor_ids = torch.randint(0, _RS * _RS, (4,))
        result = agent.evaluate_actions(obs, tool_ids, cursor_ids)
        assert result["tool_log_prob"].shape == (4,)
        assert result["cursor_log_prob"].shape == (4,)
        assert result["value"].shape == (4,)
