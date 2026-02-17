"""Tests for the RL environment."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cadfire.env.cad_env import CADEnv
from cadfire.tasks.registry import TaskRegistry
from cadfire.utils.config import tool_to_index


class TestCADEnv:
    def setup_method(self):
        TaskRegistry.discover()
        self.env = CADEnv()

    def test_reset_no_task(self):
        obs, info = self.env.reset()
        assert "image" in obs
        assert "text_ids" in obs
        assert "state_vec" in obs
        assert obs["image"].shape == self.env.image_shape

    def test_reset_with_task(self):
        task = TaskRegistry.create("draw_circle", seed=42)
        obs, info = self.env.reset(task=task)
        assert "prompt" in info
        assert len(info["prompt"]) > 0

    def test_step_noop(self):
        self.env.reset()
        tool_idx = tool_to_index()["NOOP"]
        action = {"tool_id": tool_idx, "cursor": None, "param": 0.0}
        obs, reward, term, trunc, info = self.env.step(action)
        assert not term
        assert obs["image"].shape == self.env.image_shape

    def test_step_draw_line(self):
        self.env.reset()
        tool_idx = tool_to_index()["LINE"]
        H, W = self.env.cursor_shape

        # First click
        cursor1 = np.zeros((H, W), dtype=np.float32)
        cursor1[32, 32] = 1.0
        action1 = {"tool_id": tool_idx, "cursor": cursor1, "param": 0.0}
        self.env.step(action1)

        # Second click
        cursor2 = np.zeros((H, W), dtype=np.float32)
        cursor2[96, 96] = 1.0
        action2 = {"tool_id": tool_idx, "cursor": cursor2, "param": 0.0}
        self.env.step(action2)

        assert self.env.entity_count() == 1

    def test_step_with_task_reward(self):
        task = TaskRegistry.create("fit_view", seed=42)
        obs, info = self.env.reset(task=task)
        tool_idx = tool_to_index()["FIT_VIEW"]
        action = {"tool_id": tool_idx, "cursor": None, "param": 0.0}
        obs, reward, term, trunc, info = self.env.step(action)
        # Should get some reward for fitting view
        assert isinstance(reward, float)

    def test_truncation(self):
        self.env.reset()
        tool_idx = tool_to_index()["NOOP"]
        for _ in range(self.env.max_episode_steps):
            action = {"tool_id": tool_idx, "cursor": None, "param": 0.0}
            obs, reward, term, trunc, info = self.env.step(action)
        assert trunc

    def test_state_vector_shape(self):
        obs, _ = self.env.reset()
        assert obs["state_vec"].shape == (self.env.state_dim,)

    def test_undo_redo(self):
        self.env.reset()
        # Draw something
        tool_idx = tool_to_index()["POINT"]
        H, W = self.env.cursor_shape
        cursor = np.zeros((H, W), dtype=np.float32)
        cursor[64, 64] = 1.0
        self.env.step({"tool_id": tool_idx, "cursor": cursor, "param": 0.0})
        assert self.env.entity_count() == 1

        # Undo
        undo_idx = tool_to_index()["UNDO"]
        self.env.step({"tool_id": undo_idx, "cursor": None, "param": 0.0})
        assert self.env.entity_count() == 0

        # Redo
        redo_idx = tool_to_index()["REDO"]
        self.env.step({"tool_id": redo_idx, "cursor": None, "param": 0.0})
        assert self.env.entity_count() == 1
