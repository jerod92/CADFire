"""Tests for the task system and registry."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cadfire.engine.cad_engine import CADEngine
from cadfire.tasks.base import BaseTask
from cadfire.tasks.registry import TaskRegistry, register_task


class TestTaskRegistry:
    def setup_method(self):
        TaskRegistry.discover()

    def test_discover(self):
        assert TaskRegistry.count() > 0

    def test_list_tasks(self):
        tasks = TaskRegistry.list_tasks()
        assert "draw_line" in tasks
        assert "draw_circle" in tasks
        assert "fit_view" in tasks
        assert "select_shape" in tasks

    def test_create_task(self):
        task = TaskRegistry.create("draw_circle", seed=42)
        assert task.task_name == "draw_circle"

    def test_sample(self):
        task = TaskRegistry.sample(seed=42)
        assert isinstance(task, BaseTask)

    def test_sample_by_category(self):
        task = TaskRegistry.sample(category="draw", seed=42)
        assert task.task_category == "draw"

    def test_sample_by_difficulty(self):
        task = TaskRegistry.sample(max_difficulty=2.0, seed=42)
        assert task.difficulty <= 2.0

    def test_list_by_category(self):
        draw_tasks = TaskRegistry.list_by_category("draw")
        assert len(draw_tasks) > 0
        for name in draw_tasks:
            assert TaskRegistry.get(name).task_category == "draw"


class TestDrawTasks:
    def test_draw_line_setup(self):
        engine = CADEngine()
        task = TaskRegistry.create("draw_line", seed=42)
        setup_info = task.setup(engine)
        assert "prompt" in setup_info
        assert len(setup_info["prompt"]) > 0

    def test_draw_circle_reward(self):
        engine = CADEngine()
        task = TaskRegistry.create("draw_circle", seed=42)
        task.setup(engine)

        # No entities yet -> negative/zero reward
        result = task.compute_reward(engine, {}, 1)
        assert result["reward"] <= 0

    def test_draw_circle_success(self):
        engine = CADEngine()
        task = TaskRegistry.create("draw_circle", seed=42)
        setup_info = task.setup(engine)

        # Draw the target circle exactly
        target = setup_info["target_entities"][0]
        engine.draw_circle(target.center, target.radius)

        result = task.compute_reward(engine, {}, 1)
        assert result["reward"] > 0.5

    def test_fit_view_task(self):
        engine = CADEngine()
        task = TaskRegistry.create("fit_view", seed=42)
        task.setup(engine)

        # Before fit: possibly bad occupancy
        r1 = task.compute_reward(engine, {}, 1)

        # After fit
        engine.fit_view()
        r2 = task.compute_reward(engine, {}, 2)
        assert r2["reward"] >= r1["reward"]


class TestCustomTask:
    def test_register_custom_task(self):
        @register_task
        class TestRotateTask(BaseTask):
            task_name = "test_rotate_custom"
            task_category = "modify"
            difficulty = 4.0

            def setup(self, engine):
                return {"prompt": "Rotate the shape 45 degrees"}

            def compute_reward(self, engine, action, step):
                return {"reward": 0.5, "terminated": False}

        assert "test_rotate_custom" in TaskRegistry.list_tasks()
        task = TaskRegistry.create("test_rotate_custom")
        engine = CADEngine()
        info = task.setup(engine)
        assert info["prompt"] == "Rotate the shape 45 degrees"
        result = task.compute_reward(engine, {}, 1)
        assert result["reward"] == 0.5
