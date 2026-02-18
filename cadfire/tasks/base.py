"""
BaseTask: the contract all tasks must follow.

Every RL training task inherits from BaseTask and implements:
  1. setup(engine) -> dict: Set up initial engine state, return prompt + reference image
  2. compute_reward(engine, action, step) -> dict: Compute reward for the current step
  3. generate_target(engine) -> dict: Generate the ideal target state (for evaluation)

This is the most important extensibility point. Adding a new task should ONLY require:
  - Creating a new file in cadfire/tasks/
  - Inheriting from BaseTask
  - Using @register_task decorator
  - Implementing setup() and compute_reward()

The training loop discovers tasks via the registry; no other code changes needed.
"""

from __future__ import annotations

import abc
import random
from typing import Any, Dict, List, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine

# Tools every task should always allow
UTILITY_TOOLS = ["NOOP", "CONFIRM", "CANCEL", "UNDO", "REDO"]


class BaseTask(abc.ABC):
    """
    Abstract base class for all RL training tasks.

    Subclasses control:
      - What the initial canvas looks like
      - What prompt the agent receives
      - What (optional) reference image to provide
      - How reward is computed each step
      - When the episode terminates (success condition)
    """

    # Class-level metadata (set by subclass or decorator)
    task_name: str = "base"
    task_category: str = "misc"
    difficulty: float = 1.0  # 0-10 scale, used for curriculum

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    @abc.abstractmethod
    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        """
        Initialize the CAD engine for this task episode.

        Must return a dict with at least:
          - "prompt": str - the text instruction
        Optionally:
          - "reference_image": np.ndarray (H, W, 3) - reference image for tracing
          - "target_entities": list - target entity dicts for evaluation
          - "metadata": dict - any additional info

        The engine is already reset before this is called.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_reward(self, engine: CADEngine, action: Dict[str, Any],
                       step: int) -> Dict[str, Any]:
        """
        Compute the reward for the current step.

        Must return a dict with:
          - "reward": float
          - "terminated": bool - True if task is completed (success or failure)
        Optionally:
          - "info": dict - diagnostic information
        """
        raise NotImplementedError

    def generate_prompt_variants(self) -> List[str]:
        """
        Return a list of prompt templates for lexical variation.
        Subclasses should override this for richer training.
        Default returns a single template.
        """
        return [f"Complete the {self.task_name} task."]

    def allowed_tools(self) -> Optional[List[str]]:
        """Return list of tool names the agent may use for this task.

        Return ``None`` to allow all tools (default).  Subclasses should
        override to restrict the action space for faster learning.
        The returned list is automatically unioned with ``UTILITY_TOOLS``.
        """
        return None

    def sample_prompt(self) -> str:
        """Sample a random prompt variant."""
        variants = self.generate_prompt_variants()
        return variants[self.rng.randint(len(variants))]

    # ─── Reward Helpers (common patterns) ──────────────────────────────

    @staticmethod
    def iou_reward(entities: List, targets: List,
                   render_size: int = 128) -> float:
        """
        Compute IoU between rendered entities and target entities.
        Uses rasterized binary masks for comparison.
        """
        if not entities or not targets:
            return 0.0

        mask_a = np.zeros((render_size, render_size), dtype=bool)
        mask_b = np.zeros((render_size, render_size), dtype=bool)

        for e in entities:
            pts = e.tessellate()
            if len(pts) == 0:
                continue
            # Normalize to [0, render_size]
            px = np.clip((pts[:, 0] / 1000.0 * render_size).astype(int), 0, render_size - 1)
            py = np.clip((pts[:, 1] / 1000.0 * render_size).astype(int), 0, render_size - 1)
            mask_a[py, px] = True

        for e in targets:
            pts = e.tessellate()
            if len(pts) == 0:
                continue
            px = np.clip((pts[:, 0] / 1000.0 * render_size).astype(int), 0, render_size - 1)
            py = np.clip((pts[:, 1] / 1000.0 * render_size).astype(int), 0, render_size - 1)
            mask_b[py, px] = True

        intersection = np.sum(mask_a & mask_b)
        union = np.sum(mask_a | mask_b)
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    @staticmethod
    def entity_count_reward(current: int, target: int) -> float:
        """Reward that peaks when entity count matches target."""
        if target == 0:
            return 1.0 if current == 0 else 0.0
        return max(0.0, 1.0 - abs(current - target) / max(target, 1))

    @staticmethod
    def bbox_occupancy_reward(engine: CADEngine) -> float:
        """Reward based on how well entities fill the viewport (for FIT_VIEW)."""
        if not engine.entities:
            return 0.0

        all_min = np.array([np.inf, np.inf])
        all_max = np.array([-np.inf, -np.inf])
        for e in engine.entities:
            bb_min, bb_max = e.bbox()
            all_min = np.minimum(all_min, bb_min)
            all_max = np.maximum(all_max, bb_max)

        if np.any(np.isinf(all_min)):
            return 0.0

        vis_min, vis_max = engine.viewport.visible_bounds()
        vis_area = max(np.prod(vis_max - vis_min), 1e-6)
        entity_area = max(np.prod(all_max - all_min), 1e-6)

        # Optimal: entity bbox fills 70-90% of viewport
        ratio = entity_area / vis_area
        if ratio < 0.1:
            return ratio  # too zoomed out
        elif ratio > 1.5:
            return max(0, 1.0 - (ratio - 1.0))  # too zoomed in
        else:
            # Peak around 0.7-0.9
            return 1.0 - abs(ratio - 0.8) * 2

    @staticmethod
    def selection_reward(engine: CADEngine, target_ids: set) -> float:
        """Reward for selecting exactly the right entities."""
        if not target_ids:
            return 1.0 if not engine.selected_ids else 0.0
        correct = engine.selected_ids & target_ids
        precision = len(correct) / max(len(engine.selected_ids), 1)
        recall = len(correct) / len(target_ids)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1
