"""
Supervised single-step DrawPoint task.

Generates a simple prompt to draw a point at a random location.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import PointEntity

_POINT_PROMPTS = [
    "Draw a point here",
    "Place a point at the cursor",
    "Add a point at this location",
    "Create a point entity",
]

class DrawPointTask:
    """
    Single-step DrawPoint supervised task.

    Agent must use POINT tool and click the target location.
    """

    tool_name = "POINT"
    cursor_loss_weight = 1.0  # cursor matters here

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target: np.ndarray | None = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._target = np.array([
            float(self.rng.uniform(100, 900)),
            float(self.rng.uniform(100, 900))
        ])

        prompt = _POINT_PROMPTS[int(self.rng.randint(len(_POINT_PROMPTS)))]

        return {
            "prompt": prompt,
            "target": self._target,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": self.tool_name,
            "cursor_world": self._target,
            "cursor_weight": self.cursor_loss_weight,
        }
