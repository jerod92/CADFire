"""
Supervised single-step PAN tasks (up / down / left / right).

Setup: a shape is placed near one edge of the viewport (partially visible
       or just outside).  The prompt instructs the agent to pan in a
       direction so the shape becomes centered.

Oracle: tool = PAN, cursor = target world location we want to pan toward.
        (In practice the cursor head learns to point toward the off-screen
        region, which is the key visual cue for "pan here".)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import CircleEntity, RectangleEntity

_PAN_PROMPTS = {
    "up":    ["Pan up", "Scroll up", "Move the view up", "Pan the canvas upward",
              "Shift the viewport up"],
    "down":  ["Pan down", "Scroll down", "Move the view down", "Pan the canvas downward",
              "Shift the viewport down"],
    "left":  ["Pan left", "Scroll left", "Move the view left", "Pan the canvas to the left",
              "Shift the viewport left"],
    "right": ["Pan right", "Scroll right", "Move the view right", "Pan the canvas right",
              "Shift the viewport right"],
}

_DIRECTIONS = ["up", "down", "left", "right"]


class PanTask:
    """
    Single-step PAN supervised task.

    A shape is placed at the far side of the world (near the edge the
    agent should pan toward).  The cursor target is the world center of
    that shape so the cursor head learns which direction is meant.
    """

    tool_name = "PAN"
    cursor_loss_weight = 0.5  # cursor points toward the target region

    def __init__(self, seed: int | None = None, direction: str | None = None):
        self.rng = np.random.RandomState(seed)
        self._direction: str = direction or _DIRECTIONS[
            int(self.rng.randint(len(_DIRECTIONS)))
        ]
        self._target_world: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        world_w = engine.config["canvas"]["world_width"]
        world_h = engine.config["canvas"]["world_height"]

        # Place a circle near the edge in the pan direction
        margin = 80.0
        cx, cy = 500.0, 500.0
        if self._direction == "up":
            cy = world_h - margin
        elif self._direction == "down":
            cy = margin
        elif self._direction == "left":
            cx = margin
        elif self._direction == "right":
            cx = world_w - margin

        r = float(self.rng.uniform(40, 70))
        color = int(self.rng.randint(0, 8))
        entity = CircleEntity(
            center=np.array([cx, cy]), radius=r, color_index=color
        )
        engine.add_entity(entity, save_undo=False)
        self._target_world = np.array([cx, cy], dtype=np.float64)

        # Viewport starts centered at canvas center
        prompts = _PAN_PROMPTS[self._direction]
        prompt = prompts[int(self.rng.randint(len(prompts)))]

        return {
            "prompt": prompt,
            "direction": self._direction,
            "target_world": self._target_world,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "PAN",
            "cursor_world": self._target_world,
            "cursor_weight": self.cursor_loss_weight,
        }
