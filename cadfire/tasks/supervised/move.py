"""
Supervised single-step MOVE task (drag object to target area).

Setup: a shape is selected.  The reference image channels show a target
       marker (cross-hair region) at the destination.
Agent must: use MOVE, cursor at the destination world position.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    CircleEntity, RectangleEntity, PolygonEntity, Entity,
)

_MOVE_PROMPTS = [
    "Move the {shape} to the marked location",
    "Drag the {shape} to the target area",
    "Relocate the {shape} to the destination",
    "Move the selected {shape}",
    "Drag the {shape} to the indicated position",
    "Shift the {shape} to the marker",
]


def _make_shape(rng, cx, cy):
    kind = int(rng.randint(3))
    color = int(rng.randint(0, 8))
    if kind == 0:
        r = float(rng.uniform(50, 100))
        return CircleEntity(center=np.array([cx, cy]), radius=r, color_index=color), "circle"
    elif kind == 1:
        w = float(rng.uniform(80, 160))
        h = float(rng.uniform(80, 160))
        return RectangleEntity(
            corner=np.array([cx - w / 2, cy - h / 2]),
            width=w, height=h, color_index=color,
        ), "rectangle"
    else:
        r = float(rng.uniform(60, 110))
        sides = int(rng.randint(3, 7))
        name = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon"}.get(sides, "polygon")
        return PolygonEntity(
            center=np.array([cx, cy]), radius=r, sides=sides, color_index=color
        ), name


class MoveObjectTask:
    """
    Single-step MOVE supervised task.

    The source entity is pre-selected.  Agent learns to use MOVE and aim
    cursor at the destination centroid.
    """

    tool_name = "MOVE"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._dest_world: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        src_cx = float(self.rng.uniform(150, 400))
        src_cy = float(self.rng.uniform(150, 850))
        entity, shape_name = _make_shape(self.rng, src_cx, src_cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        # Destination
        dest_cx = float(self.rng.uniform(600, 850))
        dest_cy = float(self.rng.uniform(150, 850))
        self._dest_world = np.array([dest_cx, dest_cy], dtype=np.float64)

        template = _MOVE_PROMPTS[int(self.rng.randint(len(_MOVE_PROMPTS)))]
        prompt = template.format(shape=shape_name)

        return {
            "prompt": prompt,
            "source_entity": entity,
            "dest_world": self._dest_world,
            "shape_name": shape_name,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "MOVE",
            "cursor_world": self._dest_world,
            "cursor_weight": self.cursor_loss_weight,
        }
