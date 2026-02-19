"""
Supervised single-step ROTATE task.

Setup: a shape is selected and the engine state signals a rotation is
       needed.  The prompt specifies a rotation angle.
Agent must: use ROTATE, cursor at the rotation center (entity centroid
            or a specific pivot shown in the state).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    CircleEntity, RectangleEntity, PolygonEntity, Entity,
)

_ROTATE_PROMPTS = [
    "Rotate the {shape} by {angle} degrees",
    "Spin the {shape} {angle}°",
    "Rotate the selected {shape}",
    "Turn the {shape} clockwise",
    "Apply a {angle}-degree rotation to the {shape}",
    "Rotate the {shape}",
]


def _make_shape(rng, cx, cy):
    kind = int(rng.randint(3))
    color = int(rng.randint(0, 8))
    if kind == 0:
        w = float(rng.uniform(80, 200))
        h = float(rng.uniform(80, 200))
        return RectangleEntity(
            corner=np.array([cx - w / 2, cy - h / 2]),
            width=w, height=h, color_index=color,
        ), "rectangle"
    elif kind == 1:
        r = float(rng.uniform(60, 120))
        return PolygonEntity(
            center=np.array([cx, cy]), radius=r, sides=3, color_index=color
        ), "triangle"
    else:
        r = float(rng.uniform(60, 120))
        sides = int(rng.randint(4, 9))
        name = {4: "square", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"
                }.get(sides, "polygon")
        return PolygonEntity(
            center=np.array([cx, cy]), radius=r, sides=sides, color_index=color
        ), name


class RotateObjectTask:
    """
    Single-step ROTATE supervised task.

    Entity is pre-selected.  Cursor target is the entity centroid (the
    natural pivot point for rotation).
    """

    tool_name = "ROTATE"
    cursor_loss_weight = 0.8  # pivot point matters

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._pivot: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        entity, shape_name = _make_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        self._pivot = entity.centroid()
        angle = int(self.rng.choice([30, 45, 60, 90, 120, 135, 180]))

        template = _ROTATE_PROMPTS[int(self.rng.randint(len(_ROTATE_PROMPTS)))]
        prompt = template.format(shape=shape_name, angle=angle)

        return {
            "prompt": prompt,
            "entity": entity,
            "angle": angle,
            "pivot": self._pivot,
            "shape_name": shape_name,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "ROTATE",
            "cursor_world": self._pivot,
            "cursor_weight": self.cursor_loss_weight,
        }
