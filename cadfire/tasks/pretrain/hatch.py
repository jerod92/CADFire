"""
Supervised single-step HATCH task (infill object).

Setup: a closed shape (rectangle or polygon) is placed and selected.
       The prompt instructs the agent to fill/hatch it.
Oracle: tool = HATCH, cursor = interior centroid of the selected shape.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import RectangleEntity, PolygonEntity, Entity

_HATCH_PROMPTS = [
    "Fill the {shape}",
    "Hatch the {shape}",
    "Infill the selected {shape}",
    "Apply hatch to the {shape}",
    "Fill in the {shape}",
    "Use HATCH on the {shape}",
    "Shade the {shape}",
    "Fill the selected shape",
]

_SHAPE_NAMES = ["rectangle", "triangle", "pentagon", "hexagon", "octagon"]


class HatchObjectTask:
    """
    Single-step HATCH supervised task.

    A closed shape is pre-selected.  Agent must activate HATCH with cursor
    inside the shape.
    """

    tool_name = "HATCH"
    cursor_loss_weight = 0.8  # cursor inside shape matters

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target_entity: Optional[Entity] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        color = int(self.rng.randint(0, 8))

        kind = int(self.rng.randint(5))  # 0=rect, 1-4=polygon sides 3/5/6/8
        if kind == 0:
            w = float(self.rng.uniform(100, 250))
            h = float(self.rng.uniform(100, 250))
            entity = RectangleEntity(
                corner=np.array([cx - w / 2, cy - h / 2]),
                width=w, height=h, color_index=color,
            )
            shape_name = "rectangle"
        else:
            sides_map = {1: 3, 2: 5, 3: 6, 4: 8}
            sides = sides_map[kind]
            r = float(self.rng.uniform(70, 130))
            entity = PolygonEntity(
                center=np.array([cx, cy]), radius=r, sides=sides,
                color_index=color,
            )
            shape_name = _SHAPE_NAMES[kind]

        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._target_entity = entity

        template = _HATCH_PROMPTS[int(self.rng.randint(len(_HATCH_PROMPTS)))]
        prompt = template.format(shape=shape_name)

        return {
            "prompt": prompt,
            "target_entity": entity,
            "shape_name": shape_name,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "HATCH",
            "cursor_world": self._target_entity.centroid(),
            "cursor_weight": self.cursor_loss_weight,
        }
