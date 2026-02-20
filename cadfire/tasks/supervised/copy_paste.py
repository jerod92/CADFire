"""
Supervised single-step COPY task (paste object to target location).

Setup: one shape is selected on the canvas.  A ghost/marker shows the
       target destination (rendered into reference image channels).
Agent must: use COPY tool, cursor at the destination centroid.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    CircleEntity, RectangleEntity, PolygonEntity, EllipseEntity, Entity,
)

_COPY_PROMPTS = [
    "Copy the {shape} to the marked location",
    "Paste a copy of the {shape} at the target",
    "Duplicate the {shape} to the destination",
    "Copy and paste the {shape}",
    "Make a copy of the {shape} at the indicated position",
    "Copy the selected {shape}",
]

_SHAPE_NAMES = ["circle", "rectangle", "triangle", "hexagon", "ellipse"]


def _make_shape(rng, kind, cx, cy):
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
    elif kind == 2:
        r = float(rng.uniform(60, 110))
        return PolygonEntity(
            center=np.array([cx, cy]), radius=r, sides=3, color_index=color
        ), "triangle"
    elif kind == 3:
        r = float(rng.uniform(60, 110))
        return PolygonEntity(
            center=np.array([cx, cy]), radius=r, sides=6, color_index=color
        ), "hexagon"
    else:
        a = float(rng.uniform(60, 120))
        b = float(rng.uniform(30, 60))
        return EllipseEntity(
            center=np.array([cx, cy]), semi_major=a, semi_minor=b, color_index=color
        ), "ellipse"


class CopyObjectTask:
    """
    Single-step COPY supervised task.

    The source shape is pre-selected.  A destination region is shown.
    Agent must use COPY and click the destination.
    """

    tool_name = "COPY"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._dest_world: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        kind = int(self.rng.randint(5))
        src_cx = float(self.rng.uniform(150, 400))
        src_cy = float(self.rng.uniform(150, 850))
        entity, shape_name = _make_shape(self.rng, kind, src_cx, src_cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        # Destination: opposite side of canvas
        dest_cx = float(self.rng.uniform(600, 850))
        dest_cy = float(self.rng.uniform(150, 850))
        self._dest_world = np.array([dest_cx, dest_cy], dtype=np.float64)

        template = _COPY_PROMPTS[int(self.rng.randint(len(_COPY_PROMPTS)))]
        prompt = template.format(shape=shape_name)

        return {
            "prompt": prompt,
            "source_entity": entity,
            "dest_world": self._dest_world,
            "shape_name": shape_name,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "COPY",
            "cursor_world": self._dest_world,
            "cursor_weight": self.cursor_loss_weight,
        }
