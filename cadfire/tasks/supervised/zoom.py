"""
Supervised single-step ZOOM_IN / ZOOM_OUT tasks.

Setup: a shape exists somewhere on the canvas.
Oracle: tool = ZOOM_IN or ZOOM_OUT, cursor = viewport center (cursor
        position doesn't matter for zoom; the model must learn the tool).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import CircleEntity, RectangleEntity, PolygonEntity

_ZOOM_IN_PROMPTS = [
    "Zoom in",
    "Zoom in closer",
    "Increase zoom",
    "Make it larger",
    "Magnify the view",
    "Zoom in on the canvas",
]

_ZOOM_OUT_PROMPTS = [
    "Zoom out",
    "Zoom out further",
    "Decrease zoom",
    "Make it smaller",
    "Shrink the view",
    "Zoom out to see more",
]


def _random_shape(rng, cx, cy):
    kind = int(rng.randint(3))
    color = int(rng.randint(0, 8))
    if kind == 0:
        r = float(rng.uniform(50, 120))
        return CircleEntity(center=np.array([cx, cy]), radius=r, color_index=color)
    elif kind == 1:
        w = float(rng.uniform(80, 200))
        h = float(rng.uniform(80, 200))
        return RectangleEntity(
            corner=np.array([cx - w / 2, cy - h / 2]),
            width=w, height=h, color_index=color,
        )
    else:
        r = float(rng.uniform(60, 120))
        sides = int(rng.randint(3, 9))
        return PolygonEntity(
            center=np.array([cx, cy]), radius=r, sides=sides, color_index=color
        )


class ZoomInTask:
    """Single-step ZOOM_IN supervised task."""

    tool_name = "ZOOM_IN"
    cursor_loss_weight = 0.1  # cursor irrelevant

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        entity = _random_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)

        prompt = _ZOOM_IN_PROMPTS[int(self.rng.randint(len(_ZOOM_IN_PROMPTS)))]
        return {"prompt": prompt}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        # Cursor at viewport center (doesn't matter)
        cx = engine.viewport.center.copy()
        return {
            "tool": "ZOOM_IN",
            "cursor_world": cx,
            "cursor_weight": self.cursor_loss_weight,
        }


class ZoomOutTask:
    """Single-step ZOOM_OUT supervised task."""

    tool_name = "ZOOM_OUT"
    cursor_loss_weight = 0.1  # cursor irrelevant

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        entity = _random_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        # Start zoomed in so zooming out makes sense
        engine.zoom_in()
        engine.zoom_in()

        prompt = _ZOOM_OUT_PROMPTS[int(self.rng.randint(len(_ZOOM_OUT_PROMPTS)))]
        return {"prompt": prompt}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        cx = engine.viewport.center.copy()
        return {
            "tool": "ZOOM_OUT",
            "cursor_world": cx,
            "cursor_weight": self.cursor_loss_weight,
        }
