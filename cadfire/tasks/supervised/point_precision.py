"""
Supervised single-step point precision tasks.

These tasks require the agent to select highly specific geometric points:
Triangle vertices, line midpoints, and circle centers.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import PolygonEntity, LineEntity, CircleEntity


_TRIANGLE_PROMPTS = [
    "Select the top point of the triangle",
    "Click the top vertex of the triangle",
    "Identify the upper vertex of this triangle",
]

_MIDPOINT_PROMPTS = [
    "Select the midpoint of the line segment",
    "Click the exact center of the line",
    "Identify the midpoint of this line",
]

_CENTER_PROMPTS = [
    "Select the center of the circle",
    "Click the exact center point of the circle",
    "Identify the circle's center",
]


class TriangleVertexTask:
    """
    Agent must use SELECT (or POINT) to click the highest vertex of a randomly
    rotated/scaled triangle.
    """

    tool_name = "SELECT"
    cursor_loss_weight = 2.0  # high penalty for imprecision

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target: np.ndarray | None = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        radius = float(self.rng.uniform(50, 250))
        rotation = float(self.rng.uniform(0, 360))

        triangle = PolygonEntity(
            center=np.array([cx, cy]),
            radius=radius,
            sides=3,
            rotation=rotation,
            color_index=int(self.rng.randint(1, 8))
        )
        engine.add_entity(triangle, save_undo=False)

        # The vertices are generated internally via tessellate() / _vertices()
        verts = triangle._vertices()
        
        # Find the "top" vertex (minimum Y in image space = maximum Y in world space, 
        # or whichever convention we use. Our world space has Y going up, so max Y is top)
        # Wait, usually world space Y goes up. Let's assume max Y is top.
        top_idx = np.argmax(verts[:, 1])
        self._target = verts[top_idx].copy()

        prompt = _TRIANGLE_PROMPTS[int(self.rng.randint(len(_TRIANGLE_PROMPTS)))]

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


class LineMidpointTask:
    """
    Agent must use SELECT to click the midpoint of a line segment.
    """

    tool_name = "SELECT"
    cursor_loss_weight = 2.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target: np.ndarray | None = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        p1 = np.array([float(self.rng.uniform(100, 900)), float(self.rng.uniform(100, 900))])
        p2 = np.array([float(self.rng.uniform(100, 900)), float(self.rng.uniform(100, 900))])
        
        # ensure it's not too short
        while np.linalg.norm(p1 - p2) < 50:
            p2 = np.array([float(self.rng.uniform(100, 900)), float(self.rng.uniform(100, 900))])

        line = LineEntity(start=p1, end=p2, color_index=int(self.rng.randint(1, 8)))
        engine.add_entity(line, save_undo=False)

        self._target = (p1 + p2) / 2.0

        prompt = _MIDPOINT_PROMPTS[int(self.rng.randint(len(_MIDPOINT_PROMPTS)))]

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


class CircleCenterTask:
    """
    Agent must use SELECT to click the center of a circle.
    """

    tool_name = "SELECT"
    cursor_loss_weight = 2.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target: np.ndarray | None = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        radius = float(self.rng.uniform(50, 300))

        circle = CircleEntity(
            center=np.array([cx, cy]),
            radius=radius,
            color_index=int(self.rng.randint(1, 8))
        )
        engine.add_entity(circle, save_undo=False)

        self._target = np.array([cx, cy])

        prompt = _CENTER_PROMPTS[int(self.rng.randint(len(_CENTER_PROMPTS)))]

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
