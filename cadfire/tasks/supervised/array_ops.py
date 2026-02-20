"""
Supervised single-step tasks for array-creation tools:

  ArrayRectTask  – ARRAY_RECT : create a rectangular grid of copies
  ArrayPolarTask – ARRAY_POLAR: create a circular array of copies around a centre

Both teach the model the correct spatial pick-point for each array variant:

  ARRAY_RECT  – cursor at the far-corner of the intended array footprint.
                This mirrors how most CAD packages define a rectangular array
                (source entity + destination corner → implicit step size).
                cursor_loss_weight = 1.0  (the far-corner click is the primary
                signal; getting it wrong ruins the array layout).

  ARRAY_POLAR – cursor at the rotation centre.
                The entity is pre-selected and offset from centre; the agent
                must click the pivot around which copies should be rotated.
                cursor_loss_weight = 1.0  (the centre must be accurate).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    CircleEntity, RectangleEntity, PolygonEntity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared shape factory (reused from transform_extra pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _make_small_shape(rng: np.random.RandomState, cx: float, cy: float):
    """Return (entity, shape_name) — a small entity centred at (cx, cy)."""
    kind = int(rng.randint(3))
    color = int(rng.randint(0, 8))
    if kind == 0:
        r = float(rng.uniform(25, 55))
        return CircleEntity(
            center=np.array([cx, cy]), radius=r, color_index=color,
        ), "circle"
    elif kind == 1:
        w = float(rng.uniform(40, 90))
        h = float(rng.uniform(40, 90))
        return RectangleEntity(
            corner=np.array([cx - w / 2, cy - h / 2]),
            width=w, height=h, color_index=color,
        ), "rectangle"
    else:
        sides = int(rng.choice([3, 5, 6]))
        r = float(rng.uniform(30, 60))
        name = {3: "triangle", 5: "pentagon", 6: "hexagon"}[sides]
        return PolygonEntity(
            center=np.array([cx, cy]), radius=r, sides=sides,
            color_index=color,
        ), name


# ─────────────────────────────────────────────────────────────────────────────
# ArrayRectTask
# ─────────────────────────────────────────────────────────────────────────────

_ARRAY_RECT_PROMPTS = [
    "Create a {rows}×{cols} rectangular array of the {shape}",
    "Make a {rows} by {cols} grid of copies of the {shape}",
    "Array the {shape} into a {rows}×{cols} rectangular pattern",
    "Repeat the {shape} in a {rows}-row {cols}-column grid",
    "Create a rectangular array of the {shape}",
    "Make a grid array of the {shape}",
    "Array the {shape} in a rectangular pattern",
    "Duplicate the {shape} into a grid",
    "Create a {rows}×{cols} array",
    "Make a rectangular array",
    "Array it",
]


class ArrayRectTask:
    """
    Single-step ARRAY_RECT supervised task.

    A small shape is placed near the top-left region of the canvas and
    pre-selected.  The oracle cursor points to the far corner of the intended
    array footprint — i.e. the position that the bottom-right copy would
    occupy.  This mirrors the standard CAD paradigm where the user picks
    the source entity and then the total extent of the array.

    cursor_loss_weight = 1.0: the far-corner click fully defines the array
    layout, so it is the primary training signal.
    """

    tool_name = "ARRAY_RECT"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._far_corner: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Source entity: upper-left quadrant
        src_cx = float(self.rng.uniform(100, 250))
        src_cy = float(self.rng.uniform(100, 250))
        entity, shape_name = _make_small_shape(self.rng, src_cx, src_cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        # Array parameters
        rows = int(self.rng.randint(2, 5))   # 2–4 rows
        cols = int(self.rng.randint(2, 5))   # 2–4 cols
        step_x = float(self.rng.uniform(100, 180))
        step_y = float(self.rng.uniform(100, 180))

        # Far corner = position of the bottom-right copy's centre
        far_cx = src_cx + (cols - 1) * step_x
        far_cy = src_cy + (rows - 1) * step_y
        self._far_corner = np.clip(
            np.array([far_cx, far_cy], dtype=np.float64), 50.0, 950.0
        )

        tmpl = _ARRAY_RECT_PROMPTS[int(self.rng.randint(len(_ARRAY_RECT_PROMPTS)))]
        prompt = tmpl.format(shape=shape_name, rows=rows, cols=cols)

        return {
            "prompt": prompt,
            "entity": entity,
            "shape_name": shape_name,
            "rows": rows,
            "cols": cols,
            "step_x": step_x,
            "step_y": step_y,
            "far_corner": self._far_corner,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "ARRAY_RECT",
            "cursor_world": self._far_corner,
            "cursor_weight": self.cursor_loss_weight,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ArrayPolarTask
# ─────────────────────────────────────────────────────────────────────────────

_ARRAY_POLAR_PROMPTS = [
    "Create a polar array of {n} copies of the {shape} around the centre",
    "Arrange {n} copies of the {shape} in a circular pattern",
    "Make a {n}-item circular array of the {shape}",
    "Array the {shape} {n} times around the rotation centre",
    "Create a {n}-copy polar array",
    "Distribute {n} copies of the {shape} evenly in a circle",
    "Make a polar array of the {shape}",
    "Create a circular array",
    "Array the {shape} around the marked centre",
    "Polar array it",
    "Create a rotational array of the {shape}",
]


class ArrayPolarTask:
    """
    Single-step ARRAY_POLAR supervised task.

    A small shape is placed at a random offset from the canvas centre.  The
    entity is pre-selected.  The oracle cursor points at the rotation centre
    (canvas centre ± small jitter) — the pivot around which the copies should
    be evenly distributed.

    cursor_loss_weight = 1.0: getting the rotation centre right is the entire
    spatial challenge of a polar array.
    """

    tool_name = "ARRAY_POLAR"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._rotation_centre: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Rotation centre: near the canvas centre with small jitter
        rc_x = float(self.rng.uniform(380, 620))
        rc_y = float(self.rng.uniform(380, 620))
        self._rotation_centre = np.array([rc_x, rc_y], dtype=np.float64)

        # Place source entity at a random radius from the centre
        radius = float(self.rng.uniform(180, 320))
        angle = float(self.rng.uniform(0, 2 * np.pi))
        src_cx = rc_x + radius * np.cos(angle)
        src_cy = rc_y + radius * np.sin(angle)
        src_cx = float(np.clip(src_cx, 80.0, 920.0))
        src_cy = float(np.clip(src_cy, 80.0, 920.0))

        entity, shape_name = _make_small_shape(self.rng, src_cx, src_cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        n_copies = int(self.rng.randint(3, 9))   # 3–8 copies

        tmpl = _ARRAY_POLAR_PROMPTS[int(self.rng.randint(len(_ARRAY_POLAR_PROMPTS)))]
        prompt = tmpl.format(shape=shape_name, n=n_copies)

        return {
            "prompt": prompt,
            "entity": entity,
            "shape_name": shape_name,
            "n_copies": n_copies,
            "rotation_centre": self._rotation_centre,
            "radius": radius,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "ARRAY_POLAR",
            "cursor_world": self._rotation_centre,
            "cursor_weight": self.cursor_loss_weight,
        }
