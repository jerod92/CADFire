"""
Additional single-step supervised tasks for transformation tools not yet
covered by the original eleven Phase-2 tasks.

  ScaleObjectTask  – SCALE: scale the selected shape from its centroid pivot
  MirrorObjectTask – MIRROR: reflect selected shape across a symmetry axis
  OffsetTask       – OFFSET: expand or shrink the outline of a closed shape
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    CircleEntity, PolygonEntity, RectangleEntity,
)


def _make_closed_shape(rng: np.random.RandomState, cx: float, cy: float):
    """Return (entity, name) for a randomly chosen closed shape."""
    kind = int(rng.randint(3))
    color = int(rng.randint(0, 8))
    if kind == 0:
        r = float(rng.uniform(60, 120))
        return CircleEntity(center=np.array([cx, cy]), radius=r,
                            color_index=color), "circle"
    elif kind == 1:
        w = float(rng.uniform(80, 180))
        h = float(rng.uniform(80, 180))
        return RectangleEntity(
            corner=np.array([cx - w / 2, cy - h / 2]),
            width=w, height=h, color_index=color,
        ), "rectangle"
    else:
        sides = int(rng.choice([3, 4, 5, 6, 8]))
        r = float(rng.uniform(60, 120))
        name = {3: "triangle", 4: "square", 5: "pentagon",
                6: "hexagon", 8: "octagon"}[sides]
        return PolygonEntity(center=np.array([cx, cy]), radius=r,
                             sides=sides, color_index=color), name


# ── ScaleObjectTask ───────────────────────────────────────────────────────────

_SCALE_PROMPTS = [
    "Scale the {shape} to {pct}%",
    "Resize the {shape} to {pct} percent of its current size",
    "Make the {shape} {pct}% of its size",
    "Scale the selected {shape}",
    "Resize the {shape}",
    "Make the {shape} larger",
    "Make the {shape} smaller",
    "Enlarge the {shape}",
    "Shrink the {shape}",
    "Scale it up",
    "Scale it down",
]


class ScaleObjectTask:
    """
    Single-step SCALE supervised task (no chat history).

    Entity is pre-selected.  Cursor target is the entity centroid — the
    natural scale pivot in CAD software.
    """

    tool_name = "SCALE"
    cursor_loss_weight = 0.8

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._pivot: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        entity, shape_name = _make_closed_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._pivot = entity.centroid()

        pct = int(self.rng.choice([50, 75, 125, 150, 200]))
        tmpl = _SCALE_PROMPTS[int(self.rng.randint(len(_SCALE_PROMPTS)))]
        prompt = tmpl.format(shape=shape_name, pct=pct)

        return {"prompt": prompt, "entity": entity,
                "scale_pct": pct, "shape_name": shape_name}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "SCALE",
                "cursor_world": self._pivot,
                "cursor_weight": self.cursor_loss_weight}


# ── MirrorObjectTask ──────────────────────────────────────────────────────────

_MIRROR_PROMPTS = [
    "Mirror the {shape} horizontally",
    "Flip the {shape} vertically",
    "Mirror the selected {shape}",
    "Reflect the {shape} across the vertical axis",
    "Reflect the {shape} across the horizontal axis",
    "Flip the {shape}",
    "Create a mirrored copy of the {shape}",
    "Mirror it",
    "Flip it",
]


class MirrorObjectTask:
    """
    Single-step MIRROR supervised task.

    Entity is pre-selected.  Cursor target is the entity centroid, which
    defines the default mirror-axis passing through it.  Cursor weight is
    moderate: axis direction matters more than sub-pixel accuracy.
    """

    tool_name = "MIRROR"
    cursor_loss_weight = 0.6

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._axis_pt: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(250, 750))
        cy = float(self.rng.uniform(250, 750))
        entity, shape_name = _make_closed_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        # Mirror axis midpoint = entity centroid (agent clicks the entity
        # to define the axis through it)
        self._axis_pt = entity.centroid().copy()

        tmpl = _MIRROR_PROMPTS[int(self.rng.randint(len(_MIRROR_PROMPTS)))]
        prompt = tmpl.format(shape=shape_name)

        return {"prompt": prompt, "entity": entity,
                "shape_name": shape_name, "axis_pt": self._axis_pt}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "MIRROR",
                "cursor_world": self._axis_pt,
                "cursor_weight": self.cursor_loss_weight}


# ── OffsetTask ────────────────────────────────────────────────────────────────

_OFFSET_PROMPTS = [
    "Offset the {shape} outward by {dist} units",
    "Expand the {shape} outline by {dist}",
    "Create an outer offset of the {shape}",
    "Offset the selected {shape}",
    "Shrink the {shape} inward by {dist}",
    "Create an inner offset of the {shape}",
    "Offset the {shape}",
    "Offset it outward",
    "Expand the outline",
]


class OffsetTask:
    """
    Single-step OFFSET supervised task.

    A closed shape is selected.  The cursor points slightly outside (outward
    offset) or inside (inward offset) the shape boundary — this signals
    offset direction.  Cursor weight is moderate; direction matters more
    than precise distance.
    """

    tool_name = "OFFSET"
    cursor_loss_weight = 0.7

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._cursor: Optional[np.ndarray] = None

    def _approx_radius(self, entity) -> float:
        """Estimate the characteristic radius of an entity."""
        if hasattr(entity, "radius"):
            return float(entity.radius)
        if hasattr(entity, "width") and hasattr(entity, "height"):
            return float(max(entity.width, entity.height)) / 2.0
        return 80.0

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(250, 750))
        cy = float(self.rng.uniform(250, 750))
        entity, shape_name = _make_closed_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        centroid = entity.centroid()
        approx_r = self._approx_radius(entity)
        angle = float(self.rng.uniform(0, 2 * np.pi))
        outward = bool(self.rng.randint(2))  # True = outward, False = inward

        if outward:
            r = approx_r * 1.25   # just outside the boundary
        else:
            r = approx_r * 0.55   # just inside the boundary

        cursor = centroid + np.array([r * np.cos(angle), r * np.sin(angle)])
        self._cursor = np.clip(cursor, 50.0, 950.0)

        dist_label = int(self.rng.choice([10, 20, 30, 50]))
        tmpl = _OFFSET_PROMPTS[int(self.rng.randint(len(_OFFSET_PROMPTS)))]
        prompt = tmpl.format(shape=shape_name, dist=dist_label)

        return {"prompt": prompt, "entity": entity, "shape_name": shape_name,
                "outward": outward}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "OFFSET",
                "cursor_world": self._cursor,
                "cursor_weight": self.cursor_loss_weight}
