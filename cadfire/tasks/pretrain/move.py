"""
Supervised single-step MOVE tasks.

MoveObjectTask
    Marker-driven move: the reference image shows a cross-hair destination.

PrepositionalMoveTask
    Language-driven move using directional or relational instructions:
      Directional – "drag the circle up and to the left"
      Relational  – "move the square to the right of the triangle"

Also exports `prepositional_move_step()`, a helper used by the Phase-3
teacher-forcing builder to attach a prepositional MOVE step to an
already-selected entity from an arbitrary scene.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


# ── Direction / relation tables ────────────────────────────────────────────────

# (dx_sign, dy_sign, label_variants)
_DIRECTIONS = [
    ( 0,  1, ["up", "upward", "north"]),
    ( 0, -1, ["down", "downward", "south"]),
    (-1,  0, ["left", "to the left", "westward"]),
    ( 1,  0, ["right", "to the right", "eastward"]),
    (-1,  1, ["up and to the left", "upper-left", "diagonally up-left"]),
    ( 1,  1, ["up and to the right", "upper-right", "diagonally up-right"]),
    (-1, -1, ["down and to the left", "lower-left", "diagonally down-left"]),
    ( 1, -1, ["down and to the right", "lower-right", "diagonally down-right"]),
]

_DIR_TEMPLATES = [
    "Move the {shape} {direction}",
    "Drag the {shape} {direction}",
    "Shift the {shape} {direction}",
    "Push the {shape} {direction}",
    "Slide the {shape} {direction}",
    "Move the selected {shape} {direction}",
]

# (relation_phrase, fn(anchor_centroid, anchor_r, src_r) -> dest_centroid)
_RELATIONS = [
    ("above",           lambda ac, ar, sr: ac + np.array([0.0,   ar + sr + 60.0])),
    ("below",           lambda ac, ar, sr: ac + np.array([0.0, -(ar + sr + 60.0)])),
    ("to the left of",  lambda ac, ar, sr: ac + np.array([-(ar + sr + 60.0), 0.0])),
    ("to the right of", lambda ac, ar, sr: ac + np.array([ ar + sr + 60.0,  0.0])),
    ("above and left of",
        lambda ac, ar, sr: ac + np.array([-(ar + sr + 40.0),  ar + sr + 40.0])),
    ("above and right of",
        lambda ac, ar, sr: ac + np.array([ ar + sr + 40.0,   ar + sr + 40.0])),
]

_REL_TEMPLATES = [
    "Move the {shape} {relation} the {anchor}",
    "Drag the {shape} {relation} the {anchor}",
    "Place the {shape} {relation} the {anchor}",
    "Position the {shape} {relation} the {anchor}",
    "Shift the {shape} {relation} the {anchor}",
    "Reposition the {shape} so it is {relation} the {anchor}",
]

_CANVAS_MIN = 100.0
_CANVAS_MAX = 900.0


def _approx_radius(entity: Entity) -> float:
    """Rough half-size of any entity (used for clearance calculations)."""
    bb_min, bb_max = entity.bbox()
    return float(np.linalg.norm(bb_max - bb_min)) / 2.0 + 10.0


def prepositional_move_step(
    entity: Entity,
    shape_name: str,
    rng: np.random.RandomState,
) -> Tuple[str, np.ndarray]:
    """
    Given a *pre-selected* entity already on the canvas, pick a random
    directional movement and return ``(prompt, dest_world)``.

    Only directional variants are used here (no second anchor entity).
    Intended for Phase-3 teacher-forcing builders where the scene was
    created by a separate task and the MOVE step is appended.
    """
    dx_sign, dy_sign, labels = _DIRECTIONS[int(rng.randint(len(_DIRECTIONS)))]
    label = labels[int(rng.randint(len(labels)))]
    delta = float(rng.uniform(150, 280))
    cx, cy = entity.centroid()
    dest = np.clip(
        np.array([cx + dx_sign * delta, cy + dy_sign * delta]),
        _CANVAS_MIN, _CANVAS_MAX,
    )
    template = _DIR_TEMPLATES[int(rng.randint(len(_DIR_TEMPLATES)))]
    prompt = template.format(shape=shape_name, direction=label)
    return prompt, dest


# ── PrepositionalMoveTask ──────────────────────────────────────────────────────

class PrepositionalMoveTask:
    """
    Single-step MOVE task using prepositional / relational language.

    Randomly chooses one of two subtypes each call to ``setup()``:
      Directional (50 %) – "drag the circle up and to the left"
      Relational  (50 %) – "move the square to the right of the triangle"

    The source entity is always pre-selected.  The oracle cursor is at
    the computed destination centroid.  Cursor loss weight = 1.0 (precise
    placement is the entire point of the task).
    """

    tool_name = "MOVE"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._dest_world: Optional[np.ndarray] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, _CANVAS_MIN, _CANVAS_MAX)

    def _gen_directional(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(300, 700))
        cy = float(self.rng.uniform(300, 700))
        entity, shape_name = _make_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)

        dx_sign, dy_sign, labels = _DIRECTIONS[int(self.rng.randint(len(_DIRECTIONS)))]
        label = labels[int(self.rng.randint(len(labels)))]
        delta = float(self.rng.uniform(150, 260))
        dest = self._clamp(np.array([cx + dx_sign * delta, cy + dy_sign * delta]))
        self._dest_world = dest

        template = _DIR_TEMPLATES[int(self.rng.randint(len(_DIR_TEMPLATES)))]
        prompt = template.format(shape=shape_name, direction=label)
        return {
            "prompt": prompt,
            "source_entity": entity,
            "dest_world": dest,
            "shape_name": shape_name,
            "subtype": "directional",
        }

    def _gen_relational(self, engine: CADEngine) -> Dict[str, Any]:
        # Place anchor near centre
        acx = float(self.rng.uniform(380, 620))
        acy = float(self.rng.uniform(380, 620))
        anchor, anchor_name = _make_shape(self.rng, acx, acy)
        engine.add_entity(anchor, save_undo=False)
        anchor_r = _approx_radius(anchor)

        relation_name, dest_fn = _RELATIONS[int(self.rng.randint(len(_RELATIONS)))]

        # Source starts off-centre (lower-left corner)
        scx = float(self.rng.uniform(100, 200))
        scy = float(self.rng.uniform(100, 200))
        src_entity, src_name = _make_shape(self.rng, scx, scy)
        engine.add_entity(src_entity, save_undo=False)
        engine.selected_ids.add(src_entity.id)
        src_r = _approx_radius(src_entity)

        dest = self._clamp(dest_fn(np.array([acx, acy]), anchor_r, src_r))
        self._dest_world = dest

        template = _REL_TEMPLATES[int(self.rng.randint(len(_REL_TEMPLATES)))]
        prompt = template.format(
            shape=src_name, relation=relation_name, anchor=anchor_name
        )
        return {
            "prompt": prompt,
            "source_entity": src_entity,
            "anchor_entity": anchor,
            "dest_world": dest,
            "shape_name": src_name,
            "subtype": "relational",
        }

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        if self.rng.rand() < 0.5:
            return self._gen_directional(engine)
        return self._gen_relational(engine)

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "MOVE",
            "cursor_world": self._dest_world,
            "cursor_weight": self.cursor_loss_weight,
        }
