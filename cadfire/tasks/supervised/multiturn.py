"""
Supervised multi-turn chat tasks.

The prompt encodes two conversation turns joined by " | ":

    "<first_turn> | <second_turn>"

The scene always reflects the state *after* the first turn was executed:
  • The entity described in turn 1 already exists on the canvas.
  • It is pre-selected (engine.selected_ids contains it).
  • The agent must predict the tool + cursor that satisfies turn 2.

This trains the model to exploit conversational context alongside visual
state — the key capability that separates a mere shape-drawer from an
interactive CAD assistant.

Tasks
─────
  ScaleFromChatTask        – "Draw a {shape} | make it smaller/larger"   → SCALE
  MoveFromChatTask         – "Draw a {shape} | move it {direction}"      → MOVE
  RotateFromChatTask       – "Draw a {shape} | rotate it {angle}°"       → ROTATE
  EraseFromChatTask        – "Draw a {shape} | delete / erase it"        → ERASE
  ChangeColorFromChatTask  – "Draw a {shape} | change it to {color}"     → COLOR_SET
  CopyFromChatTask         – "Draw a {shape} | copy it to the {dir}"     → COPY
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    CircleEntity, EllipseEntity, PolygonEntity, RectangleEntity,
)

# ── Colour index → human-readable name (matches config.json palette order) ──
_COLOR_NAMES = ["white", "red", "yellow", "green", "cyan", "blue", "magenta", "gray"]

# ── Shared shape factory ──────────────────────────────────────────────────────

def _make_shape(rng: np.random.RandomState, cx: float, cy: float):
    """Return (entity, name) for a random shape centred near (cx, cy)."""
    kind = int(rng.randint(5))
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
    elif kind == 2:
        r = float(rng.uniform(60, 110))
        return PolygonEntity(center=np.array([cx, cy]), radius=r,
                             sides=3, color_index=color), "triangle"
    elif kind == 3:
        r = float(rng.uniform(60, 110))
        return PolygonEntity(center=np.array([cx, cy]), radius=r,
                             sides=6, color_index=color), "hexagon"
    else:
        a = float(rng.uniform(60, 110))
        b = float(rng.uniform(30, 60))
        return EllipseEntity(center=np.array([cx, cy]),
                             semi_major=a, semi_minor=b,
                             color_index=color), "ellipse"


# ── Multi-turn base mixin ─────────────────────────────────────────────────────

class _MultiTurnBase:
    """Shared helpers for two-turn prompt construction."""

    tool_name: str = ""
    cursor_loss_weight: float = 1.0

    _DRAW_TURNS = [
        "Draw a {shape}",
        "Create a {shape}",
        "Add a {shape} to the canvas",
        "Place a {shape} here",
        "I need a {shape}",
        "Give me a {shape}",
    ]

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._entity: Optional[Any] = None

    def _draw_turn(self, shape_name: str) -> str:
        t = self._DRAW_TURNS[int(self.rng.randint(len(self._DRAW_TURNS)))]
        return t.format(shape=shape_name)

    def _build_prompt(self, shape_name: str, second_turn: str) -> str:
        return f"{self._draw_turn(shape_name)} | {second_turn}"


# ── ScaleFromChatTask ─────────────────────────────────────────────────────────

_SCALE_SHRINK = [
    "make it smaller",
    "scale it down",
    "shrink it",
    "make the {shape} smaller",
    "reduce its size",
    "make it a bit smaller",
]
_SCALE_GROW = [
    "make it larger",
    "scale it up",
    "enlarge it",
    "make the {shape} bigger",
    "increase its size",
    "make it a bit bigger",
]


class ScaleFromChatTask(_MultiTurnBase):
    """
    Multi-turn SCALE task.

    Turn 1: "Draw a {shape}"
    Turn 2: "make it smaller / larger"
    Scene:  shape present and pre-selected.
    Oracle: SCALE, cursor at entity centroid (natural scale pivot).
    """

    tool_name = "SCALE"
    cursor_loss_weight = 0.8

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(250, 750))
        cy = float(self.rng.uniform(250, 750))
        entity, shape_name = _make_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._entity = entity
        self._centroid = entity.centroid()

        pool = _SCALE_SHRINK if self.rng.randint(2) == 0 else _SCALE_GROW
        second = pool[int(self.rng.randint(len(pool)))].format(shape=shape_name)
        return {"prompt": self._build_prompt(shape_name, second),
                "entity": entity, "shape_name": shape_name}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "SCALE",
                "cursor_world": self._centroid,
                "cursor_weight": self.cursor_loss_weight}


# ── MoveFromChatTask ──────────────────────────────────────────────────────────

_MOVE_DIRS = {
    "right": ( 280,    0),
    "left":  (-280,    0),
    "up":    (   0,  280),
    "down":  (   0, -280),
}
_MOVE_SECOND_TURNS = [
    "move it to the {dir}",
    "shift it {dir}",
    "push the {shape} {dir}",
    "slide it {dir}",
    "move the {shape} to the {dir}",
    "drag it {dir}",
]


class MoveFromChatTask(_MultiTurnBase):
    """
    Multi-turn MOVE task.

    Turn 1: "Draw a {shape}"
    Turn 2: "move it to the right / left / up / down"
    Scene:  shape placed near canvas centre, pre-selected.
    Oracle: MOVE, cursor at the destination world position.
    """

    tool_name = "MOVE"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self._dest: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        dir_name = list(_MOVE_DIRS.keys())[int(self.rng.randint(len(_MOVE_DIRS)))]
        dx, dy = _MOVE_DIRS[dir_name]

        # Place entity so destination stays on canvas
        cx = float(np.clip(float(self.rng.uniform(300, 700)) - dx / 2, 150, 850))
        cy = float(np.clip(float(self.rng.uniform(300, 700)) - dy / 2, 150, 850))
        entity, shape_name = _make_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._entity = entity

        self._dest = np.array(
            [np.clip(cx + dx, 100, 900), np.clip(cy + dy, 100, 900)],
            dtype=np.float64,
        )

        tmpl = _MOVE_SECOND_TURNS[int(self.rng.randint(len(_MOVE_SECOND_TURNS)))]
        second = tmpl.format(dir=dir_name, shape=shape_name)
        return {"prompt": self._build_prompt(shape_name, second),
                "entity": entity, "dest_world": self._dest,
                "shape_name": shape_name, "direction": dir_name}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "MOVE",
                "cursor_world": self._dest,
                "cursor_weight": self.cursor_loss_weight}


# ── RotateFromChatTask ────────────────────────────────────────────────────────

_ROTATE_SECOND_TURNS = [
    "rotate it {angle} degrees",
    "spin it {angle}°",
    "rotate the {shape} {angle} degrees",
    "turn it {angle} degrees clockwise",
    "apply a {angle}-degree rotation",
    "rotate it",
    "give it a quarter turn",
]


class RotateFromChatTask(_MultiTurnBase):
    """
    Multi-turn ROTATE task.

    Turn 1: "Draw a {shape}"
    Turn 2: "rotate it {angle} degrees"
    Scene:  shape present and pre-selected.
    Oracle: ROTATE, cursor at entity centroid (natural rotation pivot).
    """

    tool_name = "ROTATE"
    cursor_loss_weight = 0.8

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(250, 750))
        cy = float(self.rng.uniform(250, 750))
        entity, shape_name = _make_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._entity = entity
        self._centroid = entity.centroid()

        angle = int(self.rng.choice([30, 45, 60, 90, 120, 135, 180]))
        tmpl = _ROTATE_SECOND_TURNS[int(self.rng.randint(len(_ROTATE_SECOND_TURNS)))]
        second = tmpl.format(angle=angle, shape=shape_name)
        return {"prompt": self._build_prompt(shape_name, second),
                "entity": entity, "angle": angle, "shape_name": shape_name}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "ROTATE",
                "cursor_world": self._centroid,
                "cursor_weight": self.cursor_loss_weight}


# ── EraseFromChatTask ─────────────────────────────────────────────────────────

_ERASE_SECOND_TURNS = [
    "delete it",
    "erase it",
    "remove it",
    "get rid of it",
    "get rid of the {shape}",
    "delete the {shape}",
    "erase the {shape}",
    "remove the {shape}",
    "I don't want it anymore",
    "undo the {shape}",
]


class EraseFromChatTask(_MultiTurnBase):
    """
    Multi-turn ERASE task.

    Turn 1: "Draw a {shape}"
    Turn 2: "delete / erase / remove it"
    Scene:  shape present and pre-selected; 0-2 distractor shapes present
            so the selection mask is the agent's only cue for WHICH entity.
    Oracle: ERASE (cursor nearly irrelevant – cursor_weight=0.05).
    """

    tool_name = "ERASE"
    cursor_loss_weight = 0.05

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Add 0–2 distractors first so they are behind the selected entity
        n_extra = int(self.rng.randint(0, 3))
        for _ in range(n_extra):
            ex = float(self.rng.uniform(100, 900))
            ey = float(self.rng.uniform(100, 900))
            extra, _ = _make_shape(self.rng, ex, ey)
            engine.add_entity(extra, save_undo=False)

        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        entity, shape_name = _make_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._entity = entity

        tmpl = _ERASE_SECOND_TURNS[int(self.rng.randint(len(_ERASE_SECOND_TURNS)))]
        second = tmpl.format(shape=shape_name)
        return {"prompt": self._build_prompt(shape_name, second),
                "entity": entity, "shape_name": shape_name}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "ERASE",
                "cursor_world": self._entity.centroid(),  # nominal; weight is near-zero
                "cursor_weight": self.cursor_loss_weight}


# ── ChangeColorFromChatTask ───────────────────────────────────────────────────

_COLOR_SECOND_TURNS = [
    "change it to {color}",
    "make it {color}",
    "change the color to {color}",
    "recolor the {shape} {color}",
    "paint it {color}",
    "set its color to {color}",
    "I want it {color} instead",
    "color it {color}",
]


class ChangeColorFromChatTask(_MultiTurnBase):
    """
    Multi-turn COLOR_SET task.

    Turn 1: "Draw a {shape}"
    Turn 2: "change it to {color}"
    Scene:  shape present and pre-selected; its current color differs from
            the requested color so there is a genuine visual change the agent
            can learn to anticipate.
    Oracle: COLOR_SET (no cursor needed – cursor_weight=0.05).
    """

    tool_name = "COLOR_SET"
    cursor_loss_weight = 0.05

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        orig_color = int(self.rng.randint(0, 8))
        target_color = int(
            self.rng.choice([c for c in range(8) if c != orig_color])
        )
        entity, shape_name = _make_shape(self.rng, cx, cy)
        entity.color_index = orig_color
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._entity = entity

        color_name = _COLOR_NAMES[target_color]
        tmpl = _COLOR_SECOND_TURNS[int(self.rng.randint(len(_COLOR_SECOND_TURNS)))]
        second = tmpl.format(color=color_name, shape=shape_name)
        return {"prompt": self._build_prompt(shape_name, second),
                "entity": entity, "shape_name": shape_name,
                "target_color": target_color, "color_name": color_name}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "COLOR_SET",
                "cursor_world": None,
                "cursor_weight": self.cursor_loss_weight}


# ── CopyFromChatTask ──────────────────────────────────────────────────────────

_COPY_DIRS = {
    "right":  ( 300,    0),
    "left":   (-300,    0),
    "above":  (   0,  300),
    "below":  (   0, -300),
}
_COPY_SECOND_TURNS = [
    "copy it to the {dir}",
    "make a copy to the {dir}",
    "duplicate the {shape} to the {dir}",
    "place a copy {dir}",
    "copy and paste it {dir}",
    "put a copy {dir}",
]


class CopyFromChatTask(_MultiTurnBase):
    """
    Multi-turn COPY task.

    Turn 1: "Draw a {shape}"
    Turn 2: "copy it to the right / left / above / below"
    Scene:  shape placed near canvas centre, pre-selected.
    Oracle: COPY, cursor at the destination centroid.
    """

    tool_name = "COPY"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self._dest: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        dir_name = list(_COPY_DIRS.keys())[int(self.rng.randint(len(_COPY_DIRS)))]
        dx, dy = _COPY_DIRS[dir_name]

        cx = float(np.clip(float(self.rng.uniform(300, 700)) - dx / 2, 150, 850))
        cy = float(np.clip(float(self.rng.uniform(300, 700)) - dy / 2, 150, 850))
        entity, shape_name = _make_shape(self.rng, cx, cy)
        engine.add_entity(entity, save_undo=False)
        engine.selected_ids.add(entity.id)
        self._entity = entity

        self._dest = np.array(
            [np.clip(cx + dx, 100, 900), np.clip(cy + dy, 100, 900)],
            dtype=np.float64,
        )

        tmpl = _COPY_SECOND_TURNS[int(self.rng.randint(len(_COPY_SECOND_TURNS)))]
        second = tmpl.format(dir=dir_name, shape=shape_name)
        return {"prompt": self._build_prompt(shape_name, second),
                "entity": entity, "dest_world": self._dest,
                "shape_name": shape_name, "direction": dir_name}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {"tool": "COPY",
                "cursor_world": self._dest,
                "cursor_weight": self.cursor_loss_weight}
