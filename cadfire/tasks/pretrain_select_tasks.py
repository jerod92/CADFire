"""
Semantic pretraining select tasks.

Two single-step tasks designed specifically for the supervised semantic-cursor
pretraining phase (pretrain_semantic.py).  They are also valid RL tasks and
will be discovered by the registry, but their primary purpose is to supply
(observation, tool_id, cursor_mask) tuples for Phase-2 pretraining.

Task overview
─────────────
SemanticSelectTask
    One shape is placed in each of 4 quadrants.  Every shape has a DISTINCT
    semantic type chosen from SEMANTIC_SHAPES (20 types).  The prompt names
    exactly one shape; the agent must respond in a single step:
        tool  = SELECT
        cursor = anywhere inside / on the target entity

SemanticMultiSelectTask
    Multiple instances of the SAME shape type are scattered across the canvas,
    mixed with a few distractors of different types.  The prompt says
    "Select all <shape>s".  The agent must respond in a single step:
        tool  = MULTISELECT
        cursor = a mask activating the centroid of every target entity
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    ArcEntity, CircleEntity, EllipseEntity,
    LineEntity, PolygonEntity, RectangleEntity, Entity,
)
from cadfire.tasks.base import BaseTask, UTILITY_TOOLS
from cadfire.tasks.registry import register_task


# ── Semantic shape catalogue (20 visually distinct types) ─────────────────────
#
# Each entry: (display_name, factory(rng, cx, cy) -> Entity)
# cx, cy are the WORLD-SPACE centre of the shape's bounding area.
# Factories must produce entities whose centroid is close to (cx, cy).

def _circle(rng, cx, cy):
    r = float(rng.uniform(50, 90))
    return CircleEntity(center=np.array([cx, cy]), radius=r,
                        color_index=int(rng.randint(0, 8)))

def _small_circle(rng, cx, cy):
    r = float(rng.uniform(20, 40))
    return CircleEntity(center=np.array([cx, cy]), radius=r,
                        color_index=int(rng.randint(0, 8)))

def _large_circle(rng, cx, cy):
    r = float(rng.uniform(90, 130))
    return CircleEntity(center=np.array([cx, cy]), radius=r,
                        color_index=int(rng.randint(0, 8)))

def _square(rng, cx, cy):
    s = float(rng.uniform(70, 120))
    return RectangleEntity(corner=np.array([cx - s / 2, cy - s / 2]),
                           width=s, height=s,
                           color_index=int(rng.randint(0, 8)))

def _wide_rectangle(rng, cx, cy):
    w = float(rng.uniform(120, 180))
    h = float(rng.uniform(40, 70))
    return RectangleEntity(corner=np.array([cx - w / 2, cy - h / 2]),
                           width=w, height=h,
                           color_index=int(rng.randint(0, 8)))

def _tall_rectangle(rng, cx, cy):
    w = float(rng.uniform(40, 70))
    h = float(rng.uniform(120, 180))
    return RectangleEntity(corner=np.array([cx - w / 2, cy - h / 2]),
                           width=w, height=h,
                           color_index=int(rng.randint(0, 8)))

def _ellipse(rng, cx, cy):
    a = float(rng.uniform(80, 120))
    b = float(rng.uniform(30, 55))
    return EllipseEntity(center=np.array([cx, cy]),
                         semi_major=a, semi_minor=b,
                         color_index=int(rng.randint(0, 8)))

def _narrow_ellipse(rng, cx, cy):
    a = float(rng.uniform(30, 50))
    b = float(rng.uniform(80, 120))
    return EllipseEntity(center=np.array([cx, cy]),
                         semi_major=a, semi_minor=b,
                         color_index=int(rng.randint(0, 8)))

def _triangle(rng, cx, cy):
    r = float(rng.uniform(60, 100))
    return PolygonEntity(center=np.array([cx, cy]), radius=r, sides=3,
                         color_index=int(rng.randint(0, 8)))

def _pentagon(rng, cx, cy):
    r = float(rng.uniform(55, 95))
    return PolygonEntity(center=np.array([cx, cy]), radius=r, sides=5,
                         color_index=int(rng.randint(0, 8)))

def _hexagon(rng, cx, cy):
    r = float(rng.uniform(55, 95))
    return PolygonEntity(center=np.array([cx, cy]), radius=r, sides=6,
                         color_index=int(rng.randint(0, 8)))

def _heptagon(rng, cx, cy):
    r = float(rng.uniform(55, 90))
    return PolygonEntity(center=np.array([cx, cy]), radius=r, sides=7,
                         color_index=int(rng.randint(0, 8)))

def _octagon(rng, cx, cy):
    r = float(rng.uniform(55, 90))
    return PolygonEntity(center=np.array([cx, cy]), radius=r, sides=8,
                         color_index=int(rng.randint(0, 8)))

def _nonagon(rng, cx, cy):
    r = float(rng.uniform(55, 85))
    return PolygonEntity(center=np.array([cx, cy]), radius=r, sides=9,
                         color_index=int(rng.randint(0, 8)))

def _decagon(rng, cx, cy):
    r = float(rng.uniform(55, 85))
    return PolygonEntity(center=np.array([cx, cy]), radius=r, sides=10,
                         color_index=int(rng.randint(0, 8)))

def _arc(rng, cx, cy):
    r = float(rng.uniform(55, 90))
    sa = float(rng.uniform(0, 60))
    return ArcEntity(center=np.array([cx, cy]), radius=r,
                     start_angle=sa, end_angle=sa + 90.0,
                     color_index=int(rng.randint(0, 8)))

def _semicircle(rng, cx, cy):
    r = float(rng.uniform(55, 90))
    sa = float(rng.uniform(0, 180))
    return ArcEntity(center=np.array([cx, cy]), radius=r,
                     start_angle=sa, end_angle=sa + 180.0,
                     color_index=int(rng.randint(0, 8)))

def _short_line(rng, cx, cy):
    angle = float(rng.uniform(0, 360))
    rad = np.radians(angle)
    half = float(rng.uniform(30, 60))
    dx, dy = np.cos(rad) * half, np.sin(rad) * half
    return LineEntity(start=np.array([cx - dx, cy - dy]),
                      end=np.array([cx + dx, cy + dy]),
                      color_index=int(rng.randint(0, 8)))

def _long_line(rng, cx, cy):
    angle = float(rng.uniform(0, 360))
    rad = np.radians(angle)
    half = float(rng.uniform(100, 160))
    dx, dy = np.cos(rad) * half, np.sin(rad) * half
    return LineEntity(start=np.array([cx - dx, cy - dy]),
                      end=np.array([cx + dx, cy + dy]),
                      color_index=int(rng.randint(0, 8)))

def _diagonal_line(rng, cx, cy):
    half = float(rng.uniform(70, 110))
    return LineEntity(start=np.array([cx - half, cy - half]),
                      end=np.array([cx + half, cy + half]),
                      color_index=int(rng.randint(0, 8)))


# Complete catalogue: (name, factory_fn)
SEMANTIC_SHAPES: List[Tuple[str, Any]] = [
    ("circle",          _circle),
    ("small circle",    _small_circle),
    ("large circle",    _large_circle),
    ("square",          _square),
    ("wide rectangle",  _wide_rectangle),
    ("tall rectangle",  _tall_rectangle),
    ("ellipse",         _ellipse),
    ("narrow ellipse",  _narrow_ellipse),
    ("triangle",        _triangle),
    ("pentagon",        _pentagon),
    ("hexagon",         _hexagon),
    ("heptagon",        _heptagon),
    ("octagon",         _octagon),
    ("nonagon",         _nonagon),
    ("decagon",         _decagon),
    ("arc",             _arc),
    ("semicircle",      _semicircle),
    ("short line",      _short_line),
    ("long line",       _long_line),
    ("diagonal line",   _diagonal_line),
]

# Number of distinct semantic types
NUM_SEMANTIC_SHAPES = len(SEMANTIC_SHAPES)

# World-space quadrant centres (world is 1000×1000)
# Each quadrant is a 500×500 region; centres sit at ¼ and ¾ marks.
_QUADRANT_CENTRES: List[Tuple[float, float]] = [
    (250.0, 750.0),  # top-left
    (750.0, 750.0),  # top-right
    (250.0, 250.0),  # bottom-left
    (750.0, 250.0),  # bottom-right
]

# Prompt templates – both tasks share these patterns
_SELECT_PROMPTS = [
    "Select the {shape}",
    "Click on the {shape}",
    "Find and select the {shape}",
    "Pick the {shape}",
    "Choose the {shape}",
    "Highlight the {shape}",
    "Use SELECT on the {shape}",
]

_MULTISELECT_PROMPTS = [
    "Select all {shape}s",
    "Highlight every {shape}",
    "Find and select all {shape}s",
    "Pick all the {shape}s",
    "Multi-select every {shape}",
    "Use MULTISELECT to grab all {shape}s",
    "Choose all {shape}s on the canvas",
]


# ── Helper ────────────────────────────────────────────────────────────────────

def _place_shape(rng, name_idx: int, cx: float, cy: float,
                 jitter: float = 60.0) -> Tuple[str, Entity]:
    """Place a semantic shape near (cx, cy) with random jitter."""
    name, factory = SEMANTIC_SHAPES[name_idx]
    jx = float(rng.uniform(-jitter, jitter))
    jy = float(rng.uniform(-jitter, jitter))
    entity = factory(rng, cx + jx, cy + jy)
    return name, entity


# ── Task 1: SemanticSelectTask ────────────────────────────────────────────────

@register_task
class SemanticSelectTask(BaseTask):
    """
    Single-step SELECT pretraining task.

    Layout: one distinct semantic shape per quadrant (4 shapes total).
    Target: one of the four shapes, identified by name in the prompt.

    Oracle action  (used by pretrain_semantic.py):
        tool   = SELECT
        cursor = Gaussian blob centred on the target entity's centroid
    """

    task_name = "semantic_select"
    task_category = "select"
    difficulty = 1.0

    def allowed_tools(self) -> List[str]:
        return UTILITY_TOOLS + ["SELECT", "DESELECT"]

    def generate_prompt_variants(self) -> List[str]:
        return _SELECT_PROMPTS

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Pick 4 distinct shape types (one per quadrant)
        n_quad = len(_QUADRANT_CENTRES)
        type_indices = list(self.rng.choice(NUM_SEMANTIC_SHAPES, size=n_quad, replace=False))

        self._entities: List[Entity] = []
        self._shape_names: List[str] = []

        for i, (cx, cy) in enumerate(_QUADRANT_CENTRES):
            name, entity = _place_shape(self.rng, type_indices[i], cx, cy)
            engine.add_entity(entity, save_undo=False)
            self._entities.append(entity)
            self._shape_names.append(name)

        # Pick one quadrant as the target
        target_q = int(self.rng.randint(0, n_quad))
        self._target_entity = self._entities[target_q]
        self._target_id = self._target_entity.id
        self._target_name = self._shape_names[target_q]

        template = _SELECT_PROMPTS[int(self.rng.randint(len(_SELECT_PROMPTS)))]
        prompt = template.format(shape=self._target_name)

        return {
            "prompt": prompt,
            "target_entity": self._target_entity,
            "target_name": self._target_name,
        }

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        f1 = self.selection_reward(engine, {self._target_id})
        terminated = f1 > 0.9
        return {
            "reward": f1 - 0.01,
            "terminated": terminated,
            "info": {"f1": f1, "target": self._target_name},
        }


# ── Task 2: SemanticMultiSelectTask ──────────────────────────────────────────

@register_task
class SemanticMultiSelectTask(BaseTask):
    """
    Single-step MULTISELECT pretraining task.

    Layout: 3-5 instances of the SAME semantic shape type placed across the
    canvas, mixed with 2-3 distractor shapes of different types.

    Target: all instances of the repeated shape type.

    Oracle action  (used by pretrain_semantic.py):
        tool   = MULTISELECT
        cursor = sum of Gaussian blobs at every target entity's centroid,
                 thresholded to [0, 1]
    """

    task_name = "semantic_multi_select"
    task_category = "select"
    difficulty = 2.0

    def allowed_tools(self) -> List[str]:
        return UTILITY_TOOLS + ["SELECT", "MULTISELECT", "DESELECT"]

    def generate_prompt_variants(self) -> List[str]:
        return _MULTISELECT_PROMPTS

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Choose one shape type as "target" (the one to select all of)
        target_type_idx = int(self.rng.randint(0, NUM_SEMANTIC_SHAPES))
        target_name, _ = SEMANTIC_SHAPES[target_type_idx]

        # Choose 2-3 distinct distractor types
        n_distractor_types = int(self.rng.randint(2, 4))
        distractor_pool = [i for i in range(NUM_SEMANTIC_SHAPES) if i != target_type_idx]
        distractor_type_indices = list(
            self.rng.choice(distractor_pool, size=n_distractor_types, replace=False)
        )

        # Place 3-5 target instances at random positions across the canvas
        n_targets = int(self.rng.randint(3, 6))
        self._target_entities: List[Entity] = []
        self._target_ids: set = set()

        for _ in range(n_targets):
            cx = float(self.rng.uniform(120, 880))
            cy = float(self.rng.uniform(120, 880))
            _, entity = _place_shape(self.rng, target_type_idx, cx, cy, jitter=0)
            engine.add_entity(entity, save_undo=False)
            self._target_entities.append(entity)
            self._target_ids.add(entity.id)

        # Place 2-4 distractor instances (random mix of distractor types)
        n_distractors = int(self.rng.randint(2, 5))
        for _ in range(n_distractors):
            dt_idx = distractor_type_indices[int(self.rng.randint(len(distractor_type_indices)))]
            cx = float(self.rng.uniform(120, 880))
            cy = float(self.rng.uniform(120, 880))
            _, entity = _place_shape(self.rng, dt_idx, cx, cy, jitter=0)
            engine.add_entity(entity, save_undo=False)

        self._target_name = target_name

        template = _MULTISELECT_PROMPTS[int(self.rng.randint(len(_MULTISELECT_PROMPTS)))]
        prompt = template.format(shape=target_name)

        return {
            "prompt": prompt,
            "target_entities": self._target_entities,
            "target_name": target_name,
        }

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        f1 = self.selection_reward(engine, self._target_ids)
        terminated = f1 > 0.9
        return {
            "reward": f1 - 0.01,
            "terminated": terminated,
            "info": {"f1": f1, "n_targets": len(self._target_ids),
                     "target": self._target_name},
        }
