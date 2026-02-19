"""
Supervised single-step SELECT / MULTISELECT tasks.

Ported from cadfire/tasks/pretrain_select_tasks.py and extended with the
standard ``oracle_action`` interface used by the Phase-2 dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    ArcEntity, CircleEntity, EllipseEntity,
    LineEntity, PolygonEntity, RectangleEntity, Entity,
)

# Re-export the shape catalogue for convenience
from cadfire.tasks.pretrain_select_tasks import (
    SEMANTIC_SHAPES, NUM_SEMANTIC_SHAPES, _QUADRANT_CENTRES, _place_shape,
    _SELECT_PROMPTS, _MULTISELECT_PROMPTS,
)


class SemanticSelectTask:
    """
    Single-step SELECT supervised task.

    Layout: one distinct semantic shape per quadrant (4 total).
    Agent must use SELECT and click the named shape.
    """

    tool_name = "SELECT"
    cursor_loss_weight = 1.0  # cursor matters here

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target_entity: Optional[Entity] = None
        self._target_name: str = ""

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        n_quad = len(_QUADRANT_CENTRES)
        type_indices = list(self.rng.choice(NUM_SEMANTIC_SHAPES, size=n_quad, replace=False))

        entities: List[Entity] = []
        shape_names: List[str] = []

        for i, (cx, cy) in enumerate(_QUADRANT_CENTRES):
            name, entity = _place_shape(self.rng, type_indices[i], cx, cy)
            engine.add_entity(entity, save_undo=False)
            entities.append(entity)
            shape_names.append(name)

        target_q = int(self.rng.randint(0, n_quad))
        self._target_entity = entities[target_q]
        self._target_name = shape_names[target_q]

        template = _SELECT_PROMPTS[int(self.rng.randint(len(_SELECT_PROMPTS)))]
        prompt = template.format(shape=self._target_name)

        return {
            "prompt": prompt,
            "target_entity": self._target_entity,
            "target_name": self._target_name,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "SELECT",
            "cursor_world": self._target_entity.centroid(),
            "cursor_weight": self.cursor_loss_weight,
        }


class SemanticMultiSelectTask:
    """
    Single-step MULTISELECT supervised task.

    3-5 instances of one shape type + 2-4 distractors.
    Agent must use MULTISELECT and mark all target instances.
    """

    tool_name = "MULTISELECT"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target_entities: List[Entity] = []
        self._target_name: str = ""

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        target_type_idx = int(self.rng.randint(0, NUM_SEMANTIC_SHAPES))
        target_name, _ = SEMANTIC_SHAPES[target_type_idx]

        n_distractor_types = int(self.rng.randint(2, 4))
        distractor_pool = [i for i in range(NUM_SEMANTIC_SHAPES) if i != target_type_idx]
        distractor_type_indices = list(
            self.rng.choice(distractor_pool, size=n_distractor_types, replace=False)
        )

        n_targets = int(self.rng.randint(3, 6))
        self._target_entities = []

        for _ in range(n_targets):
            cx = float(self.rng.uniform(120, 880))
            cy = float(self.rng.uniform(120, 880))
            _, entity = _place_shape(self.rng, target_type_idx, cx, cy, jitter=0)
            engine.add_entity(entity, save_undo=False)
            self._target_entities.append(entity)

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

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        # Return centroids of ALL target entities as a list
        centroids = [e.centroid() for e in self._target_entities]
        return {
            "tool": "MULTISELECT",
            "cursor_world": centroids,   # list of points → multiple blobs
            "cursor_weight": self.cursor_loss_weight,
        }
