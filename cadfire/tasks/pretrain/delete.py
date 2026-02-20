"""
Supervised single-step ERASE task.

Setup: one shape is already selected (highlighted in the selection mask).
       0-2 other shapes are present as context.
Agent must: use ERASE tool (no cursor needed, but we still provide a
            target at the selected entity's centroid so the cursor head
            learns to attend to the selected region).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import Entity
from cadfire.tasks.pretrain.pretrain_select_tasks import (
    SEMANTIC_SHAPES, NUM_SEMANTIC_SHAPES, _place_shape,
)

_ERASE_PROMPTS = [
    "Delete the selected {shape}",
    "Erase the {shape}",
    "Remove the selected {shape}",
    "Use ERASE to delete the {shape}",
    "Get rid of the {shape}",
    "Delete it",
    "Erase the selected object",
]

_QUADRANT_CENTRES = [
    (250.0, 750.0),
    (750.0, 750.0),
    (250.0, 250.0),
    (750.0, 250.0),
]


class DeleteObjectTask:
    """
    Single-step ERASE supervised task.

    The target entity is pre-selected (its centroid shown in the selection
    mask channel).  Agent must activate ERASE.
    """

    tool_name = "ERASE"
    cursor_loss_weight = 0.1  # cursor doesn't matter for ERASE

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target_entity: Optional[Entity] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        n = int(self.rng.randint(1, 4))  # 1-3 total shapes
        n_quad = min(n, len(_QUADRANT_CENTRES))

        type_indices = list(
            self.rng.choice(NUM_SEMANTIC_SHAPES, size=n_quad, replace=False)
        )
        entities: List[Entity] = []
        shape_names: List[str] = []
        centres = self.rng.choice(len(_QUADRANT_CENTRES), size=n_quad, replace=False)

        for i in range(n_quad):
            cx, cy = _QUADRANT_CENTRES[centres[i]]
            name, entity = _place_shape(self.rng, type_indices[i], cx, cy)
            engine.add_entity(entity, save_undo=False)
            entities.append(entity)
            shape_names.append(name)

        # Pre-select the first entity
        target_idx = 0
        self._target_entity = entities[target_idx]
        engine.selected_ids.add(self._target_entity.id)
        target_name = shape_names[target_idx]

        template = _ERASE_PROMPTS[int(self.rng.randint(len(_ERASE_PROMPTS)))]
        prompt = template.format(shape=target_name)

        return {
            "prompt": prompt,
            "target_entity": self._target_entity,
            "target_name": target_name,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "ERASE",
            "cursor_world": self._target_entity.centroid(),
            "cursor_weight": self.cursor_loss_weight,
        }
