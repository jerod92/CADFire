"""
Selection tasks: the agent must select specific entities.

Tests:
  - Selecting a specific shape by clicking near it
  - Selecting all shapes of a specific color
  - Multi-selecting with a mask
  - Deselecting all
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import CircleEntity, RectangleEntity, LineEntity
from cadfire.tasks.base import BaseTask
from cadfire.tasks.registry import register_task


COLOR_NAMES = ["white", "red", "yellow", "green", "cyan", "blue", "magenta", "gray"]


@register_task
class SelectShapeTask(BaseTask):
    """Select a specific shape by its position/type."""
    task_name = "select_shape"
    task_category = "select"
    difficulty = 1.5

    def generate_prompt_variants(self):
        return [
            "Select the {shape} at ({x:.0f},{y:.0f})",
            "Click on the {shape} near ({x:.0f},{y:.0f})",
            "Find and select the {color} {shape}",
            "Select the {shape}",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Create 3-5 shapes, one is the target
        n = self.rng.randint(3, 6)
        self._shapes = []
        for i in range(n):
            color_idx = int(self.rng.randint(0, 8))
            cx, cy = self.rng.uniform(100, 900, 2)
            r = self.rng.uniform(30, 100)
            if self.rng.rand() > 0.5:
                e = CircleEntity(center=np.array([cx, cy]), radius=r, color_index=color_idx)
            else:
                e = RectangleEntity(corner=np.array([cx, cy]), width=r*2, height=r*1.5, color_index=color_idx)
            engine.add_entity(e, save_undo=False)
            self._shapes.append(e)

        self._target_idx = int(self.rng.randint(0, n))
        target = self._shapes[self._target_idx]
        self._target_id = target.id

        c = target.centroid()
        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape=target.entity_type.lower(),
            x=c[0], y=c[1],
            color=COLOR_NAMES[target.color_index],
        )
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        f1 = self.selection_reward(engine, {self._target_id})
        terminated = f1 > 0.9
        return {"reward": f1 - 0.01, "terminated": terminated,
                "info": {"f1": f1, "selected": len(engine.selected_ids)}}


@register_task
class SelectByColorTask(BaseTask):
    """Select all entities of a specific color."""
    task_name = "select_by_color"
    task_category = "select"
    difficulty = 3.0

    def generate_prompt_variants(self):
        return [
            "Select all {color} shapes",
            "Find and select every {color} object",
            "Select all entities colored {color}",
            "Highlight all {color} items",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        target_color = int(self.rng.randint(0, 8))
        n = self.rng.randint(5, 10)
        self._target_ids = set()

        for i in range(n):
            color_idx = target_color if self.rng.rand() > 0.5 else int(self.rng.randint(0, 8))
            cx, cy = self.rng.uniform(100, 900, 2)
            r = self.rng.uniform(30, 100)
            e = CircleEntity(center=np.array([cx, cy]), radius=r, color_index=color_idx)
            engine.add_entity(e, save_undo=False)
            if color_idx == target_color:
                self._target_ids.add(e.id)

        # Ensure at least one target
        if not self._target_ids:
            e = CircleEntity(
                center=np.array([500.0, 500.0]), radius=50,
                color_index=target_color,
            )
            engine.add_entity(e, save_undo=False)
            self._target_ids.add(e.id)

        self._color_name = COLOR_NAMES[target_color]
        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(color=self._color_name)
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        f1 = self.selection_reward(engine, self._target_ids)
        terminated = f1 > 0.9
        return {"reward": f1 - 0.01, "terminated": terminated,
                "info": {"f1": f1, "target_count": len(self._target_ids)}}


@register_task
class EraseSelectionTask(BaseTask):
    """Select and erase specific entities."""
    task_name = "erase_selection"
    task_category = "modify"
    difficulty = 2.5

    def generate_prompt_variants(self):
        return [
            "Delete the {color} {shape}",
            "Erase the {shape} at ({x:.0f},{y:.0f})",
            "Remove the {color} {shape} from the drawing",
            "Select and delete the {shape}",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        n = self.rng.randint(3, 6)
        self._all_ids = set()
        shapes_info = []

        for i in range(n):
            color_idx = int(self.rng.randint(0, 8))
            cx, cy = self.rng.uniform(100, 900, 2)
            r = self.rng.uniform(30, 100)
            e = CircleEntity(center=np.array([cx, cy]), radius=r, color_index=color_idx)
            engine.add_entity(e, save_undo=False)
            self._all_ids.add(e.id)
            shapes_info.append((e.id, e.entity_type, color_idx, np.array([cx, cy])))

        # Pick target to erase
        self._target_idx = int(self.rng.randint(0, n))
        tid, ttype, tcolor, tpos = shapes_info[self._target_idx]
        self._erase_id = tid
        self._initial_count = n

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape=ttype.lower(), color=COLOR_NAMES[tcolor],
            x=tpos[0], y=tpos[1],
        )
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        # Check if the target was erased
        target_exists = any(e.id == self._erase_id for e in engine.entities)
        others_exist = sum(1 for e in engine.entities if e.id != self._erase_id and e.id in self._all_ids)
        expected_others = self._initial_count - 1

        if not target_exists and others_exist == expected_others:
            return {"reward": 1.0, "terminated": True, "info": {"erased": True}}
        elif not target_exists:
            # Erased target but also some others
            penalty = abs(others_exist - expected_others) * 0.2
            return {"reward": 0.5 - penalty, "terminated": True, "info": {"erased": True, "collateral": True}}
        else:
            return {"reward": -0.01, "terminated": False, "info": {"erased": False}}
