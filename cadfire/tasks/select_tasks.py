"""
Selection tasks: the agent must select specific entities.

Tests:
  - Selecting a specific shape by clicking near it
  - Selecting all shapes of a specific color
  - Multi-selecting with a mask
  - Deselecting all

Ambiguity-free design: when a prompt refers to a target by type alone
(e.g. "Select the rectangle"), only one entity of that type exists on
the canvas so the instruction is unambiguous.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    CircleEntity, RectangleEntity, LineEntity, EllipseEntity,
    PolygonEntity, ArcEntity, Entity,
)
from cadfire.tasks.base import BaseTask, UTILITY_TOOLS
from cadfire.tasks.registry import register_task


COLOR_NAMES = ["white", "red", "yellow", "green", "cyan", "blue", "magenta", "gray"]

# Shape factory helpers used to guarantee type-uniqueness.
# Each entry is (entity_type_name, factory_callable(rng) -> Entity).
_SHAPE_FACTORIES: List[Tuple[str, Any]] = [
    ("circle", lambda rng, cx, cy, r, ci: CircleEntity(
        center=np.array([cx, cy]), radius=r, color_index=ci)),
    ("rectangle", lambda rng, cx, cy, r, ci: RectangleEntity(
        corner=np.array([cx - r, cy - r * 0.75]), width=r * 2, height=r * 1.5, color_index=ci)),
    ("ellipse", lambda rng, cx, cy, r, ci: EllipseEntity(
        center=np.array([cx, cy]), semi_major=r * 1.3, semi_minor=r * 0.7, color_index=ci)),
    ("polygon", lambda rng, cx, cy, r, ci: PolygonEntity(
        center=np.array([cx, cy]), radius=r, sides=int(rng.randint(5, 9)), color_index=ci)),
    ("line", lambda rng, cx, cy, r, ci: LineEntity(
        start=np.array([cx - r, cy - r]), end=np.array([cx + r, cy + r]), color_index=ci)),
    ("arc", lambda rng, cx, cy, r, ci: ArcEntity(
        center=np.array([cx, cy]), radius=r,
        start_angle=float(rng.uniform(0, 90)),
        end_angle=float(rng.uniform(180, 270)),
        color_index=ci)),
]


def _make_unique_shapes(
    rng: np.random.RandomState,
    n: int,
    target_type_idx: int | None = None,
) -> Tuple[List[Entity], int]:
    """Create *n* shapes where every entity has a distinct type.

    Returns ``(shapes_list, target_index)`` where ``target_index`` is the
    index of the target shape inside the returned list.

    ``target_type_idx`` (if given) forces the target to use that factory
    index; otherwise one is chosen at random.
    """
    assert 2 <= n <= len(_SHAPE_FACTORIES)
    # Pick n distinct factory indices
    idxs = list(rng.choice(len(_SHAPE_FACTORIES), size=n, replace=False))

    # Decide which one is the target
    if target_type_idx is not None and target_type_idx in idxs:
        target_pos = idxs.index(target_type_idx)
    else:
        target_pos = int(rng.randint(0, n))

    shapes: List[Entity] = []
    for factory_idx in idxs:
        _, factory = _SHAPE_FACTORIES[factory_idx]
        cx, cy = rng.uniform(150, 850, 2)
        r = rng.uniform(40, 120)
        ci = int(rng.randint(0, 8))
        shapes.append(factory(rng, float(cx), float(cy), float(r), ci))

    return shapes, target_pos


@register_task
class SelectShapeTask(BaseTask):
    """Select a specific shape by its position/type.

    Each shape type appears at most once so that prompts like
    "Select the rectangle" are unambiguous.
    """
    task_name = "select_shape"
    task_category = "select"
    difficulty = 1.5

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "DESELECT"]

    def generate_prompt_variants(self):
        return [
            "Select the {shape} at ({x:.0f},{y:.0f})",
            "Click on the {shape} near ({x:.0f},{y:.0f})",
            "Find and select the {color} {shape}",
            "Select the {shape}",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Create 3-5 shapes, each with a *unique* entity type
        n = int(self.rng.randint(3, min(6, len(_SHAPE_FACTORIES) + 1)))
        self._shapes, self._target_idx = _make_unique_shapes(self.rng, n)

        for e in self._shapes:
            engine.add_entity(e, save_undo=False)

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

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "MULTISELECT", "DESELECT"]

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
    """Select and erase specific entities.

    Each shape type appears at most once so prompts like
    "Delete the circle" are unambiguous.
    """
    task_name = "erase_selection"
    task_category = "modify"
    difficulty = 2.5

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "ERASE"]

    def generate_prompt_variants(self):
        return [
            "Delete the {color} {shape}",
            "Erase the {shape} at ({x:.0f},{y:.0f})",
            "Remove the {color} {shape} from the drawing",
            "Select and delete the {shape}",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        n = int(self.rng.randint(3, min(6, len(_SHAPE_FACTORIES) + 1)))
        shapes, target_idx = _make_unique_shapes(self.rng, n)

        self._all_ids = set()
        for e in shapes:
            engine.add_entity(e, save_undo=False)
            self._all_ids.add(e.id)

        target = shapes[target_idx]
        self._erase_id = target.id
        self._initial_count = n

        c = target.centroid()
        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape=target.entity_type.lower(),
            color=COLOR_NAMES[target.color_index],
            x=c[0], y=c[1],
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
