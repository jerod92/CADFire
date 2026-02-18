"""
Modify tasks: move, copy, rotate, scale, mirror selected objects.

These require the agent to:
  1. Select the right entity
  2. Apply the correct transformation
  3. Achieve a target configuration
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import CircleEntity, RectangleEntity
from cadfire.tasks.base import BaseTask, UTILITY_TOOLS
from cadfire.tasks.registry import register_task


COLOR_NAMES = ["white", "red", "yellow", "green", "cyan", "blue", "magenta", "gray"]


@register_task
class MoveShapeTask(BaseTask):
    """Move a shape from one position to another."""
    task_name = "move_shape"
    task_category = "modify"
    difficulty = 3.0

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "MOVE"]

    def generate_prompt_variants(self):
        return [
            "Move the {shape} to ({tx:.0f},{ty:.0f})",
            "Drag the {color} {shape} to ({tx:.0f},{ty:.0f})",
            "Relocate the {shape} from ({sx:.0f},{sy:.0f}) to ({tx:.0f},{ty:.0f})",
            "Move the {shape} by ({dx:.0f},{dy:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Create the shape at a starting position
        self._start = np.array([self.rng.uniform(100, 500), self.rng.uniform(100, 500)])
        self._target_pos = np.array([self.rng.uniform(500, 900), self.rng.uniform(500, 900)])
        self._radius = float(self.rng.uniform(30, 80))
        self._color = int(self.rng.randint(0, 8))

        e = CircleEntity(center=self._start.copy(), radius=self._radius, color_index=self._color)
        engine.add_entity(e, save_undo=False)
        self._entity_id = e.id

        # Target: where the circle should end up
        self._target_entity = CircleEntity(
            center=self._target_pos.copy(), radius=self._radius, color_index=self._color,
        )

        dx = self._target_pos[0] - self._start[0]
        dy = self._target_pos[1] - self._start[1]
        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape="circle", color=COLOR_NAMES[self._color],
            sx=self._start[0], sy=self._start[1],
            tx=self._target_pos[0], ty=self._target_pos[1],
            dx=dx, dy=dy,
        )
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        e = engine.get_entity(self._entity_id)
        if e is None:
            return {"reward": -0.5, "terminated": True, "info": {"error": "entity_deleted"}}

        dist = np.linalg.norm(e.centroid() - self._target_pos)
        tolerance = self._radius * 0.3
        reward = max(0, 1.0 - dist / 500.0)
        terminated = dist < tolerance

        if terminated:
            reward = 1.0

        return {"reward": reward, "terminated": terminated,
                "info": {"dist": float(dist), "tolerance": tolerance}}


@register_task
class RotateShapeTask(BaseTask):
    """Rotate a shape by a specified angle."""
    task_name = "rotate_shape"
    task_category = "modify"
    difficulty = 4.0

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "ROTATE"]

    def generate_prompt_variants(self):
        return [
            "Rotate the {shape} by {angle:.0f} degrees",
            "Turn the {color} {shape} {angle:.0f} degrees",
            "Rotate the selected shape {angle:.0f} degrees around its center",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Use a rectangle so rotation is visually meaningful
        self._center = np.array([500.0, 500.0])
        self._w = float(self.rng.uniform(100, 300))
        self._h = float(self.rng.uniform(50, 150))
        self._angle = float(self.rng.choice([45, 90, 135, 180, -45, -90]))
        self._color = int(self.rng.randint(0, 8))

        corner = self._center - np.array([self._w/2, self._h/2])
        e = RectangleEntity(corner=corner, width=self._w, height=self._h, color_index=self._color)
        engine.add_entity(e, save_undo=False)
        self._entity_id = e.id

        # Create target (rotated version)
        self._target = e.clone()
        self._target.rotate(self._angle, self._center[0], self._center[1])

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape="rectangle", color=COLOR_NAMES[self._color],
            angle=self._angle,
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        e = engine.get_entity(self._entity_id)
        if e is None:
            return {"reward": -0.5, "terminated": True, "info": {}}
        iou = self.iou_reward([e], [self._target])
        reward = iou
        terminated = iou > 0.7
        return {"reward": reward, "terminated": terminated, "info": {"iou": iou}}


@register_task
class ScaleShapeTask(BaseTask):
    """Scale a shape by a specified factor."""
    task_name = "scale_shape"
    task_category = "modify"
    difficulty = 3.5

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "SCALE"]

    def generate_prompt_variants(self):
        return [
            "Scale the {shape} by a factor of {factor:.1f}",
            "Resize the {color} {shape} to {factor:.1f}x",
            "Make the {shape} {factor:.1f} times its current size",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._center = np.array([500.0, 500.0])
        self._radius = float(self.rng.uniform(40, 100))
        self._factor = float(self.rng.choice([0.5, 1.5, 2.0, 2.5]))
        self._color = int(self.rng.randint(0, 8))

        e = CircleEntity(center=self._center.copy(), radius=self._radius, color_index=self._color)
        engine.add_entity(e, save_undo=False)
        self._entity_id = e.id

        self._target = CircleEntity(
            center=self._center.copy(), radius=self._radius * self._factor,
            color_index=self._color,
        )

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape="circle", color=COLOR_NAMES[self._color], factor=self._factor,
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        e = engine.get_entity(self._entity_id)
        if e is None:
            return {"reward": -0.5, "terminated": True, "info": {}}
        iou = self.iou_reward([e], [self._target])
        reward = iou
        return {"reward": reward, "terminated": iou > 0.7, "info": {"iou": iou}}


@register_task
class CopyShapeTask(BaseTask):
    """Copy a shape to a new position."""
    task_name = "copy_shape"
    task_category = "modify"
    difficulty = 3.5

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "COPY"]

    def generate_prompt_variants(self):
        return [
            "Copy the {shape} to ({tx:.0f},{ty:.0f})",
            "Duplicate the {color} {shape} at ({tx:.0f},{ty:.0f})",
            "Make a copy of the {shape} at ({tx:.0f},{ty:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._src = np.array([300.0, 300.0])
        self._dst = np.array([self.rng.uniform(500, 900), self.rng.uniform(500, 900)])
        self._radius = float(self.rng.uniform(30, 80))
        self._color = int(self.rng.randint(0, 8))

        e = CircleEntity(center=self._src.copy(), radius=self._radius, color_index=self._color)
        engine.add_entity(e, save_undo=False)

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape="circle", color=COLOR_NAMES[self._color],
            tx=self._dst[0], ty=self._dst[1],
        )
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        circles = [e for e in engine.entities if e.entity_type == "CIRCLE"]
        # Need at least 2 circles (original + copy)
        if len(circles) < 2:
            return {"reward": -0.01, "terminated": False, "info": {"count": len(circles)}}

        # Check if any circle is near the target
        dists = [np.linalg.norm(c.centroid() - self._dst) for c in circles]
        best = min(dists)
        tolerance = self._radius * 0.5
        reward = max(0, 1.0 - best / 500.0)
        terminated = best < tolerance and len(circles) >= 2

        if terminated:
            reward = 1.0

        return {"reward": reward, "terminated": terminated,
                "info": {"best_dist": float(best), "count": len(circles)}}


@register_task
class ChangeLayerTask(BaseTask):
    """Change the layer of selected objects."""
    task_name = "change_layer"
    task_category = "property"
    difficulty = 2.0

    def allowed_tools(self):
        return UTILITY_TOOLS + ["SELECT", "CHANGE_LAYER"]

    def generate_prompt_variants(self):
        return [
            "Move the {shape} to layer {layer}",
            "Change the layer of the {color} {shape} to {layer}",
            "Put the {shape} on layer {layer}",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Create entity on layer 0
        self._color = int(self.rng.randint(0, 8))
        e = CircleEntity(
            center=np.array([500.0, 500.0]),
            radius=float(self.rng.uniform(40, 100)),
            color_index=self._color,
        )
        e.layer = 0
        engine.add_entity(e, save_undo=False)
        self._entity_id = e.id
        self._target_layer = int(self.rng.randint(1, 8))

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            shape="circle", color=COLOR_NAMES[self._color],
            layer=self._target_layer,
        )
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        e = engine.get_entity(self._entity_id)
        if e is None:
            return {"reward": -0.5, "terminated": True, "info": {}}
        correct = e.layer == self._target_layer
        return {"reward": 1.0 if correct else -0.01, "terminated": correct,
                "info": {"current_layer": e.layer, "target": self._target_layer}}
