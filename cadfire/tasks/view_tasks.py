"""
Viewport manipulation tasks: zoom, pan, fit view.

These teach the agent to navigate the canvas efficiently.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import CircleEntity, RectangleEntity, LineEntity
from cadfire.tasks.base import BaseTask
from cadfire.tasks.registry import register_task


@register_task
class FitViewTask(BaseTask):
    """Adjust viewport to show all geometry with appropriate margin."""
    task_name = "fit_view"
    task_category = "view"
    difficulty = 1.0

    def generate_prompt_variants(self):
        return [
            "Fit all geometry to view",
            "Zoom to show everything",
            "Fit the drawing to the viewport",
            "Use FIT_VIEW to see all objects",
            "Zoom extents",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Create scattered geometry
        n = self.rng.randint(3, 8)
        for _ in range(n):
            cx, cy = self.rng.uniform(50, 950, 2)
            r = self.rng.uniform(20, 80)
            e = CircleEntity(center=np.array([cx, cy]), radius=r,
                             color_index=int(self.rng.randint(0, 8)))
            engine.add_entity(e, save_undo=False)

        # Offset the viewport so things aren't visible
        engine.viewport.center = np.array([
            self.rng.uniform(-500, 1500),
            self.rng.uniform(-500, 1500),
        ])
        engine.viewport.zoom = self.rng.uniform(0.1, 5.0)

        prompt = self.rng.choice(self.generate_prompt_variants())
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        occ = self.bbox_occupancy_reward(engine)
        terminated = occ > 0.8
        return {"reward": occ, "terminated": terminated, "info": {"occupancy": occ}}


@register_task
class ZoomToCenterTask(BaseTask):
    """Zoom/pan to bring a target coordinate into the center of the view."""
    task_name = "zoom_to_center"
    task_category = "view"
    difficulty = 2.0

    def generate_prompt_variants(self):
        return [
            "Navigate to ({tx:.0f},{ty:.0f})",
            "Pan to show the area around ({tx:.0f},{ty:.0f})",
            "Center the view on ({tx:.0f},{ty:.0f})",
            "Zoom to ({tx:.0f},{ty:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Place a target marker
        self._target = np.array([
            self.rng.uniform(100, 900),
            self.rng.uniform(100, 900),
        ])
        marker = CircleEntity(center=self._target.copy(), radius=10, color_index=1)
        engine.add_entity(marker, save_undo=False)

        # Start viewport far from target
        engine.viewport.center = np.array([
            self.rng.uniform(0, 1000),
            self.rng.uniform(0, 1000),
        ])

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(tx=self._target[0], ty=self._target[1])
        return {"prompt": prompt}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        # How close is target to viewport center?
        dist = np.linalg.norm(engine.viewport.center - self._target)
        vis_min, vis_max = engine.viewport.visible_bounds()
        vis_size = np.linalg.norm(vis_max - vis_min)

        # Normalized distance (0 = perfect centering)
        norm_dist = dist / max(vis_size, 1e-6)

        # Is target in center 50% of viewport?
        in_center = norm_dist < 0.25
        reward = max(0, 1.0 - norm_dist * 2)
        terminated = in_center

        return {"reward": reward, "terminated": terminated,
                "info": {"dist": float(dist), "in_center": in_center}}
