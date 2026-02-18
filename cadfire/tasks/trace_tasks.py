"""
Tracing tasks: the agent traces over a reference image.

The reference image is provided in channels 3-5 of the observation.
The agent must draw geometry that overlaps with the reference.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import LineEntity, CircleEntity, PolylineEntity
from cadfire.tasks.base import BaseTask, UTILITY_TOOLS
from cadfire.tasks.registry import register_task


def _render_target_to_image(entities, render_size=128, world_size=1000.0):
    """Render entities to a binary reference image."""
    img = np.zeros((render_size, render_size, 3), dtype=np.uint8)
    color = np.array([255, 255, 255], dtype=np.uint8)

    for e in entities:
        pts = e.tessellate()
        if len(pts) < 2:
            continue
        px = np.clip((pts[:, 0] / world_size * render_size).astype(int), 0, render_size - 1)
        py = np.clip((pts[:, 1] / world_size * render_size).astype(int), 0, render_size - 1)
        # Draw line segments
        for i in range(len(px) - 1):
            _draw_line_on_img(img, px[i], py[i], px[i+1], py[i+1], color)
    return img


def _draw_line_on_img(img, x0, y0, x1, y1, color):
    """Simple Bresenham."""
    H, W = img.shape[:2]
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < H and 0 <= x0 < W:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy


@register_task
class TraceLineTask(BaseTask):
    """Trace a line from a reference image."""
    task_name = "trace_line"
    task_category = "trace"
    difficulty = 2.0

    def allowed_tools(self):
        return UTILITY_TOOLS + ["LINE"]

    def generate_prompt_variants(self):
        return [
            "Trace the line shown in the reference image",
            "Draw over the reference line",
            "Replicate the line from the reference",
            "Trace the displayed line segment",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        start = np.array([self.rng.uniform(100, 400), self.rng.uniform(100, 400)])
        end = np.array([self.rng.uniform(600, 900), self.rng.uniform(600, 900)])
        self._target = LineEntity(start=start, end=end)
        ref_img = _render_target_to_image([self._target])

        prompt = self.rng.choice(self.generate_prompt_variants())
        return {"prompt": prompt, "reference_image": ref_img,
                "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        lines = [e for e in engine.entities if e.entity_type == "LINE"]
        if not lines:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}
        iou = self.iou_reward(lines, [self._target])
        reward = iou + (0.1 if len(lines) == 1 else -0.05)
        return {"reward": reward, "terminated": iou > 0.7, "info": {"iou": iou}}


@register_task
class TraceCircleTask(BaseTask):
    """Trace a circle from a reference image."""
    task_name = "trace_circle"
    task_category = "trace"
    difficulty = 2.5

    def allowed_tools(self):
        return UTILITY_TOOLS + ["CIRCLE"]

    def generate_prompt_variants(self):
        return [
            "Trace the circle shown in the reference",
            "Draw a circle matching the reference image",
            "Replicate the reference circle",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        center = np.array([self.rng.uniform(200, 800), self.rng.uniform(200, 800)])
        radius = float(self.rng.uniform(50, 200))
        self._target = CircleEntity(center=center, radius=radius)
        ref_img = _render_target_to_image([self._target])

        prompt = self.rng.choice(self.generate_prompt_variants())
        return {"prompt": prompt, "reference_image": ref_img,
                "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        circles = [e for e in engine.entities if e.entity_type == "CIRCLE"]
        if not circles:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}
        iou = self.iou_reward(circles, [self._target])
        reward = iou + (0.1 if len(circles) == 1 else -0.05)
        return {"reward": reward, "terminated": iou > 0.65, "info": {"iou": iou}}


@register_task
class TraceCompositeTask(BaseTask):
    """Trace a composite shape (multiple primitives) from reference."""
    task_name = "trace_composite"
    task_category = "trace"
    difficulty = 5.0

    def allowed_tools(self):
        return UTILITY_TOOLS + ["LINE", "CIRCLE", "POLYLINE"]

    def generate_prompt_variants(self):
        return [
            "Trace all shapes shown in the reference image",
            "Replicate the drawing from the reference",
            "Draw over the reference geometry",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._targets = []
        n = self.rng.randint(2, 5)
        for _ in range(n):
            if self.rng.rand() > 0.5:
                start = np.array([self.rng.uniform(100, 900), self.rng.uniform(100, 900)])
                end = np.array([self.rng.uniform(100, 900), self.rng.uniform(100, 900)])
                self._targets.append(LineEntity(start=start, end=end))
            else:
                center = np.array([self.rng.uniform(200, 800), self.rng.uniform(200, 800)])
                radius = float(self.rng.uniform(30, 150))
                self._targets.append(CircleEntity(center=center, radius=radius))

        ref_img = _render_target_to_image(self._targets)
        prompt = self.rng.choice(self.generate_prompt_variants())
        return {"prompt": prompt, "reference_image": ref_img,
                "target_entities": self._targets}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        if not engine.entities:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}
        iou = self.iou_reward(engine.entities, self._targets)
        reward = iou
        return {"reward": reward, "terminated": iou > 0.5, "info": {"iou": iou}}
