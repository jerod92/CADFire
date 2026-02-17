"""
Drawing tasks: the agent must create specific geometric primitives.

Each task:
  1. Tells the agent what to draw (via text prompt)
  2. Generates a random target configuration
  3. Optionally provides a reference image
  4. Rewards based on geometric IoU / proximity to target
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    LineEntity, CircleEntity, RectangleEntity, ArcEntity,
    PolygonEntity, EllipseEntity, PolylineEntity,
)
from cadfire.tasks.base import BaseTask
from cadfire.tasks.registry import register_task


COLOR_NAMES = ["white", "red", "yellow", "green", "cyan", "blue", "magenta", "gray"]


def _random_pos(rng, margin=100, world=1000):
    return rng.uniform(margin, world - margin, size=2)


def _random_radius(rng, min_r=20, max_r=200):
    return rng.uniform(min_r, max_r)


@register_task
class DrawLineTask(BaseTask):
    task_name = "draw_line"
    task_category = "draw"
    difficulty = 1.0

    def generate_prompt_variants(self):
        return [
            "Draw a line from ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})",
            "Create a line segment between ({x1:.0f},{y1:.0f}) and ({x2:.0f},{y2:.0f})",
            "Use the LINE tool to connect ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})",
            "Draw a {color} line from ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._start = _random_pos(self.rng)
        self._end = _random_pos(self.rng)
        self._color_idx = int(self.rng.randint(0, len(COLOR_NAMES)))
        self._target = LineEntity(
            start=self._start.copy(), end=self._end.copy(),
            color_index=self._color_idx,
        )

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            x1=self._start[0], y1=self._start[1],
            x2=self._end[0], y2=self._end[1],
            color=COLOR_NAMES[self._color_idx],
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        lines = [e for e in engine.entities if e.entity_type == "LINE"]
        if not lines:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}

        iou = self.iou_reward(lines, [self._target])
        # Bonus for having exactly one line
        count_bonus = 0.1 if len(lines) == 1 else -0.05 * abs(len(lines) - 1)
        reward = iou + count_bonus
        terminated = iou > 0.8

        return {"reward": reward, "terminated": terminated,
                "info": {"iou": iou, "num_lines": len(lines)}}


@register_task
class DrawCircleTask(BaseTask):
    task_name = "draw_circle"
    task_category = "draw"
    difficulty = 1.0

    def generate_prompt_variants(self):
        return [
            "Draw a circle at ({cx:.0f},{cy:.0f}) with radius {r:.0f}",
            "Create a circle centered at ({cx:.0f},{cy:.0f}), radius {r:.0f}",
            "Make a {color} circle at ({cx:.0f},{cy:.0f}) with r={r:.0f}",
            "Use the CIRCLE tool at center ({cx:.0f},{cy:.0f}), radius {r:.0f}",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._center = _random_pos(self.rng, margin=200)
        self._radius = _random_radius(self.rng)
        self._color_idx = int(self.rng.randint(0, len(COLOR_NAMES)))
        self._target = CircleEntity(
            center=self._center.copy(), radius=self._radius,
            color_index=self._color_idx,
        )

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            cx=self._center[0], cy=self._center[1],
            r=self._radius, color=COLOR_NAMES[self._color_idx],
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        circles = [e for e in engine.entities if e.entity_type == "CIRCLE"]
        if not circles:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}

        iou = self.iou_reward(circles, [self._target])
        count_bonus = 0.1 if len(circles) == 1 else -0.05 * abs(len(circles) - 1)
        reward = iou + count_bonus
        terminated = iou > 0.75

        return {"reward": reward, "terminated": terminated,
                "info": {"iou": iou, "num_circles": len(circles)}}


@register_task
class DrawRectangleTask(BaseTask):
    task_name = "draw_rectangle"
    task_category = "draw"
    difficulty = 1.5

    def generate_prompt_variants(self):
        return [
            "Draw a rectangle at ({x:.0f},{y:.0f}) with width {w:.0f} and height {h:.0f}",
            "Create a {w:.0f}x{h:.0f} rectangle at ({x:.0f},{y:.0f})",
            "Make a {color} rectangle from ({x:.0f},{y:.0f}) sized {w:.0f} by {h:.0f}",
            "Draw a box at ({x:.0f},{y:.0f}), {w:.0f} wide and {h:.0f} tall",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._corner = _random_pos(self.rng, margin=200)
        self._w = float(self.rng.uniform(50, 400))
        self._h = float(self.rng.uniform(50, 400))
        self._color_idx = int(self.rng.randint(0, len(COLOR_NAMES)))
        self._target = RectangleEntity(
            corner=self._corner.copy(), width=self._w, height=self._h,
            color_index=self._color_idx,
        )

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            x=self._corner[0], y=self._corner[1],
            w=self._w, h=self._h, color=COLOR_NAMES[self._color_idx],
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        rects = [e for e in engine.entities if e.entity_type == "RECTANGLE"]
        if not rects:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}

        iou = self.iou_reward(rects, [self._target])
        reward = iou + (0.1 if len(rects) == 1 else -0.05 * abs(len(rects) - 1))
        return {"reward": reward, "terminated": iou > 0.75,
                "info": {"iou": iou}}


@register_task
class DrawPolygonTask(BaseTask):
    task_name = "draw_polygon"
    task_category = "draw"
    difficulty = 2.0

    def generate_prompt_variants(self):
        return [
            "Draw a {sides}-sided polygon at ({cx:.0f},{cy:.0f}) with radius {r:.0f}",
            "Create a regular {sides}-gon centered at ({cx:.0f},{cy:.0f}), r={r:.0f}",
            "Make a {color} {sides}-sided shape at ({cx:.0f},{cy:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._center = _random_pos(self.rng, margin=200)
        self._radius = _random_radius(self.rng, 40, 200)
        self._sides = int(self.rng.randint(3, 9))
        self._color_idx = int(self.rng.randint(0, len(COLOR_NAMES)))
        self._target = PolygonEntity(
            center=self._center.copy(), radius=self._radius,
            sides=self._sides, color_index=self._color_idx,
        )

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            cx=self._center[0], cy=self._center[1],
            r=self._radius, sides=self._sides,
            color=COLOR_NAMES[self._color_idx],
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        polys = [e for e in engine.entities if e.entity_type == "POLYGON"]
        if not polys:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}
        iou = self.iou_reward(polys, [self._target])
        reward = iou + (0.1 if len(polys) == 1 else -0.05)
        return {"reward": reward, "terminated": iou > 0.7, "info": {"iou": iou}}


@register_task
class DrawEllipseTask(BaseTask):
    task_name = "draw_ellipse"
    task_category = "draw"
    difficulty = 2.5

    def generate_prompt_variants(self):
        return [
            "Draw an ellipse at ({cx:.0f},{cy:.0f}) with axes {a:.0f} and {b:.0f}",
            "Create an ellipse centered at ({cx:.0f},{cy:.0f}), {a:.0f}x{b:.0f}",
            "Make a {color} ellipse at ({cx:.0f},{cy:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._center = _random_pos(self.rng, margin=200)
        self._a = float(self.rng.uniform(40, 250))
        self._b = float(self.rng.uniform(20, self._a * 0.8))
        self._color_idx = int(self.rng.randint(0, len(COLOR_NAMES)))
        self._target = EllipseEntity(
            center=self._center.copy(), semi_major=self._a, semi_minor=self._b,
            color_index=self._color_idx,
        )

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            cx=self._center[0], cy=self._center[1],
            a=self._a, b=self._b, color=COLOR_NAMES[self._color_idx],
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        ells = [e for e in engine.entities if e.entity_type == "ELLIPSE"]
        if not ells:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}
        iou = self.iou_reward(ells, [self._target])
        reward = iou + (0.1 if len(ells) == 1 else -0.05)
        return {"reward": reward, "terminated": iou > 0.7, "info": {"iou": iou}}


@register_task
class DrawArcTask(BaseTask):
    task_name = "draw_arc"
    task_category = "draw"
    difficulty = 3.0

    def generate_prompt_variants(self):
        return [
            "Draw an arc at ({cx:.0f},{cy:.0f}), r={r:.0f}, from {sa:.0f} to {ea:.0f} degrees",
            "Create an arc centered at ({cx:.0f},{cy:.0f}) with radius {r:.0f}",
            "Make a {color} arc at ({cx:.0f},{cy:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._center = _random_pos(self.rng, margin=200)
        self._radius = _random_radius(self.rng, 40, 200)
        self._sa = float(self.rng.uniform(0, 270))
        self._ea = float(self._sa + self.rng.uniform(30, 300))
        self._color_idx = int(self.rng.randint(0, len(COLOR_NAMES)))
        self._target = ArcEntity(
            center=self._center.copy(), radius=self._radius,
            start_angle=self._sa, end_angle=self._ea,
            color_index=self._color_idx,
        )

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            cx=self._center[0], cy=self._center[1],
            r=self._radius, sa=self._sa, ea=self._ea,
            color=COLOR_NAMES[self._color_idx],
        )
        return {"prompt": prompt, "target_entities": [self._target]}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        arcs = [e for e in engine.entities if e.entity_type == "ARC"]
        if not arcs:
            return {"reward": -0.01, "terminated": False, "info": {"iou": 0}}
        iou = self.iou_reward(arcs, [self._target])
        reward = iou + (0.1 if len(arcs) == 1 else -0.05)
        return {"reward": reward, "terminated": iou > 0.65, "info": {"iou": iou}}


@register_task
class DrawMultiPrimitiveTask(BaseTask):
    """Draw multiple primitives in one episode."""
    task_name = "draw_multi_primitive"
    task_category = "draw"
    difficulty = 4.0

    def generate_prompt_variants(self):
        return [
            "Draw a {color1} circle at ({cx:.0f},{cy:.0f}) and a {color2} rectangle at ({rx:.0f},{ry:.0f})",
            "Create a circle and a rectangle on the canvas",
            "Draw both a {color1} circle (r={r:.0f}) and a {color2} box ({w:.0f}x{h:.0f})",
        ]

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        self._c_center = _random_pos(self.rng, margin=250)
        self._c_radius = _random_radius(self.rng, 30, 150)
        self._r_corner = _random_pos(self.rng, margin=250)
        self._r_w = float(self.rng.uniform(40, 300))
        self._r_h = float(self.rng.uniform(40, 300))
        self._c1 = int(self.rng.randint(0, len(COLOR_NAMES)))
        self._c2 = int(self.rng.randint(0, len(COLOR_NAMES)))

        self._targets = [
            CircleEntity(center=self._c_center.copy(), radius=self._c_radius, color_index=self._c1),
            RectangleEntity(corner=self._r_corner.copy(), width=self._r_w, height=self._r_h, color_index=self._c2),
        ]

        template = self.rng.choice(self.generate_prompt_variants())
        prompt = template.format(
            cx=self._c_center[0], cy=self._c_center[1],
            r=self._c_radius,
            rx=self._r_corner[0], ry=self._r_corner[1],
            w=self._r_w, h=self._r_h,
            color1=COLOR_NAMES[self._c1], color2=COLOR_NAMES[self._c2],
        )
        return {"prompt": prompt, "target_entities": self._targets}

    def compute_reward(self, engine: CADEngine, action: Dict, step: int) -> Dict:
        iou = self.iou_reward(engine.entities, self._targets)
        has_circle = any(e.entity_type == "CIRCLE" for e in engine.entities)
        has_rect = any(e.entity_type == "RECTANGLE" for e in engine.entities)
        bonus = 0.1 * (has_circle + has_rect)
        reward = iou + bonus
        return {"reward": reward, "terminated": iou > 0.6,
                "info": {"iou": iou, "has_circle": has_circle, "has_rect": has_rect}}
