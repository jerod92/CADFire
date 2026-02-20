"""
Supervised single-step tasks for geometric-editing tools:

  FilletTask   – FILLET : round the corner where two pre-selected lines meet
  ChamferTask  – CHAMFER: bevel the corner where two pre-selected lines meet
  TrimTask     – TRIM   : trim one line back to its intersection with a cutter

All three teach the model to aim the cursor at a geometrically meaningful
point (the shared corner or the segment to remove) and call the right tool.

Cursor weights
──────────────
FILLET / CHAMFER : 0.9  – spatial precision matters; the agent clicks the
                          exact corner vertex to identify which corner to round.
TRIM             : 0.85 – agent clicks the segment it wants removed (on the
                          far side of the intersection from the "keep" end).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import LineEntity


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_corner_scene(
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (corner, endA, endB) — two line segments meeting at *corner*.

    The corner is placed near the canvas centre; the two free ends point
    outward in random directions that are at least 60° apart so they look
    like a genuine corner.
    """
    cx = float(rng.uniform(300, 700))
    cy = float(rng.uniform(300, 700))
    corner = np.array([cx, cy], dtype=np.float64)

    angle_a = float(rng.uniform(0, 2 * np.pi))
    # Ensure the two arms span at least 60° and at most 150° (avoid straight lines)
    spread = float(rng.uniform(np.pi / 3, 5 * np.pi / 6))
    angle_b = angle_a + spread

    length_a = float(rng.uniform(150, 300))
    length_b = float(rng.uniform(150, 300))

    end_a = corner + length_a * np.array([np.cos(angle_a), np.sin(angle_a)])
    end_b = corner + length_b * np.array([np.cos(angle_b), np.sin(angle_b)])

    end_a = np.clip(end_a, 50.0, 950.0)
    end_b = np.clip(end_b, 50.0, 950.0)

    return corner, end_a, end_b


_CORNER_SHAPES = ["corner", "angle", "joint", "intersection", "vertex"]


# ─────────────────────────────────────────────────────────────────────────────
# FilletTask
# ─────────────────────────────────────────────────────────────────────────────

_FILLET_PROMPTS = [
    "Add a fillet to the {corner}",
    "Round the {corner} with a fillet",
    "Apply a fillet at the {corner}",
    "Fillet the {corner}",
    "Create a rounded corner at the {corner}",
    "Round the sharp {corner}",
    "Apply a radius fillet to the {corner}",
    "Smooth the {corner} with a fillet",
    "Fillet the corner where the lines meet",
    "Round the joint between the two lines",
    "Add a fillet here",
    "Round this corner",
    "Fillet it",
]


class FilletTask:
    """
    Single-step FILLET supervised task.

    Two line segments are placed sharing a corner vertex.  Both are
    pre-selected.  The oracle cursor points at the shared corner — the natural
    pick-point for a fillet operation (the agent clicks the corner it wants
    rounded).

    cursor_loss_weight = 0.9: spatial accuracy is critical for this tool.
    """

    tool_name = "FILLET"
    cursor_loss_weight = 0.9

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._corner: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        corner, end_a, end_b = _rand_corner_scene(self.rng)
        self._corner = corner

        color = int(self.rng.randint(0, 8))
        line_a = LineEntity(start=corner.copy(), end=end_a, color_index=color)
        line_b = LineEntity(start=corner.copy(), end=end_b, color_index=color)

        engine.add_entity(line_a, save_undo=False)
        engine.add_entity(line_b, save_undo=False)
        engine.selected_ids.add(line_a.id)
        engine.selected_ids.add(line_b.id)

        noun = _CORNER_SHAPES[int(self.rng.randint(len(_CORNER_SHAPES)))]
        tmpl = _FILLET_PROMPTS[int(self.rng.randint(len(_FILLET_PROMPTS)))]
        prompt = tmpl.format(corner=noun)

        return {
            "prompt": prompt,
            "line_a": line_a,
            "line_b": line_b,
            "corner": corner,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "FILLET",
            "cursor_world": self._corner,
            "cursor_weight": self.cursor_loss_weight,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ChamferTask
# ─────────────────────────────────────────────────────────────────────────────

_CHAMFER_PROMPTS = [
    "Chamfer the {corner}",
    "Bevel the {corner}",
    "Apply a chamfer to the {corner}",
    "Add a chamfer at the {corner}",
    "Create a beveled edge at the {corner}",
    "Cut the {corner} with a chamfer",
    "Apply a 45-degree chamfer to the {corner}",
    "Bevel the sharp {corner}",
    "Chamfer the corner where the lines meet",
    "Add a bevel here",
    "Chamfer it",
    "Bevel the joint",
]


class ChamferTask:
    """
    Single-step CHAMFER supervised task.

    Identical scene layout to FilletTask (two lines sharing a corner), but
    teaches the CHAMFER tool instead.  The cursor again targets the corner
    vertex — the pick-point that identifies which corner to bevel.

    cursor_loss_weight = 0.9: same reasoning as FilletTask.
    """

    tool_name = "CHAMFER"
    cursor_loss_weight = 0.9

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._corner: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        corner, end_a, end_b = _rand_corner_scene(self.rng)
        self._corner = corner

        color = int(self.rng.randint(0, 8))
        line_a = LineEntity(start=corner.copy(), end=end_a, color_index=color)
        line_b = LineEntity(start=corner.copy(), end=end_b, color_index=color)

        engine.add_entity(line_a, save_undo=False)
        engine.add_entity(line_b, save_undo=False)
        engine.selected_ids.add(line_a.id)
        engine.selected_ids.add(line_b.id)

        noun = _CORNER_SHAPES[int(self.rng.randint(len(_CORNER_SHAPES)))]
        tmpl = _CHAMFER_PROMPTS[int(self.rng.randint(len(_CHAMFER_PROMPTS)))]
        prompt = tmpl.format(corner=noun)

        return {
            "prompt": prompt,
            "line_a": line_a,
            "line_b": line_b,
            "corner": corner,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "CHAMFER",
            "cursor_world": self._corner,
            "cursor_weight": self.cursor_loss_weight,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TrimTask
# ─────────────────────────────────────────────────────────────────────────────

_TRIM_PROMPTS = [
    "Trim the line at the intersection",
    "Cut the line back to where it crosses the other line",
    "Trim the overlapping segment",
    "Remove the part of the line past the intersection",
    "Trim the line",
    "Cut back the line to the boundary",
    "Trim the extending segment",
    "Remove the excess line beyond the crossing",
    "Trim it at the intersection",
    "Shorten the line to the boundary",
    "Trim the line where it crosses",
    "Cut the line segment here",
]


def _intersect_lines(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Return the intersection of two infinite lines or None if parallel.

    Line 1: p1 + t * d1
    Line 2: p2 + s * d2
    """
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    diff = p2 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    return p1 + t * d1


class TrimTask:
    """
    Single-step TRIM supervised task.

    Scene: two line segments that cross each other.  One line (the "cutter")
    is placed in a fixed orientation; the other (the "subject") passes through
    it.  The agent must call TRIM and click the segment of the *subject* line
    that should be removed — i.e. the short tail on the far side of the
    intersection from the "keep" endpoint.

    The cutter is pre-selected to signal which boundary to trim to.
    The oracle cursor is placed at the midpoint of the unwanted tail.

    cursor_loss_weight = 0.85: the click must land on the correct side of the
    intersection, but sub-pixel accuracy is not required.
    """

    tool_name = "TRIM"
    cursor_loss_weight = 0.85

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._trim_cursor: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        # Place a horizontal cutter line across the mid-canvas
        cy = float(self.rng.uniform(350, 650))
        cutter_x0 = float(self.rng.uniform(100, 300))
        cutter_x1 = float(self.rng.uniform(700, 900))
        cutter_color = int(self.rng.randint(0, 8))
        cutter = LineEntity(
            start=np.array([cutter_x0, cy]),
            end=np.array([cutter_x1, cy]),
            color_index=cutter_color,
        )

        # Subject line: runs from above the cutter to below it
        sx = float(self.rng.uniform(cutter_x0 + 80, cutter_x1 - 80))
        keep_y = float(self.rng.uniform(cy + 120, cy + 300))   # "keep" end (below)
        tail_y = float(self.rng.uniform(cy - 300, cy - 120))   # tail end (above, to trim)
        subject_color = int(self.rng.randint(0, 8))
        subject = LineEntity(
            start=np.array([sx, keep_y]),
            end=np.array([sx, tail_y]),
            color_index=subject_color,
        )

        engine.add_entity(cutter, save_undo=False)
        engine.add_entity(subject, save_undo=False)
        # Pre-select the cutter so the agent knows which boundary to trim to
        engine.selected_ids.add(cutter.id)

        # Oracle cursor: midpoint of the tail (above the cutter)
        self._trim_cursor = np.array([sx, (cy + tail_y) / 2.0], dtype=np.float64)

        tmpl = _TRIM_PROMPTS[int(self.rng.randint(len(_TRIM_PROMPTS)))]
        prompt = tmpl

        return {
            "prompt": prompt,
            "cutter": cutter,
            "subject": subject,
            "intersection": np.array([sx, cy]),
            "trim_cursor": self._trim_cursor,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "TRIM",
            "cursor_world": self._trim_cursor,
            "cursor_weight": self.cursor_loss_weight,
        }
