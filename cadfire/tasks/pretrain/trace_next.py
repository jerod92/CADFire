"""
Supervised single-step POLYLINE next-vertex task.

This is the core supervised skill for polygon tracing.

Setup:
  - A target polygon (N vertices, N = 3-8) is chosen at random.
  - Its outline is rendered into the reference image channels so the agent
    can "see" what it needs to trace.
  - k vertices (0 ≤ k < N) have already been traced as a committed
    PolylineEntity on the canvas.
  - engine.pending_points holds the k already-clicked world coords (so the
    renderer can show a ghost preview).
  - engine.active_tool = "POLYLINE" (agent is mid-command).

Agent must:
  - Predict tool = POLYLINE (continue clicking) if k+1 < N
  - Predict tool = CONFIRM  if k+1 == N (close / finish the polygon)
  - cursor = vertex[k] in world space  (the NEXT vertex to click)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import PolylineEntity, LineEntity


def _make_polygon_vertices(rng, cx: float, cy: float,
                           n: int, radius: float) -> np.ndarray:
    """
    Generate N vertices of a regular-ish polygon with random perturbation.
    Returns (N, 2) array of world-space coords.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += rng.uniform(0, 2 * np.pi)  # random rotation
    radii = rng.uniform(radius * 0.7, radius * 1.3, size=n)
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _render_target_reference(vertices: np.ndarray,
                              render_h: int, render_w: int,
                              engine: CADEngine) -> np.ndarray:
    """
    Rasterize the closed polygon outline into an (H, W, 3) uint8 image.
    Uses simple Bresenham segments.
    """
    img = np.zeros((render_h, render_w, 3), dtype=np.uint8)
    n = len(vertices)

    def world_to_px(pt):
        ndc = engine.viewport.world_to_ndc(pt.reshape(1, 2))[0]
        col = int(np.clip(ndc[0] * render_w, 0, render_w - 1))
        row = int(np.clip((1.0 - ndc[1]) * render_h, 0, render_h - 1))
        return row, col

    def draw_line(r0, c0, r1, c1):
        # Bresenham
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
        err = dr - dc
        while True:
            if 0 <= r0 < render_h and 0 <= c0 < render_w:
                img[r0, c0] = [255, 255, 255]
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc; r0 += sr
            if e2 < dr:
                err += dr; c0 += sc

    for i in range(n):
        r0, c0 = world_to_px(vertices[i])
        r1, c1 = world_to_px(vertices[(i + 1) % n])
        draw_line(r0, c0, r1, c1)

    return img


_TRACE_PROMPTS = [
    "Trace the polygon shown in the reference image",
    "Draw the polygon outline from the reference",
    "Click the next vertex of the polygon",
    "Continue tracing the shape in the reference",
    "Place the next point on the polygon",
    "Trace the outline step by step",
    "Continue drawing the polygon",
]

_CONFIRM_PROMPTS = [
    "Finish the polygon",
    "Close the polygon",
    "Confirm and complete the polygon",
    "Press CONFIRM to finish tracing",
    "Complete the polygon outline",
]


class TraceNextPointTask:
    """
    Single-step supervised task: click the next vertex of a polygon.

    Exposed oracle:
      tool   = POLYLINE  (or CONFIRM on the final step)
      cursor = next vertex world-space position
    """

    tool_name = "POLYLINE"       # may be overridden to CONFIRM on last step
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._vertices: np.ndarray = np.zeros((0, 2))
        self._step: int = 0       # which vertex is next
        self._n_verts: int = 0
        self._reference_image: Optional[np.ndarray] = None
        self._next_vertex: Optional[np.ndarray] = None
        self._final_step: bool = False

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        canvas = engine.config["canvas"]
        H = canvas["render_height"]
        W = canvas["render_width"]

        n = int(self.rng.randint(3, 9))       # polygon sides
        cx = float(self.rng.uniform(180, 820))
        cy = float(self.rng.uniform(180, 820))
        radius = float(self.rng.uniform(80, 200))

        verts = _make_polygon_vertices(self.rng, cx, cy, n, radius)
        verts = np.clip(verts, 30, 970)       # keep inside world bounds

        self._vertices = verts
        self._n_verts = n

        # Render target into reference image
        self._reference_image = _render_target_reference(verts, H, W, engine)

        # Pick a random step k: how many vertices have already been placed
        k = int(self.rng.randint(0, n))       # 0 → starting fresh
        self._step = k

        # Draw the partial polyline (k vertices already committed)
        if k >= 2:
            partial = PolylineEntity(
                points=verts[:k].copy(), closed=False,
                color_index=int(self.rng.randint(1, 8)),
            )
            engine.add_entity(partial, save_undo=False)

        # Put engine in POLYLINE mid-command state
        engine.active_tool = "POLYLINE"
        engine.pending_points = list(verts[:k])   # ghost for renderer

        # Determine next step
        self._final_step = (k == n - 1)
        self._next_vertex = verts[k].copy()

        if self._final_step:
            prompt = _CONFIRM_PROMPTS[int(self.rng.randint(len(_CONFIRM_PROMPTS)))]
        else:
            prompt = _TRACE_PROMPTS[int(self.rng.randint(len(_TRACE_PROMPTS)))]

        return {
            "prompt": prompt,
            "reference_image": self._reference_image,
            "vertices": verts,
            "step": k,
            "next_vertex": self._next_vertex,
            "is_final": self._final_step,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        tool = "CONFIRM" if self._final_step else "POLYLINE"
        return {
            "tool": tool,
            "cursor_world": self._next_vertex,
            "cursor_weight": self.cursor_loss_weight,
        }
