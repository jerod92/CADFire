"""
Multi-step polygon tracing task for Phase-3 teacher-forced pretraining.

This task generates a complete trajectory for tracing an N-vertex polygon:
  Step 0:       tool=POLYLINE, cursor=vertex[0]   (start the polyline)
  Step 1:       tool=POLYLINE, cursor=vertex[1]
  ...
  Step N-1:     tool=POLYLINE, cursor=vertex[N-1]  (last vertex)
  Step N:       tool=CONFIRM                        (close / commit)

The trainer uses teacher forcing: even if the agent's click at step k is
wrong, the ORACLE click is used to advance to step k+1.  This means the
dataset only needs to generate the oracle trajectory; the trainer handles
rolling out with substitution.

Each trajectory element is a dict:
  {
    "obs":           {...}   – rendered observation at this step
    "tool_id":       int     – oracle tool index
    "cursor_world":  ndarray – oracle click world coords  (or None for CONFIRM)
    "cursor_weight": float   – loss weight for cursor head
  }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import PolylineEntity
from cadfire.renderer.rasterizer import Renderer
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_to_index
from .trace_next import _make_polygon_vertices, _render_target_reference


_TRACE_PROMPTS = [
    "Trace the polygon shown in the reference image",
    "Draw the polygon outline from the reference",
    "Trace the shape in the reference image step by step",
    "Reproduce the polygon shown in the reference",
    "Click each vertex of the polygon in order",
]


class PolygonTraceTask:
    """
    Generates a complete oracle trajectory for polygon tracing.

    Calling ``generate_trajectory(engine, renderer, tokenizer)`` returns a
    list of ``(obs_dict, tool_id, cursor_world, cursor_weight)`` tuples –
    one per step (N POLYLINE clicks + 1 CONFIRM).

    This is used by ``pretrain_teacher.py`` which rolls out the trajectory
    with teacher forcing.
    """

    def __init__(self, seed: int | None = None, config: Dict | None = None):
        self.rng = np.random.RandomState(seed)
        self.config = config or load_config()
        self._tool_idx = tool_to_index()

    def generate_trajectory(
        self,
        engine: CADEngine,
        renderer: Renderer,
        tokenizer: BPETokenizer,
    ) -> List[Dict[str, Any]]:
        """
        Build and return a full oracle trajectory.

        Returns a list of step-dicts, each containing:
          obs          – dict(image, text_ids, state_vec)
          tool_id      – int
          cursor_world – np.ndarray (2,) or None
          cursor_weight – float
          is_final     – bool  (True only on the CONFIRM step)
        """
        canvas = self.config["canvas"]
        H = canvas["render_height"]
        W = canvas["render_width"]
        max_len = self.config["model"]["text_max_len"]
        state_dim = self.config["model"]["state_dim"]
        num_tools = len(self._tool_idx)

        # Sample polygon
        n = int(self.rng.randint(3, 9))
        cx = float(self.rng.uniform(200, 800))
        cy = float(self.rng.uniform(200, 800))
        radius = float(self.rng.uniform(80, 200))
        verts = _make_polygon_vertices(self.rng, cx, cy, n, radius)
        verts = np.clip(verts, 30, 970)

        # Reference image (full polygon outline)
        ref_image = _render_target_reference(verts, H, W, engine)

        # Prompt
        prompt_template = _TRACE_PROMPTS[int(self.rng.randint(len(_TRACE_PROMPTS)))]
        text_ids = np.array(
            tokenizer.encode_padded(prompt_template), dtype=np.int32
        )

        # Reset engine for fresh episode
        engine.reset()
        engine.active_tool = "POLYLINE"

        trajectory: List[Dict[str, Any]] = []

        for step in range(n + 1):   # N vertex clicks + 1 CONFIRM
            is_confirm = (step == n)

            # Build observation at current state
            image = renderer.render(engine)
            # Inject reference image into channels 3-5
            # (renderer uses channels 3-5 for reference; we copy manually)
            ref_float = ref_image.astype(np.float32) / 255.0
            # image shape: (H, W, C) – channels 3-5 are reference
            if image.shape[2] > 5:
                image[:, :, 3:6] = ref_float

            state_vec = self._build_state_vec(engine, num_tools, state_dim)

            obs = {
                "image":    image,
                "text_ids": text_ids.copy(),
                "state_vec": state_vec,
            }

            if is_confirm:
                tool_id = self._tool_idx["CONFIRM"]
                cursor_world = verts[-1].copy()   # cursor near last vertex
                cursor_weight = 0.05              # doesn't matter for CONFIRM
            else:
                tool_id = self._tool_idx["POLYLINE"]
                cursor_world = verts[step].copy()
                cursor_weight = 1.0

            trajectory.append({
                "obs": obs,
                "tool_id": tool_id,
                "cursor_world": cursor_world,
                "cursor_weight": cursor_weight,
                "is_final": is_confirm,
                "step": step,
                "vertices": verts,
            })

            # Advance engine state using oracle action (teacher forcing)
            if not is_confirm:
                engine.pending_points.append(verts[step].copy())
                # After 2+ points, draw partial polyline for rendering
                if len(engine.pending_points) >= 2:
                    # Remove previous partial entity
                    engine.entities = [
                        e for e in engine.entities
                        if not getattr(e, '_partial_trace', False)
                    ]
                    partial = PolylineEntity(
                        points=np.array(engine.pending_points), closed=False,
                        color_index=2,
                    )
                    partial._partial_trace = True  # type: ignore[attr-defined]
                    engine.entities.append(partial)
            else:
                # CONFIRM: commit the full polygon as a closed polyline
                engine.entities = [
                    e for e in engine.entities
                    if not getattr(e, '_partial_trace', False)
                ]
                full = PolylineEntity(
                    points=np.vstack([verts, verts[:1]]), closed=True,
                    color_index=2,
                )
                engine.add_entity(full, save_undo=False)
                engine.pending_points.clear()
                engine.active_tool = "NOOP"

        return trajectory

    def _build_state_vec(self, engine: CADEngine, num_tools: int,
                          state_dim: int) -> np.ndarray:
        canvas = self.config["canvas"]
        vec = np.zeros(state_dim, dtype=np.float32)
        vec[0] = self._tool_idx.get(engine.active_tool, 0) / max(num_tools, 1)
        vec[1] = np.log1p(engine.viewport.zoom) / 5.0
        vec[2] = engine.viewport.center[0] / canvas["world_width"]
        vec[3] = engine.viewport.center[1] / canvas["world_height"]
        vec[4] = engine.active_layer / max(len(engine.layers), 1)
        vec[5] = engine.active_color / 8.0
        vec[6] = min(len(engine.entities), 100) / 100.0
        vec[7] = min(len(engine.selected_ids), 50) / 50.0
        vec[8] = min(len(engine.pending_points), 10) / 10.0
        return vec
