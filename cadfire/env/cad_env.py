"""
Gym-compatible CAD RL Environment.

Observation space:
  - image: (H, W, C) float32 multi-channel tensor
  - text_ids: (max_len,) int32 tokenized prompt
  - state_vec: (state_dim,) float32 vector state

Action space:
  - tool_id: int, index into config tools list
  - cursor: (H, W) float32 heatmap (argmax for single point, threshold for multi-select)

The environment delegates reward computation entirely to the active Task object.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.renderer.rasterizer import Renderer
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_to_index, index_to_tool, num_tools


class CADEnv:
    """
    RL environment for CAD drafting.

    Interface mirrors gymnasium.Env but without the dependency:
      obs, info = env.reset(task=...)
      obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, config: Dict[str, Any] | None = None,
                 tokenizer: BPETokenizer | None = None):
        self.config = config or load_config()
        self.engine = CADEngine(self.config)
        self.renderer = Renderer(self.config)
        self.tokenizer = tokenizer or BPETokenizer(
            vocab_size=self.config["model"]["text_vocab_size"],
            max_len=self.config["model"]["text_max_len"],
        )

        self._tool_to_idx = tool_to_index()
        self._idx_to_tool = index_to_tool()
        self._num_tools = num_tools()

        model_cfg = self.config["model"]
        self.render_h = self.config["canvas"]["render_height"]
        self.render_w = self.config["canvas"]["render_width"]
        self.state_dim = model_cfg["state_dim"]
        self.max_episode_steps = self.config["training"]["max_episode_steps"]

        # Episode state
        self._task = None
        self._step_count = 0
        self._prompt_text = ""
        self._prompt_ids = None
        self._reference_image = None
        self._episode_reward = 0.0
        self._tool_mask = np.ones(self._num_tools, dtype=np.float32)  # all allowed by default

    # ─── Spaces info ────────────────────────────────────────────────────

    @property
    def num_tools(self) -> int:
        return self._num_tools

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (self.render_h, self.render_w, self.renderer.num_channels())

    @property
    def cursor_shape(self) -> Tuple[int, int]:
        return (self.render_h, self.render_w)

    # ─── Reset / Step ───────────────────────────────────────────────────

    def reset(self, task=None, seed: int | None = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment. Optionally provide a Task object.
        If no task, environment starts empty (useful for free-play).
        """
        if seed is not None:
            np.random.seed(seed)

        self.engine.reset()
        self._step_count = 0
        self._episode_reward = 0.0
        self._task = task

        if task is not None:
            # Let the task set up initial state
            setup_info = task.setup(self.engine)
            self._prompt_text = setup_info.get("prompt", "")
            self._reference_image = setup_info.get("reference_image", None)

            # Build tool mask from task
            allowed = task.allowed_tools()
            if allowed is not None:
                self._tool_mask = np.zeros(self._num_tools, dtype=np.float32)
                for name in allowed:
                    idx = self._tool_to_idx.get(name)
                    if idx is not None:
                        self._tool_mask[idx] = 1.0
            else:
                self._tool_mask = np.ones(self._num_tools, dtype=np.float32)
        else:
            self._prompt_text = ""
            self._reference_image = None
            self._tool_mask = np.ones(self._num_tools, dtype=np.float32)

        self._prompt_ids = np.array(
            self.tokenizer.encode_padded(self._prompt_text), dtype=np.int32
        )

        obs = self._build_obs()
        info = {"prompt": self._prompt_text, "step": 0}
        if task is not None:
            # Merge all setup info (including targets/metadata) into info
            info.update(setup_info)
        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one action step.

        action = {
            "tool_id": int,          # index into tools list
            "cursor": np.ndarray,    # (H, W) heatmap or None
            "param": float,          # optional numeric parameter
        }
        """
        self._step_count += 1
        tool_id = action.get("tool_id", 0)
        cursor = action.get("cursor", None)
        param = action.get("param", 0.0)

        tool_name = self._idx_to_tool.get(tool_id, "NOOP")

        # Convert cursor heatmap to world coordinate(s)
        cursor_world = None
        cursor_mask = None
        if cursor is not None:
            if tool_name == "MULTISELECT":
                cursor_mask = (cursor > 0.5).astype(np.float32)
            else:
                # argmax -> pixel -> world
                flat_idx = np.argmax(cursor)
                py, px = divmod(flat_idx, self.render_w)
                ndc = np.array([[px / self.render_w, 1.0 - py / self.render_h]])
                cursor_world = self.engine.viewport.ndc_to_world(ndc)[0]

        # Execute the tool command
        self._execute_tool(tool_name, cursor_world, cursor_mask, param)

        # Compute reward from task
        reward = 0.0
        terminated = False
        info = {"step": self._step_count, "tool": tool_name}

        if self._task is not None:
            # Inject tool_name so compute_reward can give dense bonuses
            action["tool_name"] = tool_name
            reward_info = self._task.compute_reward(self.engine, action, self._step_count)
            reward = reward_info.get("reward", 0.0)
            terminated = reward_info.get("terminated", False)
            info.update(reward_info.get("info", {}))

        self._episode_reward += reward
        truncated = self._step_count >= self.max_episode_steps

        obs = self._build_obs()
        info["episode_reward"] = self._episode_reward
        return obs, reward, terminated, truncated, info

    # ─── Tool Execution ─────────────────────────────────────────────────

    def _execute_tool(self, tool: str, cursor_world: np.ndarray | None,
                      cursor_mask: np.ndarray | None, param: float):
        """Dispatch tool command to engine."""
        eng = self.engine

        # Drawing tools
        if tool == "LINE" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
                eng.active_tool = "LINE"
            else:
                start = eng.pending_points.pop(0)
                eng.draw_line(start, cursor_world)
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "POLYLINE" and cursor_world is not None:
            eng.pending_points.append(cursor_world)
            eng.active_tool = "POLYLINE"

        elif tool == "CIRCLE" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
                eng.active_tool = "CIRCLE"
            else:
                center = eng.pending_points.pop(0)
                radius = float(np.linalg.norm(cursor_world - center))
                eng.draw_circle(center, max(radius, 1.0))
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "ARC" and cursor_world is not None:
            eng.pending_points.append(cursor_world)
            if len(eng.pending_points) >= 3:
                # 3 points: center, start-radius-point, end-radius-point
                center = eng.pending_points[0]
                r = float(np.linalg.norm(eng.pending_points[1] - center))
                sa = np.degrees(np.arctan2(
                    eng.pending_points[1][1] - center[1],
                    eng.pending_points[1][0] - center[0]))
                ea = np.degrees(np.arctan2(
                    eng.pending_points[2][1] - center[1],
                    eng.pending_points[2][0] - center[0]))
                eng.draw_arc(center, max(r, 1.0), sa, ea)
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "RECTANGLE" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
                eng.active_tool = "RECTANGLE"
            else:
                p1 = eng.pending_points.pop(0)
                corner = np.minimum(p1, cursor_world)
                size = np.abs(cursor_world - p1)
                eng.draw_rectangle(corner, max(size[0], 1.0), max(size[1], 1.0))
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "POLYGON" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
                eng.active_tool = "POLYGON"
            else:
                center = eng.pending_points.pop(0)
                r = float(np.linalg.norm(cursor_world - center))
                sides = max(3, min(12, int(abs(param)) if param > 0 else 6))
                eng.draw_polygon(center, max(r, 1.0), sides)
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "ELLIPSE" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
                eng.active_tool = "ELLIPSE"
            else:
                center = eng.pending_points.pop(0)
                diff = cursor_world - center
                eng.draw_ellipse(center, max(abs(diff[0]), 1.0), max(abs(diff[1]), 1.0))
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "SPLINE" and cursor_world is not None:
            eng.pending_points.append(cursor_world)
            eng.active_tool = "SPLINE"

        elif tool == "POINT" and cursor_world is not None:
            eng.draw_point(cursor_world)

        elif tool == "HATCH" and cursor_world is not None:
            eng.pending_points.append(cursor_world)
            eng.active_tool = "HATCH"

        elif tool == "MTEXT" and cursor_world is not None:
            eng.draw_text(cursor_world, "Text", height=10.0, multiline=True)

        elif tool == "DTEXT" and cursor_world is not None:
            eng.draw_text(cursor_world, "Text", height=10.0, multiline=False)

        elif tool in ("DIM_LINEAR", "DIM_ALIGNED", "DIM_ANGULAR",
                       "DIM_RADIUS", "DIM_DIAMETER") and cursor_world is not None:
            eng.pending_points.append(cursor_world)
            if len(eng.pending_points) >= 2:
                p1, p2 = eng.pending_points[0], eng.pending_points[1]
                text_pos = (p1 + p2) / 2 + np.array([0, 20])
                dim_type = tool.replace("DIM_", "")
                eng.draw_dimension(p1, p2, text_pos, dim_type)
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        # Modify tools
        elif tool == "MOVE" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
            else:
                base = eng.pending_points.pop(0)
                delta = cursor_world - base
                eng.move_selected(delta[0], delta[1])
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "COPY" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
            else:
                base = eng.pending_points.pop(0)
                delta = cursor_world - base
                eng.copy_selected(delta[0], delta[1])
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "ROTATE" and cursor_world is not None:
            eng.rotate_selected(param if param != 0 else 90.0, cursor_world)

        elif tool == "SCALE" and cursor_world is not None:
            factor = param if param > 0 else 2.0
            eng.scale_selected(factor, cursor_world)

        elif tool == "MIRROR" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
            else:
                p1 = eng.pending_points.pop(0)
                eng.mirror_selected(p1, cursor_world)
                eng.pending_points.clear()
                eng.active_tool = "NOOP"

        elif tool == "OFFSET":
            dist = param if param != 0 else 10.0
            eng.offset_selected(dist)

        elif tool == "ERASE":
            eng.erase_selected()

        elif tool == "EXPLODE":
            eng.explode_selected()

        elif tool == "MATCHPROP" and cursor_world is not None:
            hit = eng.select_at_point(cursor_world)
            if hit:
                eng.matchprop(hit)

        # Selection tools
        elif tool == "SELECT" and cursor_world is not None:
            eng.select_at_point(cursor_world, tolerance=20.0)

        elif tool == "MULTISELECT" and cursor_mask is not None:
            eng.select_in_region(cursor_mask, self.render_w, self.render_h)

        elif tool == "DESELECT":
            eng.deselect_all()

        # Layer tools
        elif tool == "LAYER_SET":
            layer_idx = max(0, min(int(param), len(eng.layers) - 1))
            eng.set_layer(layer_idx)

        elif tool == "LAYER_OFF":
            eng.layer_off(int(param))

        elif tool == "LAYER_ON":
            eng.layer_on_all()

        elif tool == "LAYER_FREEZE":
            eng.layer_freeze(int(param))

        elif tool == "LAYER_THAW":
            eng.layer_thaw_all()

        elif tool == "COLOR_SET":
            eng.active_color = max(0, min(int(param), len(self.renderer.palette) - 1))

        elif tool == "LINETYPE_SET":
            eng.active_linetype = "CONTINUOUS"  # simplified

        elif tool == "LINEWEIGHT_SET":
            eng.active_lineweight = max(0.09, min(float(param), 2.11))

        # Viewport tools
        elif tool == "ZOOM_IN":
            eng.zoom_in()

        elif tool == "ZOOM_OUT":
            eng.zoom_out()

        elif tool == "ZOOM_EXTENTS":
            eng.zoom_extents()

        elif tool == "PAN" and cursor_world is not None:
            if len(eng.pending_points) == 0:
                eng.pending_points.append(cursor_world)
            else:
                base = eng.pending_points.pop(0)
                delta = cursor_world - base
                vis_min, vis_max = eng.viewport.visible_bounds()
                extent = vis_max - vis_min
                eng.pan(-delta[0] / max(extent[0], 1e-6), -delta[1] / max(extent[1], 1e-6))
                eng.pending_points.clear()

        elif tool == "FIT_VIEW":
            eng.fit_view()

        # Undo / Redo
        elif tool == "UNDO":
            eng.undo()

        elif tool == "REDO":
            eng.redo()

        # Confirm (finalize multi-step commands)
        elif tool == "CONFIRM":
            if eng.active_tool == "POLYLINE" and len(eng.pending_points) >= 2:
                eng.draw_polyline(np.array(eng.pending_points))
            elif eng.active_tool == "SPLINE" and len(eng.pending_points) >= 2:
                eng.draw_spline(np.array(eng.pending_points))
            elif eng.active_tool == "HATCH" and len(eng.pending_points) >= 3:
                eng.draw_hatch(np.array(eng.pending_points))
            eng.pending_points.clear()
            eng.active_tool = "NOOP"

        elif tool == "CANCEL":
            eng.pending_points.clear()
            eng.ghost_entities.clear()
            eng.active_tool = "NOOP"

        # NOOP
        elif tool == "NOOP":
            pass

    # ─── Observation Building ───────────────────────────────────────────

    def _build_obs(self) -> Dict[str, np.ndarray]:
        """Build the full observation dict."""
        image = self.renderer.render(self.engine, self._reference_image)
        state_vec = self._build_state_vector()
        return {
            "image": image,
            "text_ids": self._prompt_ids,
            "state_vec": state_vec,
            "tool_mask": self._tool_mask,
        }

    def _build_state_vector(self) -> np.ndarray:
        """
        Build the vector state input.
        Contents:
          [0]: active tool index (normalized)
          [1]: zoom level (log-normalized)
          [2-3]: viewport center (normalized to world)
          [4]: active layer (normalized)
          [5]: active color (normalized)
          [6]: number of entities (normalized)
          [7]: number selected (normalized)
          [8]: pending points count (normalized)
          [9-15]: reserved / padding
        """
        eng = self.engine
        canvas = self.config["canvas"]
        vec = np.zeros(self.state_dim, dtype=np.float32)

        vec[0] = self._tool_to_idx.get(eng.active_tool, 0) / max(self._num_tools, 1)
        vec[1] = np.log1p(eng.viewport.zoom) / 5.0  # log normalize
        vec[2] = eng.viewport.center[0] / canvas["world_width"]
        vec[3] = eng.viewport.center[1] / canvas["world_height"]
        vec[4] = eng.active_layer / max(len(eng.layers), 1)
        vec[5] = eng.active_color / 8.0
        vec[6] = min(len(eng.entities), 100) / 100.0
        vec[7] = min(len(eng.selected_ids), 50) / 50.0
        vec[8] = min(len(eng.pending_points), 10) / 10.0

        return vec

    # ─── Utility ────────────────────────────────────────────────────────

    def get_engine(self) -> CADEngine:
        return self.engine

    def get_prompt(self) -> str:
        return self._prompt_text

    def entity_count(self) -> int:
        return self.engine.entity_count()
