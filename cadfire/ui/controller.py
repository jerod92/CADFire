"""
AI inference controller for the CADFire demo.

Runs CADAgent inference in a background thread so the Tkinter main loop
stays responsive.  Also owns the ``ToolMask`` that drives the
Training-Wheels feature.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

# Tool categories used by the Training-Wheels panel.
# Each value is the list of tool names that belong to the category.
TOOL_CATEGORIES: Dict[str, List[str]] = {
    "Drawing": [
        "LINE", "POLYLINE", "CIRCLE", "ARC", "RECTANGLE",
        "POLYGON", "ELLIPSE", "SPLINE", "POINT",
    ],
    "Annotation": [
        "HATCH", "MTEXT", "DTEXT",
        "DIM_LINEAR", "DIM_ALIGNED", "DIM_ANGULAR",
        "DIM_RADIUS", "DIM_DIAMETER",
    ],
    "Modify": [
        "MOVE", "COPY", "ROTATE", "SCALE", "MIRROR",
        "OFFSET", "TRIM", "EXTEND", "FILLET", "CHAMFER",
        "EXPLODE", "JOIN", "BREAK", "LENGTHEN",
    ],
    "Arrays": ["ARRAY_RECT", "ARRAY_POLAR"],
    "Selection": ["SELECT", "MULTISELECT", "DESELECT", "ERASE"],
    "Properties": [
        "LAYER_SET", "LAYER_OFF", "LAYER_ON", "LAYER_FREEZE", "LAYER_THAW",
        "COLOR_SET", "LINETYPE_SET", "LINEWEIGHT_SET", "MATCHPROP",
    ],
    "View": ["ZOOM_IN", "ZOOM_OUT", "ZOOM_EXTENTS", "PAN", "FIT_VIEW"],
    "Control": ["UNDO", "REDO", "CONFIRM", "CANCEL", "NOOP"],
}


class ToolMask:
    """
    Tracks which tool categories are enabled (Training-Wheels).

    Produces a float32 numpy array mask ``(num_tools,)`` where
    1.0 = allowed and 0.0 = blocked.  The mask is passed to
    ``CADAgent.act(tool_mask=...)`` to constrain the AI.
    """

    def __init__(self, tool_list: List[str]):
        self._tool_list = tool_list
        # Map tool name → index for fast lookup
        self._idx: Dict[str, int] = {t: i for i, t in enumerate(tool_list)}
        # All categories enabled by default
        self._enabled: Set[str] = set(TOOL_CATEGORIES.keys())

    @property
    def category_names(self) -> List[str]:
        return list(TOOL_CATEGORIES.keys())

    def set_category(self, category: str, enabled: bool) -> None:
        if enabled:
            self._enabled.add(category)
        else:
            self._enabled.discard(category)

    def is_category_enabled(self, category: str) -> bool:
        return category in self._enabled

    def as_numpy(self) -> np.ndarray:
        """Return the ``(num_tools,)`` float32 mask."""
        mask = np.zeros(len(self._tool_list), dtype=np.float32)
        for cat, enabled in [(c, c in self._enabled) for c in TOOL_CATEGORIES]:
            if not enabled:
                continue
            for name in TOOL_CATEGORIES[cat]:
                if name in self._idx:
                    mask[self._idx[name]] = 1.0
        # Ensure NOOP is always allowed so model can always do something
        if "NOOP" in self._idx:
            mask[self._idx["NOOP"]] = 1.0
        # If mask is all-zero (all cats disabled), allow everything
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask


class AIController:
    """
    Wraps a ``CADAgent`` and runs inference asynchronously.

    Usage::

        ctrl = AIController(tool_list)
        ctrl.load_model("model_saves")
        # Later (from UI thread):
        ctrl.step_async(obs, callback=lambda result: ...)
    """

    def __init__(self, tool_list: List[str]):
        self.tool_mask = ToolMask(tool_list)
        self._agent = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._running = False
        self.model_info: str = "No model loaded"

    # ── Model loading ────────────────────────────────────────────────────

    def load_model(self, checkpoint_dir: str, config: dict | None = None) -> bool:
        """
        Try to load the latest checkpoint from *checkpoint_dir*.

        Returns True on success, False if no checkpoint was found.
        """
        try:
            import torch
            from cadfire.model.cad_agent import CADAgent
            from cadfire.training.checkpoint import CheckpointManager

            cfg = config
            if cfg is None:
                from cadfire.utils.config import load_config
                cfg = load_config()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = CADAgent(cfg)
            ckpt = CheckpointManager(checkpoint_dir, cfg)
            meta = ckpt.load(agent, optimizer=None, device=device)

            agent.eval()
            with self._lock:
                self._agent = agent
                self._device = device
                step = meta.get("step", 0)
                self.model_info = f"Loaded – step {step:,}  [{device}]"
            return True

        except Exception as exc:
            self.model_info = f"Load failed: {exc}"
            return False

    @property
    def has_model(self) -> bool:
        with self._lock:
            return self._agent is not None

    # ── Inference ────────────────────────────────────────────────────────

    def step_sync(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
    ) -> Dict[str, Any] | None:
        """
        Run one synchronous inference step.

        Returns the action dict (tool_id int, cursor ndarray, etc.)
        or None if no model is loaded.
        """
        with self._lock:
            agent = self._agent
            device = self._device
        if agent is None:
            return None

        try:
            import torch

            def _to_tensor(arr, dtype=torch.float32):
                return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)

            obs_t = {
                "image":    _to_tensor(obs["image"]),
                "text_ids": _to_tensor(obs["text_ids"], dtype=torch.long),
                "state_vec": _to_tensor(obs["state_vec"]),
            }
            mask_np = self.tool_mask.as_numpy()
            mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)

            result = agent.act(obs_t, deterministic=deterministic, tool_mask=mask_t)

            # Unwrap tensors to plain Python / numpy
            return {
                "tool_id":       int(result["tool_id"][0].item()),
                "cursor":        result["cursor"][0].cpu().numpy(),  # (H, W)
                "param":         float(result["param"][0].item()),
                "value":         float(result["value"][0].item()),
                "tool_entropy":  float(result["tool_entropy"][0].item()),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def step_async(
        self,
        obs: Dict[str, Any],
        callback: Callable[[Dict[str, Any] | None], None],
        deterministic: bool = False,
    ) -> None:
        """
        Run inference in a background thread and call *callback* with the result.
        """
        def _worker():
            result = self.step_sync(obs, deterministic=deterministic)
            callback(result)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
