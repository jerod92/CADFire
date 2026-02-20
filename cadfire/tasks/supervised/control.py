"""
Supervised single-step control tasks (Undo/Redo).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cadfire.engine.cad_engine import CADEngine

_UNDO_PROMPTS = [
    "Undo the last action",
    "Go back one step",
    "Revert that change",
    "Undo",
]

_REDO_PROMPTS = [
    "Redo the previous step",
    "Put that back",
    "Redo the action",
    "Redo",
]

class UndoTask:
    """
    Single-step Undo supervised task.
    Agent must use UNDO tool. Cursor is irrelevant.
    """

    tool_name = "UNDO"
    cursor_loss_weight = 0.05

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        prompt = _UNDO_PROMPTS[int(self.rng.randint(len(_UNDO_PROMPTS)))]
        return {"prompt": prompt}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": self.tool_name,
            "cursor_world": None,
            "cursor_weight": self.cursor_loss_weight,
        }


class RedoTask:
    """
    Single-step Redo supervised task.
    Agent must use REDO tool. Cursor is irrelevant.
    """

    tool_name = "REDO"
    cursor_loss_weight = 0.05

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        prompt = _REDO_PROMPTS[int(self.rng.randint(len(_REDO_PROMPTS)))]
        return {"prompt": prompt}

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": self.tool_name,
            "cursor_world": None,
            "cursor_weight": self.cursor_loss_weight,
        }
