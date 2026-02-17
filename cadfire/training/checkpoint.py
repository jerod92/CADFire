"""
Checkpoint manager for saving/loading training state.

Handles:
  - Model weights
  - Optimizer state
  - Training metadata (step, episode, task stats)
  - Tool map for compatibility checking
  - Diagnostics JSON for monitoring
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from cadfire.utils.config import load_config, tool_list


class CheckpointManager:
    """Manages model checkpoints and training diagnostics."""

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 config: Dict[str, Any] | None = None):
        self.config = config or load_config()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_path = self.checkpoint_dir / "diagnostics.json"
        self._diagnostics: Dict[str, Any] = {
            "training_log": [],
            "best_reward": float("-inf"),
            "total_steps": 0,
            "total_episodes": 0,
            "tool_list": tool_list(),
        }

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             step: int, episode: int, extra: Dict[str, Any] | None = None,
             tag: str = "latest"):
        """Save a checkpoint."""
        path = self.checkpoint_dir / f"{tag}.pt"
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "episode": episode,
            "tool_list": tool_list(),
            "num_tools": len(tool_list()),
            "timestamp": time.time(),
        }
        if extra:
            state["extra"] = extra
        torch.save(state, path)

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
             tag: str = "latest", device: str = "cpu") -> Dict[str, Any]:
        """Load a checkpoint. Returns metadata dict."""
        path = self.checkpoint_dir / f"{tag}.pt"
        if not path.exists():
            return {"step": 0, "episode": 0}

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle tool list growth
        saved_tools = checkpoint.get("num_tools", len(tool_list()))
        current_tools = len(tool_list())
        if current_tools > saved_tools:
            # Load what we can, then extend the tool head
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.extend_tools(current_tools)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError):
                pass  # optimizer shape mismatch after tool extension, reset

        return {
            "step": checkpoint.get("step", 0),
            "episode": checkpoint.get("episode", 0),
            "extra": checkpoint.get("extra", {}),
        }

    def save_best(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  step: int, episode: int, reward: float):
        """Save as 'best' if reward exceeds previous best."""
        if reward > self._diagnostics["best_reward"]:
            self._diagnostics["best_reward"] = reward
            self.save(model, optimizer, step, episode, tag="best",
                      extra={"best_reward": reward})

    def log_step(self, step: int, episode: int, metrics: Dict[str, float]):
        """Log training metrics."""
        entry = {
            "step": step,
            "episode": episode,
            "timestamp": time.time(),
            **metrics,
        }
        self._diagnostics["training_log"].append(entry)
        self._diagnostics["total_steps"] = step
        self._diagnostics["total_episodes"] = episode

        # Keep log from growing unbounded
        max_log = 10000
        if len(self._diagnostics["training_log"]) > max_log:
            self._diagnostics["training_log"] = self._diagnostics["training_log"][-max_log:]

    def save_diagnostics(self):
        """Write diagnostics.json to disk."""
        with open(self.diagnostics_path, "w") as f:
            json.dump(self._diagnostics, f, indent=2, default=str)

    def load_diagnostics(self) -> Dict[str, Any]:
        """Load diagnostics from disk."""
        if self.diagnostics_path.exists():
            with open(self.diagnostics_path) as f:
                self._diagnostics = json.load(f)
        return self._diagnostics

    def list_checkpoints(self) -> list:
        """List available checkpoint files."""
        return sorted(self.checkpoint_dir.glob("*.pt"))
