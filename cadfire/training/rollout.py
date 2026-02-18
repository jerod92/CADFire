"""
Rollout buffer for PPO training.

Stores trajectories from environment interaction and computes
advantages using Generalized Advantage Estimation (GAE).
"""

from __future__ import annotations

from typing import Dict, Generator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """
    Stores rollout data for PPO updates.

    Collected data per step:
      - image observation
      - text_ids
      - state_vec
      - tool_id action
      - cursor_flat_id action
      - param
      - tool_log_prob
      - cursor_log_prob
      - value
      - reward
      - done
    """

    def __init__(self, buffer_size: int, image_shape: Tuple[int, ...],
                 text_len: int, state_dim: int, num_tools: int = 56,
                 device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False

        H, W, C = image_shape
        self.images = np.zeros((buffer_size, H, W, C), dtype=np.float32)
        self.text_ids = np.zeros((buffer_size, text_len), dtype=np.int32)
        self.state_vecs = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.tool_ids = np.zeros(buffer_size, dtype=np.int64)
        self.cursor_flat_ids = np.zeros(buffer_size, dtype=np.int64)
        self.params = np.zeros(buffer_size, dtype=np.float32)
        self.tool_log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.cursor_log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.tool_masks = np.ones((buffer_size, num_tools), dtype=np.float32)

        # Computed during finalize
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def add(self, obs: Dict[str, np.ndarray], tool_id: int, cursor_flat_id: int,
            param: float, tool_log_prob: float, cursor_log_prob: float,
            value: float, reward: float, done: bool):
        """Add a single transition."""
        idx = self.ptr
        self.images[idx] = obs["image"]
        self.text_ids[idx] = obs["text_ids"]
        self.state_vecs[idx] = obs["state_vec"]
        self.tool_masks[idx] = obs.get("tool_mask", np.ones(self.tool_masks.shape[1], dtype=np.float32))
        self.tool_ids[idx] = tool_id
        self.cursor_flat_ids[idx] = cursor_flat_id
        self.params[idx] = param
        self.tool_log_probs[idx] = tool_log_prob
        self.cursor_log_probs[idx] = cursor_log_prob
        self.values[idx] = value
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def finalize(self, last_value: float, gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        """Compute GAE advantages and returns."""
        size = self.buffer_size if self.full else self.ptr

        # GAE computation
        gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def iterate_batches(self, batch_size: int,
                        device: str = "cpu") -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield mini-batches as torch tensors."""
        size = self.buffer_size if self.full else self.ptr
        indices = np.arange(size)
        np.random.shuffle(indices)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_idx = indices[start:end]

            yield {
                "images": torch.tensor(self.images[batch_idx], device=device),
                "text_ids": torch.tensor(self.text_ids[batch_idx], dtype=torch.long, device=device),
                "state_vecs": torch.tensor(self.state_vecs[batch_idx], device=device),
                "tool_masks": torch.tensor(self.tool_masks[batch_idx], device=device),
                "tool_ids": torch.tensor(self.tool_ids[batch_idx], dtype=torch.long, device=device),
                "cursor_flat_ids": torch.tensor(self.cursor_flat_ids[batch_idx], dtype=torch.long, device=device),
                "old_tool_log_probs": torch.tensor(self.tool_log_probs[batch_idx], device=device),
                "old_cursor_log_probs": torch.tensor(self.cursor_log_probs[batch_idx], device=device),
                "advantages": torch.tensor(self.advantages[batch_idx], device=device),
                "returns": torch.tensor(self.returns[batch_idx], device=device),
            }

    def reset(self):
        """Clear the buffer."""
        self.ptr = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.ptr
