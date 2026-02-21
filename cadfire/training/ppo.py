"""
PPO (Proximal Policy Optimization) trainer for the CAD agent.

Implements the standard PPO-Clip algorithm with:
  - Dual action space (tool + cursor) with combined loss
  - Generalized Advantage Estimation (GAE)
  - Value function clipping
  - Entropy bonus for exploration
  - Gradient clipping

The trainer owns the training loop but delegates episode management
to the environment and task system.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cadfire.env.cad_env import CADEnv
from cadfire.model.cad_agent import CADAgent
from cadfire.tasks.registry import TaskRegistry
from cadfire.training.rollout import RolloutBuffer
from cadfire.training.checkpoint import CheckpointManager
from cadfire.utils.config import load_config


class PPOTrainer:
    """
    PPO training loop for the CAD RL agent.

    Usage:
        trainer = PPOTrainer(config)
        trainer.train(num_steps=100000)
    """

    def __init__(self, config: Dict[str, Any] | None = None,
                 device: str | None = None):
        self.config = config or load_config()
        t = self.config["training"]

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Discover all tasks
        TaskRegistry.discover()

        # Create environment
        self.env = CADEnv(self.config)

        # Create model
        self.agent = CADAgent(self.config).to(self.device)

        # Freeze the text encoder to prevent catastrophic forgetting during RL
        for param in self.agent.text.parameters():
            param.requires_grad = False

        # Optimizer: only optimize parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.agent.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=t["lr"])

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=t["rollout_steps"],
            image_shape=self.env.image_shape,
            text_len=self.config["model"]["text_max_len"],
            state_dim=self.config["model"]["state_dim"],
            num_tools=self.env.num_tools,
            device=self.device,
        )

        # Checkpoint manager
        self.ckpt = CheckpointManager(
            checkpoint_dir=t.get("checkpoint_dir", "model_saves"),
            config=self.config,
        )

        # Training hyperparameters
        self.gamma = t["gamma"]
        self.gae_lambda = t["gae_lambda"]
        self.clip_epsilon = t["clip_epsilon"]
        self.value_coeff = t["value_coeff"]
        self.entropy_coeff = t["entropy_coeff"]
        self.max_grad_norm = t["max_grad_norm"]
        self.ppo_epochs = t["ppo_epochs"]
        self.batch_size = t["batch_size"]
        self.rollout_steps = t["rollout_steps"]
        self.save_interval = t["save_interval"]
        self.log_interval = t["log_interval"]

        # Training state
        self.global_step = 0
        self.episode_count = 0
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []

        # Curriculum: start with easy tasks, increase when performance gates
        self.max_difficulty = 2.0
        self.difficulty_step = 0.5
        self.curriculum_reward_threshold = t.get("curriculum_reward_threshold", 5.0)
        self.curriculum_window = 100  # episodes to average over

        # Entropy annealing: high initial exploration, decay over time
        self.entropy_coeff_start = t.get("entropy_coeff_start", 0.05)
        self.entropy_coeff_end = t.get("entropy_coeff_end", 0.005)
        self.entropy_anneal_steps = t.get("entropy_anneal_steps", 200_000)

    def train(self, num_steps: int = 100000,
              resume: bool = True,
              task_weights: Dict[str, float] | None = None,
              max_difficulty: float | None = None,
              callback=None):
        """
        Main training loop.

        Args:
            num_steps: total environment steps to train
            resume: whether to load latest checkpoint
            task_weights: optional per-task sampling weights
            max_difficulty: override max difficulty for curriculum
            callback: optional function called every log_interval with metrics dict
        """
        if max_difficulty is not None:
            self.max_difficulty = max_difficulty

        # Resume from checkpoint
        if resume:
            meta = self.ckpt.load(self.agent, self.optimizer, device=self.device)
            self.global_step = meta.get("step", 0)
            self.episode_count = meta.get("episode", 0)
            self.ckpt.load_diagnostics()

        print(f"Starting training from step {self.global_step}")
        print(f"Device: {self.device}")
        print(f"Available tasks: {TaskRegistry.list_tasks()}")
        print(f"Initial difficulty cap: {self.max_difficulty}")

        # Initialize episode
        task = self._sample_task(task_weights)
        obs, info = self.env.reset(task=task)
        episode_reward = 0.0
        episode_length = 0

        start_step = self.global_step
        start_time = time.time()

        while self.global_step < start_step + num_steps:
            # Collect rollout
            self.buffer.reset()
            for _ in range(self.rollout_steps):
                # Convert obs to torch
                obs_t = self._obs_to_torch(obs)

                # Extract tool mask for this step
                tool_mask_t = obs_t.get("tool_mask", None)

                # Get action from policy
                action_info = self.agent.act(obs_t, tool_mask=tool_mask_t)

                tool_id = action_info["tool_id"].item()
                cursor = action_info["cursor"].cpu().numpy()[0]
                param = action_info["param"].item()

                # Flatten cursor for storage
                cursor_flat_id = np.argmax(cursor)

                # Step environment
                action = {"tool_id": tool_id, "cursor": cursor, "param": param}
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)

                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                # Store in buffer
                self.buffer.add(
                    obs, tool_id, int(cursor_flat_id), param,
                    action_info["tool_log_prob"].item(),
                    action_info["cursor_log_prob"].item(),
                    action_info["value"].item(),
                    reward, done,
                )

                obs = next_obs
                self.global_step += 1

                if done:
                    self._episode_rewards.append(episode_reward)
                    self._episode_lengths.append(episode_length)
                    self.episode_count += 1

                    # Start new episode
                    task = self._sample_task(task_weights)
                    obs, info = self.env.reset(task=task)
                    episode_reward = 0.0
                    episode_length = 0

                # Entropy annealing
                frac = min(1.0, self.global_step / max(self.entropy_anneal_steps, 1))
                self.entropy_coeff = (self.entropy_coeff_start
                                      + frac * (self.entropy_coeff_end - self.entropy_coeff_start))

            # Compute GAE
            with torch.no_grad():
                obs_t = self._obs_to_torch(obs)
                last_action = self.agent.act(obs_t, deterministic=True)
                last_value = last_action["value"].item()
            self.buffer.finalize(last_value, self.gamma, self.gae_lambda)

            # PPO update
            metrics = self._ppo_update()

            # Performance-gated curriculum (once per rollout, not per step)
            if (len(self._episode_rewards) >= self.curriculum_window
                    and self.max_difficulty < 10.0):
                recent_avg = np.mean(self._episode_rewards[-self.curriculum_window:])
                if recent_avg > self.curriculum_reward_threshold:
                    self.max_difficulty = min(10.0, self.max_difficulty + self.difficulty_step)
                    print(f"  [Curriculum] avg_reward={recent_avg:.3f} > threshold "
                          f" → difficulty {self.max_difficulty:.1f}")

            # Logging
            if self.global_step % self.log_interval < self.rollout_steps:
                avg_reward = np.mean(self._episode_rewards[-100:]) if self._episode_rewards else 0
                avg_length = np.mean(self._episode_lengths[-100:]) if self._episode_lengths else 0
                elapsed = time.time() - start_time
                sps = (self.global_step - start_step) / max(elapsed, 1)

                log_metrics = {
                    "avg_reward": float(avg_reward),
                    "avg_episode_length": float(avg_length),
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "steps_per_second": sps,
                    "difficulty": self.max_difficulty,
                    "episodes": self.episode_count,
                }

                self.ckpt.log_step(self.global_step, self.episode_count, log_metrics)

                print(f"Step {self.global_step:>8d} | "
                      f"Ep {self.episode_count:>6d} | "
                      f"R {avg_reward:>7.3f} | "
                      f"L {avg_length:>5.1f} | "
                      f"PL {metrics['policy_loss']:>7.4f} | "
                      f"VL {metrics['value_loss']:>7.4f} | "
                      f"H {metrics['entropy']:>5.3f} | "
                      f"SPS {sps:>5.0f} | "
                      f"D {self.max_difficulty:.1f}")

                if callback:
                    callback(log_metrics)

            # Save checkpoint
            if self.global_step % self.save_interval < self.rollout_steps:
                self.ckpt.save(self.agent, self.optimizer,
                               self.global_step, self.episode_count)
                self.ckpt.save_diagnostics()
                if self._episode_rewards:
                    self.ckpt.save_best(
                        self.agent, self.optimizer,
                        self.global_step, self.episode_count,
                        float(np.mean(self._episode_rewards[-100:])),
                    )

        # Final save
        self.ckpt.save(self.agent, self.optimizer,
                       self.global_step, self.episode_count)
        self.ckpt.save_diagnostics()
        print(f"Training complete. Total steps: {self.global_step}")

    def _ppo_update(self) -> Dict[str, float]:
        """Run PPO update epochs on the current rollout buffer."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.iterate_batches(self.batch_size, self.device):
                obs_batch = {
                    "image": batch["images"],
                    "text_ids": batch["text_ids"],
                    "state_vec": batch["state_vecs"],
                }

                # Evaluate current policy on old actions (with tool mask)
                eval_out = self.agent.evaluate_actions(
                    obs_batch,
                    batch["tool_ids"],
                    batch["cursor_flat_ids"],
                    tool_mask=batch["tool_masks"],
                )

                # Combined log probability
                new_log_prob = eval_out["tool_log_prob"] + eval_out["cursor_log_prob"]
                old_log_prob = batch["old_tool_log_probs"] + batch["old_cursor_log_probs"]

                # PPO ratio
                ratio = torch.exp(new_log_prob - old_log_prob)

                # Normalize advantages
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred = eval_out["value"]
                value_loss = nn.functional.mse_loss(value_pred, batch["returns"])

                # Entropy bonus
                entropy = (eval_out["tool_entropy"] + eval_out["cursor_entropy"]).mean()

                # Total loss
                loss = (policy_loss
                        + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1

        num_batches = max(num_batches, 1)
        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches,
        }

    def _sample_task(self, weights=None):
        """Sample a task respecting difficulty curriculum."""
        try:
            if weights:
                return TaskRegistry.sample_weighted(weights, seed=None)
            return TaskRegistry.sample(max_difficulty=self.max_difficulty)
        except ValueError:
            # Fallback: sample any task
            return TaskRegistry.sample()

    def _obs_to_torch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy observation dict to batched torch tensors."""
        result = {
            "image": torch.tensor(obs["image"], device=self.device).unsqueeze(0),
            "text_ids": torch.tensor(obs["text_ids"], dtype=torch.long, device=self.device).unsqueeze(0),
            "state_vec": torch.tensor(obs["state_vec"], device=self.device).unsqueeze(0),
        }
        if "tool_mask" in obs:
            result["tool_mask"] = torch.tensor(obs["tool_mask"], device=self.device).unsqueeze(0)
        return result
