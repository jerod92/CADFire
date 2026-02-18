#!/usr/bin/env python3
"""
CADFire Training Script.

Can be run directly or imported into a Jupyter notebook.
Designed for vast.ai GPU instances.

Usage:
    python train.py                    # Train with defaults
    python train.py --steps 500000     # Train for 500k steps
    python train.py --resume           # Resume from latest checkpoint
    python train.py --device cuda      # Force GPU
    python train.py --difficulty 5.0   # Start with higher difficulty

From notebook:
    from train import run_training
    run_training(num_steps=100000)
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent))


def run_training(
    num_steps: int = 100000,
    resume: bool = True,
    device: str | None = None,
    max_difficulty: float | None = None,
    config_path: str | None = None,
    checkpoint_dir: str = "checkpoints_1",
    task_weights: dict | None = None,
    callback=None,
):
    """
    Run the PPO training loop.

    Args:
        num_steps: Total environment steps to train.
        resume: If True, load latest checkpoint and continue.
        device: 'cuda', 'cpu', or None (auto-detect).
        max_difficulty: Starting difficulty cap for curriculum learning.
        config_path: Path to config.json (default: repo root).
        checkpoint_dir: Where to save checkpoints.
        task_weights: Optional dict mapping task_name -> sampling weight.
        callback: Optional function(metrics_dict) called each log interval.
    """
    from cadfire.utils.config import load_config, reload
    from cadfire.training.ppo import PPOTrainer

    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()

    config["training"]["checkpoint_dir"] = checkpoint_dir

    # Create and run trainer
    trainer = PPOTrainer(config=config, device=device)
    trainer.train(
        num_steps=num_steps,
        resume=resume,
        task_weights=task_weights,
        max_difficulty=max_difficulty,
        callback=callback,
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(description="CADFire RL Training")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from latest checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="Starting difficulty cap")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_1",
                        help="Checkpoint directory")
    args = parser.parse_args()

    run_training(
        num_steps=args.steps,
        resume=not args.no_resume,
        device=args.device,
        max_difficulty=args.difficulty,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
