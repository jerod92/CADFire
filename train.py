#!/usr/bin/env python3
"""
CADFire Training Script.

Can be run directly or imported into a Jupyter notebook.
Designed for vast.ai GPU instances.

Three-Phase Pretraining Pipeline
─────────────────────────────────
  Phase 1 – Tool classifier  (text → tool, no vision)
  Phase 2 – Semantic cursor  (vision + text → SELECT/MULTISELECT + cursor mask)
  Phase 3 – PPO RL           (full agent, curriculum learning)

Phases 1 and 2 are supervised warm-starts that prevent catastrophic forgetting
when Phase-3 RL begins.  Each phase can be run independently.  Existing
checkpoints are loaded before each phase so weights are always preserved.

Usage
─────
    # Full pipeline from scratch:
    python train.py --pretrain-tool --pretrain-semantic --steps 100000

    # Resume PPO from existing checkpoint (skip pretraining):
    python train.py --steps 500000 --resume

    # Phase 2 only (semantic cursor), then PPO:
    python train.py --pretrain-semantic --steps 100000 --resume

    # PPO only:
    python train.py --steps 100000

    # Phase 1 only (no RL):
    python train.py --pretrain-tool --steps 0

From notebook:
    from train import run_training, run_pretrain_tool, run_pretrain_semantic
    agent = run_pretrain_tool(num_epochs=30)
    agent = run_pretrain_semantic(agent=agent, num_samples=20000, num_epochs=20)
    run_training(num_steps=100000, resume=True)
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent))


# ── Phase 1: Tool Classifier Pretraining ─────────────────────────────────────

def run_pretrain_tool(
    agent=None,
    config=None,
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str | None = None,
    checkpoint_dir: str = "checkpoints_1",
    verbose: bool = True,
):
    """
    Phase 1: supervised text → tool pretraining.

    Trains text encoder + fusion bridge + tool head using cross-entropy
    against (prompt, tool_id) pairs.  Vision encoder and cursor head are
    frozen throughout.

    Args:
        agent         : Existing CADAgent (reused if provided, else created).
        config        : Config dict (defaults to config.json).
        num_epochs    : Supervised epochs over the prompt dataset.
        lr            : Adam learning rate.
        batch_size    : Mini-batch size.
        device        : 'cuda' / 'cpu' / None (auto).
        checkpoint_dir: Where to load/save the checkpoint.
        verbose       : Print per-epoch stats.

    Returns:
        The trained CADAgent (all weights unfrozen).
    """
    import torch
    from cadfire.model.cad_agent import CADAgent
    from cadfire.training.checkpoint import CheckpointManager
    from cadfire.training.pretrain_tools import pretrain_tool_classifier
    from cadfire.utils.config import load_config

    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent is None:
        agent = CADAgent(config)
        ckpt = CheckpointManager(checkpoint_dir, config)
        meta = ckpt.load(agent, optimizer=None, device=device)
        if meta.get("step", 0) > 0:
            print(f"  Loaded checkpoint (step {meta['step']}) for Phase-1 warm-start")

    if verbose:
        print("=" * 60)
        print("Phase 1 – Tool Classifier Pretraining")
        print(f"  Epochs     : {num_epochs}")
        print(f"  LR         : {lr}")
        print(f"  Batch size : {batch_size}")
        print("=" * 60)

    history = pretrain_tool_classifier(
        agent, config,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    # Save phase-1 checkpoint
    import torch.optim as optim
    dummy_opt = optim.Adam(agent.parameters(), lr=lr)
    ckpt = CheckpointManager(checkpoint_dir, config)
    ckpt.save(agent, dummy_opt, step=0, episode=0, extra={
        "pretrain_phase": "tool_classifier",
        "pretrain_epochs": num_epochs,
        "final_acc": history["accuracies"][-1],
    })

    if verbose:
        print(f"\nPhase-1 complete | "
              f"accuracy {history['accuracies'][-1]:.3f} | "
              f"loss {history['losses'][-1]:.4f}")
        print(f"Checkpoint saved to {checkpoint_dir}/")

    return agent


# ── Phase 2: Semantic Cursor Pretraining ─────────────────────────────────────

def run_pretrain_semantic(
    agent=None,
    config=None,
    num_samples: int = 20_000,
    num_epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 32,
    sigma: float = 12.0,
    multi_ratio: float = 0.4,
    cursor_weight: float = 1.0,
    num_workers: int = 0,
    device: str | None = None,
    checkpoint_dir: str = "checkpoints_1",
    verbose: bool = True,
):
    """
    Phase 2: semantic cursor pretraining (SELECT + MULTISELECT, single-step).

    Trains vision encoder + fusion bridge + tool head + cursor head using
    focal-BCE loss on Gaussian cursor masks.  Text encoder is frozen to
    preserve Phase-1 weights.

    Two sample types (controlled by ``multi_ratio``):
      • SemanticSelectTask    – "Select the <shape>" → SELECT + single-entity mask
      • SemanticMultiSelectTask – "Select all <shape>s" → MULTISELECT + multi-entity mask

    Args:
        agent         : Existing CADAgent (reused if provided, else created).
        config        : Config dict (defaults to config.json).
        num_samples   : Generated samples per epoch.
        num_epochs    : Training epochs.
        lr            : Adam learning rate.
        batch_size    : Mini-batch size.
        sigma         : Gaussian blob radius (pixels) for cursor targets.
        multi_ratio   : Fraction of MULTISELECT samples (0-1).
        cursor_weight : Relative weight of cursor BCE vs tool CE loss.
        num_workers   : DataLoader worker processes (0 = main process).
        device        : 'cuda' / 'cpu' / None (auto).
        checkpoint_dir: Where to load/save the checkpoint.
        verbose       : Print per-epoch stats.

    Returns:
        The trained CADAgent (all weights unfrozen).
    """
    import torch
    import torch.optim as optim
    from cadfire.model.cad_agent import CADAgent
    from cadfire.training.checkpoint import CheckpointManager
    from cadfire.training.pretrain_semantic import pretrain_semantic_cursor
    from cadfire.utils.config import load_config

    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent is None:
        agent = CADAgent(config)
        ckpt = CheckpointManager(checkpoint_dir, config)
        meta = ckpt.load(agent, optimizer=None, device=device)
        if meta.get("step", 0) > 0 or meta.get("extra", {}):
            print(f"  Loaded checkpoint for Phase-2 warm-start")

    if verbose:
        print("=" * 60)
        print("Phase 2 – Semantic Cursor Pretraining")
        print(f"  Samples/epoch  : {num_samples:,}")
        print(f"  Epochs         : {num_epochs}")
        print(f"  LR             : {lr}")
        print(f"  Batch size     : {batch_size}")
        print(f"  Gaussian sigma : {sigma:.1f} px")
        print(f"  Multi-ratio    : {multi_ratio:.0%}")
        print(f"  Cursor weight  : {cursor_weight}")
        print("=" * 60)

    history = pretrain_semantic_cursor(
        agent, config,
        num_samples=num_samples,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        sigma=sigma,
        multi_ratio=multi_ratio,
        cursor_weight=cursor_weight,
        num_workers=num_workers,
        device=device,
        verbose=verbose,
    )

    # Save phase-2 checkpoint
    dummy_opt = optim.Adam(agent.parameters(), lr=lr)
    ckpt = CheckpointManager(checkpoint_dir, config)
    ckpt.save(agent, dummy_opt, step=0, episode=0, extra={
        "pretrain_phase": "semantic_cursor",
        "pretrain_epochs": num_epochs,
        "final_tool_acc": history["tool_accuracies"][-1],
        "final_cursor_loss": history["cursor_losses"][-1],
    })

    if verbose:
        print(f"\nPhase-2 complete | "
              f"tool acc {history['tool_accuracies'][-1]:.3f} | "
              f"cursor loss {history['cursor_losses'][-1]:.4f}")
        print(f"Checkpoint saved to {checkpoint_dir}/")

    return agent


# ── Phase 3: PPO RL Training ─────────────────────────────────────────────────

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
    Phase 3: PPO RL training loop.

    Args:
        num_steps     : Total environment steps to train.
        resume        : If True, load latest checkpoint and continue.
        device        : 'cuda', 'cpu', or None (auto-detect).
        max_difficulty: Starting difficulty cap for curriculum learning.
        config_path   : Path to config.json (default: repo root).
        checkpoint_dir: Where to save checkpoints.
        task_weights  : Optional dict mapping task_name -> sampling weight.
        callback      : Optional function(metrics_dict) called each log interval.
    """
    from cadfire.utils.config import load_config
    from cadfire.training.ppo import PPOTrainer

    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()

    config["training"]["checkpoint_dir"] = checkpoint_dir

    trainer = PPOTrainer(config=config, device=device)
    trainer.train(
        num_steps=num_steps,
        resume=resume,
        task_weights=task_weights,
        max_difficulty=max_difficulty,
        callback=callback,
    )

    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CADFire – three-phase training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline from scratch:
    python train.py --pretrain-tool --pretrain-semantic --steps 100000

  Phase 2 + PPO (skip Phase 1):
    python train.py --pretrain-semantic --steps 100000 --resume

  PPO only (resume from checkpoint):
    python train.py --steps 500000 --resume

  Phase 1 only (no RL):
    python train.py --pretrain-tool --steps 0
""",
    )

    # ── Which phases to run ──────────────────────────────────────────────
    parser.add_argument("--pretrain-tool", action="store_true",
                        help="Run Phase 1: text→tool supervised pretraining")
    parser.add_argument("--pretrain-semantic", action="store_true",
                        help="Run Phase 2: semantic cursor supervised pretraining")

    # ── PPO options ──────────────────────────────────────────────────────
    parser.add_argument("--steps", type=int, default=100000,
                        help="PPO environment steps (0 = skip RL)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from latest checkpoint before PPO")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start PPO from scratch (ignore checkpoint)")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="Starting difficulty cap for curriculum")

    # ── Phase 1 options ──────────────────────────────────────────────────
    parser.add_argument("--tool-epochs", type=int, default=30,
                        help="[Phase 1] Supervised epochs for tool classifier")
    parser.add_argument("--tool-lr", type=float, default=1e-3,
                        help="[Phase 1] Learning rate")
    parser.add_argument("--tool-batch", type=int, default=64,
                        help="[Phase 1] Batch size")

    # ── Phase 2 options ──────────────────────────────────────────────────
    parser.add_argument("--sem-samples", type=int, default=20_000,
                        help="[Phase 2] Generated samples per epoch")
    parser.add_argument("--sem-epochs", type=int, default=20,
                        help="[Phase 2] Training epochs")
    parser.add_argument("--sem-lr", type=float, default=3e-4,
                        help="[Phase 2] Learning rate")
    parser.add_argument("--sem-batch", type=int, default=32,
                        help="[Phase 2] Batch size")
    parser.add_argument("--sem-sigma", type=float, default=12.0,
                        help="[Phase 2] Gaussian blob radius (pixels)")
    parser.add_argument("--sem-multi-ratio", type=float, default=0.4,
                        help="[Phase 2] Fraction of MULTISELECT samples (0-1)")
    parser.add_argument("--sem-cursor-weight", type=float, default=1.0,
                        help="[Phase 2] Cursor loss weight vs tool loss")
    parser.add_argument("--sem-workers", type=int, default=0,
                        help="[Phase 2] DataLoader worker processes")

    # ── Shared options ───────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default=None,
                        help="torch device: cuda / cpu / None (auto)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_1",
                        help="Checkpoint directory")

    args = parser.parse_args()

    # Load config once
    from cadfire.utils.config import load_config
    config = load_config(args.config) if args.config else load_config()
    config["training"]["checkpoint_dir"] = args.checkpoint_dir

    agent = None  # passed between phases so weights accumulate

    # ── Phase 1 ─────────────────────────────────────────────────────────
    if args.pretrain_tool:
        agent = run_pretrain_tool(
            agent=agent,
            config=config,
            num_epochs=args.tool_epochs,
            lr=args.tool_lr,
            batch_size=args.tool_batch,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # ── Phase 2 ─────────────────────────────────────────────────────────
    if args.pretrain_semantic:
        agent = run_pretrain_semantic(
            agent=agent,
            config=config,
            num_samples=args.sem_samples,
            num_epochs=args.sem_epochs,
            lr=args.sem_lr,
            batch_size=args.sem_batch,
            sigma=args.sem_sigma,
            multi_ratio=args.sem_multi_ratio,
            cursor_weight=args.sem_cursor_weight,
            num_workers=args.sem_workers,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # ── Phase 3 (PPO) ────────────────────────────────────────────────────
    if args.steps > 0:
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
