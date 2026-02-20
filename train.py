#!/usr/bin/env python3
"""
CADFire Training Script  –  Four-Phase Pipeline.

Can be run directly or imported into a Jupyter notebook.
Designed for vast.ai GPU instances.

Four-Phase Pretraining Pipeline
────────────────────────────────
  Phase 1 – Tool classifier      (text → tool, no vision)
  Phase 2 – Semantic cursor      (vision + text → tool + cursor, single-step,
                                   ALL parameters unfrozen including text encoder)
  Phase 3 – Teacher forcing      (vision + text → tool + cursor, 2–9-step
                                   trajectories, oracle advances environment)
  [After Phase 3] → diagnostic GIFs to verify polygon-tracing capability
  Phase 4 – PPO RL               (full agent, curriculum learning)

Phases 1–3 are supervised warm-starts.  Phases progress in difficulty:
  1: learn tool names from text
  2: learn to identify & point at objects (single-step)
  3: learn to sequence actions (multi-step, teacher-forced)
  4: learn from sparse reward signals (RL)

Each phase can be run independently.  Existing checkpoints are loaded
before each phase so weights always accumulate.

Usage
─────
    # Full pipeline from scratch:
    python train.py --pretrain-tool --pretrain-semantic --pretrain-teacher \\
                    --generate-gifs --steps 100000

    # Phase 2 + Phase 3 + PPO (skip Phase 1):
    python train.py --pretrain-semantic --pretrain-teacher --steps 100000 --resume

    # Phase 3 (teacher forcing) only, then GIFs:
    python train.py --pretrain-teacher --generate-gifs --steps 0

    # PPO only (resume from checkpoint):
    python train.py --steps 500000 --resume

    # Phase 1 only (no RL):
    python train.py --pretrain-tool --steps 0

From notebook:
    from train import (run_pretrain_tool, run_pretrain_semantic,
                       run_pretrain_teacher, run_diagnostics, run_training)
    agent = run_pretrain_tool(num_epochs=30)
    agent = run_pretrain_semantic(agent=agent, num_samples=20000, num_epochs=20)
    agent = run_pretrain_teacher(agent=agent, num_trajectories=5000, num_epochs=15)
    run_diagnostics(agent=agent, n_episodes=6, output_dir="diagnostics/")
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
    device=None,
    checkpoint_dir: str = "model_saves",
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

    # Normalise key names to match Phase 2/3 for consistent notebook plotting
    history_out = {
        "tool_losses":      history["losses"],
        "tool_accuracies":  history["accuracies"],
    }
    return agent, history_out


# ── Phase 2: Semantic Cursor Pretraining ─────────────────────────────────────

def run_pretrain_semantic(
    agent=None,
    config=None,
    num_samples: int = 20_000,
    num_epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 32,
    sigma: float = 12.0,
    cursor_weight: float = 1.0,
    num_workers: int = 0,
    device=None,
    checkpoint_dir: str = "model_saves",
    verbose: bool = True,
):
    """
    Phase 2: semantic cursor pretraining (single-step, all 11 task types).

    Trains ALL model parameters (including text encoder) on one-step
    supervised tasks covering SELECT, MULTISELECT, ERASE, PAN, ZOOM_IN,
    ZOOM_OUT, HATCH, POLYLINE (trace next vertex), COPY, MOVE, ROTATE.

    The text encoder is intentionally NOT frozen so the model can learn to
    associate object names ("hexagon", "circle", etc.) with their visual
    appearances and the tools used to interact with them.
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
        print("Phase 2 – Semantic Cursor Pretraining (all params unfrozen)")
        print(f"  Samples/epoch  : {num_samples:,}")
        print(f"  Epochs         : {num_epochs}")
        print(f"  LR             : {lr}")
        print(f"  Batch size     : {batch_size}")
        print(f"  Gaussian sigma : {sigma:.1f} px")
        print(f"  Cursor weight  : {cursor_weight}")
        from cadfire.training.pretrain_semantic import _TASK_REGISTRY as _REG
        print(f"  Tasks          : {len(_REG)} supervised task types")
        print("=" * 60)

    history = pretrain_semantic_cursor(
        agent, config,
        num_samples=num_samples,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        sigma=sigma,
        cursor_weight=cursor_weight,
        num_workers=num_workers,
        device=device,
        verbose=verbose,
    )

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

    return agent, history


# ── Phase 3: Teacher-Forced Multi-Step Pretraining ───────────────────────────

def run_pretrain_teacher(
    agent=None,
    config=None,
    num_trajectories: int = 5_000,
    num_epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 8,
    sigma: float = 12.0,
    cursor_weight: float = 1.5,
    polygon_ratio: float = 0.7,
    device=None,
    checkpoint_dir: str = "model_saves",
    verbose: bool = True,
):
    """
    Phase 3: teacher-forced multi-step pretraining (2–9-step trajectories).

    The agent sees sequential observations from trajectories including polygon
    tracing (primary) and short 2-step tool chains (select→erase, select→rotate,
    select→copy).  At each step, the ORACLE action is used to advance the
    environment (teacher forcing), and loss is computed per-step.

    No long-horizon reward: still purely supervised, with immediate per-step
    feedback.  Bridges single-step Phase 2 and full RL Phase 4.
    """
    import torch
    import torch.optim as optim
    from cadfire.model.cad_agent import CADAgent
    from cadfire.training.checkpoint import CheckpointManager
    from cadfire.training.pretrain_teacher import pretrain_teacher_forcing
    from cadfire.utils.config import load_config

    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent is None:
        agent = CADAgent(config)
        ckpt = CheckpointManager(checkpoint_dir, config)
        meta = ckpt.load(agent, optimizer=None, device=device)
        if meta.get("step", 0) > 0 or meta.get("extra", {}):
            print(f"  Loaded checkpoint for Phase-3 warm-start")

    if verbose:
        print("=" * 60)
        print("Phase 3 – Teacher-Forced Multi-Step Pretraining")
        print(f"  Trajectories/epoch : {num_trajectories:,}")
        print(f"  Epochs             : {num_epochs}")
        print(f"  LR                 : {lr}")
        print(f"  Polygon ratio      : {polygon_ratio:.0%}")
        print(f"  Cursor weight      : {cursor_weight}")
        print("=" * 60)

    history = pretrain_teacher_forcing(
        agent, config,
        num_trajectories=num_trajectories,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        sigma=sigma,
        cursor_weight=cursor_weight,
        polygon_ratio=polygon_ratio,
        device=device,
        verbose=verbose,
    )

    dummy_opt = optim.Adam(agent.parameters(), lr=lr)
    ckpt = CheckpointManager(checkpoint_dir, config)
    ckpt.save(agent, dummy_opt, step=0, episode=0, extra={
        "pretrain_phase": "teacher_forcing",
        "pretrain_epochs": num_epochs,
        "final_tool_acc": history["tool_accuracies"][-1],
        "final_cursor_loss": history["cursor_losses"][-1],
        "avg_traj_length": history["traj_lengths"][-1],
    })

    if verbose:
        print(f"\nPhase-3 complete | "
              f"tool acc {history['tool_accuracies'][-1]:.3f} | "
              f"cursor loss {history['cursor_losses'][-1]:.4f} | "
              f"avg traj len {history['traj_lengths'][-1]:.1f}")
        print(f"Checkpoint saved to {checkpoint_dir}/")

    return agent, history


# ── Supervised Learning Task Diagnostics ─────────────────────────────────────

def run_supervised_diagnostics(
    agent=None,
    config=None,
    output_dir: str = "supervised_diagnostics",
    n_per_task: int = 4,
    n_trajectories: int = 3,
    phases=(1, 2, 3),
    panel_size: int = 128,
    device=None,
    checkpoint_dir: str = "model_saves",
    seed: int = 0,
    verbose: bool = True,
):
    """
    Generate rich PNG diagnostics for all supervised learning tasks.

    For each supervised task type (Phases 1-3), produces images showing
    the agent's full observation space and the desired target outputs:

    Observation inputs (per panel):
      • Viewport RGB       – drawing with ghosts and selection highlights
      • Reference / Raster – raster reference image (tracing tasks)
      • Selection Mask     – which entities are currently selected
      • Layer Composite    – all layer masks in distinct colours
      • State vector info  – active tool, zoom, viewport, entity/selection/
                             pending-point counts (cursor history)
      • Text prompt        – full decoded prompt including past turn history

    Target outputs (per panel):
      • Oracle tool selection – the tool the agent should predict
      • Oracle cursor heatmap – Gaussian blob(s) for single-point or
                                multi-select cursor targets
      • Cursor weight         – 0 = tool-only; 1 = cursor-critical

    Args:
        agent          : Optional CADAgent (reserved for future overlays).
        config         : Config dict (defaults to config.json).
        output_dir     : Root output directory.
        n_per_task     : Phase-2 samples per task type.
        n_trajectories : Phase-3 trajectories to visualise.
        phases         : Tuple of phase numbers to run: (1,), (2,), (3,), or (1,2,3).
        panel_size     : Pixel size of each image sub-panel.
        device         : Torch device string.
        checkpoint_dir : Checkpoint directory to load agent from (if agent=None).
        seed           : RNG seed for reproducibility.
        verbose        : Print progress.

    Returns:
        Dict with paths/counts for each phase generated.
    """
    import torch
    from cadfire.model.cad_agent import CADAgent
    from cadfire.training.checkpoint import CheckpointManager
    from cadfire.training.supervised_diagnostics import generate_supervised_diagnostics
    from cadfire.utils.config import load_config

    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent is None:
        agent = CADAgent(config)
        ckpt = CheckpointManager(checkpoint_dir, config)
        meta = ckpt.load(agent, optimizer=None, device=device)
        if verbose and (meta.get("step", 0) > 0 or meta.get("extra", {})):
            print(f"  Loaded checkpoint (step {meta.get('step', 0)}) for supervised diagnostics")

    if verbose:
        print("=" * 60)
        print("Supervised Learning Task Diagnostics")
        print(f"  Phases      : {phases}")
        print(f"  Per task    : {n_per_task} samples")
        print(f"  Trajectories: {n_trajectories}")
        print(f"  Panel size  : {panel_size}px")
        print(f"  Output dir  : {output_dir}/")
        print("=" * 60)

    return generate_supervised_diagnostics(
        agent=agent,
        config=config,
        output_dir=output_dir,
        n_per_task=n_per_task,
        n_trajectories=n_trajectories,
        phases=tuple(phases),
        panel_size=panel_size,
        device=device,
        seed=seed,
        verbose=verbose,
    )


# ── Diagnostics: GIF generation after Phase 3 ────────────────────────────────

def run_diagnostics(
    agent=None,
    config=None,
    n_episodes: int = 6,
    output_dir: str = "diagnostics",
    fps: float = 1.5,
    device=None,
    checkpoint_dir: str = "model_saves",
    verbose: bool = True,
):
    """
    Generate diagnostic GIFs after Phase-3 teacher-forced pretraining.

    Produces two GIF types per episode:
      • oracle_ep<N>.gif – oracle-driven rollout with agent attention overlay
      • free_ep<N>.gif   – fully autonomous agent rollout (no teacher forcing)

    The free rollout shows whether the framework can trace an arbitrary polygon
    end-to-end without any oracle guidance.
    """
    import torch
    from cadfire.model.cad_agent import CADAgent
    from cadfire.training.checkpoint import CheckpointManager
    from cadfire.training.diagnostics import generate_diagnostic_gifs
    from cadfire.utils.config import load_config

    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent is None:
        agent = CADAgent(config)
        ckpt = CheckpointManager(checkpoint_dir, config)
        meta = ckpt.load(agent, optimizer=None, device=device)
        if verbose:
            print(f"  Loaded checkpoint (step {meta.get('step', 0)}) for diagnostics")

    if verbose:
        print("=" * 60)
        print("Post-Phase-3 Diagnostics – Polygon Tracing GIFs")
        print(f"  Episodes   : {n_episodes}")
        print(f"  Output dir : {output_dir}/")
        print(f"  FPS        : {fps}")
        print("=" * 60)

    metrics = generate_diagnostic_gifs(
        agent, config,
        output_dir=output_dir,
        n_episodes=n_episodes,
        device=device,
        fps=fps,
        verbose=verbose,
    )

    return metrics


# ── Phase 4: PPO RL Training ─────────────────────────────────────────────────

def run_training(
    num_steps: int = 100000,
    resume: bool = True,
    device=None,
    max_difficulty=None,
    config_path=None,
    checkpoint_dir: str = "model_saves",
    task_weights: dict | None = None,
    callback=None,
):
    """
    Phase 4: PPO RL training loop.

    Full agent training with curriculum learning.  Loads from Phase-3
    checkpoint so all prior supervised learning is preserved.

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
        description="CADFire – four-phase training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline from scratch:
    python train.py --pretrain-tool --pretrain-semantic --pretrain-teacher \\
                    --generate-gifs --steps 100000

  Phase 2 + Phase 3 + PPO (skip Phase 1):
    python train.py --pretrain-semantic --pretrain-teacher --steps 100000 --resume

  Phase 3 only + diagnostics:
    python train.py --pretrain-teacher --generate-gifs --steps 0

  PPO only (resume from checkpoint):
    python train.py --steps 500000 --resume

  Phase 1 only (no RL):
    python train.py --pretrain-tool --steps 0
""",
    )

    # ── Which phases to run ──────────────────────────────────────────────
    parser.add_argument("--pretrain-tool",     action="store_true",
                        help="Run Phase 1: text→tool supervised pretraining")
    parser.add_argument("--pretrain-semantic", action="store_true",
                        help="Run Phase 2: single-step semantic cursor pretraining "
                             "(all 11 task types, text encoder unfrozen)")
    parser.add_argument("--pretrain-teacher",  action="store_true",
                        help="Run Phase 3: teacher-forced multi-step pretraining "
                             "(polygon tracing + short 2-step chains)")
    parser.add_argument("--generate-gifs",     action="store_true",
                        help="Generate diagnostic GIFs after Phase 3")
    parser.add_argument("--supervised-diag",   action="store_true",
                        help="Generate supervised learning task diagnostics (all phases)")

    # ── PPO options ──────────────────────────────────────────────────────
    parser.add_argument("--steps", type=int, default=100000,
                        help="Phase 4 PPO environment steps (0 = skip RL)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from latest checkpoint before PPO")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start PPO from scratch (ignore checkpoint)")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="Starting difficulty cap for curriculum")

    # ── Phase 1 options ──────────────────────────────────────────────────
    parser.add_argument("--tool-epochs", type=int, default=30,
                        help="[Phase 1] Supervised epochs for tool classifier")
    parser.add_argument("--tool-lr",    type=float, default=1e-3,
                        help="[Phase 1] Learning rate")
    parser.add_argument("--tool-batch", type=int, default=64,
                        help="[Phase 1] Batch size")

    # ── Phase 2 options ──────────────────────────────────────────────────
    parser.add_argument("--sem-samples",      type=int,   default=20_000,
                        help="[Phase 2] Generated samples per epoch")
    parser.add_argument("--sem-epochs",       type=int,   default=20,
                        help="[Phase 2] Training epochs")
    parser.add_argument("--sem-lr",           type=float, default=3e-4,
                        help="[Phase 2] Learning rate")
    parser.add_argument("--sem-batch",        type=int,   default=32,
                        help="[Phase 2] Batch size")
    parser.add_argument("--sem-sigma",        type=float, default=12.0,
                        help="[Phase 2] Gaussian blob radius (pixels)")
    parser.add_argument("--sem-cursor-weight",type=float, default=1.0,
                        help="[Phase 2] Cursor loss weight vs tool loss")
    parser.add_argument("--sem-workers",      type=int,   default=0,
                        help="[Phase 2] DataLoader worker processes")

    # ── Phase 3 options ──────────────────────────────────────────────────
    parser.add_argument("--teacher-trajectories", type=int,   default=5_000,
                        help="[Phase 3] Trajectories generated per epoch")
    parser.add_argument("--teacher-epochs",       type=int,   default=15,
                        help="[Phase 3] Training epochs")
    parser.add_argument("--teacher-lr",           type=float, default=1e-4,
                        help="[Phase 3] Learning rate")
    parser.add_argument("--teacher-batch",        type=int,   default=8,
                        help="[Phase 3] Batch size for trajectory lengths")
    parser.add_argument("--teacher-sigma",        type=float, default=12.0,
                        help="[Phase 3] Gaussian blob radius (pixels)")
    parser.add_argument("--teacher-cursor-weight",type=float, default=1.5,
                        help="[Phase 3] Cursor loss weight")
    parser.add_argument("--teacher-poly-ratio",   type=float, default=0.7,
                        help="[Phase 3] Fraction of polygon-trace trajectories")

    # ── Diagnostics options ──────────────────────────────────────────────
    parser.add_argument("--gif-episodes",  type=int,   default=6,
                        help="[Diagnostics] Number of polygon episodes to render")
    parser.add_argument("--gif-output",    type=str,   default="diagnostics",
                        help="[Diagnostics] Output directory for GIFs")
    parser.add_argument("--gif-fps",       type=float, default=1.5,
                        help="[Diagnostics] GIF frames per second")

    # ── Supervised diagnostics options ───────────────────────────────────
    parser.add_argument("--sup-diag-output",    type=str,   default="supervised_diagnostics",
                        help="[SupDiag] Output directory for supervised task diagnostics")
    parser.add_argument("--sup-diag-per-task",  type=int,   default=4,
                        help="[SupDiag] Samples per Phase-2 task type")
    parser.add_argument("--sup-diag-trajs",     type=int,   default=3,
                        help="[SupDiag] Phase-3 trajectories to visualise")
    parser.add_argument("--sup-diag-phases",    type=int,   nargs="+", default=[1, 2, 3],
                        help="[SupDiag] Which phases to include (e.g. 2 3)")
    parser.add_argument("--sup-diag-panel",     type=int,   default=128,
                        help="[SupDiag] Pixel size for each image sub-panel")

    # ── Shared options ───────────────────────────────────────────────────
    parser.add_argument("--device",        type=str, default=None,
                        help="torch device: cuda / cpu / None (auto)")
    parser.add_argument("--config",        type=str, default=None,
                        help="Path to config.json")
    parser.add_argument("--checkpoint-dir",type=str, default="model_saves",
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
            cursor_weight=args.sem_cursor_weight,
            num_workers=args.sem_workers,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # ── Phase 3 ─────────────────────────────────────────────────────────
    if args.pretrain_teacher:
        agent = run_pretrain_teacher(
            agent=agent,
            config=config,
            num_trajectories=args.teacher_trajectories,
            num_epochs=args.teacher_epochs,
            lr=args.teacher_lr,
            batch_size=args.teacher_batch,
            sigma=args.teacher_sigma,
            cursor_weight=args.teacher_cursor_weight,
            polygon_ratio=args.teacher_poly_ratio,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # ── Post-Phase-3 Diagnostics ─────────────────────────────────────────
    if args.generate_gifs:
        run_diagnostics(
            agent=agent,
            config=config,
            n_episodes=args.gif_episodes,
            output_dir=args.gif_output,
            fps=args.gif_fps,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # ── Supervised Learning Task Diagnostics ─────────────────────────────
    if args.supervised_diag:
        run_supervised_diagnostics(
            agent=agent,
            config=config,
            output_dir=args.sup_diag_output,
            n_per_task=args.sup_diag_per_task,
            n_trajectories=args.sup_diag_trajs,
            phases=args.sup_diag_phases,
            panel_size=args.sup_diag_panel,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # ── Phase 4 (PPO) ────────────────────────────────────────────────────
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
