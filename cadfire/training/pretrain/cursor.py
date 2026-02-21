"""
Cursor imitation pretraining (Phase 2 of the warm-start pipeline).

Goal: teach the vision encoder + cursor head to point at the correct pixel
*before* RL training begins, using behavioral cloning from oracle trajectories.

Each task already knows the answer (target_entities from setup()). We convert
those world-space coordinates to pixel indices and train with supervised
cross-entropy on the flattened 256x256 cursor heatmap (65536 classes).

The tool head is trained simultaneously so both heads stay aligned.

Design:
  - OracleDataset: generates (obs, tool_id, cursor_flat_id) triples
  - pretrain_cursor_imitation(): trains vision + cursor + tool heads
  - Text encoder is kept frozen (already trained in Phase 1)

Usage:
    from cadfire.training.pretrain.cursor import pretrain_cursor_imitation
    history = pretrain_cursor_imitation(agent, config, num_samples=20000, num_epochs=20)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cadfire.engine.cad_engine import CADEngine
from cadfire.env.cad_env import CADEnv
from cadfire.model.cad_agent import CADAgent
from cadfire.renderer.rasterizer import Renderer
from cadfire.tasks.registry import TaskRegistry
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.training.checkpoint import CheckpointManager
from cadfire.utils.config import load_config, tool_to_index


# ── Oracle step generation ────────────────────────────────────────────

def _world_to_flat(world_pt: np.ndarray, engine: CADEngine,
                   render_w: int, render_h: int) -> int:
    """Convert a world-space point to a flat pixel index (row-major)."""
    ndc = engine.viewport.world_to_ndc(world_pt.reshape(1, 2))
    px = int(np.clip(ndc[0, 0] * render_w, 0, render_w - 1))
    py = int(np.clip((1.0 - ndc[0, 1]) * render_h, 0, render_h - 1))
    return py * render_w + px


def _oracle_steps_for_task(task, engine: CADEngine,
                            render_w: int, render_h: int,
                            tool_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Return a list of (tool_id, cursor_flat_id) oracle steps for a task.

    Each tuple is one action the oracle would take. Multi-step tools
    (e.g. LINE needs two clicks) produce multiple tuples.

    Returns empty list if the task type is not supported.
    """
    steps = []
    task_name = task.task_name

    # Helper: get tool id safely
    def tid(name: str) -> int:
        return tool_idx.get(name, tool_idx.get("NOOP", 0))

    # ── Draw tasks ────────────────────────────────────────────────────
    if task_name == "draw_line":
        t = task._target
        steps.append((tid("LINE"), _world_to_flat(t.start, engine, render_w, render_h)))
        steps.append((tid("LINE"), _world_to_flat(t.end,   engine, render_w, render_h)))

    elif task_name == "draw_circle":
        t = task._target
        center = t.center
        # Click 1: center; Click 2: point on circumference (right side)
        rim = center + np.array([t.radius, 0.0])
        steps.append((tid("CIRCLE"), _world_to_flat(center, engine, render_w, render_h)))
        steps.append((tid("CIRCLE"), _world_to_flat(rim,    engine, render_w, render_h)))

    elif task_name == "draw_rectangle":
        t = task._target
        corner = t.corner
        opposite = corner + np.array([t.width, t.height])
        steps.append((tid("RECTANGLE"), _world_to_flat(corner,   engine, render_w, render_h)))
        steps.append((tid("RECTANGLE"), _world_to_flat(opposite, engine, render_w, render_h)))

    elif task_name == "draw_polygon":
        t = task._target
        center = t.center
        rim = center + np.array([t.radius, 0.0])
        steps.append((tid("POLYGON"), _world_to_flat(center, engine, render_w, render_h)))
        steps.append((tid("POLYGON"), _world_to_flat(rim,    engine, render_w, render_h)))

    elif task_name == "draw_ellipse":
        t = task._target
        center = t.center
        rim = center + np.array([t.semi_major, 0.0])
        steps.append((tid("ELLIPSE"), _world_to_flat(center, engine, render_w, render_h)))
        steps.append((tid("ELLIPSE"), _world_to_flat(rim,    engine, render_w, render_h)))

    elif task_name == "draw_arc":
        t = task._target
        center = t.center
        # Click 1: center; Click 2: start-angle point; Click 3: end-angle point
        sa_rad = math.radians(t.start_angle)
        ea_rad = math.radians(t.end_angle)
        p_start = center + t.radius * np.array([math.cos(sa_rad), math.sin(sa_rad)])
        p_end   = center + t.radius * np.array([math.cos(ea_rad), math.sin(ea_rad)])
        steps.append((tid("ARC"), _world_to_flat(center,  engine, render_w, render_h)))
        steps.append((tid("ARC"), _world_to_flat(p_start, engine, render_w, render_h)))
        steps.append((tid("ARC"), _world_to_flat(p_end,   engine, render_w, render_h)))

    elif task_name == "draw_multi_primitive":
        # Circle first, then rectangle
        c_center = task._c_center
        c_rim    = c_center + np.array([task._c_radius, 0.0])
        r_corner = task._r_corner
        r_opp    = r_corner + np.array([task._r_w, task._r_h])
        steps.append((tid("CIRCLE"),    _world_to_flat(c_center, engine, render_w, render_h)))
        steps.append((tid("CIRCLE"),    _world_to_flat(c_rim,    engine, render_w, render_h)))
        steps.append((tid("RECTANGLE"), _world_to_flat(r_corner, engine, render_w, render_h)))
        steps.append((tid("RECTANGLE"), _world_to_flat(r_opp,    engine, render_w, render_h)))

    # ── Select tasks ──────────────────────────────────────────────────
    elif task_name == "select_shape":
        # Click on the entity's bounding box center
        if hasattr(task, "_entity") and task._entity is not None:
            bb_min, bb_max = task._entity.bbox()
            center = (bb_min + bb_max) / 2.0
            steps.append((tid("SELECT"), _world_to_flat(center, engine, render_w, render_h)))

    elif task_name == "select_by_color":
        if hasattr(task, "_target_entity") and task._target_entity is not None:
            bb_min, bb_max = task._target_entity.bbox()
            center = (bb_min + bb_max) / 2.0
            steps.append((tid("SELECT"), _world_to_flat(center, engine, render_w, render_h)))

    # ── View tasks ────────────────────────────────────────────────────
    elif task_name == "fit_view":
        # No cursor needed — just the tool
        center_px = (render_h // 2) * render_w + (render_w // 2)
        steps.append((tid("FIT_VIEW"), center_px))

    # ── Modify tasks (select then transform) ──────────────────────────
    elif task_name in ("move_shape", "copy_shape"):
        if hasattr(task, "_entity") and task._entity is not None:
            bb_min, bb_max = task._entity.bbox()
            entity_center = (bb_min + bb_max) / 2.0
            # Step 1: select the entity
            steps.append((tid("SELECT"), _world_to_flat(entity_center, engine, render_w, render_h)))
            # Step 2: pick tool; Step 3: base point; Step 4: destination
            tool_name = "MOVE" if task_name == "move_shape" else "COPY"
            dest = entity_center + getattr(task, "_delta", np.array([50.0, 50.0]))
            steps.append((tid(tool_name), _world_to_flat(entity_center, engine, render_w, render_h)))
            steps.append((tid(tool_name), _world_to_flat(dest,          engine, render_w, render_h)))

    elif task_name == "rotate_shape":
        if hasattr(task, "_entity") and task._entity is not None:
            bb_min, bb_max = task._entity.bbox()
            entity_center = (bb_min + bb_max) / 2.0
            steps.append((tid("SELECT"), _world_to_flat(entity_center, engine, render_w, render_h)))
            steps.append((tid("ROTATE"), _world_to_flat(entity_center, engine, render_w, render_h)))

    elif task_name == "scale_shape":
        if hasattr(task, "_entity") and task._entity is not None:
            bb_min, bb_max = task._entity.bbox()
            entity_center = (bb_min + bb_max) / 2.0
            steps.append((tid("SELECT"), _world_to_flat(entity_center, engine, render_w, render_h)))
            steps.append((tid("SCALE"),  _world_to_flat(entity_center, engine, render_w, render_h)))

    elif task_name == "erase_selection":
        if hasattr(task, "_entity") and task._entity is not None:
            bb_min, bb_max = task._entity.bbox()
            entity_center = (bb_min + bb_max) / 2.0
            steps.append((tid("SELECT"), _world_to_flat(entity_center, engine, render_w, render_h)))
            steps.append((tid("ERASE"),  (render_h // 2) * render_w + (render_w // 2)))

    elif task_name == "change_layer":
        center_px = (render_h // 2) * render_w + (render_w // 2)
        steps.append((tid("LAYER_SET"), center_px))

    # ── Trace tasks ───────────────────────────────────────────────────
    elif task_name == "trace_line":
        if hasattr(task, "_target") and task._target is not None:
            t = task._target
            steps.append((tid("LINE"), _world_to_flat(t.start, engine, render_w, render_h)))
            steps.append((tid("LINE"), _world_to_flat(t.end,   engine, render_w, render_h)))

    elif task_name == "trace_circle":
        if hasattr(task, "_target") and task._target is not None:
            t = task._target
            rim = t.center + np.array([t.radius, 0.0])
            steps.append((tid("CIRCLE"), _world_to_flat(t.center, engine, render_w, render_h)))
            steps.append((tid("CIRCLE"), _world_to_flat(rim,      engine, render_w, render_h)))

    elif task_name == "trace_composite":
        # Best effort: click center of canvas
        center_px = (render_h // 2) * render_w + (render_w // 2)
        steps.append((tid("LINE"), center_px))

    elif task_name == "zoom_to_center":
        center_px = (render_h // 2) * render_w + (render_w // 2)
        steps.append((tid("ZOOM_IN"), center_px))

    return steps


# ── Dataset ───────────────────────────────────────────────────────────

class OracleDataset(Dataset):
    """
    Dataset of (image, text_ids, state_vec, tool_id, cursor_flat_id) tuples
    generated from oracle task trajectories.

    Each sample is one step from an oracle episode. Multi-step tasks
    (e.g. draw_circle needs 2 clicks) contribute multiple samples.
    """

    def __init__(self, config: Dict[str, Any] | None = None,
                 num_samples: int = 20000,
                 seed: int = 42):
        self.config = config or load_config()
        self.samples: List[Dict[str, Any]] = []

        TaskRegistry.discover()
        task_names = TaskRegistry.list_tasks()

        renderer = Renderer(self.config)
        tokenizer = BPETokenizer(
            vocab_size=self.config["model"]["text_vocab_size"],
            max_len=self.config["model"]["text_max_len"],
        )
        tool_idx = tool_to_index()
        render_w = self.config["canvas"]["render_width"]
        render_h = self.config["canvas"]["render_height"]

        rng = np.random.RandomState(seed)
        attempts = 0
        max_attempts = num_samples * 10

        while len(self.samples) < num_samples and attempts < max_attempts:
            attempts += 1
            task_name = rng.choice(task_names)
            task_seed = int(rng.randint(0, 2**31))

            try:
                task = TaskRegistry.create(task_name, seed=task_seed)
                engine = CADEngine(self.config)
                setup_info = task.setup(engine)

                prompt = setup_info.get("prompt", "")
                text_ids = np.array(tokenizer.encode_padded(prompt), dtype=np.int32)
                reference_image = setup_info.get("reference_image", None)

                # Build initial observation
                obs_image = renderer.render(engine, reference_image)

                # Build state vector (mirrors CADEnv._build_state_vector)
                state_vec = _build_state_vec(engine, self.config, tool_idx)

                # Get oracle steps
                oracle_steps = _oracle_steps_for_task(
                    task, engine, render_w, render_h, tool_idx
                )

                if not oracle_steps:
                    continue

                # Add each oracle step as a separate sample
                for tool_id, cursor_flat_id in oracle_steps:
                    if len(self.samples) >= num_samples:
                        break
                    self.samples.append({
                        "image": obs_image.copy(),
                        "text_ids": text_ids.copy(),
                        "state_vec": state_vec.copy(),
                        "tool_id": tool_id,
                        "cursor_flat_id": cursor_flat_id,
                    })

            except Exception:
                # Skip tasks that fail to set up (e.g. missing attributes)
                continue

        if len(self.samples) == 0:
            raise RuntimeError("OracleDataset: no samples generated. "
                               "Check that tasks are discoverable.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "image":          torch.tensor(s["image"],          dtype=torch.float32),
            "text_ids":       torch.tensor(s["text_ids"],       dtype=torch.long),
            "state_vec":      torch.tensor(s["state_vec"],      dtype=torch.float32),
            "tool_id":        torch.tensor(s["tool_id"],        dtype=torch.long),
            "cursor_flat_id": torch.tensor(s["cursor_flat_id"], dtype=torch.long),
        }


def _build_state_vec(engine: CADEngine, config: Dict[str, Any],
                     tool_idx: Dict[str, int]) -> np.ndarray:
    """Mirrors CADEnv._build_state_vector without needing a full env."""
    num_tools = len(tool_idx)
    canvas = config["canvas"]
    state_dim = config["model"]["state_dim"]
    vec = np.zeros(state_dim, dtype=np.float32)
    vec[0] = tool_idx.get(engine.active_tool, 0) / max(num_tools, 1)
    vec[1] = np.log1p(engine.viewport.zoom) / 5.0
    vec[2] = engine.viewport.center[0] / canvas["world_width"]
    vec[3] = engine.viewport.center[1] / canvas["world_height"]
    vec[4] = engine.active_layer / max(len(engine.layers), 1)
    vec[5] = engine.active_color / 8.0
    vec[6] = min(len(engine.entities), 100) / 100.0
    vec[7] = min(len(engine.selected_ids), 50) / 50.0
    vec[8] = min(len(engine.pending_points), 10) / 10.0
    return vec


# ── Pretraining loop ──────────────────────────────────────────────────

def pretrain_cursor_imitation(
    agent: CADAgent,
    config: Dict[str, Any] | None = None,
    num_samples: int = 20000,
    num_epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: str | None = None,
    verbose: bool = True,
    dataset_seed: int = 42,
    checkpoint_dir: str | None = None,
) -> Dict[str, List[float]]:
    """
    Run cursor imitation pretraining (behavioral cloning).

    Trains the vision encoder, cursor head, and tool head simultaneously
    using supervised cross-entropy on oracle (tool_id, cursor_flat_id) pairs.
    The text encoder is kept frozen (already trained in Phase 1).

    Checkpointing: if ``checkpoint_dir`` is provided, the function saves an
    epoch-level checkpoint after every epoch (tag ``cursor_pretrain_epoch_N``)
    and a rolling ``cursor_pretrain_latest`` tag. On restart it automatically
    detects the latest completed epoch and resumes from there, skipping
    dataset generation only when the dataset is already cached.

    Args:
        agent:          The CADAgent model (modified in-place).
        config:         Optional config dict.
        num_samples:    Number of oracle samples to generate.
        num_epochs:     Training epochs over the dataset.
        lr:             Learning rate.
        batch_size:     Mini-batch size (keep small — images are large).
        device:         torch device string.
        verbose:        Print per-epoch stats.
        dataset_seed:   RNG seed for reproducible dataset generation.
        checkpoint_dir: Directory for epoch checkpoints and resume. If None,
                        no checkpoints are written.

    Returns:
        Dict with 'tool_losses', 'cursor_losses', 'tool_accuracies' lists.
    """
    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = agent.to(device)

    # Freeze text encoder — it was trained in Phase 1
    for param in agent.text.parameters():
        param.requires_grad = False

    # Unfreeze everything else
    for module in [agent.vision, agent.fusion, agent.tool_head, agent.cursor_head]:
        for param in module.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()), lr=lr
    )

    # ── Checkpoint manager & resume ───────────────────────────────────
    ckpt_mgr: CheckpointManager | None = None
    start_epoch = 0
    history: Dict[str, List[float]] = {
        "tool_losses":     [],
        "cursor_losses":   [],
        "tool_accuracies": [],
    }

    if checkpoint_dir is not None:
        ckpt_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir, config=config)
        # Find the highest completed epoch checkpoint
        import re
        epoch_tags = []
        for pt in ckpt_mgr.list_checkpoints():
            m = re.match(r"cursor_pretrain_epoch_(\d+)\.pt$", pt.name)
            if m:
                epoch_tags.append(int(m.group(1)))
        if epoch_tags:
            last_epoch = max(epoch_tags)
            tag = f"cursor_pretrain_epoch_{last_epoch}"
            meta = ckpt_mgr.load(agent, optimizer, tag=tag, device=device)
            start_epoch = last_epoch  # resume from NEXT epoch
            history = meta.get("extra", {}).get("history", history)
            if verbose:
                print(f"  Resuming cursor pretraining from epoch {last_epoch} "
                      f"(checkpoint: {tag}.pt)")

    if verbose:
        print(f"Generating oracle dataset ({num_samples} samples)...")

    dataset = OracleDataset(config=config, num_samples=num_samples, seed=dataset_seed)
    actual_samples = len(dataset)

    if verbose:
        print(f"  Generated {actual_samples} oracle samples from "
              f"{len(TaskRegistry.list_tasks())} task types")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0,
    )

    tool_criterion   = nn.CrossEntropyLoss()
    cursor_criterion = nn.CrossEntropyLoss()

    render_h = config["canvas"]["render_height"]
    render_w = config["canvas"]["render_width"]
    num_cursor_classes = render_h * render_w  # 65536 for 256x256

    if start_epoch >= num_epochs:
        if verbose:
            print(f"  All {num_epochs} epochs already completed — nothing to do.")
        return history

    agent.train()
    for epoch in range(start_epoch, num_epochs):
        total_tool_loss   = 0.0
        total_cursor_loss = 0.0
        correct_tool      = 0
        total             = 0

        for batch in loader:
            images          = batch["image"].to(device)          # (B, H, W, C)
            text_ids        = batch["text_ids"].to(device)       # (B, seq_len)
            state_vecs      = batch["state_vec"].to(device)      # (B, state_dim)
            tool_targets    = batch["tool_id"].to(device)        # (B,)
            cursor_targets  = batch["cursor_flat_id"].to(device) # (B,)

            B = images.size(0)

            obs = {
                "image":     images,
                "text_ids":  text_ids,
                "state_vec": state_vecs,
            }

            out = agent(obs)
            tool_logits    = out["tool_logits"]     # (B, num_tools)
            cursor_heatmap = out["cursor_heatmap"]  # (B, 1, H, W)

            # Flatten cursor heatmap to (B, H*W) for CE loss
            cursor_logits = cursor_heatmap.squeeze(1).reshape(B, num_cursor_classes)

            tool_loss   = tool_criterion(tool_logits, tool_targets)
            cursor_loss = cursor_criterion(cursor_logits, cursor_targets)

            # Weight cursor loss lower initially — it's a much harder problem
            loss = tool_loss + 0.5 * cursor_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, agent.parameters()), 1.0
            )
            optimizer.step()

            total_tool_loss   += tool_loss.item()   * B
            total_cursor_loss += cursor_loss.item() * B
            correct_tool      += (tool_logits.argmax(dim=-1) == tool_targets).sum().item()
            total             += B

        n = max(total, 1)
        avg_tool_loss   = total_tool_loss   / n
        avg_cursor_loss = total_cursor_loss / n
        tool_acc        = correct_tool      / n

        history["tool_losses"].append(avg_tool_loss)
        history["cursor_losses"].append(avg_cursor_loss)
        history["tool_accuracies"].append(tool_acc)

        if verbose:
            print(f"  Cursor pretrain epoch {epoch + 1:>3d}/{num_epochs} | "
                  f"tool_loss {avg_tool_loss:.4f} | "
                  f"cursor_loss {avg_cursor_loss:.4f} | "
                  f"tool_acc {tool_acc:.3f}")

        # ── Save epoch checkpoint ─────────────────────────────────────
        if ckpt_mgr is not None:
            epoch_tag = f"cursor_pretrain_epoch_{epoch + 1}"
            ckpt_mgr.save(
                agent, optimizer,
                step=epoch + 1, episode=0,
                tag=epoch_tag,
                extra={"history": history, "num_epochs": num_epochs,
                       "num_samples": num_samples, "dataset_seed": dataset_seed},
            )
            # Also keep a rolling 'latest' tag for quick resume
            ckpt_mgr.save(
                agent, optimizer,
                step=epoch + 1, episode=0,
                tag="cursor_pretrain_latest",
                extra={"history": history, "num_epochs": num_epochs,
                       "num_samples": num_samples, "dataset_seed": dataset_seed},
            )
            # Remove the previous epoch's checkpoint to save disk space
            if epoch > start_epoch:
                prev_tag = f"cursor_pretrain_epoch_{epoch}"
                prev_path = ckpt_mgr.checkpoint_dir / f"{prev_tag}.pt"
                if prev_path.exists():
                    prev_path.unlink()

    # Unfreeze text encoder for subsequent RL training
    for param in agent.text.parameters():
        param.requires_grad = True

    return history


# ── CLI entry-point ───────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cursor imitation pretraining")
    parser.add_argument("--samples",  type=int,   default=20000, help="Oracle samples")
    parser.add_argument("--epochs",   type=int,   default=20,    help="Training epochs")
    parser.add_argument("--lr",       type=float, default=3e-4,  help="Learning rate")
    parser.add_argument("--batch",    type=int,   default=32,    help="Batch size")
    parser.add_argument("--save",     type=str,   default=None,  help="Save checkpoint path")
    parser.add_argument("--device",   type=str,   default=None,  help="torch device")
    args = parser.parse_args()

    config = load_config()
    agent  = CADAgent(config)

    print(f"Cursor imitation pretraining")
    print(f"  Samples: {args.samples}, Epochs: {args.epochs}")

    history = pretrain_cursor_imitation(
        agent, config,
        num_samples=args.samples,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        device=args.device,
    )

    print(f"\nFinal tool accuracy:  {history['tool_accuracies'][-1]:.3f}")
    print(f"Final cursor loss:    {history['cursor_losses'][-1]:.4f}")

    if args.save:
        agent.save_checkpoint(args.save, extra_meta={
            "cursor_pretrain_epochs":  args.epochs,
            "cursor_pretrain_samples": args.samples,
            "cursor_final_tool_acc":   history["tool_accuracies"][-1],
        })
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
