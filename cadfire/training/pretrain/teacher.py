"""
Phase-3: Teacher-Forced Multi-Step Pretraining.

Goal
────
Bridge between Phase-2 (single-step supervised) and Phase-4 (PPO RL).

Phase-3 trains the agent on 2–5 step supervised trajectories using teacher
forcing: at each step the ORACLE action is executed to advance the environment,
regardless of what the agent predicted.  This means:
  – The agent always sees a correctly-progressed state (no error accumulation).
  – Loss is computed at every step vs the oracle action.
  – No long-horizon reward signal (purely supervised, step-level feedback).

Teacher forcing avoids the sparse reward and exploration problems of RL while
teaching the model to handle *sequences* of actions rather than isolated steps.

Primary trajectory: polygon tracing (the gateway to arbitrary geometry)
────────────────────────────────────────────────────────────────────────
A random N-vertex polygon is chosen.  The oracle trajectory is:
  step 0:     POLYLINE  + click vertex[0]
  step 1:     POLYLINE  + click vertex[1]
  ...
  step N-1:   POLYLINE  + click vertex[N-1]
  step N:     CONFIRM   (close the polygon)

Total steps per trajectory: N+1 (N = 3–8, so 4–9 steps).

We also add shorter 2–3-step sequences for other tool chains:
  • Select → ERASE  (2 steps)
  • Select → ROTATE (2 steps)
  • Select → COPY   (2 steps: select + click dest)

Loss design
───────────
Same as Phase 2:
  loss_step = CrossEntropy(tool) + cursor_weight × FocalBCE(cursor_heatmap)
  loss_traj = mean over all steps in trajectory

Checkpoint safety
─────────────────
Loads Phase-2 checkpoint (all weights).  All parameters remain trainable.
After training saves checkpoint for Phase-4 (PPO).

Usage
─────
    # Standalone
    python -m cadfire.training.pretrain.teacher --trajectories 5000 --epochs 15

    # From train.py / notebook
    from cadfire.training.pretrain.teacher import pretrain_teacher_forcing
    history = pretrain_teacher_forcing(agent, config, num_trajectories=5000, num_epochs=15)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cadfire.engine.cad_engine import CADEngine
from cadfire.model.cad_agent import CADAgent
from cadfire.renderer.rasterizer import Renderer
from cadfire.tasks.pretrain.polygon_trace import PolygonTraceTask
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.training.pretrain.semantic import (
    oracle_to_cursor_mask, focal_bce_loss,
)
from cadfire.utils.config import load_config, tool_to_index

# ── Short two-step trajectory builders ────────────────────────────────────────

from cadfire.tasks.pretrain.select import SemanticSelectTask
from cadfire.tasks.pretrain.delete import DeleteObjectTask
from cadfire.tasks.pretrain.rotate import RotateObjectTask
from cadfire.tasks.pretrain.copy_paste import CopyObjectTask
from cadfire.tasks.pretrain.move import MoveObjectTask, prepositional_move_step
from cadfire.tasks.pretrain.conditional import AndSelectTrajectory


def _build_select_then_erase(rng, engine, renderer, tokenizer, tool_idx,
                              H, W, sigma, state_dim, config) -> List[Dict]:
    """2-step: SELECT target, then ERASE."""
    select_task = SemanticSelectTask(seed=int(rng.randint(0, 2**31)))
    engine.reset()
    setup = select_task.setup(engine)
    image = renderer.render(engine)
    text_ids = np.array(tokenizer.encode_padded(setup["prompt"]), dtype=np.int32)
    sv = _state_vec(engine, tool_idx, state_dim, config)

    # Step 1: SELECT
    oracle1 = select_task.oracle_action(engine, setup)
    mask1 = oracle_to_cursor_mask(oracle1["cursor_world"], engine, H, W, sigma)
    step1 = _make_step(image, text_ids, sv, tool_idx[oracle1["tool"]],
                        mask1, oracle1.get("cursor_weight", 1.0))

    # Apply oracle: select the entity
    engine.selected_ids.add(setup["target_entity"].id)

    # Re-render after selection
    image2 = renderer.render(engine)
    erase_prompt = "Erase the selected shape"
    text_ids2 = np.array(tokenizer.encode_padded(erase_prompt), dtype=np.int32)
    sv2 = _state_vec(engine, tool_idx, state_dim, config)
    centroid = setup["target_entity"].centroid()
    mask2 = oracle_to_cursor_mask(centroid, engine, H, W, sigma)
    step2 = _make_step(image2, text_ids2, sv2, tool_idx["ERASE"], mask2, 0.1)

    return [step1, step2]


def _build_select_then_rotate(rng, engine, renderer, tokenizer, tool_idx,
                               H, W, sigma, state_dim, config) -> List[Dict]:
    """2-step: SELECT target, then ROTATE."""
    select_task = SemanticSelectTask(seed=int(rng.randint(0, 2**31)))
    engine.reset()
    setup = select_task.setup(engine)
    image = renderer.render(engine)
    text_ids = np.array(tokenizer.encode_padded(setup["prompt"]), dtype=np.int32)
    sv = _state_vec(engine, tool_idx, state_dim, config)
    oracle1 = select_task.oracle_action(engine, setup)
    mask1 = oracle_to_cursor_mask(oracle1["cursor_world"], engine, H, W, sigma)
    step1 = _make_step(image, text_ids, sv, tool_idx[oracle1["tool"]],
                        mask1, oracle1.get("cursor_weight", 1.0))

    engine.selected_ids.add(setup["target_entity"].id)

    image2 = renderer.render(engine)
    rotate_prompt = "Rotate the selected shape"
    text_ids2 = np.array(tokenizer.encode_padded(rotate_prompt), dtype=np.int32)
    sv2 = _state_vec(engine, tool_idx, state_dim, config)
    pivot = setup["target_entity"].centroid()
    mask2 = oracle_to_cursor_mask(pivot, engine, H, W, sigma)
    step2 = _make_step(image2, text_ids2, sv2, tool_idx["ROTATE"], mask2, 0.8)

    return [step1, step2]


def _build_select_then_copy(rng, engine, renderer, tokenizer, tool_idx,
                             H, W, sigma, state_dim, config) -> List[Dict]:
    """2-step: SELECT target, then COPY to destination."""
    select_task = SemanticSelectTask(seed=int(rng.randint(0, 2**31)))
    engine.reset()
    setup = select_task.setup(engine)
    image = renderer.render(engine)
    text_ids = np.array(tokenizer.encode_padded(setup["prompt"]), dtype=np.int32)
    sv = _state_vec(engine, tool_idx, state_dim, config)
    oracle1 = select_task.oracle_action(engine, setup)
    mask1 = oracle_to_cursor_mask(oracle1["cursor_world"], engine, H, W, sigma)
    step1 = _make_step(image, text_ids, sv, tool_idx[oracle1["tool"]],
                        mask1, oracle1.get("cursor_weight", 1.0))

    engine.selected_ids.add(setup["target_entity"].id)

    image2 = renderer.render(engine)
    dest = np.array([float(rng.uniform(600, 850)), float(rng.uniform(200, 800))])
    copy_prompt = f"Copy the {setup['target_name']} to the right"
    text_ids2 = np.array(tokenizer.encode_padded(copy_prompt), dtype=np.int32)
    sv2 = _state_vec(engine, tool_idx, state_dim, config)
    mask2 = oracle_to_cursor_mask(dest, engine, H, W, sigma)
    step2 = _make_step(image2, text_ids2, sv2, tool_idx["COPY"], mask2, 1.0)

    return [step1, step2]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_vec(engine: CADEngine, tool_idx: Dict, state_dim: int,
               config: Dict) -> np.ndarray:
    canvas = config["canvas"]
    num_tools = len(tool_idx)
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


def _make_step(image, text_ids, state_vec, tool_id, cursor_mask,
               cursor_weight) -> Dict:
    return {
        "image":         image,
        "text_ids":      text_ids,
        "state_vec":     state_vec,
        "tool_id":       tool_id,
        "cursor_mask":   cursor_mask,
        "cursor_weight": float(cursor_weight),
    }


def _build_select_then_move(rng, engine, renderer, tokenizer, tool_idx,
                             H, W, sigma, state_dim, config) -> List[Dict]:
    """2-step: SELECT target, then MOVE it prepositionally (directional)."""
    select_task = SemanticSelectTask(seed=int(rng.randint(0, 2**31)))
    engine.reset()
    setup = select_task.setup(engine)
    image = renderer.render(engine)
    text_ids = np.array(tokenizer.encode_padded(setup["prompt"]), dtype=np.int32)
    sv = _state_vec(engine, tool_idx, state_dim, config)
    oracle1 = select_task.oracle_action(engine, setup)
    mask1 = oracle_to_cursor_mask(oracle1["cursor_world"], engine, H, W, sigma)
    step1 = _make_step(image, text_ids, sv, tool_idx[oracle1["tool"]],
                        mask1, oracle1.get("cursor_weight", 1.0))

    # Apply oracle: select the entity
    engine.selected_ids.add(setup["target_entity"].id)

    # Generate prepositional MOVE prompt & destination
    move_prompt, dest = prepositional_move_step(
        setup["target_entity"], setup["target_name"], rng
    )

    image2 = renderer.render(engine)
    text_ids2 = np.array(tokenizer.encode_padded(move_prompt), dtype=np.int32)
    sv2 = _state_vec(engine, tool_idx, state_dim, config)
    mask2 = oracle_to_cursor_mask(dest, engine, H, W, sigma)
    step2 = _make_step(image2, text_ids2, sv2, tool_idx["MOVE"], mask2, 1.0)

    return [step1, step2]


def _build_and_select(rng, engine, renderer, tokenizer, tool_idx,
                      H, W, sigma, state_dim, config) -> List[Dict]:
    """2-step: SELECT shape1, then MULTISELECT shape2 (AND-selection)."""
    traj_task = AndSelectTrajectory(seed=int(rng.randint(0, 2**31)))
    engine.reset()
    setup = traj_task.setup(engine)
    prompt = setup["prompt"]

    # Step 1: SELECT shape1
    image = renderer.render(engine)
    text_ids = np.array(tokenizer.encode_padded(prompt), dtype=np.int32)
    sv = _state_vec(engine, tool_idx, state_dim, config)
    oracle1 = traj_task.oracle_step1()
    mask1 = oracle_to_cursor_mask(oracle1["cursor_world"], engine, H, W, sigma)
    step1 = _make_step(image, text_ids, sv, tool_idx[oracle1["tool"]],
                        mask1, oracle1["cursor_weight"])

    # Apply oracle: register shape1 as selected
    engine.selected_ids.add(setup["shape1_entity"].id)

    # Step 2: MULTISELECT shape2
    image2 = renderer.render(engine)
    text_ids2 = np.array(tokenizer.encode_padded(prompt), dtype=np.int32)
    sv2 = _state_vec(engine, tool_idx, state_dim, config)
    oracle2 = traj_task.oracle_step2()
    mask2 = oracle_to_cursor_mask(oracle2["cursor_world"], engine, H, W, sigma)
    step2 = _make_step(image2, text_ids2, sv2, tool_idx[oracle2["tool"]],
                        mask2, oracle2["cursor_weight"])

    return [step1, step2]


# ── Short trajectory builders registry ───────────────────────────────────────

_SHORT_BUILDERS = [
    _build_select_then_erase,
    _build_select_then_rotate,
    _build_select_then_copy,
    _build_select_then_move,
    _build_and_select,
]


# ── Dataset (on-the-fly trajectory generation) ────────────────────────────────

class TeacherForcingDataset:
    """
    On-the-fly generator of multi-step trajectories.

    Each call to ``__getitem__`` produces one complete trajectory: a list of
    per-step dicts with (obs, tool_id, cursor_mask, cursor_weight).

    Trajectories are NOT batched across time steps (variable length).
    The DataLoader is therefore used with batch_size=1 and trajectories are
    collated manually in the training loop.

    Mix of trajectory types (controlled by ``polygon_ratio``):
      • Polygon tracing:          (N+1 steps, N = 3–8)
      • Short 2-step sequences:   (select→erase / rotate / copy)
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        num_trajectories: int = 5_000,
        sigma: float = 12.0,
        polygon_ratio: float = 0.7,
        seed: int | None = None,
    ):
        self.config = config or load_config()
        self.num_trajectories = num_trajectories
        self.sigma = sigma
        self.polygon_ratio = polygon_ratio
        self.rng = np.random.RandomState(seed)

        canvas = self.config["canvas"]
        self.H = canvas["render_height"]
        self.W = canvas["render_width"]
        self.state_dim = self.config["model"]["state_dim"]

        self.tokenizer = BPETokenizer(
            vocab_size=self.config["model"]["text_vocab_size"],
            max_len=self.config["model"]["text_max_len"],
        )
        self.tool_idx = tool_to_index()

    def __len__(self) -> int:
        return self.num_trajectories

    def __getitem__(self, idx: int) -> List[Dict]:
        engine = CADEngine(self.config)
        renderer = Renderer(self.config)

        if self.rng.rand() < self.polygon_ratio:
            return self._gen_polygon(engine, renderer)
        else:
            return self._gen_short(engine, renderer)

    def _gen_polygon(self, engine, renderer) -> List[Dict]:
        task = PolygonTraceTask(
            seed=int(self.rng.randint(0, 2**31)),
            config=self.config,
        )
        traj = task.generate_trajectory(engine, renderer, self.tokenizer)
        # Convert trajectory dicts to training step format
        steps = []
        for t in traj:
            obs = t["obs"]
            mask = oracle_to_cursor_mask(
                t["cursor_world"], engine, self.H, self.W, self.sigma
            )
            steps.append(_make_step(
                obs["image"], obs["text_ids"], obs["state_vec"],
                t["tool_id"], mask, t["cursor_weight"],
            ))
        return steps

    def _gen_short(self, engine, renderer) -> List[Dict]:
        builder = _SHORT_BUILDERS[int(self.rng.randint(len(_SHORT_BUILDERS)))]
        return builder(
            self.rng, engine, renderer, self.tokenizer, self.tool_idx,
            self.H, self.W, self.sigma, self.state_dim, self.config,
        )


# ── Training loop ─────────────────────────────────────────────────────────────

def pretrain_teacher_forcing(
    agent: CADAgent,
    config: Dict[str, Any] | None = None,
    num_trajectories: int = 5_000,
    num_epochs: int = 15,
    lr: float = 1e-4,
    sigma: float = 12.0,
    cursor_weight: float = 1.5,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    polygon_ratio: float = 0.7,
    device: str | None = None,
    verbose: bool = True,
    seed: int | None = None,
) -> Dict[str, List[float]]:
    """
    Phase-3 teacher-forced multi-step pretraining.

    Trains ALL agent parameters on 2–9-step supervised trajectories with
    teacher forcing.  Polygon tracing is the primary trajectory type.

    Args:
        agent            : CADAgent instance (modified in-place).
        config           : Config dict.
        num_trajectories : Trajectories generated per epoch.
        num_epochs       : Training epochs.
        lr               : Adam learning rate (lower than Phase 2).
        sigma            : Gaussian blob radius for cursor targets.
        cursor_weight    : Global cursor BCE loss scale.
        focal_gamma      : Focal loss gamma.
        focal_alpha      : Focal loss alpha.
        polygon_ratio    : Fraction of polygon-tracing trajectories (rest are
                           short 2-step sequences).
        device           : 'cuda' / 'cpu' / None (auto).
        verbose          : Print per-epoch stats.
        seed             : RNG seed.

    Returns:
        History dict with per-epoch averages:
        'tool_losses', 'cursor_losses', 'total_losses', 'tool_accuracies',
        'traj_lengths'.
    """
    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = agent.to(device)
    for param in agent.parameters():
        param.requires_grad = True

    trainable = [p for p in agent.parameters() if p.requires_grad]
    if verbose:
        n = sum(p.numel() for p in trainable)
        print(f"  Teacher forcing: {n:,} trainable parameters")

    optimizer = optim.Adam(trainable, lr=lr)
    tool_criterion = nn.CrossEntropyLoss()

    dataset = TeacherForcingDataset(
        config=config,
        num_trajectories=num_trajectories,
        sigma=sigma,
        polygon_ratio=polygon_ratio,
        seed=seed,
    )

    history: Dict[str, List[float]] = {
        "tool_losses":    [],
        "cursor_losses":  [],
        "total_losses":   [],
        "tool_accuracies": [],
        "traj_lengths":   [],
    }

    for epoch in range(num_epochs):
        epoch_tool   = 0.0
        epoch_cursor = 0.0
        epoch_total  = 0.0
        epoch_corr   = 0
        epoch_steps  = 0
        epoch_trajs  = 0

        # Shuffle trajectory indices
        indices = list(range(len(dataset)))
        rng_ep  = np.random.RandomState(seed if seed is None else seed + epoch)
        rng_ep.shuffle(indices)

        agent.train()

        for traj_idx in indices:
            trajectory = dataset[traj_idx]
            if not trajectory:
                continue

            traj_tool   = 0.0
            traj_cursor = 0.0
            traj_loss   = torch.tensor(0.0, device=device)
            traj_corr   = 0

            for step in trajectory:
                image      = torch.from_numpy(step["image"]).float().unsqueeze(0).to(device)
                text_ids   = torch.from_numpy(step["text_ids"]).long().unsqueeze(0).to(device)
                state_vec  = torch.from_numpy(step["state_vec"]).float().unsqueeze(0).to(device)
                tool_id    = torch.tensor([step["tool_id"]], dtype=torch.long, device=device)
                cursor_tgt = torch.from_numpy(step["cursor_mask"]).float().unsqueeze(0).to(device)
                c_w        = step["cursor_weight"]

                obs = {"image": image, "text_ids": text_ids, "state_vec": state_vec}
                out = agent(obs)

                tool_logits    = out["tool_logits"]    # (1, num_tools)
                cursor_heatmap = out["cursor_heatmap"] # (1, 1, H, W)

                t_loss = tool_criterion(tool_logits, tool_id)
                c_loss = focal_bce_loss(
                    cursor_heatmap.squeeze(1),  # (1, H, W)
                    cursor_tgt,
                    gamma=focal_gamma,
                    alpha=focal_alpha,
                )
                step_loss = t_loss + cursor_weight * c_w * c_loss
                traj_loss = traj_loss + step_loss

                traj_tool   += t_loss.item()
                traj_cursor += c_loss.item()
                traj_corr   += int((tool_logits.argmax(dim=-1) == tool_id).item())

            n_steps = len(trajectory)
            optimizer.zero_grad()
            (traj_loss / max(n_steps, 1)).backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_tool   += traj_tool
            epoch_cursor += traj_cursor
            epoch_total  += (traj_loss.item() / max(n_steps, 1))
            epoch_corr   += traj_corr
            epoch_steps  += n_steps
            epoch_trajs  += 1

        n = max(epoch_steps, 1)
        avg_tool   = epoch_tool   / n
        avg_cursor = epoch_cursor / n
        avg_total  = epoch_total  / max(epoch_trajs, 1)
        avg_acc    = epoch_corr   / n
        avg_len    = epoch_steps  / max(epoch_trajs, 1)

        history["tool_losses"].append(avg_tool)
        history["cursor_losses"].append(avg_cursor)
        history["total_losses"].append(avg_total)
        history["tool_accuracies"].append(avg_acc)
        history["traj_lengths"].append(avg_len)

        if verbose:
            print(
                f"  Teacher epoch {epoch + 1:>3d}/{num_epochs} | "
                f"total {avg_total:.4f} | "
                f"tool {avg_tool:.4f} (acc {avg_acc:.3f}) | "
                f"cursor {avg_cursor:.4f} | "
                f"avg_len {avg_len:.1f}"
            )

    return history


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    import argparse
    from cadfire.training.checkpoint import CheckpointManager

    parser = argparse.ArgumentParser(
        description="Phase-3 teacher-forced multi-step pretraining"
    )
    parser.add_argument("--trajectories", type=int,   default=5_000)
    parser.add_argument("--epochs",       type=int,   default=15)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--sigma",        type=float, default=12.0)
    parser.add_argument("--cursor-weight",type=float, default=1.5)
    parser.add_argument("--poly-ratio",   type=float, default=0.7)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--load",         type=str,   default=None)
    parser.add_argument("--save",         type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=None)
    args = parser.parse_args()

    config = load_config()
    agent  = CADAgent(config)
    dev    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.load:
        ckpt = CheckpointManager(args.load)
        ckpt.load(agent, optimizer=None, device=dev)
        print(f"Loaded checkpoint from {args.load}")

    print("=" * 60)
    print("Phase 3 – Teacher-Forced Multi-Step Pretraining")
    print(f"  Trajectories/epoch : {args.trajectories:,}")
    print(f"  Epochs             : {args.epochs}")
    print(f"  LR                 : {args.lr}")
    print(f"  Polygon ratio      : {args.poly_ratio:.0%}")
    print("=" * 60)

    history = pretrain_teacher_forcing(
        agent, config,
        num_trajectories=args.trajectories,
        num_epochs=args.epochs,
        lr=args.lr,
        sigma=args.sigma,
        cursor_weight=args.cursor_weight,
        polygon_ratio=args.poly_ratio,
        device=dev,
        seed=args.seed,
    )

    print(f"\nFinal tool accuracy : {history['tool_accuracies'][-1]:.3f}")
    print(f"Final avg traj len  : {history['traj_lengths'][-1]:.1f}")

    if args.save:
        dummy_opt = optim.Adam(agent.parameters(), lr=1e-4)
        ckpt = CheckpointManager(args.save)
        ckpt.save(agent, dummy_opt, step=0, episode=0, extra={
            "pretrain_phase": "teacher_forcing",
            "pretrain_epochs": args.epochs,
            "final_tool_acc": history["tool_accuracies"][-1],
        })
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
