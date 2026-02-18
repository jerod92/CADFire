"""
Semantic cursor pretraining – Phase 2 of the warm-start pipeline.

Goal
────
After Phase 1 (text → tool classifier), teach the model to:

  (a) Single-select  – "Select the <shape>"
        tool  = SELECT
        cursor heatmap peaks inside the named entity

  (b) Multi-select   – "Select all <shape>s"
        tool  = MULTISELECT
        cursor heatmap peaks at every instance of the named entity

Both are single-step tasks (no pending points, no multi-click sequences).
Training uses supervised behavioural cloning against oracle actions.

Loss design
───────────
  • Tool head  : CrossEntropyLoss  (same as Phase 1)
  • Cursor head: Focal BCE on the full (H × W) heatmap
      – Target  = Gaussian blob(s) centred on entity centroid(s)
      – Focal weighting suppresses easy-negative gradient
      – Works identically for SELECT (one blob) and MULTISELECT (N blobs)
  • Total loss  = tool_loss + cursor_weight × cursor_loss

Checkpoint safety
─────────────────
The function accepts an optional checkpoint path.  It loads the existing
weights (preserving Phase-1 text-encoder weights), freezes the text encoder,
and trains vision encoder + fusion + tool head + cursor head.
After training, ALL parameters are unfrozen for subsequent PPO.

Usage
─────
    # Standalone
    python -m cadfire.training.pretrain_semantic --samples 20000 --epochs 20

    # From notebook / train.py
    from cadfire.training.pretrain_semantic import pretrain_semantic_cursor
    history = pretrain_semantic_cursor(agent, config, num_samples=20000, num_epochs=20)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cadfire.engine.cad_engine import CADEngine
from cadfire.model.cad_agent import CADAgent
from cadfire.renderer.rasterizer import Renderer
from cadfire.tasks.pretrain_select_tasks import (
    SemanticSelectTask, SemanticMultiSelectTask,
)
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_to_index


# ── Cursor-mask helpers ───────────────────────────────────────────────────────

def _world_to_pixel(world_xy: np.ndarray, engine: CADEngine,
                    H: int, W: int) -> Tuple[int, int]:
    """Convert one world-space point to (row, col) pixel coordinates."""
    ndc = engine.viewport.world_to_ndc(world_xy.reshape(1, 2))[0]
    col = int(np.clip(ndc[0] * W, 0, W - 1))
    row = int(np.clip((1.0 - ndc[1]) * H, 0, H - 1))
    return row, col


def _gaussian_blob(row: int, col: int, H: int, W: int,
                   sigma: float = 12.0) -> np.ndarray:
    """Return an (H, W) float32 array with a Gaussian peak at (row, col)."""
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    return np.exp(-((ys - row) ** 2 + (xs - col) ** 2) / (2 * sigma ** 2))


def _make_cursor_mask(centroids_px: List[Tuple[int, int]],
                      H: int, W: int,
                      sigma: float = 12.0) -> np.ndarray:
    """
    Build a float32 cursor-mask for one or more entity centroids.

    Each centroid contributes a Gaussian blob; the result is clipped to [0, 1].
    """
    mask = np.zeros((H, W), dtype=np.float32)
    for row, col in centroids_px:
        mask += _gaussian_blob(row, col, H, W, sigma)
    return np.clip(mask, 0.0, 1.0)


# ── Focal BCE loss ────────────────────────────────────────────────────────────

def focal_bce_loss(pred_logits: torch.Tensor,
                   target: torch.Tensor,
                   gamma: float = 2.0,
                   alpha: float = 0.75) -> torch.Tensor:
    """
    Focal Binary Cross-Entropy loss for imbalanced cursor-mask prediction.

    Args:
        pred_logits : (B, H, W) – raw logits from cursor head (squeezed)
        target      : (B, H, W) – soft Gaussian masks in [0, 1]
        gamma       : focusing parameter (2.0 standard)
        alpha       : weight for positive class (higher = focus on positives)

    The Gaussian target values act as soft labels so the loss is already
    smooth; gamma then down-weights easy negatives further.
    """
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none"
    )
    prob = torch.sigmoid(pred_logits)
    # p_t: probability of the "correct" class
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    # alpha_t: weighting
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SemanticDataset(Dataset):
    """
    Supervised dataset for semantic cursor pretraining.

    Each sample is generated on-the-fly:
      1. Instantiate a random SemanticSelectTask or SemanticMultiSelectTask.
      2. Run setup() to populate the engine.
      3. Render the observation image.
      4. Build a Gaussian cursor-mask targeting the relevant entity centroid(s).
      5. Record the oracle tool id (SELECT or MULTISELECT).

    Returns a dict with:
        image       : (H, W, C)   float32
        text_ids    : (max_len,)  int32
        state_vec   : (state_dim,) float32
        tool_id     : ()          int64
        cursor_mask : (H, W)      float32  – Gaussian blob(s) in [0, 1]
        is_multiselect : ()       bool
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        num_samples: int = 20_000,
        sigma: float = 12.0,
        multi_ratio: float = 0.4,   # fraction of samples that are MULTISELECT
        seed: int | None = None,
    ):
        self.config = config or load_config()
        self.num_samples = num_samples
        self.sigma = sigma
        self.multi_ratio = multi_ratio
        self.rng = np.random.RandomState(seed)

        canvas = self.config["canvas"]
        self.H = canvas["render_height"]
        self.W = canvas["render_width"]

        self.tokenizer = BPETokenizer(
            vocab_size=self.config["model"]["text_vocab_size"],
            max_len=self.config["model"]["text_max_len"],
        )
        self.max_len = self.config["model"]["text_max_len"]
        self.state_dim = self.config["model"]["state_dim"]

        self._tool_idx = tool_to_index()
        self._select_id = self._tool_idx["SELECT"]
        self._multiselect_id = self._tool_idx["MULTISELECT"]

    # ----- internal helpers --------------------------------------------------

    def _build_state_vec(self, engine: CADEngine) -> np.ndarray:
        """Minimal state vector (mirrors CADEnv._build_state_vector)."""
        canvas = self.config["canvas"]
        num_tools = len(self._tool_idx)
        vec = np.zeros(self.state_dim, dtype=np.float32)
        vec[0] = self._tool_idx.get(engine.active_tool, 0) / max(num_tools, 1)
        vec[1] = np.log1p(engine.viewport.zoom) / 5.0
        vec[2] = engine.viewport.center[0] / canvas["world_width"]
        vec[3] = engine.viewport.center[1] / canvas["world_height"]
        vec[4] = engine.active_layer / max(len(engine.layers), 1)
        vec[5] = engine.active_color / 8.0
        vec[6] = min(len(engine.entities), 100) / 100.0
        vec[7] = min(len(engine.selected_ids), 50) / 50.0
        vec[8] = min(len(engine.pending_points), 10) / 10.0
        return vec

    def _generate_single_select(
        self, engine: CADEngine, renderer: Renderer
    ) -> Dict[str, Any]:
        """Generate one SELECT sample."""
        seed = int(self.rng.randint(0, 2 ** 31))
        task = SemanticSelectTask(seed=seed)
        engine.reset()
        setup_info = task.setup(engine)

        # Render observation (no selection active yet)
        image = renderer.render(engine)

        # Oracle cursor: Gaussian blob at the target entity's centroid
        target = setup_info["target_entity"]
        centroid = target.centroid()
        row, col = _world_to_pixel(centroid, engine, self.H, self.W)
        cursor_mask = _make_cursor_mask([(row, col)], self.H, self.W, self.sigma)

        text_ids = np.array(
            self.tokenizer.encode_padded(setup_info["prompt"]), dtype=np.int32
        )
        state_vec = self._build_state_vec(engine)

        return {
            "image": image,
            "text_ids": text_ids,
            "state_vec": state_vec,
            "tool_id": self._select_id,
            "cursor_mask": cursor_mask,
            "is_multiselect": False,
        }

    def _generate_multi_select(
        self, engine: CADEngine, renderer: Renderer
    ) -> Dict[str, Any]:
        """Generate one MULTISELECT sample."""
        seed = int(self.rng.randint(0, 2 ** 31))
        task = SemanticMultiSelectTask(seed=seed)
        engine.reset()
        setup_info = task.setup(engine)

        # Render observation (no selection active yet)
        image = renderer.render(engine)

        # Oracle cursor: Gaussian blobs at ALL target entity centroids
        target_entities = setup_info["target_entities"]
        centroids_px = []
        for ent in target_entities:
            c = ent.centroid()
            row, col = _world_to_pixel(c, engine, self.H, self.W)
            centroids_px.append((row, col))
        cursor_mask = _make_cursor_mask(centroids_px, self.H, self.W, self.sigma)

        text_ids = np.array(
            self.tokenizer.encode_padded(setup_info["prompt"]), dtype=np.int32
        )
        state_vec = self._build_state_vec(engine)

        return {
            "image": image,
            "text_ids": text_ids,
            "state_vec": state_vec,
            "tool_id": self._multiselect_id,
            "cursor_mask": cursor_mask,
            "is_multiselect": True,
        }

    # ----- Dataset interface -------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Each worker gets its own engine + renderer (no shared state)
        engine = CADEngine(self.config)
        renderer = Renderer(self.config)

        use_multi = (self.rng.rand() < self.multi_ratio)
        if use_multi:
            sample = self._generate_multi_select(engine, renderer)
        else:
            sample = self._generate_single_select(engine, renderer)

        return {
            "image":          torch.from_numpy(sample["image"]).float(),
            "text_ids":       torch.from_numpy(sample["text_ids"]).long(),
            "state_vec":      torch.from_numpy(sample["state_vec"]).float(),
            "tool_id":        torch.tensor(sample["tool_id"], dtype=torch.long),
            "cursor_mask":    torch.from_numpy(sample["cursor_mask"]).float(),
            "is_multiselect": torch.tensor(sample["is_multiselect"], dtype=torch.bool),
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def pretrain_semantic_cursor(
    agent: CADAgent,
    config: Dict[str, Any] | None = None,
    num_samples: int = 20_000,
    num_epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 32,
    sigma: float = 12.0,
    multi_ratio: float = 0.4,
    cursor_weight: float = 1.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    num_workers: int = 0,
    device: str | None = None,
    verbose: bool = True,
    seed: int | None = None,
) -> Dict[str, List[float]]:
    """
    Phase-2 semantic cursor pretraining via supervised behavioural cloning.

    Freezes the text encoder (preserving Phase-1 weights) and trains:
        vision encoder, fusion bridge, tool head, cursor head

    Args:
        agent         : CADAgent instance (modified in-place).
        config        : Config dict (defaults to global config.json).
        num_samples   : Number of generated samples per epoch.
        num_epochs    : Training epochs over the dataset.
        lr            : Adam learning rate.
        batch_size    : Mini-batch size.
        sigma         : Gaussian blob radius (pixels) for cursor targets.
        multi_ratio   : Fraction of samples that are MULTISELECT (vs SELECT).
        cursor_weight : Loss weight for cursor BCE vs tool CE.
        focal_gamma   : Focal-loss gamma (focusing on hard examples).
        focal_alpha   : Focal-loss alpha (positive-class weight).
        num_workers   : DataLoader worker processes (0 = main process only).
        device        : 'cuda' / 'cpu' / None (auto-detect).
        verbose       : Print per-epoch stats.
        seed          : Optional RNG seed.

    Returns:
        Dict with lists: 'tool_losses', 'cursor_losses', 'total_losses',
        'tool_accuracies'.
    """
    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = agent.to(device)

    # ── Freeze text encoder; unfreeze everything else ──────────────────────
    for param in agent.parameters():
        param.requires_grad = False
    for module in [agent.vision, agent.fusion, agent.tool_head, agent.cursor_head]:
        for param in module.parameters():
            param.requires_grad = True

    trainable = [p for p in agent.parameters() if p.requires_grad]
    if verbose:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Semantic pretrain: {n_trainable:,} trainable parameters "
              f"(text encoder frozen)")

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset = SemanticDataset(
        config=config,
        num_samples=num_samples,
        sigma=sigma,
        multi_ratio=multi_ratio,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        # Each worker needs its own engine/renderer – stateless dataset handles this
        persistent_workers=(num_workers > 0),
    )

    optimizer = optim.Adam(trainable, lr=lr)
    tool_criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "tool_losses": [],
        "cursor_losses": [],
        "total_losses": [],
        "tool_accuracies": [],
    }

    agent.train()

    for epoch in range(num_epochs):
        epoch_tool_loss = 0.0
        epoch_cursor_loss = 0.0
        epoch_total_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for batch in loader:
            image      = batch["image"].to(device)           # (B, H, W, C)
            text_ids   = batch["text_ids"].to(device)        # (B, max_len)
            state_vec  = batch["state_vec"].to(device)       # (B, state_dim)
            tool_ids   = batch["tool_id"].to(device)         # (B,)
            cursor_tgt = batch["cursor_mask"].to(device)     # (B, H, W)

            obs = {
                "image":     image,
                "text_ids":  text_ids,
                "state_vec": state_vec,
            }

            out = agent(obs)
            tool_logits    = out["tool_logits"]    # (B, num_tools)
            cursor_heatmap = out["cursor_heatmap"] # (B, 1, H, W)

            # Tool classification loss (cross-entropy)
            t_loss = tool_criterion(tool_logits, tool_ids)

            # Cursor focal BCE loss  (heatmap vs Gaussian mask)
            c_loss = focal_bce_loss(
                cursor_heatmap.squeeze(1),  # (B, H, W)
                cursor_tgt,                 # (B, H, W)
                gamma=focal_gamma,
                alpha=focal_alpha,
            )

            loss = t_loss + cursor_weight * c_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            B = tool_ids.size(0)
            epoch_tool_loss   += t_loss.item() * B
            epoch_cursor_loss += c_loss.item() * B
            epoch_total_loss  += loss.item() * B
            epoch_correct     += (tool_logits.argmax(dim=-1) == tool_ids).sum().item()
            n_batches         += B

        n = max(n_batches, 1)
        avg_tool   = epoch_tool_loss   / n
        avg_cursor = epoch_cursor_loss / n
        avg_total  = epoch_total_loss  / n
        avg_acc    = epoch_correct      / n

        history["tool_losses"].append(avg_tool)
        history["cursor_losses"].append(avg_cursor)
        history["total_losses"].append(avg_total)
        history["tool_accuracies"].append(avg_acc)

        if verbose:
            print(
                f"  Semantic epoch {epoch + 1:>3d}/{num_epochs} | "
                f"total {avg_total:.4f} | "
                f"tool {avg_tool:.4f} (acc {avg_acc:.3f}) | "
                f"cursor {avg_cursor:.4f}"
            )

    # ── Unfreeze everything for PPO ────────────────────────────────────────
    for param in agent.parameters():
        param.requires_grad = True

    return history


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    import argparse
    from cadfire.training.checkpoint import CheckpointManager

    parser = argparse.ArgumentParser(
        description="Phase-2 semantic cursor pretraining"
    )
    parser.add_argument("--samples",  type=int,   default=20_000,
                        help="Generated samples per epoch")
    parser.add_argument("--epochs",   type=int,   default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr",       type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch",    type=int,   default=32,
                        help="Batch size")
    parser.add_argument("--sigma",    type=float, default=12.0,
                        help="Gaussian blob radius for cursor targets (pixels)")
    parser.add_argument("--multi-ratio", type=float, default=0.4,
                        help="Fraction of MULTISELECT samples (0-1)")
    parser.add_argument("--cursor-weight", type=float, default=1.0,
                        help="Relative weight of cursor loss vs tool loss")
    parser.add_argument("--workers",  type=int,   default=0,
                        help="DataLoader worker processes")
    parser.add_argument("--device",   type=str,   default=None,
                        help="torch device (cuda/cpu/auto)")
    parser.add_argument("--load",     type=str,   default=None,
                        help="Load checkpoint before training")
    parser.add_argument("--save",     type=str,   default=None,
                        help="Save checkpoint after training")
    parser.add_argument("--seed",     type=int,   default=None,
                        help="Random seed")
    args = parser.parse_args()

    config = load_config()
    agent  = CADAgent(config)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.load:
        ckpt = CheckpointManager(args.load)
        ckpt.load(agent, optimizer=None, device=dev)
        print(f"Loaded checkpoint from {args.load}")

    print("=" * 60)
    print("Phase 2 – Semantic Cursor Pretraining")
    print(f"  Samples/epoch : {args.samples:,}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch}")
    print(f"  LR            : {args.lr}")
    print(f"  Sigma (px)    : {args.sigma}")
    print(f"  Multi-ratio   : {args.multi_ratio:.0%}")
    print("=" * 60)

    history = pretrain_semantic_cursor(
        agent, config,
        num_samples=args.samples,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        sigma=args.sigma,
        multi_ratio=args.multi_ratio,
        cursor_weight=args.cursor_weight,
        num_workers=args.workers,
        device=dev,
        seed=args.seed,
    )

    print(f"\nFinal tool accuracy : {history['tool_accuracies'][-1]:.3f}")
    print(f"Final cursor loss   : {history['cursor_losses'][-1]:.4f}")

    if args.save:
        # Save using a minimal dummy-optimizer state so CheckpointManager is happy
        dummy_opt = optim.Adam(agent.parameters(), lr=1e-3)
        ckpt = CheckpointManager(args.save)
        ckpt.save(agent, dummy_opt, step=0, episode=0, extra={
            "pretrain_phase": "semantic_cursor",
            "pretrain_epochs": args.epochs,
            "final_tool_acc": history["tool_accuracies"][-1],
            "final_cursor_loss": history["cursor_losses"][-1],
        })
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
