"""
Semantic cursor pretraining – Phase 2 of the warm-start pipeline.

Goal
────
After Phase 1 (text → tool classifier), teach the model to jointly predict:
  (a) the correct tool for a given visual + text observation
  (b) the precise cursor location (heatmap) for cursor-critical tools

ALL parameters are trained in Phase 2 (including the text encoder) because
the model must simultaneously learn:
  – the visual meaning of shape types
  – the linguistic names of objects and actions
Freezing the text encoder here would prevent the model from associating
"hexagon" → the six-sided shape it sees, etc.

Supervised task coverage (one task per major tool)
───────────────────────────────────────────────────
  SELECT          – SemanticSelectTask
  MULTISELECT     – SemanticMultiSelectTask
  ERASE           – DeleteObjectTask
  PAN             – PanTask (up/down/left/right)
  ZOOM_IN         – ZoomInTask
  ZOOM_OUT        – ZoomOutTask
  HATCH           – HatchObjectTask
  POLYLINE        – TraceNextPointTask  (click next polygon vertex)
  COPY            – CopyObjectTask
  MOVE            – MoveObjectTask
  ROTATE          – RotateObjectTask
  SCALE           – ScaleObjectTask         (single-step)
  MIRROR          – MirrorObjectTask        (single-step)
  OFFSET          – OffsetTask              (single-step)
  SCALE (chat)    – ScaleFromChatTask       (multi-turn: "Draw X | make it smaller")
  MOVE  (chat)    – MoveFromChatTask        (multi-turn: "Draw X | move it right")
  ROTATE(chat)    – RotateFromChatTask      (multi-turn: "Draw X | rotate it")
  ERASE (chat)    – EraseFromChatTask       (multi-turn: "Draw X | delete it")
  COLOR (chat)    – ChangeColorFromChatTask (multi-turn: "Draw X | change it to red")
  COPY  (chat)    – CopyFromChatTask        (multi-turn: "Draw X | copy it to the right")

Loss design
───────────
  • Tool head  : CrossEntropyLoss (same as Phase 1)
  • Cursor head: Focal BCE on the full (H × W) heatmap
      – Target  = Gaussian blob(s) centred on oracle cursor coords
      – Focal weighting suppresses easy-negative gradient
  • Each task carries a ``cursor_loss_weight`` scalar (0.05–1.0) that
    down-weights the cursor BCE for tool-only actions (ERASE, ZOOM, etc.)
  • Total loss per sample = tool_loss + cursor_loss_weight × cursor_bce

Usage
─────
    # Standalone
    python -m cadfire.training.pretrain_semantic --samples 20000 --epochs 20

    # From train.py
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
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_to_index

# ── Supervised task imports ────────────────────────────────────────────────────
from cadfire.tasks.supervised.select import SemanticSelectTask, SemanticMultiSelectTask
from cadfire.tasks.supervised.delete import DeleteObjectTask
from cadfire.tasks.supervised.pan import PanTask
from cadfire.tasks.supervised.zoom import ZoomInTask, ZoomOutTask
from cadfire.tasks.supervised.hatch import HatchObjectTask
from cadfire.tasks.supervised.trace_next import TraceNextPointTask
from cadfire.tasks.supervised.copy_paste import CopyObjectTask
from cadfire.tasks.supervised.move import MoveObjectTask
from cadfire.tasks.supervised.rotate import RotateObjectTask
from cadfire.tasks.supervised.multiturn import (
    ScaleFromChatTask, MoveFromChatTask, RotateFromChatTask,
    EraseFromChatTask, ChangeColorFromChatTask, CopyFromChatTask,
)
from cadfire.tasks.supervised.transform_extra import (
    ScaleObjectTask, MirrorObjectTask, OffsetTask,
)


# ── Cursor-mask helpers ────────────────────────────────────────────────────────

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
    Build a float32 cursor-mask for one or more pixel positions.
    Each position contributes a Gaussian blob; result is clipped to [0, 1].
    """
    mask = np.zeros((H, W), dtype=np.float32)
    for row, col in centroids_px:
        mask += _gaussian_blob(row, col, H, W, sigma)
    return np.clip(mask, 0.0, 1.0)


def oracle_to_cursor_mask(cursor_world, engine: CADEngine,
                          H: int, W: int, sigma: float) -> np.ndarray:
    """
    Convert an oracle cursor specification to a (H, W) Gaussian mask.

    cursor_world may be:
      – np.ndarray (2,)          → single blob
      – list of np.ndarray (2,)  → multiple blobs (MULTISELECT)
      – None                      → uniform zero mask (tool-only action)
    """
    if cursor_world is None:
        return np.zeros((H, W), dtype=np.float32)

    if isinstance(cursor_world, list):
        pxs = [_world_to_pixel(pt, engine, H, W) for pt in cursor_world]
    else:
        pxs = [_world_to_pixel(cursor_world, engine, H, W)]

    return _make_cursor_mask(pxs, H, W, sigma)


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
        gamma       : focusing parameter
        alpha       : weight for positive class
    """
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none"
    )
    prob = torch.sigmoid(pred_logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


# ── Task registry for Phase 2 ─────────────────────────────────────────────────

# Each entry: (weight, task_class, constructor_kwargs)
# Weight controls sampling probability relative to total.
_TASK_REGISTRY = [
    # ── Original 11 tasks ──────────────────────────────────────────────────
    (2.0, SemanticSelectTask,       {}),
    (2.0, SemanticMultiSelectTask,  {}),
    (1.5, DeleteObjectTask,         {}),
    (1.0, PanTask,                  {}),
    (0.8, ZoomInTask,               {}),
    (0.8, ZoomOutTask,              {}),
    (1.0, HatchObjectTask,          {}),
    (2.5, TraceNextPointTask,       {}),  # highest weight – critical skill
    (1.0, CopyObjectTask,           {}),
    (1.0, MoveObjectTask,           {}),
    (1.0, RotateObjectTask,         {}),
    # ── New single-step transform tasks ────────────────────────────────────
    (1.0, ScaleObjectTask,          {}),  # SCALE with pivot cursor
    (0.8, MirrorObjectTask,         {}),  # MIRROR with axis cursor
    (0.8, OffsetTask,               {}),  # OFFSET with direction cursor
    # ── Multi-turn chat tasks ───────────────────────────────────────────────
    # These teach the model to read conversation history:
    # prompt = "<first turn> | <second turn>"; entity already exists + selected
    (1.2, ScaleFromChatTask,        {}),  # "Draw X | make it smaller/bigger"
    (1.2, MoveFromChatTask,         {}),  # "Draw X | move it right/left/…"
    (1.2, RotateFromChatTask,       {}),  # "Draw X | rotate it N degrees"
    (1.0, EraseFromChatTask,        {}),  # "Draw X | delete it"
    (0.8, ChangeColorFromChatTask,  {}),  # "Draw X | change it to {color}"
    (1.2, CopyFromChatTask,         {}),  # "Draw X | copy it to the right"
]

_WEIGHTS = np.array([w for w, _, _ in _TASK_REGISTRY], dtype=np.float64)
_WEIGHTS /= _WEIGHTS.sum()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SemanticDataset(Dataset):
    """
    Supervised dataset for Phase-2 semantic cursor pretraining.

    Samples are generated on-the-fly from the full supervised task registry.
    Each sample is a single-step (observation, oracle_tool, oracle_cursor).

    Returns a dict with:
        image        : (H, W, C)     float32
        text_ids     : (max_len,)    int32
        state_vec    : (state_dim,)  float32
        tool_id      : ()            int64
        cursor_mask  : (H, W)        float32  Gaussian blob(s) in [0, 1]
        cursor_weight: ()            float32  per-sample cursor loss weight
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        num_samples: int = 20_000,
        sigma: float = 12.0,
        seed: int | None = None,
    ):
        self.config = config or load_config()
        self.num_samples = num_samples
        self.sigma = sigma
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

    def _build_state_vec(self, engine: CADEngine) -> np.ndarray:
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

    def _generate_sample(self, engine: CADEngine, renderer: Renderer) -> Dict:
        """Pick a random supervised task and generate one training sample."""
        task_idx = int(self.rng.choice(len(_TASK_REGISTRY), p=_WEIGHTS))
        _, task_class, kwargs = _TASK_REGISTRY[task_idx]

        seed = int(self.rng.randint(0, 2 ** 31))
        task = task_class(seed=seed, **kwargs)

        engine.reset()
        setup_info = task.setup(engine)

        # Render observation
        image = renderer.render(engine)

        # Inject reference image if the task provides one
        ref = setup_info.get("reference_image")
        if ref is not None and image.shape[2] > 5:
            image[:, :, 3:6] = ref.astype(np.float32) / 255.0

        # Oracle action
        oracle = task.oracle_action(engine, setup_info)
        tool_name = oracle["tool"]
        cursor_world = oracle["cursor_world"]
        cursor_weight = float(oracle.get("cursor_weight", 1.0))

        tool_id = self._tool_idx.get(tool_name, 0)

        cursor_mask = oracle_to_cursor_mask(
            cursor_world, engine, self.H, self.W, self.sigma
        )

        text_ids = np.array(
            self.tokenizer.encode_padded(setup_info["prompt"]), dtype=np.int32
        )
        state_vec = self._build_state_vec(engine)

        return {
            "image":         image,
            "text_ids":      text_ids,
            "state_vec":     state_vec,
            "tool_id":       tool_id,
            "cursor_mask":   cursor_mask,
            "cursor_weight": cursor_weight,
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        engine = CADEngine(self.config)
        renderer = Renderer(self.config)
        sample = self._generate_sample(engine, renderer)

        return {
            "image":         torch.from_numpy(sample["image"]).float(),
            "text_ids":      torch.from_numpy(sample["text_ids"]).long(),
            "state_vec":     torch.from_numpy(sample["state_vec"]).float(),
            "tool_id":       torch.tensor(sample["tool_id"], dtype=torch.long),
            "cursor_mask":   torch.from_numpy(sample["cursor_mask"]).float(),
            "cursor_weight": torch.tensor(sample["cursor_weight"], dtype=torch.float32),
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

    Trains ALL parameters (text encoder, vision encoder, fusion, tool head,
    cursor head).  The text encoder must learn to associate names like
    "hexagon" with the visual shape it sees – freezing it would break this.

    Args:
        agent         : CADAgent instance (modified in-place).
        config        : Config dict (defaults to global config.json).
        num_samples   : Number of generated samples per epoch.
        num_epochs    : Training epochs over the dataset.
        lr            : Adam learning rate.
        batch_size    : Mini-batch size.
        sigma         : Gaussian blob radius (pixels) for cursor targets.
        cursor_weight : Global scale applied to per-sample cursor BCE weight.
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

    # ── Train ALL parameters (text encoder included) ───────────────────────
    for param in agent.parameters():
        param.requires_grad = True

    trainable = [p for p in agent.parameters() if p.requires_grad]
    if verbose:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Semantic pretrain: {n_trainable:,} trainable parameters "
              f"(all modules unfrozen – text encoder trains here)")

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset = SemanticDataset(
        config=config,
        num_samples=num_samples,
        sigma=sigma,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
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
        epoch_tool_loss   = 0.0
        epoch_cursor_loss = 0.0
        epoch_total_loss  = 0.0
        epoch_correct     = 0
        n_batches         = 0

        for batch in loader:
            image        = batch["image"].to(device)          # (B, H, W, C)
            text_ids     = batch["text_ids"].to(device)       # (B, max_len)
            state_vec    = batch["state_vec"].to(device)      # (B, state_dim)
            tool_ids     = batch["tool_id"].to(device)        # (B,)
            cursor_tgt   = batch["cursor_mask"].to(device)    # (B, H, W)
            c_weights    = batch["cursor_weight"].to(device)  # (B,) per-sample

            obs = {
                "image":     image,
                "text_ids":  text_ids,
                "state_vec": state_vec,
            }

            out = agent(obs)
            tool_logits    = out["tool_logits"]    # (B, num_tools)
            cursor_heatmap = out["cursor_heatmap"] # (B, 1, H, W)

            # Tool classification loss
            t_loss = tool_criterion(tool_logits, tool_ids)

            # Per-sample weighted cursor focal BCE
            # Compute per-pixel BCE, then average over spatial dims,
            # then weight by per-sample cursor_weight, then average over batch.
            c_logits = cursor_heatmap.squeeze(1)           # (B, H, W)
            bce_raw  = F.binary_cross_entropy_with_logits(
                c_logits, cursor_tgt, reduction="none"
            )                                              # (B, H, W)
            prob  = torch.sigmoid(c_logits)
            p_t   = prob * cursor_tgt + (1.0 - prob) * (1.0 - cursor_tgt)
            a_t   = focal_alpha * cursor_tgt + (1.0 - focal_alpha) * (1.0 - cursor_tgt)
            focal = a_t * (1.0 - p_t) ** focal_gamma * bce_raw
            c_loss_per_sample = focal.mean(dim=(-2, -1))  # (B,)
            c_loss = (c_weights * c_loss_per_sample).mean()

            loss = t_loss + cursor_weight * c_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            B = tool_ids.size(0)
            epoch_tool_loss   += t_loss.item() * B
            epoch_cursor_loss += c_loss.item() * B
            epoch_total_loss  += loss.item()   * B
            epoch_correct     += (tool_logits.argmax(dim=-1) == tool_ids).sum().item()
            n_batches         += B

        n = max(n_batches, 1)
        avg_tool   = epoch_tool_loss   / n
        avg_cursor = epoch_cursor_loss / n
        avg_total  = epoch_total_loss  / n
        avg_acc    = epoch_correct     / n

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

    # All parameters remain trainable for Phase 3 / PPO
    return history


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    import argparse
    from cadfire.training.checkpoint import CheckpointManager

    parser = argparse.ArgumentParser(
        description="Phase-2 semantic cursor pretraining"
    )
    parser.add_argument("--samples",       type=int,   default=20_000)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--batch",         type=int,   default=32)
    parser.add_argument("--sigma",         type=float, default=12.0)
    parser.add_argument("--cursor-weight", type=float, default=1.0)
    parser.add_argument("--workers",       type=int,   default=0)
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--load",          type=str,   default=None)
    parser.add_argument("--save",          type=str,   default=None)
    parser.add_argument("--seed",          type=int,   default=None)
    args = parser.parse_args()

    config = load_config()
    agent  = CADAgent(config)
    dev    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.load:
        ckpt = CheckpointManager(args.load)
        ckpt.load(agent, optimizer=None, device=dev)
        print(f"Loaded checkpoint from {args.load}")

    print("=" * 60)
    print("Phase 2 – Semantic Cursor Pretraining (all params unfrozen)")
    print(f"  Samples/epoch : {args.samples:,}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch}")
    print(f"  LR            : {args.lr}")
    print(f"  Sigma (px)    : {args.sigma}")
    print(f"  Tasks         : {len(_TASK_REGISTRY)} supervised task types")
    print("=" * 60)

    history = pretrain_semantic_cursor(
        agent, config,
        num_samples=args.samples,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        sigma=args.sigma,
        cursor_weight=args.cursor_weight,
        num_workers=args.workers,
        device=dev,
        seed=args.seed,
    )

    print(f"\nFinal tool accuracy : {history['tool_accuracies'][-1]:.3f}")
    print(f"Final cursor loss   : {history['cursor_losses'][-1]:.4f}")

    if args.save:
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
