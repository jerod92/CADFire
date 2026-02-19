"""
Diagnostic GIF generator for the CADFire training pipeline.

Generates animated GIFs after Phase-3 teacher-forced pretraining to visualise
how well the agent can trace an arbitrary polygon.

Two modes
─────────
1. ``oracle_rollout`` – teacher forcing: oracle actions advance the environment,
   but the agent's cursor heatmap is overlaid so we can see what it attends to
   at each step (useful during training to diagnose whether the cursor head is
   learning even if tool prediction is not perfect).

2. ``free_rollout``   – autonomous rollout: agent's own predictions drive the
   environment.  Ideal for evaluating end-to-end polygon tracing.

Output layout (per frame)
─────────────────────────
┌──────────────────────┬──────────────────────┐
│  Viewport + cursor   │  Reference image     │
│  heatmap overlay     │  (target polygon)    │
└──────────────────────┴──────────────────────┘
  Caption: step N │ tool: POLYLINE │ action_acc: 1.0

GIF is saved to ``output_dir/polygon_trace_<mode>_<seed>.gif``.

Usage
─────
    # After Phase-3 training:
    from cadfire.training.diagnostics import generate_diagnostic_gifs
    generate_diagnostic_gifs(
        agent, config,
        output_dir="diagnostics/",
        n_episodes=6,
        device="cuda",
    )

    # From CLI:
    python -m cadfire.training.diagnostics --checkpoint checkpoints_1/ --episodes 6
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.model.cad_agent import CADAgent
from cadfire.renderer.rasterizer import Renderer
from cadfire.tasks.supervised.polygon_trace import (
    PolygonTraceTask, _render_target_reference, _make_polygon_vertices,
)
from cadfire.tasks.supervised.trace_next import _TRACE_PROMPTS
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_to_index, index_to_tool


# ── Colour helpers ────────────────────────────────────────────────────────────

def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """Convert (H, W) float [0,1] heatmap to (H, W, 3) uint8 jet colormap."""
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray,
                      alpha: float = 0.45) -> np.ndarray:
    """Overlay jet heatmap on RGB frame with given alpha blend."""
    hmap_rgb = _heatmap_to_rgb(heatmap).astype(np.float32)
    out = rgb.astype(np.float32) * (1.0 - alpha) + hmap_rgb * alpha
    return out.clip(0, 255).astype(np.uint8)


def _draw_cursor_dot(rgb: np.ndarray, row: int, col: int,
                     color=(0, 255, 0), radius: int = 4) -> np.ndarray:
    """Draw a small filled circle at (row, col) in the given color."""
    H, W = rgb.shape[:2]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                r, c = row + dr, col + dc
                if 0 <= r < H and 0 <= c < W:
                    rgb[r, c] = color
    return rgb


def _draw_crosshair(rgb: np.ndarray, row: int, col: int,
                    color=(255, 200, 0), size: int = 8, thickness: int = 1) -> np.ndarray:
    """Draw a crosshair marker at (row, col)."""
    H, W = rgb.shape[:2]
    for d in range(-size, size + 1):
        r, c = row + d, col
        if 0 <= r < H and 0 <= c < W:
            rgb[r, c] = color
        r, c = row, col + d
        if 0 <= r < H and 0 <= c < W:
            rgb[r, c] = color
    return rgb


def _make_text_bar(text: str, W: int, bar_h: int = 18,
                   bg=(30, 30, 30), fg=(230, 230, 230)) -> np.ndarray:
    """Create a simple text-bar image using block pixels (no font rendering dep)."""
    bar = np.full((bar_h, W, 3), bg, dtype=np.uint8)
    # Simple 5-wide pixel font for digits/letters is complex; instead
    # embed text as semi-transparent caption using ASCII-art blobs.
    # Fallback: just coloured bar with no text rendering dependency.
    _ = text  # text is unused for rendering but kept for caller documentation
    return bar


def _stack_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Side-by-side concatenation of two (H, W, 3) uint8 arrays."""
    assert left.shape[0] == right.shape[0], "height mismatch"
    return np.concatenate([left, right], axis=1)


def _world_to_pixel(world_xy: np.ndarray, engine: CADEngine,
                    H: int, W: int) -> Tuple[int, int]:
    ndc = engine.viewport.world_to_ndc(world_xy.reshape(1, 2))[0]
    col = int(np.clip(ndc[0] * W, 0, W - 1))
    row = int(np.clip((1.0 - ndc[1]) * H, 0, H - 1))
    return row, col


# ── GIF writer (pure numpy, no external dep if imageio unavailable) ───────────

def _save_gif(frames: List[np.ndarray], path: Path, fps: float = 2.0):
    """
    Save ``frames`` (list of H×W×3 uint8 arrays) as an animated GIF.

    Tries imageio first (common in scientific Python), then PIL/Pillow.
    If neither is available, saves individual PNG frames instead.
    """
    duration_ms = int(1000 / fps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio
        imageio.mimwrite(str(path), frames, format="GIF",
                         duration=duration_ms / 1000.0, loop=0)
        return
    except ImportError:
        pass

    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            str(path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return
    except ImportError:
        pass

    # Fallback: save individual PNGs (works with numpy only via npy files)
    stem = path.stem
    png_dir = path.parent / stem
    png_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(str(png_dir / f"frame_{i:04d}.png"))
    except ImportError:
        # Absolute last resort: save as .npy
        np.save(str(png_dir / "frames.npy"), np.stack(frames))
    print(f"  [diagnostics] Note: GIF deps unavailable – "
          f"frames saved to {png_dir}/")


# ── Agent inference helpers ───────────────────────────────────────────────────

import torch


def _agent_predict(agent, obs_np: Dict, device: str) -> Dict:
    """Run agent forward pass and return numpy predictions."""
    import torch
    agent.eval()
    with torch.no_grad():
        image     = torch.from_numpy(obs_np["image"]).float().unsqueeze(0).to(device)
        text_ids  = torch.from_numpy(obs_np["text_ids"]).long().unsqueeze(0).to(device)
        state_vec = torch.from_numpy(obs_np["state_vec"]).float().unsqueeze(0).to(device)
        obs = {"image": image, "text_ids": text_ids, "state_vec": state_vec}
        out = agent(obs)

    tool_logits    = out["tool_logits"].squeeze(0).cpu().numpy()   # (num_tools,)
    cursor_heatmap = out["cursor_heatmap"].squeeze().cpu().numpy()  # (H, W)
    cursor_sigmoid = 1.0 / (1.0 + np.exp(-cursor_heatmap))

    pred_tool_id = int(np.argmax(tool_logits))
    pred_cursor_px = np.unravel_index(np.argmax(cursor_sigmoid), cursor_sigmoid.shape)
    # pred_cursor_px is (row, col)

    return {
        "tool_id":       pred_tool_id,
        "cursor_px":     pred_cursor_px,       # (row, col) in pixel space
        "cursor_heatmap": cursor_sigmoid,
    }


def _state_vec_from_engine(engine: CADEngine, tool_idx: Dict,
                            state_dim: int, config: Dict) -> np.ndarray:
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


# ── Oracle-rollout GIF ────────────────────────────────────────────────────────

def oracle_rollout_gif(
    agent: CADAgent,
    config: Dict,
    seed: int,
    device: str,
    output_path: Path,
    fps: float = 1.5,
) -> Dict:
    """
    Teacher-forced rollout GIF.

    Oracle actions drive the environment.  Agent cursor heatmap is overlaid
    to show where the model attends at each step.  Both the oracle cursor
    position (crosshair) and agent's predicted position (dot) are marked.

    Returns per-step accuracy metrics.
    """
    canvas = config["canvas"]
    H, W = canvas["render_height"], canvas["render_width"]
    state_dim = config["model"]["state_dim"]
    max_len   = config["model"]["text_max_len"]
    tool_idx  = tool_to_index()
    idx_tool  = index_to_tool()

    engine   = CADEngine(config)
    renderer = Renderer(config)
    tokenizer = BPETokenizer(
        vocab_size=config["model"]["text_vocab_size"],
        max_len=max_len,
    )

    task = PolygonTraceTask(seed=seed, config=config)
    trajectory = task.generate_trajectory(engine, renderer, tokenizer)

    n_verts = len(trajectory) - 1  # last step is CONFIRM
    prompt_text = _TRACE_PROMPTS[seed % len(_TRACE_PROMPTS)]
    text_ids = np.array(tokenizer.encode_padded(prompt_text), dtype=np.int32)

    # Rebuild the polygon vertices from the first trajectory step
    verts = trajectory[0]["vertices"]
    ref_image = _render_target_reference(verts, H, W, engine)

    frames = []
    step_accs = []

    for t_step, traj_step in enumerate(trajectory):
        obs_np = traj_step["obs"]
        oracle_tool_id = traj_step["tool_id"]
        oracle_cursor  = traj_step["cursor_world"]  # world space

        # Get agent prediction
        pred = _agent_predict(agent, obs_np, device)
        pred_tool_id = pred["tool_id"]
        pred_cursor_px = pred["cursor_px"]
        heatmap = pred["cursor_heatmap"]

        # Accuracy
        tool_correct = int(pred_tool_id == oracle_tool_id)
        step_accs.append(tool_correct)

        # Build RGB viewport frame (channels 0-2)
        obs_img = obs_np["image"]
        viewport_rgb = (obs_img[:, :, :3] * 255).astype(np.uint8)

        # Overlay heatmap
        frame_left = _overlay_heatmap(viewport_rgb, heatmap, alpha=0.40)

        # Draw oracle cursor (gold crosshair)
        oracle_px = _world_to_pixel(oracle_cursor, engine, H, W)
        frame_left = _draw_crosshair(frame_left, oracle_px[0], oracle_px[1],
                                     color=(255, 200, 0), size=7)

        # Draw predicted cursor (green dot if correct, red if wrong)
        dot_color = (0, 255, 80) if tool_correct else (255, 60, 60)
        frame_left = _draw_cursor_dot(frame_left, pred_cursor_px[0],
                                       pred_cursor_px[1], color=dot_color)

        # Reference image (right panel)
        frame_right = ref_image.copy()

        # Mark already-traced vertices on reference
        for k in range(t_step):
            r, c = _world_to_pixel(verts[k], engine, H, W)
            frame_right = _draw_cursor_dot(frame_right, r, c,
                                           color=(100, 255, 100), radius=3)
        if t_step < len(verts):
            r, c = _world_to_pixel(verts[t_step], engine, H, W)
            frame_right = _draw_crosshair(frame_right, r, c,
                                          color=(255, 200, 0), size=6)

        # Text bar
        oracle_name = idx_tool.get(oracle_tool_id, "?")
        pred_name   = idx_tool.get(pred_tool_id, "?")
        bar_text = (f"step {t_step} | oracle:{oracle_name} "
                    f"pred:{pred_name} | {'OK' if tool_correct else 'MISS'}")
        bar = _make_text_bar(bar_text, W * 2, bar_h=20)

        combined = np.concatenate([
            _stack_frames(frame_left, frame_right),
            bar,
        ], axis=0)
        frames.append(combined)

    _save_gif(frames, output_path, fps=fps)
    overall_acc = float(np.mean(step_accs)) if step_accs else 0.0
    return {"tool_accuracy": overall_acc, "n_steps": len(trajectory), "seed": seed}


# ── Free-rollout GIF ─────────────────────────────────────────────────────────

def free_rollout_gif(
    agent: CADAgent,
    config: Dict,
    seed: int,
    device: str,
    output_path: Path,
    max_steps: int = 15,
    fps: float = 1.5,
) -> Dict:
    """
    Autonomous (no teacher forcing) rollout GIF.

    The agent drives the environment with its own predicted actions.
    We compare against the oracle trajectory to compute accuracy.

    Returns per-step accuracy and whether the polygon was completed.
    """
    import torch
    from cadfire.engine.geometry import PolylineEntity

    canvas = config["canvas"]
    H, W = canvas["render_height"], canvas["render_width"]
    state_dim = config["model"]["state_dim"]
    max_len   = config["model"]["text_max_len"]
    tool_idx  = tool_to_index()
    idx_tool  = index_to_tool()
    polyline_id = tool_idx.get("POLYLINE", 0)
    confirm_id  = tool_idx.get("CONFIRM",  1)

    engine    = CADEngine(config)
    renderer  = Renderer(config)
    tokenizer = BPETokenizer(
        vocab_size=config["model"]["text_vocab_size"],
        max_len=max_len,
    )

    # Generate oracle trajectory to get polygon shape
    ref_engine = CADEngine(config)
    ref_task = PolygonTraceTask(seed=seed, config=config)
    oracle_traj = ref_task.generate_trajectory(ref_engine, renderer, tokenizer)
    verts     = oracle_traj[0]["vertices"]
    n_verts   = len(verts)
    ref_image = _render_target_reference(verts, H, W, engine)

    prompt_text = _TRACE_PROMPTS[seed % len(_TRACE_PROMPTS)]
    text_ids = np.array(tokenizer.encode_padded(prompt_text), dtype=np.int32)

    # Reset engine for free rollout
    engine.reset()
    engine.active_tool = "POLYLINE"

    frames = []
    step_accs = []
    completed = False

    for step in range(max_steps):
        # Build current observation
        image = renderer.render(engine)
        if image.shape[2] > 5:
            image[:, :, 3:6] = ref_image.astype(np.float32) / 255.0
        state_vec = _state_vec_from_engine(engine, tool_idx, state_dim, config)
        obs_np = {"image": image, "text_ids": text_ids, "state_vec": state_vec}

        # Oracle reference for this step
        oracle_step = oracle_traj[min(step, len(oracle_traj) - 1)]
        oracle_tool_id = oracle_step["tool_id"]
        oracle_cursor  = oracle_step["cursor_world"]

        # Agent prediction
        pred = _agent_predict(agent, obs_np, device)
        pred_tool_id   = pred["tool_id"]
        pred_cursor_px = pred["cursor_px"]   # (row, col)
        heatmap        = pred["cursor_heatmap"]

        tool_correct = int(pred_tool_id == oracle_tool_id)
        step_accs.append(tool_correct)

        # Viewport RGB + heatmap overlay
        viewport_rgb = (image[:, :, :3] * 255).astype(np.uint8)
        frame_left = _overlay_heatmap(viewport_rgb, heatmap, alpha=0.40)

        # Oracle cursor (gold crosshair)
        oracle_px = _world_to_pixel(oracle_cursor, engine, H, W)
        frame_left = _draw_crosshair(frame_left, oracle_px[0], oracle_px[1],
                                     color=(255, 200, 0), size=7)
        # Agent cursor (green or red)
        dot_color = (0, 255, 80) if tool_correct else (255, 60, 60)
        frame_left = _draw_cursor_dot(frame_left, pred_cursor_px[0],
                                       pred_cursor_px[1], color=dot_color)

        # Reference panel
        frame_right = ref_image.copy()
        for k in range(step):
            if k < n_verts:
                r, c = _world_to_pixel(verts[k], engine, H, W)
                frame_right = _draw_cursor_dot(frame_right, r, c,
                                               color=(100, 255, 100), radius=3)

        bar_text = (f"step {step} | oracle:{idx_tool.get(oracle_tool_id, '?')} "
                    f"pred:{idx_tool.get(pred_tool_id, '?')} | "
                    f"{'OK' if tool_correct else 'MISS'}")
        bar = _make_text_bar(bar_text, W * 2, bar_h=20)
        combined = np.concatenate([
            _stack_frames(frame_left, frame_right), bar
        ], axis=0)
        frames.append(combined)

        # Apply AGENT's predicted action to advance state
        if pred_tool_id == confirm_id:
            # Agent chose to confirm – commit polygon
            engine.entities = [
                e for e in engine.entities
                if not getattr(e, '_partial_trace', False)
            ]
            if len(engine.pending_points) >= 2:
                full = PolylineEntity(
                    points=np.array(engine.pending_points), closed=True,
                    color_index=2,
                )
                engine.add_entity(full, save_undo=False)
            engine.pending_points.clear()
            engine.active_tool = "NOOP"
            completed = True
            break
        elif pred_tool_id == polyline_id:
            # Convert predicted pixel cursor to world coords
            ndc_col = pred_cursor_px[1] / max(W - 1, 1)
            ndc_row = 1.0 - pred_cursor_px[0] / max(H - 1, 1)
            ndc = np.array([[ndc_col, ndc_row]])
            world_pt = engine.viewport.ndc_to_world(ndc)[0]
            engine.pending_points.append(world_pt)

            # Update partial polyline entity
            engine.entities = [
                e for e in engine.entities
                if not getattr(e, '_partial_trace', False)
            ]
            if len(engine.pending_points) >= 2:
                partial = PolylineEntity(
                    points=np.array(engine.pending_points), closed=False,
                    color_index=2,
                )
                partial._partial_trace = True  # type: ignore[attr-defined]
                engine.entities.append(partial)
        else:
            # Unexpected tool – stop
            break

    _save_gif(frames, output_path, fps=fps)
    return {
        "tool_accuracy": float(np.mean(step_accs)) if step_accs else 0.0,
        "n_steps":       len(frames),
        "completed":     completed,
        "seed":          seed,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def generate_diagnostic_gifs(
    agent: CADAgent,
    config: Dict | None = None,
    output_dir: str = "diagnostics",
    n_episodes: int = 6,
    device: str | None = None,
    fps: float = 1.5,
    seeds: List[int] | None = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate diagnostic GIFs after Phase-3 teacher-forced pretraining.

    For each episode, produces TWO GIFs:
      • ``oracle_ep<N>.gif`` – oracle-driven with agent attention overlaid
      • ``free_ep<N>.gif``   – fully autonomous agent rollout

    Args:
        agent      : Trained CADAgent.
        config     : Config dict.
        output_dir : Directory to write GIFs (created if missing).
        n_episodes : Number of polygon episodes to render.
        device     : Torch device.
        fps        : GIF frames per second.
        seeds      : Optional list of per-episode RNG seeds.
        verbose    : Print per-episode stats.

    Returns:
        List of dicts with per-episode metrics.
    """
    import torch
    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = list(range(n_episodes))
    else:
        seeds = seeds[:n_episodes]
    if len(seeds) < n_episodes:
        seeds += list(range(len(seeds), n_episodes))

    all_metrics = []

    for ep, seed in enumerate(seeds):
        if verbose:
            print(f"  [diag] Episode {ep + 1}/{n_episodes}  (seed={seed})")

        oracle_path = out / f"oracle_ep{ep:02d}.gif"
        free_path   = out / f"free_ep{ep:02d}.gif"

        m_oracle = oracle_rollout_gif(
            agent, config, seed=seed, device=device,
            output_path=oracle_path, fps=fps,
        )
        m_free = free_rollout_gif(
            agent, config, seed=seed, device=device,
            output_path=free_path, fps=fps,
        )

        metrics = {
            "episode":           ep,
            "seed":              seed,
            "oracle_tool_acc":   m_oracle["tool_accuracy"],
            "oracle_n_steps":    m_oracle["n_steps"],
            "free_tool_acc":     m_free["tool_accuracy"],
            "free_n_steps":      m_free["n_steps"],
            "free_completed":    m_free["completed"],
        }
        all_metrics.append(metrics)

        if verbose:
            print(
                f"    oracle acc={m_oracle['tool_accuracy']:.3f} "
                f"| free acc={m_free['tool_accuracy']:.3f} "
                f"completed={m_free['completed']} "
                f"steps={m_free['n_steps']}"
            )

    # Summary
    if verbose and all_metrics:
        oracle_acc = np.mean([m["oracle_tool_acc"] for m in all_metrics])
        free_acc   = np.mean([m["free_tool_acc"]   for m in all_metrics])
        completed  = np.mean([float(m["free_completed"]) for m in all_metrics])
        print(f"\n  [diag] SUMMARY  oracle_acc={oracle_acc:.3f} | "
              f"free_acc={free_acc:.3f} | "
              f"completed={completed:.0%} ({int(completed * n_episodes)}/{n_episodes})")
        print(f"  [diag] GIFs written to: {out.resolve()}/")

    return all_metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import torch
    from cadfire.training.checkpoint import CheckpointManager

    parser = argparse.ArgumentParser(
        description="Generate diagnostic GIFs for polygon tracing"
    )
    parser.add_argument("--checkpoint",  type=str, default="checkpoints_1",
                        help="Checkpoint directory to load agent from")
    parser.add_argument("--output-dir", type=str, default="diagnostics",
                        help="Output directory for GIFs")
    parser.add_argument("--episodes",   type=int, default=6,
                        help="Number of episodes to render")
    parser.add_argument("--fps",        type=float, default=1.5,
                        help="GIF frames per second")
    parser.add_argument("--device",     type=str, default=None,
                        help="torch device: cuda / cpu / None (auto)")
    args = parser.parse_args()

    config = load_config()
    agent  = CADAgent(config)
    dev    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = CheckpointManager(args.checkpoint, config)
    meta = ckpt.load(agent, optimizer=None, device=dev)
    print(f"Loaded checkpoint (step {meta.get('step', 0)}) from {args.checkpoint}/")
    print(f"Generating {args.episodes} diagnostic GIF episodes → {args.output_dir}/")

    metrics = generate_diagnostic_gifs(
        agent, config,
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        device=dev,
        fps=args.fps,
        verbose=True,
    )


if __name__ == "__main__":
    main()
