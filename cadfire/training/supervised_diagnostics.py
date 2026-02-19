"""
Supervised learning task diagnostics.

Generates rich PNG visualisations of supervised task samples, exposing the
full observation space the agent sees and the target outputs it must predict.
Covers all three pretraining phases.

Observation panels shown per sample (left → right)
───────────────────────────────────────────────────
1. Viewport RGB        (image ch 0-2)   – drawing with ghosts & selection highlights
2. Reference / Raster  (image ch 3-5)   – trace reference (black if unused)
3. Selection Mask      (image ch 14)    – jet-coloured where entities are selected
4. Layer Composite     (image ch 6-13)  – all layer masks in distinct colours
5. Oracle Cursor       (target heatmap) – Gaussian blob(s) the agent must predict

State / Prompt panel (right-most column)
─────────────────────────────────────────
• Full text prompt decoded from BPE token ids (includes past conversation history
  for multi-turn tasks – e.g. "Draw a circle | make it smaller")
• Active tool, zoom, viewport centre, active layer, active colour
• Entity count, selection count, pending cursor points (cursor history)
• Target: oracle tool name, cursor weight, peak pixel, cursor type

Output structure
────────────────
{output_dir}/
  phase1/
    tool_prompts.png               – text grid: tool → all prompt variants
  phase2/
    {TaskClassName}.png            – n_per_task samples stacked vertically
  phase3/
    trajectory_{i:02d}.png         – all steps of a trajectory stacked vertically

Usage
─────
  # After any pretraining phase:
  from cadfire.training.supervised_diagnostics import generate_supervised_diagnostics
  generate_supervised_diagnostics(agent=agent, config=config, output_dir="sup_diag/")

  # CLI:
  python -m cadfire.training.supervised_diagnostics --checkpoint checkpoints_1/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.renderer.rasterizer import Renderer
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_to_index, index_to_tool


# ── Colour constants ───────────────────────────────────────────────────────────

# Distinct per-layer colours (8 layers)
_LAYER_COLORS: List[Tuple[int, int, int]] = [
    (255, 100, 100),  # red
    (100, 255, 100),  # green
    (100, 100, 255),  # blue
    (255, 255, 100),  # yellow
    (255, 100, 255),  # magenta
    (100, 255, 255),  # cyan
    (255, 180, 100),  # orange
    (180, 100, 255),  # purple
]


# ── Image helpers ──────────────────────────────────────────────────────────────

def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """Jet colormap: (H, W) float32 [0,1] → (H, W, 3) uint8."""
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _layer_composite_rgb(image: np.ndarray, n_layers: int = 8) -> np.ndarray:
    """
    Composite layer mask channels into a single coloured RGB image.

    Channels 6..6+n_layers-1 are float32 binary masks.
    Each layer gets a distinct colour; overlaps are additive-clipped.
    """
    H, W = image.shape[:2]
    composite = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(min(n_layers, image.shape[2] - 6)):
        mask = image[:, :, 6 + i]
        color = np.array(_LAYER_COLORS[i % len(_LAYER_COLORS)], dtype=np.float32) / 255.0
        composite += mask[:, :, None] * color
    return np.clip(composite * 255, 0, 255).astype(np.uint8)


def _resize_nearest(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Pure-numpy nearest-neighbour resize. Works for (H,W) and (H,W,3)."""
    H, W = arr.shape[:2]
    row_idx = (np.arange(new_h) * H / new_h).astype(int).clip(0, H - 1)
    col_idx = (np.arange(new_w) * W / new_w).astype(int).clip(0, W - 1)
    if arr.ndim == 3:
        return arr[np.ix_(row_idx, col_idx)]
    return arr[np.ix_(row_idx, col_idx)]


def _draw_cross(rgb: np.ndarray, row: int, col: int,
                color: Tuple[int, int, int] = (255, 255, 255),
                size: int = 5) -> np.ndarray:
    """Draw a crosshair marker at (row, col) on an RGB image (in-place copy)."""
    H, W = rgb.shape[:2]
    out = rgb.copy()
    for d in range(-size, size + 1):
        if 0 <= row + d < H:
            out[row + d, col] = color
        if 0 <= col + d < W:
            out[row, col + d] = color
    return out


def _draw_dot(rgb: np.ndarray, row: int, col: int,
              color: Tuple[int, int, int] = (255, 200, 0),
              radius: int = 3) -> np.ndarray:
    """Draw a small filled circle at (row, col)."""
    H, W = rgb.shape[:2]
    out = rgb.copy()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                r, c = row + dr, col + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r, c] = color
    return out


# ── Text helpers ───────────────────────────────────────────────────────────────

def _make_label_bar(text: str, W: int, bar_h: int = 18,
                    bg: Tuple = (30, 30, 30),
                    fg: Tuple = (220, 220, 0)) -> np.ndarray:
    """Render a narrow text label bar. Uses PIL if available, else solid bar."""
    bar = np.full((bar_h, W, 3), bg, dtype=np.uint8)
    try:
        from PIL import Image as _I, ImageDraw as _D, ImageFont as _F
        pil = _I.fromarray(bar)
        draw = _D.Draw(pil)
        try:
            font = _F.truetype("arial.ttf", 11)
        except (OSError, IOError):
            font = _F.load_default()
        draw.text((3, 2), text, fill=fg, font=font)
        bar = np.array(pil)
    except ImportError:
        pass
    return bar


def _make_text_panel(
    lines: List[Tuple[str, Tuple[int, int, int]]],
    W: int,
    H: int,
) -> np.ndarray:
    """
    Render a text panel (H, W, 3).

    lines: list of (text, rgb_color) tuples drawn top-to-bottom.
    Uses PIL if available; returns a blank dark panel if not installed.
    """
    panel = np.full((H, W, 3), (20, 20, 30), dtype=np.uint8)
    try:
        from PIL import Image as _I, ImageDraw as _D, ImageFont as _F
        pil = _I.fromarray(panel)
        draw = _D.Draw(pil)
        try:
            font = _F.truetype("arial.ttf", 11)
        except (OSError, IOError):
            font = _F.load_default()
        y = 4
        for text, color in lines:
            if y + 14 > H:
                break
            draw.text((4, y), str(text)[:70], fill=color, font=font)
            y += 14
        panel = np.array(pil)
    except ImportError:
        pass
    return panel


# ── State vector helpers ───────────────────────────────────────────────────────

def _build_state_vec(engine: CADEngine, tool_idx: Dict[str, int],
                     config: Dict) -> np.ndarray:
    """Build the normalised state vector from a live engine instance."""
    state_dim = config["model"]["state_dim"]
    canvas = config["canvas"]
    num_tools = max(len(tool_idx), 1)
    vec = np.zeros(state_dim, dtype=np.float32)
    vec[0] = tool_idx.get(engine.active_tool, 0) / num_tools
    vec[1] = np.log1p(engine.viewport.zoom) / 5.0
    vec[2] = engine.viewport.center[0] / canvas["world_width"]
    vec[3] = engine.viewport.center[1] / canvas["world_height"]
    vec[4] = engine.active_layer / max(len(engine.layers), 1)
    vec[5] = engine.active_color / 8.0
    vec[6] = min(len(engine.entities), 100) / 100.0
    vec[7] = min(len(engine.selected_ids), 50) / 50.0
    vec[8] = min(len(engine.pending_points), 10) / 10.0
    return vec


def _decode_state_vec(state_vec: np.ndarray, config: Dict) -> Dict[str, str]:
    """Decode a normalised state vector into human-readable field values."""
    from cadfire.utils.config import tool_list as _tl
    tools = _tl()
    num_tools = max(len(tools), 1)
    n_layers = config.get("layers", {}).get("max_layers", 8)

    tidx = int(round(float(state_vec[0]) * num_tools))
    tidx = max(0, min(tidx, num_tools - 1))
    tool_name = tools[tidx] if tools else "?"

    zoom = float(np.expm1(float(state_vec[1]) * 5.0))
    vp_x = float(state_vec[2]) * config["canvas"]["world_width"]
    vp_y = float(state_vec[3]) * config["canvas"]["world_height"]
    layer = int(round(float(state_vec[4]) * n_layers))
    color = int(round(float(state_vec[5]) * 8.0))
    entities = int(round(float(state_vec[6]) * 100))
    selected = int(round(float(state_vec[7]) * 50))
    pending = int(round(float(state_vec[8]) * 10))

    return {
        "active_tool": tool_name,
        "zoom": f"{zoom:.2f}x",
        "viewport": f"({vp_x:.0f},{vp_y:.0f})",
        "layer": str(layer),
        "color_idx": str(color),
        "entities": str(entities),
        "selected": str(selected),
        "pending_pts": str(pending),
    }


def _decode_prompt(text_ids: np.ndarray, tokenizer: BPETokenizer) -> str:
    """Decode BPE token ids back to the original prompt text."""
    try:
        ids = [int(i) for i in text_ids if int(i) != 0]
        return tokenizer.decode(ids)
    except Exception:
        return "<decode error>"


# ── Frame builder ──────────────────────────────────────────────────────────────

def _build_sample_frame(
    image: np.ndarray,
    cursor_mask: np.ndarray,
    text_ids: np.ndarray,
    state_vec: np.ndarray,
    oracle_tool: str,
    cursor_weight: float,
    task_name: str,
    tokenizer: BPETokenizer,
    config: Dict,
    panel_size: int = 128,
    step_idx: Optional[int] = None,
    n_steps: Optional[int] = None,
) -> np.ndarray:
    """
    Build one diagnostic frame for a supervised training sample.

    Panels (left → right):
      Viewport RGB | Reference/Raster | Selection Mask | Layer Composite
      | Oracle Cursor | Text (prompt + state + target)

    Returns (H_frame, W_frame, 3) uint8 RGB.
    """
    n_layers = config.get("layers", {}).get("max_layers", 8)
    sel_ch = 6 + n_layers  # channel index for the selection mask

    # ── 1. Viewport RGB (ch 0-2): drawing with ghosts & selection highlights ──
    viewport_rgb = (image[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

    # ── 2. Reference / Raster (ch 3-5): trace reference image ─────────────────
    if image.shape[2] > 5:
        raster_raw = image[:, :, 3:6]
        raster_rgb = (raster_raw * 255).clip(0, 255).astype(np.uint8)
    else:
        raster_rgb = np.zeros_like(viewport_rgb)

    # ── 3. Selection Mask (ch 6+n_layers): jet-coloured ───────────────────────
    if image.shape[2] > sel_ch:
        sel_rgb = _heatmap_to_rgb(image[:, :, sel_ch])
    else:
        sel_rgb = np.zeros_like(viewport_rgb)

    # ── 4. Layer Composite (ch 6..6+n_layers-1): distinct colours per layer ───
    layer_rgb = _layer_composite_rgb(image, n_layers)

    # ── 5. Oracle cursor heatmap: what the agent must predict ──────────────────
    cursor_rgb = _heatmap_to_rgb(cursor_mask)
    peak_px_str = "none"
    if cursor_mask.max() > 1e-6:
        peak = np.unravel_index(int(np.argmax(cursor_mask)), cursor_mask.shape)
        cursor_rgb = _draw_cross(cursor_rgb, peak[0], peak[1],
                                 color=(255, 255, 255), size=5)
        peak_px_str = f"({peak[1]},{peak[0]})"  # (col, row) = (x, y)

    cursor_type = "multi-blob" if oracle_tool == "MULTISELECT" else "single-point"

    # ── Resize each image panel to panel_size ──────────────────────────────────
    # Determine raster label based on content
    raster_has_content = image.shape[2] > 5 and image[:, :, 3:6].max() > 0.01
    raster_label = "Raster" if raster_has_content else "Raster (none)"

    panels: List[np.ndarray] = []
    for arr, label in [
        (viewport_rgb, "Viewport RGB"),
        (raster_rgb,   raster_label),
        (sel_rgb,      "Selection"),
        (layer_rgb,    "Layers"),
        (cursor_rgb,   f"Oracle: {oracle_tool}"),
    ]:
        resized = _resize_nearest(arr, panel_size, panel_size)
        bar = _make_label_bar(label, panel_size, bar_h=16)
        panels.append(np.concatenate([bar, resized], axis=0))

    # ── Text panel: prompt (text history) + state vector + target ─────────────
    prompt = _decode_prompt(text_ids, tokenizer)
    state = _decode_state_vec(state_vec, config)

    step_header = ""
    if step_idx is not None and n_steps is not None:
        step_header = f"Step {step_idx + 1}/{n_steps}"

    # Text lines: (content, colour)
    W_TEXT = 220
    Y = (220, 220, 220)   # white
    YL = (255, 220, 80)   # yellow
    CY = (100, 200, 255)  # cyan
    OR = (255, 160, 80)   # orange
    GN = (100, 255, 150)  # green

    # Multi-turn tasks encode history as "Turn1 | Turn2" - show on two lines
    if " | " in prompt:
        parts = prompt.split(" | ", 1)
        p_lines = [(f"  {parts[0]}", Y), (f"  | {parts[1]}", Y)]
    else:
        p_lines = [(f"  {prompt[:55]}", Y)]
        if len(prompt) > 55:
            p_lines.append((f"  {prompt[55:110]}", Y))

    lines: List[Tuple[str, Tuple]] = []
    if step_header:
        lines += [(step_header, GN), ("", Y)]
    lines += (
        [("PROMPT (text history):", YL)]
        + p_lines
        + [
            ("", Y),
            ("-- OBSERVATION --", CY),
            (f"  Active tool:  {state['active_tool']}", Y),
            (f"  Zoom:         {state['zoom']}", Y),
            (f"  Viewport ctr: {state['viewport']}", Y),
            (f"  Layer:        {state['layer']}", Y),
            (f"  Color idx:    {state['color_idx']}", Y),
            (f"  Entities:     {state['entities']}", Y),
            (f"  Selected:     {state['selected']}", Y),
            (f"  Pending pts:  {state['pending_pts']}  (cursor history)", Y),
            ("", Y),
            ("-- TARGET --", CY),
            (f"  Tool:         {oracle_tool}", OR),
            (f"  Cursor wt:    {cursor_weight:.2f}", Y),
            (f"  Type:         {cursor_type}", Y),
            (f"  Peak (x,y):   {peak_px_str}", Y),
        ]
    )

    text_h = panel_size + 16
    text_panel = _make_text_panel(lines, W=W_TEXT, H=text_h)
    bar_t = _make_label_bar("State / Prompt / Target", W_TEXT, bar_h=16)
    panels.append(np.concatenate([bar_t, text_panel], axis=0))

    # ── Compose horizontal strip ───────────────────────────────────────────────
    total_h = max(p.shape[0] for p in panels)
    padded: List[np.ndarray] = []
    for p in panels:
        if p.shape[0] < total_h:
            pad = np.zeros((total_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
    row = np.concatenate(padded, axis=1)

    # ── Title bar ──────────────────────────────────────────────────────────────
    title = f"Task: {task_name}  |  Oracle tool: {oracle_tool}  |  cursor_wt: {cursor_weight:.2f}"
    if step_header:
        title = f"{step_header}  |  {title}"
    title_bar = _make_label_bar(title, row.shape[1], bar_h=20,
                                bg=(10, 20, 40), fg=(255, 220, 100))
    return np.concatenate([title_bar, row], axis=0)


# ── Image savers ───────────────────────────────────────────────────────────────

def _save_png(frame: np.ndarray, path: Path) -> None:
    """Save (H, W, 3) uint8 as PNG. Falls back to .npy if PIL is absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(frame).save(str(path))
        return
    except ImportError:
        pass
    np.save(str(path.with_suffix(".npy")), frame)


# ── Phase generators ───────────────────────────────────────────────────────────

def _gen_phase1_grid(config: Dict, output_dir: Path, verbose: bool) -> None:
    """
    Phase 1 diagnostic: tool-prompt text grid.

    For every tool in the config, shows all its natural-language prompt
    variants. Saved as a single PNG (or .npy fallback).
    """
    from cadfire.training.pretrain_tools import _TOOL_PROMPTS
    from cadfire.utils.config import tool_list

    tools = tool_list()
    row_h = 16
    W = 920
    total_H = (len(tools) + 2) * row_h + 10

    # (text, colour) pairs
    lines: List[Tuple[str, Tuple]] = [
        ("Phase 1 – Tool Prompt Grid  (text → tool classifier)", (255, 220, 100)),
        ("Tool                   Prompt variants (first 4)", (150, 200, 255)),
        ("─" * 85, (80, 80, 100)),
    ]
    for tool in tools:
        prompts = _TOOL_PROMPTS.get(tool, [f"Activate {tool.lower()}"])
        shown = " | ".join(prompts[:4])
        lines.append((f"{tool:<22} {shown}", (180, 220, 180)))

    panel = _make_text_panel(lines, W=W, H=total_H)
    out_path = output_dir / "phase1" / "tool_prompts.png"
    _save_png(panel, out_path)
    if verbose:
        print(f"    [phase1] tool_prompts → {out_path}")


def _gen_phase2_samples(
    config: Dict,
    n_per_task: int,
    output_dir: Path,
    seed: int,
    panel_size: int,
    verbose: bool,
) -> Dict[str, int]:
    """
    Phase 2 diagnostic: one PNG per supervised task type.

    Generates ``n_per_task`` samples for each task in the Phase-2 registry,
    stacks them vertically, and saves as ``phase2/{TaskClassName}.png``.

    Returns a dict mapping task_class_name → number of frames saved.
    """
    from cadfire.training.pretrain_semantic import _TASK_REGISTRY, oracle_to_cursor_mask

    sigma = 12.0
    H_c = config["canvas"]["render_height"]
    W_c = config["canvas"]["render_width"]

    tokenizer = BPETokenizer(
        vocab_size=config["model"]["text_vocab_size"],
        max_len=config["model"]["text_max_len"],
    )
    tool_idx = tool_to_index()
    rng = np.random.RandomState(seed)
    counts: Dict[str, int] = {}

    for _wt, task_class, task_kwargs in _TASK_REGISTRY:
        task_name = task_class.__name__
        frames: List[np.ndarray] = []

        if verbose:
            print(f"    [phase2] {task_name} ({n_per_task} samples)")

        for _i in range(n_per_task):
            engine = CADEngine(config)
            renderer = Renderer(config)
            task = task_class(seed=int(rng.randint(0, 2 ** 31)), **task_kwargs)
            engine.reset()

            try:
                setup_info = task.setup(engine)
            except Exception as exc:
                if verbose:
                    print(f"      skip (setup error: {exc})")
                continue

            image = renderer.render(engine)
            ref = setup_info.get("reference_image")
            if ref is not None and image.shape[2] > 5:
                image[:, :, 3:6] = ref.astype(np.float32) / 255.0

            try:
                oracle = task.oracle_action(engine, setup_info)
            except Exception as exc:
                if verbose:
                    print(f"      skip (oracle error: {exc})")
                continue

            tool_name = oracle["tool"]
            cursor_world = oracle["cursor_world"]
            cursor_weight = float(oracle.get("cursor_weight", 1.0))
            cursor_mask = oracle_to_cursor_mask(cursor_world, engine, H_c, W_c, sigma)

            prompt = setup_info["prompt"]
            text_ids = np.array(tokenizer.encode_padded(prompt), dtype=np.int32)
            state_vec = _build_state_vec(engine, tool_idx, config)

            frame = _build_sample_frame(
                image=image,
                cursor_mask=cursor_mask,
                text_ids=text_ids,
                state_vec=state_vec,
                oracle_tool=tool_name,
                cursor_weight=cursor_weight,
                task_name=task_name,
                tokenizer=tokenizer,
                config=config,
                panel_size=panel_size,
            )
            frames.append(frame)

        if frames:
            combined = np.concatenate(frames, axis=0)
            out_path = output_dir / "phase2" / f"{task_name}.png"
            _save_png(combined, out_path)
            counts[task_name] = len(frames)
            if verbose:
                print(f"      → {out_path}")
        else:
            counts[task_name] = 0

    return counts


def _gen_phase3_trajectories(
    config: Dict,
    n_trajectories: int,
    output_dir: Path,
    seed: int,
    panel_size: int,
    verbose: bool,
) -> int:
    """
    Phase 3 diagnostic: one PNG per teacher-forcing trajectory.

    Each PNG stacks all steps of the trajectory vertically, showing how
    the oracle action and observation evolve step-by-step.

    Returns the number of trajectories saved.
    """
    from cadfire.training.pretrain_teacher import TeacherForcingDataset

    tokenizer = BPETokenizer(
        vocab_size=config["model"]["text_vocab_size"],
        max_len=config["model"]["text_max_len"],
    )
    idx_tool = index_to_tool()

    dataset = TeacherForcingDataset(
        config=config,
        num_trajectories=n_trajectories,
        sigma=12.0,
        seed=seed,
    )

    saved = 0
    for traj_idx in range(min(n_trajectories, len(dataset))):
        trajectory = dataset[traj_idx]
        if not trajectory:
            continue

        n_steps = len(trajectory)
        frames: List[np.ndarray] = []

        for step_idx, step in enumerate(trajectory):
            oracle_tool_id = step["tool_id"]
            oracle_tool_name = idx_tool.get(oracle_tool_id, f"tool_{oracle_tool_id}")
            cursor_weight = float(step["cursor_weight"])
            cursor_mask = step["cursor_mask"]

            frame = _build_sample_frame(
                image=step["image"],
                cursor_mask=cursor_mask,
                text_ids=step["text_ids"],
                state_vec=step["state_vec"],
                oracle_tool=oracle_tool_name,
                cursor_weight=cursor_weight,
                task_name="Phase3-TeacherForcing",
                tokenizer=tokenizer,
                config=config,
                panel_size=panel_size,
                step_idx=step_idx,
                n_steps=n_steps,
            )
            frames.append(frame)

        if frames:
            combined = np.concatenate(frames, axis=0)
            out_path = output_dir / "phase3" / f"trajectory_{traj_idx:02d}.png"
            _save_png(combined, out_path)
            saved += 1
            if verbose:
                print(f"    [phase3] trajectory {traj_idx:02d} "
                      f"({n_steps} steps) → {out_path}")

    return saved


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_supervised_diagnostics(
    agent=None,
    config: Dict | None = None,
    output_dir: str = "supervised_diagnostics",
    n_per_task: int = 4,
    n_trajectories: int = 3,
    phases: Tuple[int, ...] = (1, 2, 3),
    panel_size: int = 128,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate rich diagnostic images for all supervised learning tasks.

    For each supervised task type, produces a stacked PNG showing the full
    observation space (viewport, raster, selection mask, layer masks, state
    vector, prompt / text history) alongside the target outputs (oracle tool,
    oracle cursor heatmap).

    Args:
        agent         : Optional CADAgent (reserved for future prediction
                        overlay – not required for basic diagnostics).
        config        : Config dict (defaults to config.json).
        output_dir    : Root output directory (created if missing).
        n_per_task    : Phase-2 samples per supervised task type.
        n_trajectories: Phase-3 trajectories to visualise.
        phases        : Which phases to run, e.g. (1, 2, 3) or (2,).
        panel_size    : Pixel size of each image sub-panel (default 128).
        device        : Torch device string (unused without agent).
        seed          : RNG seed for reproducibility.
        verbose       : Print progress to stdout.

    Returns:
        Dict summarising what was generated:
          { "phase1": path_str,
            "phase2": {"task_count": N, "output_dir": path_str},
            "phase3": {"saved": N,     "output_dir": path_str} }
    """
    config = config or load_config()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[supervised_diagnostics] Output → {out.resolve()}")
        print(f"  Phases: {phases}  |  panel_size: {panel_size}px")

    results: Dict[str, Any] = {}

    if 1 in phases:
        if verbose:
            print("\n  Phase 1 – tool prompt grid")
        _gen_phase1_grid(config, out, verbose)
        results["phase1"] = str(out / "phase1" / "tool_prompts.png")

    if 2 in phases:
        if verbose:
            print(f"\n  Phase 2 – supervised task samples ({n_per_task} per task)")
        counts = _gen_phase2_samples(
            config=config,
            n_per_task=n_per_task,
            output_dir=out,
            seed=seed,
            panel_size=panel_size,
            verbose=verbose,
        )
        results["phase2"] = {
            "task_count": len(counts),
            "sample_counts": counts,
            "output_dir": str(out / "phase2"),
        }

    if 3 in phases:
        if verbose:
            print(f"\n  Phase 3 – teacher-forcing trajectories ({n_trajectories})")
        saved = _gen_phase3_trajectories(
            config=config,
            n_trajectories=n_trajectories,
            output_dir=out,
            seed=seed,
            panel_size=panel_size,
            verbose=verbose,
        )
        results["phase3"] = {
            "saved": saved,
            "output_dir": str(out / "phase3"),
        }

    if verbose:
        print(f"\n[supervised_diagnostics] Done → {out.resolve()}/")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import torch

    parser = argparse.ArgumentParser(
        description="Generate supervised learning task diagnostics"
    )
    parser.add_argument("--checkpoint",      type=str,   default=None,
                        help="Checkpoint dir (optional – loads agent for future overlays)")
    parser.add_argument("--output-dir",      type=str,   default="supervised_diagnostics")
    parser.add_argument("--n-per-task",      type=int,   default=4,
                        help="Phase-2 samples per task type")
    parser.add_argument("--n-trajectories",  type=int,   default=3,
                        help="Phase-3 trajectories to visualise")
    parser.add_argument("--phases",          type=int,   nargs="+", default=[1, 2, 3],
                        help="Which phases to run (1 2 3)")
    parser.add_argument("--panel-size",      type=int,   default=128,
                        help="Pixel size of each image sub-panel")
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--seed",            type=int,   default=0)
    args = parser.parse_args()

    config = load_config()
    agent = None
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        from cadfire.model.cad_agent import CADAgent
        from cadfire.training.checkpoint import CheckpointManager
        agent = CADAgent(config)
        ckpt = CheckpointManager(args.checkpoint, config)
        meta = ckpt.load(agent, optimizer=None, device=dev)
        agent = agent.to(dev)
        print(f"Loaded checkpoint (step {meta.get('step', 0)}) from {args.checkpoint}/")

    generate_supervised_diagnostics(
        agent=agent,
        config=config,
        output_dir=args.output_dir,
        n_per_task=args.n_per_task,
        n_trajectories=args.n_trajectories,
        phases=tuple(args.phases),
        panel_size=args.panel_size,
        device=dev,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
