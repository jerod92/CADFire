#!/usr/bin/env python3
"""
Task Diagnostic: shows how each task should be solved.

For every registered task, this script:
  1. Creates the environment and sets up the task
  2. Executes the *ideal* (oracle) action sequence
  3. Captures frames at each step
  4. Saves a per-task strip image and a combined summary grid

Output:
  task_diagnostics/              (directory)
    draw_line.png                (frame strip for draw_line)
    ...
    summary_grid.png             (all tasks in one image)

Usage:
  python diagnose_tasks.py
  # Or from notebook:
  from diagnose_tasks import run_diagnostics
  run_diagnostics()
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cadfire.utils.config import load_config, tool_list, tool_to_index
from cadfire.engine.cad_engine import CADEngine
from cadfire.env.cad_env import CADEnv
from cadfire.tasks.registry import TaskRegistry


# ─── Oracle action sequences per task ──────────────────────────────────
# Each entry maps task_name -> a callable(env, task, setup_info) that
# returns a list of (action_dict, label_str) tuples.

def _oracle_draw_line(env, task, info):
    """LINE: select tool, click start, click end."""
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    return [
        (_action(env, "LINE", ent.start), "LINE tool"),
        (_action(env, "LINE", ent.start), f"Start ({ent.start[0]:.0f},{ent.start[1]:.0f})"),
        (_action(env, "LINE", ent.end), f"End ({ent.end[0]:.0f},{ent.end[1]:.0f})"),
        (_action(env, "CONFIRM", None), "CONFIRM"),
    ]


def _oracle_draw_circle(env, task, info):
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    edge = ent.center.copy()
    edge[0] += ent.radius
    return [
        (_action(env, "CIRCLE", ent.center), "CIRCLE tool"),
        (_action(env, "CIRCLE", ent.center), f"Center ({ent.center[0]:.0f},{ent.center[1]:.0f})"),
        (_action(env, "CIRCLE", edge), f"Edge (r={ent.radius:.0f})"),
    ]


def _oracle_draw_rectangle(env, task, info):
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]  # Rectangle stored as polyline or has bounds
    # Rectangles have min_pt, max_pt
    if hasattr(ent, 'min_pt') and hasattr(ent, 'max_pt'):
        p1, p2 = ent.min_pt, ent.max_pt
    elif hasattr(ent, 'points') and len(ent.points) >= 2:
        p1 = ent.points[0]
        p2 = ent.points[2] if len(ent.points) >= 3 else ent.points[1]
    else:
        return []
    return [
        (_action(env, "RECTANGLE", p1), "RECT tool"),
        (_action(env, "RECTANGLE", p1), f"Corner1 ({p1[0]:.0f},{p1[1]:.0f})"),
        (_action(env, "RECTANGLE", p2), f"Corner2 ({p2[0]:.0f},{p2[1]:.0f})"),
    ]


def _oracle_draw_polygon(env, task, info):
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    center = getattr(ent, 'center', np.array([500.0, 500.0]))
    edge = center.copy()
    edge[0] += getattr(ent, 'radius', 100.0)
    return [
        (_action(env, "POLYGON", center), "POLYGON tool"),
        (_action(env, "POLYGON", center), f"Center ({center[0]:.0f},{center[1]:.0f})"),
        (_action(env, "POLYGON", edge), f"Edge"),
    ]


def _oracle_draw_ellipse(env, task, info):
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    center = getattr(ent, 'center', np.array([500.0, 500.0]))
    edge = center.copy()
    edge[0] += getattr(ent, 'semi_major', 100.0)
    end = center.copy()
    end[1] += getattr(ent, 'semi_minor', 50.0)
    return [
        (_action(env, "ELLIPSE", center), "ELLIPSE tool"),
        (_action(env, "ELLIPSE", center), f"Center"),
        (_action(env, "ELLIPSE", edge), f"Major axis"),
        (_action(env, "ELLIPSE", end), f"Minor axis"),
    ]


def _oracle_draw_arc(env, task, info):
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    center = getattr(ent, 'center', np.array([500.0, 500.0]))
    r = getattr(ent, 'radius', 100.0)
    start_angle = getattr(ent, 'start_angle', 0.0)
    end_angle = getattr(ent, 'end_angle', np.pi)
    p_start = center + r * np.array([np.cos(start_angle), np.sin(start_angle)])
    p_mid = center + r * np.array([np.cos((start_angle + end_angle) / 2),
                                    np.sin((start_angle + end_angle) / 2)])
    p_end = center + r * np.array([np.cos(end_angle), np.sin(end_angle)])
    return [
        (_action(env, "ARC", p_start), "ARC tool"),
        (_action(env, "ARC", p_start), "Start"),
        (_action(env, "ARC", p_mid), "Mid"),
        (_action(env, "ARC", p_end), "End"),
    ]


def _oracle_fit_view(env, task, info):
    return [
        (_action(env, "FIT_VIEW", None), "FIT_VIEW"),
    ]


def _oracle_noop(env, task, info):
    """Fallback: just show the initial state + NOOP."""
    return [
        (_action(env, "NOOP", None), "NOOP (no oracle)"),
    ]


# Map of task_name -> oracle function
ORACLES = {
    "draw_line": _oracle_draw_line,
    "draw_circle": _oracle_draw_circle,
    "draw_rectangle": _oracle_draw_rectangle,
    "draw_polygon": _oracle_draw_polygon,
    "draw_ellipse": _oracle_draw_ellipse,
    "draw_arc": _oracle_draw_arc,
    "fit_view": _oracle_fit_view,
}


# ─── Helpers ────────────────────────────────────────────────────────────

def _action(env, tool_name, world_pos):
    """Build an action dict from tool name and world position."""
    tool_idx = tool_to_index()
    tid = tool_idx.get(tool_name, 0)

    if world_pos is not None:
        cursor = _world_to_cursor(env, world_pos)
    else:
        cursor = np.zeros((env.render_h, env.render_w), dtype=np.float32)
        cursor[env.render_h // 2, env.render_w // 2] = 1.0

    return {"tool_id": tid, "cursor": cursor, "param": 0.0}


def _world_to_cursor(env, world_pos):
    """Convert world coords to a one-hot cursor heatmap."""
    vp = env.engine.viewport
    px = int((world_pos[0] - vp.center[0]) / (vp.visible_bounds()[1][0] - vp.visible_bounds()[0][0])
             * env.render_w + env.render_w / 2)
    py = int((world_pos[1] - vp.center[1]) / (vp.visible_bounds()[1][1] - vp.visible_bounds()[0][1])
             * env.render_h + env.render_h / 2)
    px = np.clip(px, 0, env.render_w - 1)
    py = np.clip(py, 0, env.render_h - 1)
    cursor = np.zeros((env.render_h, env.render_w), dtype=np.float32)
    cursor[int(py), int(px)] = 1.0
    return cursor



def _render_frame(obs, label, size=256, targets=None):
    """Extract viewport RGB from observation and add a label. Optionally overlay targets."""
    img = obs["image"]
    # First 3 channels are viewport RGB (H, W, C)
    rgb = img[:, :, :3]
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    pil = Image.fromarray(rgb).resize((size, size), Image.NEAREST)

    # Overlay targets if provided
    if targets:
        draw_overlay = ImageDraw.Draw(pil)
        # We need to map world coordinates to this small image
        # Assuming 1000x1000 world and standard viewport
        # For simplicity, we'll assume the viewport shows the whole 0-1000 range or use relative coords
        # But wait, we don't have the viewport transform handy here easily without the env.
        # However, trace_tasks uses a simple 0-1000 -> 0-size mapping.
        
        for ent in targets:
            if hasattr(ent, 'tessellate'):
                pts = ent.tessellate()
                if len(pts) > 1:
                    # Map 0..1000 to 0..size (approximate, assuming default view)
                    # Ideally we should use env.engine.viewport but we don't have it here.
                    # We'll use the same logic as trace_tasks: scale by size/1000
                    px = (pts[:, 0] / 1000.0 * size).astype(int)
                    py = (pts[:, 1] / 1000.0 * size).astype(int)
                    
                    # Draw points
                    xy = list(zip(px, py))
                    draw_overlay.line(xy, fill=(0, 255, 0), width=1)


    # Add label at the top
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    # Background bar
    draw.rectangle([0, 0, size, 20], fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 0), font=font)
    return pil


# ─── Main diagnostic runner ────────────────────────────────────────────

def run_diagnostics(output_dir: str = "task_diagnostics", seed: int = 42):
    """
    Run diagnostic on all tasks, save per-task frame strips and a summary grid.
    Returns dict of task_name -> list of PIL frames.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config()
    env = CADEnv(config)
    TaskRegistry.discover()
    all_tasks = sorted(TaskRegistry.list_tasks())

    print(f"Found {len(all_tasks)} tasks: {all_tasks}")
    print(f"Output dir: {out.resolve()}\n")

    frame_size = 200
    max_frames = 6  # max frames per task strip
    results = {}

    for task_name in all_tasks:
        print(f"-- {task_name} ", end="")
        try:
            task = TaskRegistry.create(task_name, seed=seed)
        except Exception as e:
            print(f"SKIP (create failed: {e})")
            continue

        obs, info = env.reset(task=task, seed=seed)
        # setup_info is now merged into info
        
        # Get targets for visualization
        targets = info.get("target_entities", [])

        # Capture initial state
        frames = [_render_frame(obs, f"[{task_name}] Initial", frame_size, targets=targets)]

        # Get allowed tools
        allowed = task.allowed_tools()
        allowed_str = ", ".join(allowed) if allowed else "ALL"

        # Get oracle or fallback
        oracle_fn = ORACLES.get(task_name, _oracle_noop)
        steps = oracle_fn(env, task, info)

        for i, (action, label) in enumerate(steps[:max_frames - 1]):
            try:
                obs, reward, terminated, truncated, step_info = env.step(action)
                tag = f"r={reward:.3f}" if reward != 0 else ""
                frames.append(_render_frame(obs, f"Step {i+1}: {label} {tag}", frame_size))
            except Exception as e:
                print(f"(step {i+1} error: {e}) ", end="")
                break

        # Build strip image
        strip_w = frame_size * len(frames)
        strip = Image.new("RGB", (strip_w, frame_size + 30), (30, 30, 30))

        # Task header
        draw = ImageDraw.Draw(strip)
        try:
            hfont = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            hfont = ImageFont.load_default()
        draw.text((4, 2), f"{task_name}  |  tools: {allowed_str}", fill=(200, 200, 200), font=hfont)

        for j, frame in enumerate(frames):
            strip.paste(frame, (j * frame_size, 30))

        strip.save(out / f"{task_name}.png")
        results[task_name] = frames
        print(f"OK ({len(frames)} frames, tools=[{allowed_str}])")

    # ── Build summary grid ──────────────────────────────────────────────
    num_tasks = len(results)
    if num_tasks == 0:
        print("No tasks produced frames.")
        return results

    cols = max_frames
    rows = num_tasks
    grid_w = cols * frame_size + 10
    grid_h = rows * (frame_size + 35) + 10
    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)

    try:
        gfont = ImageFont.truetype("arial.ttf", 11)
    except (OSError, IOError):
        gfont = ImageFont.load_default()

    for row, (tname, frames) in enumerate(sorted(results.items())):
        y = row * (frame_size + 35) + 5
        draw.text((5, y), tname, fill=(180, 255, 100), font=gfont)
        y += 15
        for col, frame in enumerate(frames):
            x = col * frame_size + 5
            grid.paste(frame, (x, y))

    grid.save(out / "summary_grid.png")
    print(f"\n[OK] Summary grid saved to {out / 'summary_grid.png'}")
    print(f"  Grid size: {grid_w}x{grid_h}")

    return results


if __name__ == "__main__":
    run_diagnostics()
