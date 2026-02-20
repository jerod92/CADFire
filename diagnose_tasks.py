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

def _action(env, tool_name, world_pos, param=0.0):
    """Build an action dict from tool name and world position."""
    tool_idx = tool_to_index()
    tid = tool_idx.get(tool_name, 0)

    # Convert world to cursor heatmap
    # NOTE: we must match the environment's internal logic for cursor positioning
    if world_pos is not None:
        # Use viewport logic directly
        vp = env.engine.viewport
        ndc = vp.world_to_ndc(np.array([world_pos]))[0]
        # Map NDC (0..1) to pixel (0..W-1)
        px = int(ndc[0] * env.render_w)
        # Flip Y for image coords (0 at top, 1 at bottom in NDC? No, NDC usually 0-1)
        # World Y up -> NDC Y up? Let's check Viewport.world_to_ndc
        # ndc[:, 1] = (points[:, 1] - center + half) / 2*half
        # So low world Y -> low NDC Y. High world Y -> High NDC Y.
        # But image (0,0) is top-left.
        # Rasterizer uses: py = ((1.0 - ndc[:, 1]) * H)
        # So we must invert Y here to match what the environment expects in the cursor map?
        # The env Step method:
        # flat_idx = np.argmax(cursor)
        # py, px = divmod(flat_idx, W)
        # ndc = [px/W, 1.0 - py/H]
        # world = ndc_to_world(ndc)
        #
        # So if we want to target World Y, we need a Py such that (1.0 - Py/H) maps to World Y.
        # NDC_Y = World_to_NDC(Y)
        # 1.0 - Py/H = NDC_Y  => Py/H = 1.0 - NDC_Y => Py = (1.0 - NDC_Y) * H
        py = int((1.0 - ndc[1]) * env.render_h)
        
        px = np.clip(px, 0, env.render_w - 1)
        py = np.clip(py, 0, env.render_h - 1)
        cursor = np.zeros((env.render_h, env.render_w), dtype=np.float32)
        cursor[py, px] = 1.0
    else:
        cursor = np.zeros((env.render_h, env.render_w), dtype=np.float32)
        cursor[env.render_h // 2, env.render_w // 2] = 1.0

    return {"tool_id": tid, "cursor": cursor, "param": float(param)}


def _oracle_draw_line(env, task, info):
    """LINE: click start, click end."""
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    return [
        (_action(env, "LINE", ent.start), f"Start"),
        (_action(env, "LINE", ent.end), f"End"),
    ]


def _oracle_draw_circle(env, task, info):
    """CIRCLE: click center, click edge."""
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    edge = ent.center.copy()
    edge[0] += ent.radius
    return [
        (_action(env, "CIRCLE", ent.center), f"Center"),
        (_action(env, "CIRCLE", edge), f"Edge (r={ent.radius:.0f})"),
    ]


def _oracle_draw_rectangle(env, task, info):
    """RECTANGLE: click corner1, click corner2."""
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    # Rectangles defined by min_pt, max_pt or corner+width/height
    if hasattr(ent, 'corner') and hasattr(ent, 'width'):
        p1 = ent.corner
        p2 = ent.corner + np.array([ent.width, ent.height])
    elif hasattr(ent, 'min_pt') and hasattr(ent, 'max_pt'):
        p1, p2 = ent.min_pt, ent.max_pt
    else:
        pts = ent.tessellate()
        p1 = pts.min(axis=0)
        p2 = pts.max(axis=0)

    return [
        (_action(env, "RECTANGLE", p1), f"Corner1"),
        (_action(env, "RECTANGLE", p2), f"Corner2"),
    ]


def _oracle_draw_polygon(env, task, info):
    """POLYGON: click center, click edge (with param=sides)."""
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    center = getattr(ent, 'center', np.array([500.0, 500.0]))
    edge = center.copy()
    edge[0] += getattr(ent, 'radius', 100.0)
    sides = getattr(ent, 'sides', 6)
    return [
        (_action(env, "POLYGON", center, param=sides), f"Center"),
        (_action(env, "POLYGON", edge, param=sides), f"Edge (n={sides})"),
    ]


def _oracle_draw_ellipse(env, task, info):
    """ELLIPSE: click center, click corner (defines axes)."""
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    center = getattr(ent, 'center', np.array([500.0, 500.0]))
    a = getattr(ent, 'semi_major', 100.0)
    b = getattr(ent, 'semi_minor', 50.0)
    # Ellipse tool uses (cursor - center) as axes sizes
    corner = center + np.array([a, b])
    
    return [
        (_action(env, "ELLIPSE", center), f"Center"),
        (_action(env, "ELLIPSE", corner), f"Size ({a:.0f}x{b:.0f})"),
    ]


def _oracle_draw_arc(env, task, info):
    """ARC: click center, click start-point, click end-point."""
    t = info.get("target_entities", [])
    if not t:
        return []
    ent = t[0]
    center = getattr(ent, 'center', np.array([500.0, 500.0]))
    r = getattr(ent, 'radius', 100.0)
    start_angle = getattr(ent, 'start_angle', 0.0)
    end_angle = getattr(ent, 'end_angle', np.pi)
    
    # 3 points: Center, Start, End
    # Note: cad_engine expects points in world space
    # It calculates angles using atan2(dy, dx)
    p_start = center + r * np.array([np.cos(np.radians(start_angle)), np.sin(np.radians(start_angle))])
    p_end = center + r * np.array([np.cos(np.radians(end_angle)), np.sin(np.radians(end_angle))])

    return [
        (_action(env, "ARC", center), "Center"),
        (_action(env, "ARC", p_start), "Start Point"),
        (_action(env, "ARC", p_end), "End Point"),
    ]


def _oracle_fit_view(env, task, info):
    return [
        (_action(env, "FIT_VIEW", None), "FIT_VIEW"),
    ]



# ─── Modify Oracles ─────────────────────────────────────────────────────

def _oracle_move_shape(env, task, info):
    # Task: Move entity from start to target
    # setup() creates entity at _start, wants it at _target_pos
    if not hasattr(task, '_start') or not hasattr(task, '_target_pos'):
        return []
    start = getattr(task, '_start')
    target = getattr(task, '_target_pos')
    radius = getattr(task, '_radius', 50.0)
    
    # Select at edge to ensure we hit the entity (center might fail tolerance)
    pt_select = start + np.array([radius, 0.0])

    return [
        (_action(env, "SELECT", pt_select), "Select object (edge)"),
        (_action(env, "MOVE", start), "MOVE base point"),
        (_action(env, "MOVE", target), "MOVE target point"),
    ]

def _oracle_rotate_shape(env, task, info):
    # Task: Rotate entity by _angle around _center
    if not hasattr(task, '_center') or not hasattr(task, '_angle'):
        return []
    center = getattr(task, '_center')
    angle = getattr(task, '_angle')
    # For rectangle, click a corner to select
    w = getattr(task, '_w', 100.0)
    h = getattr(task, '_h', 50.0)
    corner = center - np.array([w/2, h/2])

    return [
        (_action(env, "SELECT", corner), "Select object (corner)"),
        (_action(env, "ROTATE", center, param=angle), f"ROTATE {angle} deg"),
    ]

def _oracle_scale_shape(env, task, info):
    # Task: Scale entity by _factor around _center
    if not hasattr(task, '_center') or not hasattr(task, '_factor'):
        return []
    center = getattr(task, '_center')
    factor = getattr(task, '_factor')
    radius = getattr(task, '_radius', 50.0)
    pt_select = center + np.array([radius, 0.0])
    
    return [
        (_action(env, "SELECT", pt_select), "Select object (edge)"),
        (_action(env, "SCALE", center, param=factor), f"SCALE {factor}x"),
    ]

def _oracle_copy_shape(env, task, info):
    # Task: Copy entity from _src to _dst
    if not hasattr(task, '_src') or not hasattr(task, '_dst'):
        return []
    src = getattr(task, '_src')
    dst = getattr(task, '_dst')
    radius = getattr(task, '_radius', 50.0)
    pt_select = src + np.array([radius, 0.0])
    
    return [
        (_action(env, "SELECT", pt_select), "Select object (edge)"),
        (_action(env, "COPY", src), "COPY base point"),
        (_action(env, "COPY", dst), "COPY target point"),
    ]

def _oracle_erase_selection(env, task, info):
    # Task: Erase selected entity.
    # Usually based on EraseShapeTask or similar. Assuming it has a target shape.
    # If generic, let's look for any entity.
    # But typically tasks have _entity_id.
    # If we can't find specific attrs, we'll try center + offset.
    pos = getattr(task, '_center', np.array([500.0, 500.0]))
    # Assume it's a circle or similar, offset slightly
    pt_select = pos + np.array([20.0, 0.0])

    return [
        (_action(env, "SELECT", pt_select), "Select object"),
        (_action(env, "ERASE", None), "ERASE"),
    ]

# ─── Trace Oracles ──────────────────────────────────────────────────────

def _oracle_trace_composite(env, task, info):
    # Similar to draw_multi_primitive: iterate targets and draw them
    targets = info.get("target_entities", [])
    steps = []
    
    for i, ent in enumerate(targets):
        if ent.entity_type == "LINE":
            steps.append((_action(env, "LINE", ent.start), f"Line {i+1} Start"))
            steps.append((_action(env, "LINE", ent.end), f"Line {i+1} End"))
        elif ent.entity_type == "CIRCLE":
            edge = ent.center.copy()
            edge[0] += ent.radius
            steps.append((_action(env, "CIRCLE", ent.center), f"Circ {i+1} Center"))
            steps.append((_action(env, "CIRCLE", edge), f"Circ {i+1} Edge"))
        elif ent.entity_type == "POLYLINE":
             # For simplicity, if simple open polyline
             for p in ent.points:
                 steps.append((_action(env, "POLYLINE", p), f"Poly {i+1} Pt"))
             steps.append((_action(env, "CONFIRM", None), "CONFIRM"))
             
    return steps

# ─── View Oracles ───────────────────────────────────────────────────────

def _oracle_zoom_to_center(env, task, info):
    # Task: Center viewport on _target
    if not hasattr(task, '_target'):
        return []
    target = getattr(task, '_target')
    
    # We can use PAN to move current center to target
    # Current center is env.engine.viewport.center (starts random in setup)
    # But oracle sequence is static? No, it takes `env`.
    # Actually, setup() randomizes viewport. 
    # So we need to compute delta from current env state?
    # Or just use the target coordinate?
    # The PAN tool takes a relative move or start/end points?
    # _execute_tool("PAN"):
    #   if pending: delta = cursor - base
    #   pan( -delta / extent ... )
    # So we can click current center, then click target.
    # That implies delta = target - center.
    # Pan moves the VIEWPORT center.
    # If we want the VIEWPORT center to move TO the target...
    # We want new_center = target.
    # Pan logic: center += dx_frac * extent.
    # 
    # Wait, simpler approach: Use keyboard shortcuts if available? None.
    # Use PAN tool with 2 points.
    # Click 1: Current Center (500,500 in world? No, viewport center).
    # Click 2: Target.
    # Base = Center. Cursor = Target. Delta = Target - Center.
    # Pan implementation: center += -delta ...
    # This moves the camera in OPOSITE direction of drag (like dragging paper).
    # If I drag paper Left, camera moves Right?
    # Let's check `cad_engine.py`:
    # pan(dx_frac, dy_frac): center += dx * extent.
    # execute: pan(-delta.x, -delta.y).
    #
    # We want: center_new = target.
    # center_new = center_old + update.
    # target = center + update.
    # update = target - center.
    #
    # We need: -delta = target - center  => delta = center - target.
    # So:
    # Click 1: Target
    # Click 2: Center
    # Delta = Center - Target.
    # Pan(-Delta) = Pan(-(Center-Target)) = Pan(Target-Center).
    # New Center = Center + (Target-Center) = Target.
    #
    # So: Click 1 = Target, Click 2 = Viewport Center.
    
    vp_center = env.engine.viewport.center
    return [
        (_action(env, "PAN", target), "PAN Start (Target)"),
        (_action(env, "PAN", vp_center), "PAN End (View Center)"),
        # Maybe Zoom In a bit to verify?
        # (_action(env, "ZOOM_IN", None), "Zoom In"),
    ]




def _oracle_noop(env, task, info):
    """Fallback: just show the initial state + NOOP."""
    return [
        (_action(env, "NOOP", None), "NOOP"),
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
    # Modify
    "move_shape": _oracle_move_shape,
    "rotate_shape": _oracle_rotate_shape,
    "scale_shape": _oracle_scale_shape,
    "copy_shape": _oracle_copy_shape,
    "erase_selection": _oracle_erase_selection,
    # Trace
    "trace_line": _oracle_draw_line,       # Same logic as draw (target is in info)
    "trace_circle": _oracle_draw_circle,   # Same logic
    "trace_composite": _oracle_trace_composite,
    # View
    "zoom_to_center": _oracle_zoom_to_center,
}


# ─── Helpers ────────────────────────────────────────────────────────────

def _render_frame(env, obs, label, size=256, targets=None):
    """Extract viewport RGB from observation and add a label. Optionally overlay targets."""
    img = obs["image"]
    # First 3 channels are viewport RGB (H, W, C)
    rgb = img[:, :, :3]
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    pil = Image.fromarray(rgb).resize((size, size), Image.NEAREST)

    # Overlay targets if provided
    if targets:
        draw_overlay = ImageDraw.Draw(pil)
        vp = env.engine.viewport
        
        for ent in targets:
            if hasattr(ent, 'tessellate'):
                pts = ent.tessellate()
                if len(pts) > 1:
                    # Use actual viewport transformation
                    ndc = vp.world_to_ndc(pts)
                    
                    # NDC is [0,1] with (0,0) at bottom-left? No, need to check doc.
                    # viewport.world_to_ndc:
                    # ndc[:, 0] = ...
                    # ndc[:, 1] = ...
                    # Standard mathematical 0..1.
                    # Image coordinates: (0,0) is TOP-left.
                    # So x is same, y is inverted.
                    
                    px = (ndc[:, 0] * size).astype(int)
                    py = ((1.0 - ndc[:, 1]) * size).astype(int)
                    
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
        frames = [_render_frame(env, obs, f"[{task_name}] Initial", frame_size, targets=targets)]

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
                frames.append(_render_frame(env, obs, f"Step {i+1}: {label} {tag}", frame_size, targets=targets))
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
