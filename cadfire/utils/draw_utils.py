"""
Drawing and geometry utility functions for CADFire.

Covers:
  - Coordinate transforms (world ↔ screen)
  - Snap helpers (grid snap, angle snap, entity snap)
  - Geometric primitives (distance, angle, midpoint, polar)
  - Bounding-box helpers
  - Tessellation / sampling helpers

These are pure-numpy utilities with no GUI or engine dependencies,
making them usable from any layer: UI, tasks, training diagnostics.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ── Coordinate Transforms ────────────────────────────────────────────────────

def world_to_screen(
    world_xy: np.ndarray,
    viewport_center: np.ndarray,
    viewport_zoom: float,
    world_size: float,
    canvas_size: int,
) -> np.ndarray:
    """
    Convert world coordinates to screen (pixel) coordinates.

    World space: (0, 0) at bottom-left, y increases upward.
    Screen space: (0, 0) at top-left, y increases downward.

    Args:
        world_xy      : (..., 2) array of world coordinates.
        viewport_center: (2,) world coordinate at canvas centre.
        viewport_zoom  : Current zoom factor.
        world_size     : Width (= height) of the square world.
        canvas_size    : Width (= height) of the square canvas in pixels.

    Returns:
        (..., 2) float array of pixel coordinates.
    """
    world_xy = np.asarray(world_xy, dtype=np.float64)
    half_vis = (world_size / viewport_zoom) / 2.0
    ndc = (world_xy - viewport_center + half_vis) / (2.0 * half_vis)  # [0, 1]
    screen = np.empty_like(ndc)
    screen[..., 0] = ndc[..., 0] * canvas_size
    screen[..., 1] = (1.0 - ndc[..., 1]) * canvas_size  # flip y
    return screen


def screen_to_world(
    screen_xy: np.ndarray,
    viewport_center: np.ndarray,
    viewport_zoom: float,
    world_size: float,
    canvas_size: int,
) -> np.ndarray:
    """
    Convert screen (pixel) coordinates to world coordinates.

    Inverse of :func:`world_to_screen`.
    """
    screen_xy = np.asarray(screen_xy, dtype=np.float64)
    half_vis = (world_size / viewport_zoom) / 2.0
    ndc = np.empty_like(screen_xy)
    ndc[..., 0] = screen_xy[..., 0] / canvas_size
    ndc[..., 1] = 1.0 - screen_xy[..., 1] / canvas_size  # flip y
    world = ndc * (2.0 * half_vis) + viewport_center - half_vis
    return world


def ndc_to_world(ndc: np.ndarray, viewport_center: np.ndarray,
                 viewport_zoom: float, world_size: float) -> np.ndarray:
    """Convert normalised device coordinates [0,1] to world coordinates."""
    half_vis = (world_size / viewport_zoom) / 2.0
    return ndc * (2.0 * half_vis) + viewport_center - half_vis


# ── Snap Helpers ─────────────────────────────────────────────────────────────

def snap_to_grid(
    world_xy: np.ndarray,
    grid_size: float,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    """
    Snap a world-space point to the nearest grid intersection.

    Args:
        world_xy  : (2,) point to snap.
        grid_size : Grid cell size in world units.
        origin    : Grid origin (defaults to [0, 0]).

    Returns:
        (2,) snapped point.
    """
    if grid_size <= 0:
        return np.asarray(world_xy, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64) if origin is not None else np.zeros(2)
    rel = np.asarray(world_xy, dtype=np.float64) - o
    return np.round(rel / grid_size) * grid_size + o


def snap_angle(angle_deg: float, increment: float = 15.0) -> float:
    """
    Snap an angle to the nearest multiple of *increment* degrees.

    Useful for ortho / polar-tracking mode.
    """
    return round(angle_deg / increment) * increment


def snap_to_entity_points(
    world_xy: np.ndarray,
    tessellated_points: np.ndarray,
    tolerance: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Snap to the nearest point in a pre-tessellated point cloud.

    Args:
        world_xy           : (2,) query point.
        tessellated_points : (N, 2) array of candidate snap points.
        tolerance          : Maximum snap distance in world units.

    Returns:
        Snapped (2,) point or None if nothing is within tolerance.
    """
    if len(tessellated_points) == 0:
        return None
    dists = np.linalg.norm(tessellated_points - world_xy, axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] <= tolerance:
        return tessellated_points[idx].copy()
    return None


# ── Geometric Primitives ─────────────────────────────────────────────────────

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint of segment p1→p2."""
    return (np.asarray(p1, dtype=np.float64) + np.asarray(p2, dtype=np.float64)) / 2.0


def angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Angle in degrees of the vector from p1 to p2, measured
    counter-clockwise from the positive x-axis.  Range: [0, 360).
    """
    d = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    a = math.degrees(math.atan2(d[1], d[0]))
    return a % 360.0


def polar_point(origin: np.ndarray, angle_degrees: float, dist: float) -> np.ndarray:
    """
    Compute a point at *dist* world-units from *origin* at *angle_degrees*.

    Counter-clockwise from positive x-axis, matching CAD convention.
    """
    rad = math.radians(angle_degrees)
    o = np.asarray(origin, dtype=np.float64)
    return o + np.array([math.cos(rad), math.sin(rad)]) * dist


def rotate_point(point: np.ndarray, center: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate *point* around *center* by *angle_degrees* (counter-clockwise)."""
    rad = math.radians(angle_degrees)
    c, s = math.cos(rad), math.sin(rad)
    p = np.asarray(point, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]]) + center


def perpendicular_distance(
    point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
) -> float:
    """Perpendicular distance from *point* to the infinite line through line_start→line_end."""
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(line_start, dtype=np.float64)
    b = np.asarray(line_end, dtype=np.float64)
    seg = b - a
    length = np.linalg.norm(seg)
    if length < 1e-12:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(seg, a - p)) / length)


def line_intersection(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Intersection of lines p1→p2 and p3→p4 (infinite lines).

    Returns the intersection point or None if lines are parallel.
    """
    d1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    d2 = np.asarray(p4, dtype=np.float64) - np.asarray(p3, dtype=np.float64)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None  # parallel
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return np.asarray(p1, dtype=np.float64) + t * d1


# ── Bounding Box Helpers ─────────────────────────────────────────────────────

def bbox_of_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Axis-aligned bounding box of a point array.

    Returns:
        (min_xy, max_xy) each (2,) float64.
    """
    pts = np.asarray(points, dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def bbox_center(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Centre of an axis-aligned bounding box."""
    return (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2.0


def bbox_size(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """(width, height) of an axis-aligned bounding box."""
    return np.asarray(bbox_max) - np.asarray(bbox_min)


def bbox_union(
    bboxes: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned union of multiple bounding boxes."""
    mins = np.array([b[0] for b in bboxes])
    maxs = np.array([b[1] for b in bboxes])
    return mins.min(axis=0), maxs.max(axis=0)


# ── Status Bar Formatting ────────────────────────────────────────────────────

def format_world_coords(xy: np.ndarray, decimals: int = 1) -> str:
    """Format a world coordinate pair for display in a status bar."""
    return f"X: {xy[0]:.{decimals}f}  Y: {xy[1]:.{decimals}f}"


def format_distance(d: float, decimals: int = 1) -> str:
    """Format a distance value for display."""
    return f"d: {d:.{decimals}f}"


def format_angle(deg: float, decimals: int = 1) -> str:
    """Format an angle value for display."""
    return f"∠ {deg:.{decimals}f}°"


# ── Curve / Arc Sampling ─────────────────────────────────────────────────────

def sample_arc(
    center: np.ndarray,
    radius: float,
    start_deg: float,
    end_deg: float,
    n: int = 64,
) -> np.ndarray:
    """
    Sample *n* evenly-spaced points along a circular arc.

    Angles are measured counter-clockwise from positive x-axis.
    """
    c = np.asarray(center, dtype=np.float64)
    # Handle wrap-around
    if end_deg < start_deg:
        end_deg += 360.0
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), n)
    return c + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def sample_ellipse(
    center: np.ndarray,
    semi_major: float,
    semi_minor: float,
    rotation_deg: float = 0.0,
    n: int = 64,
) -> np.ndarray:
    """Sample *n* evenly-spaced points around an ellipse."""
    c = np.asarray(center, dtype=np.float64)
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.column_stack([semi_major * np.cos(t), semi_minor * np.sin(t)])
    if rotation_deg:
        rot = math.radians(rotation_deg)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        pts = pts @ R.T
    return pts + c


def sample_regular_polygon(
    center: np.ndarray,
    radius: float,
    sides: int,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """
    Vertices of a regular polygon, closed (first == last point).
    """
    c = np.asarray(center, dtype=np.float64)
    angles = [
        math.radians(rotation_deg + 360.0 * i / sides) for i in range(sides + 1)
    ]
    pts = np.array([[math.cos(a), math.sin(a)] for a in angles]) * radius + c
    return pts


# ── Heatmap / Cursor Helpers ─────────────────────────────────────────────────

def gaussian_heatmap(
    center_px: Tuple[int, int],
    height: int,
    width: int,
    sigma: float = 12.0,
) -> np.ndarray:
    """
    Create a (H, W) float32 Gaussian blob centred at *center_px*.

    Useful for visualising where the model's cursor is pointing.
    """
    cy, cx = center_px
    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


def heatmap_to_world(
    heatmap: np.ndarray,
    viewport_center: np.ndarray,
    viewport_zoom: float,
    world_size: float,
) -> np.ndarray:
    """
    Convert a (H, W) cursor heatmap argmax to a world coordinate.

    Returns the world (x, y) of the pixel with the highest activation.
    """
    H, W = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    py, px = divmod(flat_idx, W)
    screen_xy = np.array([px + 0.5, py + 0.5], dtype=np.float64)
    return screen_to_world(screen_xy, viewport_center, viewport_zoom, world_size, W)
