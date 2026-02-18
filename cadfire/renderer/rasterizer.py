"""
Pure-numpy rasterizer that produces the multi-channel image tensor
consumed by the RL model.

Output channels:
  0-2:       Current viewport RGB (with ghosting/selection highlights)
  3-5:       Reference raster image (user-provided, e.g. for tracing)
  6..6+L-1:  Layer masks (binary, one per layer)
  6+L:       Selection mask (binary)
  6+L+1:     X ground coords (tanh-scaled world x)
  6+L+2:     Y ground coords (tanh-scaled world y)
  6+L+3:     X window coords (min-max scaled, 0-1 linear ramp)
  6+L+4:     Y window coords (min-max scaled, 0-1 linear ramp)

All rendering is done via Bresenham line drawing and polygon fill,
keeping everything in numpy with no external rendering deps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import Entity, HatchEntity
from cadfire.utils.config import load_config


class Renderer:
    """Renders a CADEngine state into the multi-channel observation tensor."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or load_config()
        canvas = self.config["canvas"]
        self.W = canvas["render_width"]
        self.H = canvas["render_height"]
        colors = self.config["colors"]
        self.palette = np.array(colors["palette"], dtype=np.uint8)
        self.bg_color = np.array(colors["background"], dtype=np.uint8)
        self.sel_color = np.array(colors["selection_highlight"], dtype=np.uint8)
        self.ghost_alpha = colors["ghost_alpha"]
        self.max_layers = self.config["layers"]["max_layers"]

    def num_channels(self) -> int:
        """Total image channels: 3 (viewport) + 3 (reference) + L (layers) + 1 (selection) + 4 (coords)."""
        return 3 + 3 + self.max_layers + 1 + 4

    def render(self, engine: CADEngine,
               reference_image: np.ndarray | None = None) -> np.ndarray:
        """
        Produce the full observation tensor.
        Returns: (H, W, C) float32 array normalized to [0, 1].
        """
        H, W = self.H, self.W
        C = self.num_channels()

        obs = np.zeros((H, W, C), dtype=np.float32)

        # Channel 0-2: Viewport RGB
        viewport_rgb = self._render_viewport(engine)
        obs[:, :, 0:3] = viewport_rgb / 255.0

        # Channel 3-5: Reference image
        if reference_image is not None:
            ref = self._resize_reference(reference_image)
            obs[:, :, 3:6] = ref / 255.0

        # Channel 6..6+L-1: Layer masks
        layer_masks = self._render_layer_masks(engine)
        obs[:, :, 6:6 + self.max_layers] = layer_masks

        # Channel 6+L: Selection mask
        sel_mask = self._render_selection_mask(engine)
        obs[:, :, 6 + self.max_layers] = sel_mask

        # Channels 6+L+1 .. 6+L+4: Coordinate grids
        coord_base = 6 + self.max_layers + 1
        coord_grids = self._render_coord_channels(engine)
        obs[:, :, coord_base:coord_base + 4] = coord_grids

        return obs

    def render_rgb_only(self, engine: CADEngine) -> np.ndarray:
        """Render just the viewport RGB (H, W, 3) uint8 for visualization."""
        return self._render_viewport(engine)

    def _render_viewport(self, engine: CADEngine) -> np.ndarray:
        """Render entities to an RGB image."""
        img = np.full((self.H, self.W, 3), self.bg_color, dtype=np.uint8)

        # Draw visible entities
        for entity in engine.visible_entities():
            color = self.palette[entity.color_index % len(self.palette)]
            is_selected = entity.id in engine.selected_ids
            if is_selected:
                color = self.sel_color
            self._draw_entity(img, engine, entity, color)

        # Draw ghost entities with alpha blending
        for entity in engine.ghost_entities:
            color = self.palette[entity.color_index % len(self.palette)]
            ghost_img = np.full_like(img, self.bg_color)
            self._draw_entity(ghost_img, engine, entity, color)
            mask = np.any(ghost_img != self.bg_color, axis=-1)
            img[mask] = (img[mask] * (1 - self.ghost_alpha) +
                         ghost_img[mask] * self.ghost_alpha).astype(np.uint8)

        return img

    def _draw_entity(self, img: np.ndarray, engine: CADEngine,
                     entity: Entity, color: np.ndarray):
        """Rasterize a single entity onto img."""
        pts_world = entity.tessellate()
        if len(pts_world) == 0:
            return

        # World -> pixel coords
        ndc = engine.viewport.world_to_ndc(pts_world)
        valid = np.isfinite(ndc).all(axis=1)
        if not valid.any():
            return
        ndc = ndc[valid]
        px = (ndc[:, 0] * self.W).astype(np.int32)
        py = ((1.0 - ndc[:, 1]) * self.H).astype(np.int32)  # flip y

        # Handle filled entities
        if isinstance(entity, HatchEntity) and len(pts_world) >= 3:
            self._fill_polygon(img, px, py, color)
            return

        # Draw connected line segments
        for i in range(len(px) - 1):
            self._draw_line(img, px[i], py[i], px[i + 1], py[i + 1], color,
                            int(max(1, entity.lineweight)))

        # Draw points as small circles
        if entity.entity_type == "POINT":
            cx, cy = px[0], py[0]
            r = 2
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < self.H and 0 <= xx < self.W:
                            img[yy, xx] = color

    def _draw_line(self, img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                   color: np.ndarray, thickness: int = 1):
        """Bresenham line with thickness.

        All arithmetic is done with native Python ints to avoid numpy
        int32 overflow when ``2 * err`` is computed for large pixel
        distances (the original overflow warning at line 214).
        """
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        half_t = thickness // 2

        # Safety limit: prevent infinite loops from degenerate coordinates
        max_steps = 2 * (dx + dy) + 1

        for _ in range(max_steps):
            for ty in range(-half_t, half_t + 1):
                for tx in range(-half_t, half_t + 1):
                    yy, xx = y0 + ty, x0 + tx
                    if 0 <= yy < self.H and 0 <= xx < self.W:
                        img[yy, xx] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _fill_polygon(self, img: np.ndarray, px: np.ndarray, py: np.ndarray,
                      color: np.ndarray):
        """Scanline polygon fill."""
        min_y = max(0, py.min())
        max_y = min(self.H - 1, py.max())
        n = len(px)

        for y in range(min_y, max_y + 1):
            intersections = []
            for i in range(n):
                j = (i + 1) % n
                if (py[i] <= y < py[j]) or (py[j] <= y < py[i]):
                    if py[j] != py[i]:
                        x = px[i] + (y - py[i]) * (px[j] - px[i]) / (py[j] - py[i])
                        intersections.append(int(x))
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x_start = max(0, intersections[k])
                x_end = min(self.W - 1, intersections[k + 1])
                img[y, x_start:x_end + 1] = color

    def _render_layer_masks(self, engine: CADEngine) -> np.ndarray:
        """Render binary masks for each layer."""
        masks = np.zeros((self.H, self.W, self.max_layers), dtype=np.float32)
        for entity in engine.entities:
            if entity.layer >= self.max_layers:
                continue
            pts_world = entity.tessellate()
            if len(pts_world) == 0:
                continue
            ndc = engine.viewport.world_to_ndc(pts_world)
            valid = np.isfinite(ndc).all(axis=1)
            if not valid.any():
                continue
            ndc = ndc[valid]
            px = (ndc[:, 0] * self.W).astype(np.int32)
            py = ((1.0 - ndc[:, 1]) * self.H).astype(np.int32)
            for i in range(len(px) - 1):
                self._draw_mask_line(masks[:, :, entity.layer], px[i], py[i], px[i + 1], py[i + 1])
            # also mark single points
            if len(px) == 1:
                if 0 <= py[0] < self.H and 0 <= px[0] < self.W:
                    masks[py[0], px[0], entity.layer] = 1.0
        return masks

    def _draw_mask_line(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        """Bresenham for binary mask.

        Uses native Python ints to avoid numpy int32 overflow on
        ``2 * err`` for large pixel distances.
        """
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        max_steps = 2 * (dx + dy) + 1

        for _ in range(max_steps):
            if 0 <= y0 < self.H and 0 <= x0 < self.W:
                mask[y0, x0] = 1.0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _render_selection_mask(self, engine: CADEngine) -> np.ndarray:
        """Render binary mask of selected entities."""
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        for entity in engine.selected_entities():
            pts_world = entity.tessellate()
            if len(pts_world) == 0:
                continue
            ndc = engine.viewport.world_to_ndc(pts_world)
            valid = np.isfinite(ndc).all(axis=1)
            if not valid.any():
                continue
            ndc = ndc[valid]
            px = (ndc[:, 0] * self.W).astype(np.int32)
            py = ((1.0 - ndc[:, 1]) * self.H).astype(np.int32)
            for i in range(len(px) - 1):
                self._draw_mask_line(mask, px[i], py[i], px[i + 1], py[i + 1])
        return mask

    def _render_coord_channels(self, engine: CADEngine) -> np.ndarray:
        """Render 4 spatial coordinate channels.

        Channel 0: X ground (tanh-scaled world x, centered on viewport)
        Channel 1: Y ground (tanh-scaled world y, centered on viewport)
        Channel 2: X window (linear ramp 0-1, left to right)
        Channel 3: Y window (linear ramp 0-1, top to bottom)

        Ground channels use ``tanh(world_coord / half_extent)`` so that
        the visible viewport maps roughly to [-1, 1] while coordinates
        beyond the viewport saturate smoothly.
        """
        H, W = self.H, self.W
        coords = np.zeros((H, W, 4), dtype=np.float32)

        # Window coords: simple linear ramps [0, 1]
        wx = np.linspace(0.0, 1.0, W, dtype=np.float32)
        wy = np.linspace(0.0, 1.0, H, dtype=np.float32)
        coords[:, :, 2] = wx[np.newaxis, :]   # broadcast across rows
        coords[:, :, 3] = wy[:, np.newaxis]   # broadcast across cols

        # Ground coords: map each pixel to world space, then tanh-normalize
        vis_min, vis_max = engine.viewport.visible_bounds()
        center = (vis_min + vis_max) / 2.0
        half_ext = np.maximum((vis_max - vis_min) / 2.0, 1e-6)

        # Build world-space grids via NDC
        ndc_x = np.linspace(0.0, 1.0, W, dtype=np.float32)
        ndc_y = np.linspace(1.0, 0.0, H, dtype=np.float32)  # flip y
        world_x = vis_min[0] + ndc_x * (vis_max[0] - vis_min[0])
        world_y = vis_min[1] + ndc_y * (vis_max[1] - vis_min[1])

        # Center and scale, then tanh
        gx = np.tanh((world_x - center[0]) / half_ext[0])
        gy = np.tanh((world_y - center[1]) / half_ext[1])
        coords[:, :, 0] = gx[np.newaxis, :]
        coords[:, :, 1] = gy[:, np.newaxis]

        return coords

    def _resize_reference(self, ref: np.ndarray) -> np.ndarray:
        """Nearest-neighbor resize of reference image to render size."""
        h, w = ref.shape[:2]
        if h == self.H and w == self.W:
            return ref.astype(np.float32)
        # Nearest neighbor
        row_idx = (np.arange(self.H) * h / self.H).astype(int)
        col_idx = (np.arange(self.W) * w / self.W).astype(int)
        return ref[np.ix_(row_idx, col_idx)].astype(np.float32)
