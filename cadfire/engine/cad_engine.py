"""
Core CAD Engine: manages entities, layers, selection, undo/redo, and viewport.

This is the "world state" that the RL environment wraps. It is purely numpy-based
with no GUI dependencies. The engine processes commands and modifies state; the
Renderer reads from it to produce image tensors.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from cadfire.engine.geometry import (
    Entity, LineEntity, PolylineEntity, CircleEntity, ArcEntity,
    RectangleEntity, PolygonEntity, EllipseEntity, SplineEntity,
    PointEntity, HatchEntity, TextEntity, DimensionEntity,
)
from cadfire.utils.config import load_config


@dataclass
class LayerState:
    name: str = "0"
    visible: bool = True
    frozen: bool = False
    color_index: int = 0
    linetype: str = "CONTINUOUS"


@dataclass
class Viewport:
    center: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0]))
    zoom: float = 1.0
    width: float = 1000.0
    height: float = 1000.0

    def world_to_ndc(self, points: np.ndarray) -> np.ndarray:
        """Convert world coords to normalized device coords [0,1]."""
        half_w = (self.width / self.zoom) / 2.0
        half_h = (self.height / self.zoom) / 2.0
        ndc = np.empty_like(points)
        ndc[:, 0] = (points[:, 0] - self.center[0] + half_w) / (2 * half_w)
        ndc[:, 1] = (points[:, 1] - self.center[1] + half_h) / (2 * half_h)
        return ndc

    def ndc_to_world(self, ndc: np.ndarray) -> np.ndarray:
        """Convert normalized device coords [0,1] to world coords."""
        half_w = (self.width / self.zoom) / 2.0
        half_h = (self.height / self.zoom) / 2.0
        world = np.empty_like(ndc)
        world[:, 0] = ndc[:, 0] * (2 * half_w) + self.center[0] - half_w
        world[:, 1] = ndc[:, 1] * (2 * half_h) + self.center[1] - half_h
        return world

    def visible_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_xy, max_xy) of visible world area."""
        half_w = (self.width / self.zoom) / 2.0
        half_h = (self.height / self.zoom) / 2.0
        return (
            self.center - np.array([half_w, half_h]),
            self.center + np.array([half_w, half_h]),
        )


class CADEngine:
    """
    Stateful CAD engine. Holds all entities, layers, selection, and viewport.
    Supports undo/redo via full-state snapshots (simple and robust for RL).
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or load_config()
        canvas = self.config["canvas"]
        layer_cfg = self.config["layers"]
        vp_cfg = self.config["viewport"]

        # Entities: ordered list (draw order matters for rendering)
        self.entities: List[Entity] = []

        # Layers
        self.layers: List[LayerState] = []
        for i, name in enumerate(layer_cfg["layer_names"][:layer_cfg["max_layers"]]):
            self.layers.append(LayerState(name=name, color_index=i % 8))
        self.active_layer: int = layer_cfg["default_layer"]

        # Active drawing color (index into palette)
        self.active_color: int = 0
        self.active_linetype: str = "CONTINUOUS"

        # Selection
        self.selected_ids: Set[str] = set()

        # Viewport
        self.viewport = Viewport(
            center=np.array(vp_cfg["default_center"], dtype=np.float64),
            zoom=vp_cfg["default_zoom"],
            width=canvas["world_width"],
            height=canvas["world_height"],
        )

        # Undo/Redo stacks (store serialized snapshots)
        self._undo_stack: List[Dict] = []
        self._redo_stack: List[Dict] = []
        self._max_undo = self.config.get("undo", {}).get("max_history", 50)

        # Ghost entities (for in-progress tool preview)
        self.ghost_entities: List[Entity] = []

        # Pending tool state (for multi-step commands like polyline)
        self.pending_points: List[np.ndarray] = []
        self.active_tool: str = "NOOP"

    # ─── Snapshot / Undo / Redo ─────────────────────────────────────────

    def _snapshot(self) -> Dict:
        """Capture full engine state for undo."""
        return {
            "entities": [e.clone() for e in self.entities],
            "selected_ids": set(self.selected_ids),
            "active_layer": self.active_layer,
            "active_color": self.active_color,
            "active_linetype": self.active_linetype,
            "viewport": copy.deepcopy(self.viewport),
            "layers": copy.deepcopy(self.layers),
        }

    def _restore(self, snap: Dict):
        """Restore engine state from snapshot."""
        self.entities = snap["entities"]
        self.selected_ids = snap["selected_ids"]
        self.active_layer = snap["active_layer"]
        self.active_color = snap["active_color"]
        self.active_linetype = snap["active_linetype"]
        self.viewport = snap["viewport"]
        self.layers = snap["layers"]

    def save_undo(self):
        """Push current state to undo stack (call before any mutation)."""
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        return True

    # ─── Entity Management ──────────────────────────────────────────────

    def add_entity(self, entity: Entity, save_undo: bool = True) -> str:
        """Add entity and return its id."""
        if save_undo:
            self.save_undo()
        entity.layer = self.active_layer
        entity.color_index = self.active_color
        entity.linetype = self.active_linetype
        self.entities.append(entity)
        return entity.id

    def remove_entity(self, entity_id: str, save_undo: bool = True):
        if save_undo:
            self.save_undo()
        self.entities = [e for e in self.entities if e.id != entity_id]
        self.selected_ids.discard(entity_id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None

    def entities_on_layer(self, layer_idx: int) -> List[Entity]:
        return [e for e in self.entities if e.layer == layer_idx]

    def visible_entities(self) -> List[Entity]:
        """Return entities on visible, non-frozen layers."""
        return [
            e for e in self.entities
            if e.visible
            and e.layer < len(self.layers)
            and self.layers[e.layer].visible
            and not self.layers[e.layer].frozen
        ]

    # ─── Selection ──────────────────────────────────────────────────────

    def select(self, entity_id: str):
        self.selected_ids.add(entity_id)

    def deselect(self, entity_id: str):
        self.selected_ids.discard(entity_id)

    def deselect_all(self):
        self.selected_ids.clear()

    def select_at_point(self, world_point: np.ndarray, tolerance: float = 5.0) -> Optional[str]:
        """Select nearest visible entity within tolerance of a world point."""
        best_id = None
        best_dist = tolerance
        for e in self.visible_entities():
            pts = e.tessellate()
            if len(pts) == 0:
                continue
            dists = np.linalg.norm(pts - world_point, axis=1)
            min_d = dists.min()
            if min_d < best_dist:
                best_dist = min_d
                best_id = e.id
        if best_id:
            self.selected_ids.add(best_id)
        return best_id

    def select_in_region(self, mask: np.ndarray, render_w: int, render_h: int):
        """Select entities whose centroid falls within a binary mask (render-space)."""
        vis_min, vis_max = self.viewport.visible_bounds()
        for e in self.visible_entities():
            c = e.centroid()
            # convert centroid to pixel coords
            ndc = self.viewport.world_to_ndc(c.reshape(1, 2))[0]
            px = int(ndc[0] * render_w)
            py = int((1.0 - ndc[1]) * render_h)  # flip y
            if 0 <= px < render_w and 0 <= py < render_h:
                if mask[py, px] > 0.5:
                    self.selected_ids.add(e.id)

    def selected_entities(self) -> List[Entity]:
        return [e for e in self.entities if e.id in self.selected_ids]

    # ─── Drawing Commands ───────────────────────────────────────────────

    def draw_line(self, start: np.ndarray, end: np.ndarray) -> str:
        e = LineEntity(start=start.astype(np.float64), end=end.astype(np.float64))
        return self.add_entity(e)

    def draw_polyline(self, points: np.ndarray, closed: bool = False) -> str:
        e = PolylineEntity(points=points.astype(np.float64), closed=closed)
        return self.add_entity(e)

    def draw_circle(self, center: np.ndarray, radius: float) -> str:
        e = CircleEntity(center=center.astype(np.float64), radius=float(radius))
        return self.add_entity(e)

    def draw_arc(self, center: np.ndarray, radius: float, start_angle: float, end_angle: float) -> str:
        e = ArcEntity(
            center=center.astype(np.float64), radius=float(radius),
            start_angle=float(start_angle), end_angle=float(end_angle),
        )
        return self.add_entity(e)

    def draw_rectangle(self, corner: np.ndarray, width: float, height: float) -> str:
        e = RectangleEntity(corner=corner.astype(np.float64), width=float(width), height=float(height))
        return self.add_entity(e)

    def draw_polygon(self, center: np.ndarray, radius: float, sides: int, rotation: float = 0) -> str:
        e = PolygonEntity(
            center=center.astype(np.float64), radius=float(radius),
            sides=int(sides), rotation=float(rotation),
        )
        return self.add_entity(e)

    def draw_ellipse(self, center: np.ndarray, semi_major: float, semi_minor: float, rotation: float = 0) -> str:
        e = EllipseEntity(
            center=center.astype(np.float64),
            semi_major=float(semi_major), semi_minor=float(semi_minor),
            rotation=float(rotation),
        )
        return self.add_entity(e)

    def draw_spline(self, control_points: np.ndarray, degree: int = 3) -> str:
        e = SplineEntity(control_points=control_points.astype(np.float64), degree=degree)
        return self.add_entity(e)

    def draw_point(self, position: np.ndarray) -> str:
        e = PointEntity(position=position.astype(np.float64))
        return self.add_entity(e)

    def draw_hatch(self, boundary: np.ndarray, pattern: str = "SOLID") -> str:
        e = HatchEntity(boundary=boundary.astype(np.float64), pattern=pattern)
        return self.add_entity(e)

    def draw_text(self, position: np.ndarray, text: str, height: float = 10.0,
                  rotation: float = 0.0, multiline: bool = False) -> str:
        e = TextEntity(
            position=position.astype(np.float64), text=text,
            height=height, rotation=rotation, multiline=multiline,
        )
        return self.add_entity(e)

    def draw_dimension(self, p1: np.ndarray, p2: np.ndarray,
                       text_pos: np.ndarray, dim_type: str = "LINEAR") -> str:
        measurement = float(np.linalg.norm(p2 - p1))
        if dim_type == "ANGULAR":
            # interpret as angle between two vectors from origin
            measurement = math.degrees(math.atan2(p2[1], p2[0]) - math.atan2(p1[1], p1[0]))
        e = DimensionEntity(
            dim_type=dim_type,
            point1=p1.astype(np.float64), point2=p2.astype(np.float64),
            text_position=text_pos.astype(np.float64), measurement=measurement,
        )
        return self.add_entity(e)

    # ─── Modify Commands ────────────────────────────────────────────────

    def move_selected(self, dx: float, dy: float):
        self.save_undo()
        for e in self.selected_entities():
            e.translate(dx, dy)

    def copy_selected(self, dx: float, dy: float) -> List[str]:
        self.save_undo()
        new_ids = []
        for e in self.selected_entities():
            clone = e.clone()
            clone.translate(dx, dy)
            self.entities.append(clone)
            new_ids.append(clone.id)
        return new_ids

    def rotate_selected(self, angle_deg: float, center: np.ndarray | None = None):
        self.save_undo()
        sel = self.selected_entities()
        if not sel:
            return
        if center is None:
            # rotate around centroid of selection
            centroids = np.array([e.centroid() for e in sel])
            center = centroids.mean(axis=0)
        for e in sel:
            e.rotate(angle_deg, center[0], center[1])

    def scale_selected(self, factor: float, center: np.ndarray | None = None):
        self.save_undo()
        sel = self.selected_entities()
        if not sel:
            return
        if center is None:
            centroids = np.array([e.centroid() for e in sel])
            center = centroids.mean(axis=0)
        for e in sel:
            e.scale_entity(factor, center[0], center[1])

    def mirror_selected(self, p1: np.ndarray, p2: np.ndarray) -> List[str]:
        """Mirror selected entities across line p1->p2, creating copies."""
        self.save_undo()
        new_ids = []
        for e in self.selected_entities():
            clone = e.clone()
            clone.mirror_entity(p1, p2)
            self.entities.append(clone)
            new_ids.append(clone.id)
        return new_ids

    def offset_selected(self, distance: float):
        """Simplified offset: scale each entity from its own centroid."""
        self.save_undo()
        new_ids = []
        for e in self.selected_entities():
            clone = e.clone()
            c = clone.centroid()
            # Offset approximation: scale slightly from centroid
            bb_min, bb_max = clone.bbox()
            size = max(np.linalg.norm(bb_max - bb_min), 1e-6)
            factor = 1.0 + distance / size
            clone.scale_entity(factor, c[0], c[1])
            self.entities.append(clone)
            new_ids.append(clone.id)
        return new_ids

    def erase_selected(self):
        self.save_undo()
        self.entities = [e for e in self.entities if e.id not in self.selected_ids]
        self.selected_ids.clear()

    def explode_selected(self):
        """Explode compound entities (rectangles, polygons) into polylines."""
        self.save_undo()
        new_entities = []
        remove_ids = set()
        for e in self.selected_entities():
            if e.entity_type in ("RECTANGLE", "POLYGON", "ELLIPSE"):
                pts = e.tessellate()
                # Create line segments
                for i in range(len(pts) - 1):
                    line = LineEntity(
                        start=pts[i].copy(), end=pts[i + 1].copy(),
                        layer=e.layer, color_index=e.color_index,
                        linetype=e.linetype, lineweight=e.lineweight,
                    )
                    new_entities.append(line)
                remove_ids.add(e.id)
        self.entities = [e for e in self.entities if e.id not in remove_ids]
        self.entities.extend(new_entities)
        self.selected_ids -= remove_ids

    def matchprop(self, source_id: str):
        """Copy properties from source entity to all selected entities."""
        src = self.get_entity(source_id)
        if not src:
            return
        self.save_undo()
        for e in self.selected_entities():
            if e.id != source_id:
                e.layer = src.layer
                e.color_index = src.color_index
                e.linetype = src.linetype
                e.lineweight = src.lineweight

    # ─── Layer Commands ─────────────────────────────────────────────────

    def set_layer(self, layer_idx: int):
        if 0 <= layer_idx < len(self.layers):
            self.active_layer = layer_idx

    def layer_off(self, layer_idx: int):
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].visible = False

    def layer_on(self, layer_idx: int):
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].visible = True

    def layer_on_all(self):
        for l in self.layers:
            l.visible = True

    def layer_freeze(self, layer_idx: int):
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].frozen = True

    def layer_thaw(self, layer_idx: int):
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].frozen = False

    def layer_thaw_all(self):
        for l in self.layers:
            l.frozen = False

    def layer_isolate(self, layer_idx: int):
        """Show only the specified layer."""
        for i, l in enumerate(self.layers):
            l.visible = (i == layer_idx)

    # ─── Viewport Commands ──────────────────────────────────────────────

    def zoom_in(self):
        vp = self.config["viewport"]
        self.viewport.zoom = min(self.viewport.zoom * vp["zoom_step"], vp["max_zoom"])

    def zoom_out(self):
        vp = self.config["viewport"]
        self.viewport.zoom = max(self.viewport.zoom / vp["zoom_step"], vp["min_zoom"])

    def zoom_extents(self):
        """Fit all entities in view."""
        if not self.entities:
            return
        all_min = np.array([np.inf, np.inf])
        all_max = np.array([-np.inf, -np.inf])
        for e in self.entities:
            bb_min, bb_max = e.bbox()
            all_min = np.minimum(all_min, bb_min)
            all_max = np.maximum(all_max, bb_max)
        if np.any(np.isinf(all_min)):
            return
        self.viewport.center = (all_min + all_max) / 2.0
        extent = all_max - all_min
        margin = 1.1  # 10% margin
        zoom_x = self.viewport.width / max(extent[0] * margin, 1e-6)
        zoom_y = self.viewport.height / max(extent[1] * margin, 1e-6)
        self.viewport.zoom = min(zoom_x, zoom_y)

    def fit_view(self):
        """Alias for zoom_extents with 5% margin."""
        if not self.entities:
            return
        all_min = np.array([np.inf, np.inf])
        all_max = np.array([-np.inf, -np.inf])
        for e in self.entities:
            bb_min, bb_max = e.bbox()
            all_min = np.minimum(all_min, bb_min)
            all_max = np.maximum(all_max, bb_max)
        if np.any(np.isinf(all_min)):
            return
        self.viewport.center = (all_min + all_max) / 2.0
        extent = all_max - all_min
        margin = 1.05  # 5% margin as per spec
        zoom_x = self.viewport.width / max(extent[0] * margin, 1e-6)
        zoom_y = self.viewport.height / max(extent[1] * margin, 1e-6)
        self.viewport.zoom = min(zoom_x, zoom_y)

    def pan(self, dx_frac: float, dy_frac: float):
        """Pan by fraction of visible area."""
        vis_min, vis_max = self.viewport.visible_bounds()
        extent = vis_max - vis_min
        self.viewport.center[0] += dx_frac * extent[0]
        self.viewport.center[1] += dy_frac * extent[1]

    # ─── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire engine state."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "active_layer": self.active_layer,
            "active_color": self.active_color,
            "active_linetype": self.active_linetype,
            "viewport": {
                "center": self.viewport.center.tolist(),
                "zoom": self.viewport.zoom,
            },
            "layers": [
                {"name": l.name, "visible": l.visible, "frozen": l.frozen,
                 "color_index": l.color_index, "linetype": l.linetype}
                for l in self.layers
            ],
        }

    def entity_count(self) -> int:
        return len(self.entities)

    def selected_count(self) -> int:
        return len(self.selected_ids)

    def clear(self):
        """Reset engine to empty state."""
        self.save_undo()
        self.entities.clear()
        self.selected_ids.clear()
        self.ghost_entities.clear()
        self.pending_points.clear()
        self.active_tool = "NOOP"

    def reset(self):
        """Full reset including undo history (for new episodes)."""
        self.entities.clear()
        self.selected_ids.clear()
        self.ghost_entities.clear()
        self.pending_points.clear()
        self.active_tool = "NOOP"
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.active_layer = self.config["layers"]["default_layer"]
        self.active_color = 0
        self.active_linetype = "CONTINUOUS"
        vp_cfg = self.config["viewport"]
        self.viewport.center = np.array(vp_cfg["default_center"], dtype=np.float64)
        self.viewport.zoom = vp_cfg["default_zoom"]
        for l in self.layers:
            l.visible = True
            l.frozen = False
