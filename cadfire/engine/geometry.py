"""
Core geometry primitives for the CAD engine.

Every entity is a plain data object with numpy arrays for coordinates.
Entities know how to:
  - Return their bounding box
  - Return points for rasterization (tessellate to polyline segments)
  - Clone themselves (for undo/copy)
  - Serialize to dict (for DXF export and checkpointing)
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Entity:
    """Base class for all CAD entities."""
    entity_type: str = "ENTITY"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    layer: int = 0
    color_index: int = 0        # index into config palette
    linetype: str = "CONTINUOUS"
    lineweight: float = 1.0
    visible: bool = True
    locked: bool = False

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_xy, max_xy) bounding box."""
        raise NotImplementedError

    def tessellate(self, resolution: int = 64) -> np.ndarray:
        """Return Nx2 array of points for rasterization."""
        raise NotImplementedError

    def centroid(self) -> np.ndarray:
        """Return center point."""
        bb_min, bb_max = self.bbox()
        return (bb_min + bb_max) / 2.0

    def clone(self) -> Entity:
        """Deep copy."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for export/checkpoint."""
        return {
            "entity_type": self.entity_type,
            "id": self.id,
            "layer": self.layer,
            "color_index": self.color_index,
            "linetype": self.linetype,
            "lineweight": self.lineweight,
        }

    def transform(self, matrix: np.ndarray) -> None:
        """Apply a 3x3 affine transform in-place. Subclasses override."""
        raise NotImplementedError

    def translate(self, dx: float, dy: float) -> None:
        mat = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        self.transform(mat)

    def rotate(self, angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> None:
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        # translate to origin, rotate, translate back
        mat = np.array([
            [c, -s, cx - c * cx + s * cy],
            [s,  c, cy - s * cx - c * cy],
            [0,  0, 1]
        ], dtype=np.float64)
        self.transform(mat)

    def scale_entity(self, factor: float, cx: float = 0.0, cy: float = 0.0) -> None:
        mat = np.array([
            [factor, 0, cx * (1 - factor)],
            [0, factor, cy * (1 - factor)],
            [0, 0, 1]
        ], dtype=np.float64)
        self.transform(mat)

    def mirror_entity(self, p1: np.ndarray, p2: np.ndarray) -> None:
        """Mirror across line defined by p1->p2."""
        d = p2 - p1
        d = d / (np.linalg.norm(d) + 1e-12)
        # reflection matrix about line through origin with direction d
        # then translate
        a, b = d[0], d[1]
        # Householder-like: R = 2*d*d^T - I (for the 2x2 part)
        # But we need to handle offset. Use translate-reflect-translate.
        T1 = np.array([[1, 0, -p1[0]], [0, 1, -p1[1]], [0, 0, 1]], dtype=np.float64)
        R = np.array([
            [2*a*a - 1, 2*a*b,     0],
            [2*a*b,     2*b*b - 1, 0],
            [0,         0,         1]
        ], dtype=np.float64)
        T2 = np.array([[1, 0, p1[0]], [0, 1, p1[1]], [0, 0, 1]], dtype=np.float64)
        self.transform(T2 @ R @ T1)


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply 3x3 affine to Nx2 points, return Nx2."""
    n = len(points)
    if n == 0:
        return points
    hom = np.ones((n, 3), dtype=np.float64)
    hom[:, :2] = points
    transformed = (matrix @ hom.T).T
    return transformed[:, :2]


@dataclass
class LineEntity(Entity):
    entity_type: str = "LINE"
    start: np.ndarray = field(default_factory=lambda: np.zeros(2))
    end: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def bbox(self):
        pts = np.stack([self.start, self.end])
        return pts.min(axis=0), pts.max(axis=0)

    def tessellate(self, resolution=64):
        return np.stack([self.start, self.end])

    def transform(self, matrix):
        pts = _apply_affine(np.stack([self.start, self.end]), matrix)
        self.start, self.end = pts[0], pts[1]

    def to_dict(self):
        d = super().to_dict()
        d.update({"start": self.start.tolist(), "end": self.end.tolist()})
        return d


@dataclass
class PolylineEntity(Entity):
    entity_type: str = "POLYLINE"
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    closed: bool = False

    def bbox(self):
        if len(self.points) == 0:
            return np.zeros(2), np.zeros(2)
        return self.points.min(axis=0), self.points.max(axis=0)

    def tessellate(self, resolution=64):
        if self.closed and len(self.points) > 2:
            return np.vstack([self.points, self.points[:1]])
        return self.points

    def transform(self, matrix):
        self.points = _apply_affine(self.points, matrix)

    def to_dict(self):
        d = super().to_dict()
        d.update({"points": self.points.tolist(), "closed": self.closed})
        return d


@dataclass
class CircleEntity(Entity):
    entity_type: str = "CIRCLE"
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1.0

    def bbox(self):
        r = np.array([self.radius, self.radius])
        return self.center - r, self.center + r

    def tessellate(self, resolution=64):
        t = np.linspace(0, 2 * np.pi, resolution + 1)
        pts = np.column_stack([
            self.center[0] + self.radius * np.cos(t),
            self.center[1] + self.radius * np.sin(t),
        ])
        return pts

    def transform(self, matrix):
        c = _apply_affine(self.center.reshape(1, 2), matrix)[0]
        # approximate new radius from scale
        edge = self.center + np.array([self.radius, 0.0])
        edge_t = _apply_affine(edge.reshape(1, 2), matrix)[0]
        self.radius = float(np.linalg.norm(edge_t - c))
        self.center = c

    def to_dict(self):
        d = super().to_dict()
        d.update({"center": self.center.tolist(), "radius": self.radius})
        return d


@dataclass
class ArcEntity(Entity):
    entity_type: str = "ARC"
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1.0
    start_angle: float = 0.0   # degrees
    end_angle: float = 90.0    # degrees

    def bbox(self):
        pts = self.tessellate()
        return pts.min(axis=0), pts.max(axis=0)

    def tessellate(self, resolution=64):
        sa = math.radians(self.start_angle)
        ea = math.radians(self.end_angle)
        if ea < sa:
            ea += 2 * math.pi
        t = np.linspace(sa, ea, resolution)
        return np.column_stack([
            self.center[0] + self.radius * np.cos(t),
            self.center[1] + self.radius * np.sin(t),
        ])

    def transform(self, matrix):
        pts = self.tessellate(resolution=4)
        pts = _apply_affine(pts, matrix)
        c = _apply_affine(self.center.reshape(1, 2), matrix)[0]
        edge = self.center + np.array([self.radius, 0.0])
        edge_t = _apply_affine(edge.reshape(1, 2), matrix)[0]
        self.radius = float(np.linalg.norm(edge_t - c))
        self.center = c
        # recalculate angles from transformed start/end
        s = pts[0] - c
        e = pts[-1] - c
        self.start_angle = math.degrees(math.atan2(s[1], s[0]))
        self.end_angle = math.degrees(math.atan2(e[1], e[0]))

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "center": self.center.tolist(), "radius": self.radius,
            "start_angle": self.start_angle, "end_angle": self.end_angle,
        })
        return d


@dataclass
class RectangleEntity(Entity):
    """Stored as corner + width/height for convenience; tessellates to polyline."""
    entity_type: str = "RECTANGLE"
    corner: np.ndarray = field(default_factory=lambda: np.zeros(2))
    width: float = 1.0
    height: float = 1.0

    def _corners(self) -> np.ndarray:
        c = self.corner
        return np.array([
            c,
            c + [self.width, 0],
            c + [self.width, self.height],
            c + [0, self.height],
        ])

    def bbox(self):
        pts = self._corners()
        return pts.min(axis=0), pts.max(axis=0)

    def tessellate(self, resolution=64):
        pts = self._corners()
        return np.vstack([pts, pts[:1]])  # closed

    def transform(self, matrix):
        pts = _apply_affine(self._corners(), matrix)
        self.corner = pts.min(axis=0)
        new_max = pts.max(axis=0)
        self.width = float(new_max[0] - self.corner[0])
        self.height = float(new_max[1] - self.corner[1])

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "corner": self.corner.tolist(),
            "width": self.width, "height": self.height,
        })
        return d


@dataclass
class PolygonEntity(Entity):
    entity_type: str = "POLYGON"
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1.0
    sides: int = 6
    rotation: float = 0.0  # degrees

    def _vertices(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, self.sides + 1)[:-1] + math.radians(self.rotation)
        return np.column_stack([
            self.center[0] + self.radius * np.cos(angles),
            self.center[1] + self.radius * np.sin(angles),
        ])

    def bbox(self):
        pts = self._vertices()
        return pts.min(axis=0), pts.max(axis=0)

    def tessellate(self, resolution=64):
        v = self._vertices()
        return np.vstack([v, v[:1]])

    def transform(self, matrix):
        c = _apply_affine(self.center.reshape(1, 2), matrix)[0]
        edge = self.center + np.array([self.radius, 0.0])
        edge_t = _apply_affine(edge.reshape(1, 2), matrix)[0]
        self.radius = float(np.linalg.norm(edge_t - c))
        self.center = c

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "center": self.center.tolist(), "radius": self.radius,
            "sides": self.sides, "rotation": self.rotation,
        })
        return d


@dataclass
class EllipseEntity(Entity):
    entity_type: str = "ELLIPSE"
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    semi_major: float = 2.0
    semi_minor: float = 1.0
    rotation: float = 0.0  # degrees of major axis

    def bbox(self):
        pts = self.tessellate()
        return pts.min(axis=0), pts.max(axis=0)

    def tessellate(self, resolution=64):
        t = np.linspace(0, 2 * np.pi, resolution + 1)
        r = math.radians(self.rotation)
        cos_r, sin_r = math.cos(r), math.sin(r)
        x = self.semi_major * np.cos(t)
        y = self.semi_minor * np.sin(t)
        return np.column_stack([
            self.center[0] + cos_r * x - sin_r * y,
            self.center[1] + sin_r * x + cos_r * y,
        ])

    def transform(self, matrix):
        c = _apply_affine(self.center.reshape(1, 2), matrix)[0]
        edge = self.center + np.array([self.semi_major, 0.0])
        edge_t = _apply_affine(edge.reshape(1, 2), matrix)[0]
        self.semi_major = float(np.linalg.norm(edge_t - c))
        edge2 = self.center + np.array([0.0, self.semi_minor])
        edge2_t = _apply_affine(edge2.reshape(1, 2), matrix)[0]
        self.semi_minor = float(np.linalg.norm(edge2_t - c))
        self.center = c

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "center": self.center.tolist(),
            "semi_major": self.semi_major, "semi_minor": self.semi_minor,
            "rotation": self.rotation,
        })
        return d


@dataclass
class SplineEntity(Entity):
    entity_type: str = "SPLINE"
    control_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    degree: int = 3

    def bbox(self):
        pts = self.tessellate()
        if len(pts) == 0:
            return np.zeros(2), np.zeros(2)
        return pts.min(axis=0), pts.max(axis=0)

    def tessellate(self, resolution=64):
        cp = self.control_points
        n = len(cp)
        if n < 2:
            return cp
        # Catmull-Rom-style interpolation through control points
        t_vals = np.linspace(0, 1, resolution)
        # Simple cubic interpolation between segments
        if n == 2:
            return np.column_stack([
                np.interp(t_vals, [0, 1], cp[:, 0]),
                np.interp(t_vals, [0, 1], cp[:, 1]),
            ])
        # Use numpy polyfit per-axis for smooth curve
        param = np.linspace(0, 1, n)
        t_fine = np.linspace(0, 1, resolution)
        deg = min(self.degree, n - 1)
        cx = np.polyfit(param, cp[:, 0], deg)
        cy = np.polyfit(param, cp[:, 1], deg)
        return np.column_stack([np.polyval(cx, t_fine), np.polyval(cy, t_fine)])

    def transform(self, matrix):
        self.control_points = _apply_affine(self.control_points, matrix)

    def to_dict(self):
        d = super().to_dict()
        d.update({"control_points": self.control_points.tolist(), "degree": self.degree})
        return d


@dataclass
class PointEntity(Entity):
    entity_type: str = "POINT"
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def bbox(self):
        return self.position.copy(), self.position.copy()

    def tessellate(self, resolution=64):
        return self.position.reshape(1, 2)

    def transform(self, matrix):
        self.position = _apply_affine(self.position.reshape(1, 2), matrix)[0]

    def to_dict(self):
        d = super().to_dict()
        d.update({"position": self.position.tolist()})
        return d


@dataclass
class HatchEntity(Entity):
    """Filled region defined by a boundary polyline."""
    entity_type: str = "HATCH"
    boundary: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pattern: str = "SOLID"

    def bbox(self):
        if len(self.boundary) == 0:
            return np.zeros(2), np.zeros(2)
        return self.boundary.min(axis=0), self.boundary.max(axis=0)

    def tessellate(self, resolution=64):
        if len(self.boundary) < 3:
            return self.boundary
        return np.vstack([self.boundary, self.boundary[:1]])

    def transform(self, matrix):
        self.boundary = _apply_affine(self.boundary, matrix)

    def to_dict(self):
        d = super().to_dict()
        d.update({"boundary": self.boundary.tolist(), "pattern": self.pattern})
        return d


@dataclass
class TextEntity(Entity):
    entity_type: str = "TEXT"
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    text: str = ""
    height: float = 10.0
    rotation: float = 0.0
    multiline: bool = False

    def bbox(self):
        # approximate bbox based on text length and height
        w = len(self.text) * self.height * 0.6
        h = self.height
        if self.multiline:
            lines = self.text.count('\n') + 1
            h *= lines
        return self.position.copy(), self.position + np.array([w, h])

    def tessellate(self, resolution=64):
        # text is not tessellated for rasterization the same way
        # return bounding box corners for rendering purposes
        bb_min, bb_max = self.bbox()
        return np.array([bb_min, [bb_max[0], bb_min[1]], bb_max, [bb_min[0], bb_max[1]], bb_min])

    def transform(self, matrix):
        self.position = _apply_affine(self.position.reshape(1, 2), matrix)[0]

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "position": self.position.tolist(), "text": self.text,
            "height": self.height, "rotation": self.rotation,
            "multiline": self.multiline,
        })
        return d


@dataclass
class DimensionEntity(Entity):
    """Generic dimension entity (linear, aligned, angular, radial, diameter)."""
    entity_type: str = "DIMENSION"
    dim_type: str = "LINEAR"  # LINEAR, ALIGNED, ANGULAR, RADIUS, DIAMETER
    point1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    point2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    text_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    measurement: float = 0.0
    text_override: str = ""

    def bbox(self):
        pts = np.stack([self.point1, self.point2, self.text_position])
        return pts.min(axis=0), pts.max(axis=0)

    def tessellate(self, resolution=64):
        # dimension lines: two extension lines + dimension line
        return np.stack([self.point1, self.point2])

    def transform(self, matrix):
        pts = _apply_affine(np.stack([self.point1, self.point2, self.text_position]), matrix)
        self.point1, self.point2, self.text_position = pts[0], pts[1], pts[2]

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "dim_type": self.dim_type,
            "point1": self.point1.tolist(), "point2": self.point2.tolist(),
            "text_position": self.text_position.tolist(),
            "measurement": self.measurement, "text_override": self.text_override,
        })
        return d


# Registry of entity types for deserialization
ENTITY_TYPES = {
    "LINE": LineEntity,
    "POLYLINE": PolylineEntity,
    "CIRCLE": CircleEntity,
    "ARC": ArcEntity,
    "RECTANGLE": RectangleEntity,
    "POLYGON": PolygonEntity,
    "ELLIPSE": EllipseEntity,
    "SPLINE": SplineEntity,
    "POINT": PointEntity,
    "HATCH": HatchEntity,
    "TEXT": TextEntity,
    "DIMENSION": DimensionEntity,
}
