"""
DXF export: converts CADEngine state to a standard DXF file.

Produces DXF R2010 compatible files that can be opened in AutoCAD,
LibreCAD, FreeCAD, and other CAD software.

The DXF format is text-based with group codes. We write it directly
without any external dependencies.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import (
    Entity, LineEntity, PolylineEntity, CircleEntity, ArcEntity,
    RectangleEntity, PolygonEntity, EllipseEntity, SplineEntity,
    PointEntity, HatchEntity, TextEntity, DimensionEntity,
)

# DXF color index (ACI) mapping from our palette indices
# AutoCAD Color Index: 1=red, 2=yellow, 3=green, 4=cyan, 5=blue, 6=magenta, 7=white
ACI_MAP = {0: 7, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8}


class DXFWriter:
    """Write CAD engine state to DXF format."""

    def __init__(self):
        self._lines: List[str] = []

    def _gc(self, code: int, value: str):
        """Write a group code + value pair."""
        self._lines.append(f"{code:>3d}")
        self._lines.append(str(value))

    def _header(self):
        """Write DXF header section."""
        self._gc(0, "SECTION")
        self._gc(2, "HEADER")
        self._gc(9, "$ACADVER")
        self._gc(1, "AC1024")  # R2010
        self._gc(9, "$INSUNITS")
        self._gc(70, "4")  # millimeters
        self._gc(0, "ENDSEC")

    def _tables(self, engine: CADEngine):
        """Write tables section (layers, linetypes)."""
        self._gc(0, "SECTION")
        self._gc(2, "TABLES")

        # Linetype table
        self._gc(0, "TABLE")
        self._gc(2, "LTYPE")
        self._gc(70, "1")
        # CONTINUOUS
        self._gc(0, "LTYPE")
        self._gc(2, "CONTINUOUS")
        self._gc(70, "0")
        self._gc(3, "Solid line")
        self._gc(72, "65")
        self._gc(73, "0")
        self._gc(40, "0.0")
        self._gc(0, "ENDTAB")

        # Layer table
        self._gc(0, "TABLE")
        self._gc(2, "LAYER")
        self._gc(70, str(len(engine.layers)))
        for layer in engine.layers:
            self._gc(0, "LAYER")
            self._gc(2, layer.name)
            self._gc(70, "0" if not layer.frozen else "1")
            self._gc(62, str(ACI_MAP.get(layer.color_index, 7)))
            self._gc(6, layer.linetype)
        self._gc(0, "ENDTAB")

        self._gc(0, "ENDSEC")

    def _entity(self, e: Entity):
        """Write a single entity."""
        layer_name = str(e.layer)
        aci = ACI_MAP.get(e.color_index, 7)

        if isinstance(e, LineEntity):
            self._gc(0, "LINE")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(10, f"{e.start[0]:.6f}")
            self._gc(20, f"{e.start[1]:.6f}")
            self._gc(30, "0.0")
            self._gc(11, f"{e.end[0]:.6f}")
            self._gc(21, f"{e.end[1]:.6f}")
            self._gc(31, "0.0")

        elif isinstance(e, (PolylineEntity, RectangleEntity, PolygonEntity)):
            pts = e.tessellate()
            closed = getattr(e, "closed", True)
            self._gc(0, "LWPOLYLINE")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            n = len(pts) - 1 if closed and len(pts) > 1 else len(pts)
            self._gc(90, str(n))
            self._gc(70, "1" if closed else "0")
            for i in range(n):
                self._gc(10, f"{pts[i][0]:.6f}")
                self._gc(20, f"{pts[i][1]:.6f}")

        elif isinstance(e, CircleEntity):
            self._gc(0, "CIRCLE")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(10, f"{e.center[0]:.6f}")
            self._gc(20, f"{e.center[1]:.6f}")
            self._gc(30, "0.0")
            self._gc(40, f"{e.radius:.6f}")

        elif isinstance(e, ArcEntity):
            self._gc(0, "ARC")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(10, f"{e.center[0]:.6f}")
            self._gc(20, f"{e.center[1]:.6f}")
            self._gc(30, "0.0")
            self._gc(40, f"{e.radius:.6f}")
            self._gc(50, f"{e.start_angle:.6f}")
            self._gc(51, f"{e.end_angle:.6f}")

        elif isinstance(e, EllipseEntity):
            self._gc(0, "ELLIPSE")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(10, f"{e.center[0]:.6f}")
            self._gc(20, f"{e.center[1]:.6f}")
            self._gc(30, "0.0")
            # Major axis endpoint (relative to center)
            rad = math.radians(e.rotation)
            self._gc(11, f"{e.semi_major * math.cos(rad):.6f}")
            self._gc(21, f"{e.semi_major * math.sin(rad):.6f}")
            self._gc(31, "0.0")
            # Ratio of minor to major
            self._gc(40, f"{e.semi_minor / max(e.semi_major, 1e-6):.6f}")
            self._gc(41, "0.0")  # start param
            self._gc(42, f"{2 * math.pi:.6f}")  # end param

        elif isinstance(e, SplineEntity):
            # Export as polyline approximation
            pts = e.tessellate()
            self._gc(0, "LWPOLYLINE")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(90, str(len(pts)))
            self._gc(70, "0")
            for pt in pts:
                self._gc(10, f"{pt[0]:.6f}")
                self._gc(20, f"{pt[1]:.6f}")

        elif isinstance(e, PointEntity):
            self._gc(0, "POINT")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(10, f"{e.position[0]:.6f}")
            self._gc(20, f"{e.position[1]:.6f}")
            self._gc(30, "0.0")

        elif isinstance(e, TextEntity):
            self._gc(0, "TEXT" if not e.multiline else "MTEXT")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(10, f"{e.position[0]:.6f}")
            self._gc(20, f"{e.position[1]:.6f}")
            self._gc(30, "0.0")
            self._gc(40, f"{e.height:.6f}")
            self._gc(1, e.text)
            if e.rotation != 0:
                self._gc(50, f"{e.rotation:.6f}")

        elif isinstance(e, DimensionEntity):
            # Simplified dimension export
            self._gc(0, "DIMENSION")
            self._gc(8, layer_name)
            self._gc(62, str(aci))
            self._gc(10, f"{e.text_position[0]:.6f}")
            self._gc(20, f"{e.text_position[1]:.6f}")
            self._gc(30, "0.0")
            self._gc(13, f"{e.point1[0]:.6f}")
            self._gc(23, f"{e.point1[1]:.6f}")
            self._gc(33, "0.0")
            self._gc(14, f"{e.point2[0]:.6f}")
            self._gc(24, f"{e.point2[1]:.6f}")
            self._gc(34, "0.0")
            self._gc(1, e.text_override if e.text_override else f"{e.measurement:.2f}")

    def _entities_section(self, engine: CADEngine):
        """Write entities section."""
        self._gc(0, "SECTION")
        self._gc(2, "ENTITIES")
        for e in engine.entities:
            self._entity(e)
        self._gc(0, "ENDSEC")

    def write(self, engine: CADEngine, path: str):
        """Write complete DXF file."""
        self._lines = []
        self._header()
        self._tables(engine)
        self._entities_section(engine)
        self._gc(0, "EOF")

        with open(path, "w") as f:
            f.write("\n".join(self._lines) + "\n")

    def to_string(self, engine: CADEngine) -> str:
        """Return DXF content as string."""
        self._lines = []
        self._header()
        self._tables(engine)
        self._entities_section(engine)
        self._gc(0, "EOF")
        return "\n".join(self._lines) + "\n"
