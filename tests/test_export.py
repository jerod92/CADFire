"""Tests for DXF export."""

import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cadfire.engine.cad_engine import CADEngine
from cadfire.export.dxf_writer import DXFWriter


class TestDXFExport:
    def test_export_empty(self):
        engine = CADEngine()
        writer = DXFWriter()
        dxf = writer.to_string(engine)
        assert "SECTION" in dxf
        assert "EOF" in dxf

    def test_export_line(self):
        engine = CADEngine()
        engine.draw_line(np.array([0, 0]), np.array([100, 100]))
        writer = DXFWriter()
        dxf = writer.to_string(engine)
        assert "LINE" in dxf
        assert "100.000000" in dxf

    def test_export_circle(self):
        engine = CADEngine()
        engine.draw_circle(np.array([500, 500]), 200)
        writer = DXFWriter()
        dxf = writer.to_string(engine)
        assert "CIRCLE" in dxf
        assert "200.000000" in dxf

    def test_export_all_types(self):
        engine = CADEngine()
        engine.draw_line(np.array([0, 0]), np.array([100, 100]))
        engine.draw_circle(np.array([200, 200]), 50)
        engine.draw_arc(np.array([300, 300]), 40, 0, 90)
        engine.draw_rectangle(np.array([400, 400]), 100, 80)
        engine.draw_point(np.array([500, 500]))
        engine.draw_text(np.array([600, 600]), "Hello", height=12)

        writer = DXFWriter()
        dxf = writer.to_string(engine)
        assert "LINE" in dxf
        assert "CIRCLE" in dxf
        assert "ARC" in dxf
        assert "LWPOLYLINE" in dxf
        assert "POINT" in dxf
        assert "TEXT" in dxf

    def test_write_file(self):
        engine = CADEngine()
        engine.draw_circle(np.array([500, 500]), 100)
        writer = DXFWriter()

        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
            writer.write(engine, f.name)
            assert Path(f.name).stat().st_size > 0

    def test_layers_in_dxf(self):
        engine = CADEngine()
        engine.set_layer(3)
        engine.draw_circle(np.array([500, 500]), 100)
        writer = DXFWriter()
        dxf = writer.to_string(engine)
        assert "LAYER" in dxf
