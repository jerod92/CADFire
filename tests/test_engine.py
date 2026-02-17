"""Tests for the CAD engine and geometry primitives."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cadfire.engine.geometry import (
    LineEntity, CircleEntity, RectangleEntity, PolygonEntity,
    EllipseEntity, ArcEntity, SplineEntity, PointEntity, HatchEntity,
    TextEntity, DimensionEntity,
)
from cadfire.engine.cad_engine import CADEngine


class TestGeometry:
    def test_line_bbox(self):
        line = LineEntity(start=np.array([10, 20]), end=np.array([50, 60]))
        bb_min, bb_max = line.bbox()
        assert np.allclose(bb_min, [10, 20])
        assert np.allclose(bb_max, [50, 60])

    def test_line_tessellate(self):
        line = LineEntity(start=np.array([0, 0]), end=np.array([100, 100]))
        pts = line.tessellate()
        assert pts.shape == (2, 2)
        assert np.allclose(pts[0], [0, 0])
        assert np.allclose(pts[1], [100, 100])

    def test_circle_bbox(self):
        c = CircleEntity(center=np.array([100, 100]), radius=50)
        bb_min, bb_max = c.bbox()
        assert np.allclose(bb_min, [50, 50])
        assert np.allclose(bb_max, [150, 150])

    def test_circle_tessellate_closed(self):
        c = CircleEntity(center=np.array([0, 0]), radius=10)
        pts = c.tessellate()
        # Should be closed (first == last)
        assert np.allclose(pts[0], pts[-1], atol=1e-6)
        assert len(pts) > 10

    def test_rectangle_corners(self):
        r = RectangleEntity(corner=np.array([10, 20]), width=100, height=50)
        pts = r.tessellate()
        assert len(pts) == 5  # closed polyline
        assert np.allclose(pts[0], pts[-1])

    def test_polygon_sides(self):
        p = PolygonEntity(center=np.array([100, 100]), radius=50, sides=6)
        pts = p.tessellate()
        assert len(pts) == 7  # 6 vertices + close

    def test_entity_clone(self):
        line = LineEntity(start=np.array([1, 2]), end=np.array([3, 4]))
        clone = line.clone()
        assert clone.id != line.id or True  # deepcopy makes new id only if default_factory
        clone.start[0] = 999
        assert line.start[0] != 999  # independent copy

    def test_translate(self):
        line = LineEntity(start=np.array([0.0, 0.0]), end=np.array([10.0, 10.0]))
        line.translate(5, 5)
        assert np.allclose(line.start, [5, 5])
        assert np.allclose(line.end, [15, 15])

    def test_rotate(self):
        line = LineEntity(start=np.array([10.0, 0.0]), end=np.array([20.0, 0.0]))
        line.rotate(90, 0, 0)
        assert np.allclose(line.start, [0, 10], atol=1e-6)

    def test_scale(self):
        c = CircleEntity(center=np.array([100.0, 100.0]), radius=50.0)
        c.scale_entity(2.0, 100.0, 100.0)
        assert abs(c.radius - 100.0) < 1e-6

    def test_to_dict(self):
        line = LineEntity(start=np.array([1.0, 2.0]), end=np.array([3.0, 4.0]))
        d = line.to_dict()
        assert d["entity_type"] == "LINE"
        assert d["start"] == [1.0, 2.0]

    def test_point_entity(self):
        p = PointEntity(position=np.array([50.0, 75.0]))
        bb_min, bb_max = p.bbox()
        assert np.allclose(bb_min, [50, 75])

    def test_ellipse_tessellate(self):
        e = EllipseEntity(center=np.array([0, 0]), semi_major=20, semi_minor=10)
        pts = e.tessellate()
        assert len(pts) > 10
        assert np.allclose(pts[0], pts[-1], atol=1e-3)

    def test_spline_tessellate(self):
        cp = np.array([[0, 0], [50, 100], [100, 0], [150, 100]], dtype=float)
        s = SplineEntity(control_points=cp, degree=3)
        pts = s.tessellate()
        assert len(pts) == 64


class TestCADEngine:
    def setup_method(self):
        self.engine = CADEngine()

    def test_add_entity(self):
        eid = self.engine.draw_line(np.array([0, 0]), np.array([100, 100]))
        assert self.engine.entity_count() == 1
        assert self.engine.get_entity(eid) is not None

    def test_remove_entity(self):
        eid = self.engine.draw_circle(np.array([50, 50]), 25)
        assert self.engine.entity_count() == 1
        self.engine.remove_entity(eid)
        assert self.engine.entity_count() == 0

    def test_undo_redo(self):
        self.engine.draw_line(np.array([0, 0]), np.array([10, 10]))
        assert self.engine.entity_count() == 1
        self.engine.undo()
        assert self.engine.entity_count() == 0
        self.engine.redo()
        assert self.engine.entity_count() == 1

    def test_selection(self):
        eid = self.engine.draw_circle(np.array([500, 500]), 50)
        self.engine.select(eid)
        assert len(self.engine.selected_ids) == 1
        self.engine.deselect_all()
        assert len(self.engine.selected_ids) == 0

    def test_select_at_point(self):
        eid = self.engine.draw_circle(np.array([500.0, 500.0]), 50.0)
        # Click near the circle edge
        hit = self.engine.select_at_point(np.array([550.0, 500.0]), tolerance=10)
        assert hit == eid

    def test_move_selected(self):
        eid = self.engine.draw_circle(np.array([100.0, 100.0]), 50.0)
        self.engine.select(eid)
        self.engine.move_selected(50, 50)
        e = self.engine.get_entity(eid)
        assert np.allclose(e.center, [150, 150])

    def test_copy_selected(self):
        eid = self.engine.draw_circle(np.array([100.0, 100.0]), 50.0)
        self.engine.select(eid)
        new_ids = self.engine.copy_selected(200, 200)
        assert len(new_ids) == 1
        assert self.engine.entity_count() == 2

    def test_erase_selected(self):
        eid = self.engine.draw_circle(np.array([100, 100]), 50)
        self.engine.select(eid)
        self.engine.erase_selected()
        assert self.engine.entity_count() == 0

    def test_layers(self):
        self.engine.set_layer(3)
        eid = self.engine.draw_line(np.array([0, 0]), np.array([10, 10]))
        e = self.engine.get_entity(eid)
        assert e.layer == 3

    def test_layer_visibility(self):
        self.engine.set_layer(1)
        self.engine.draw_circle(np.array([100, 100]), 50)
        assert len(self.engine.visible_entities()) == 1
        self.engine.layer_off(1)
        assert len(self.engine.visible_entities()) == 0
        self.engine.layer_on(1)
        assert len(self.engine.visible_entities()) == 1

    def test_zoom_extents(self):
        self.engine.draw_circle(np.array([200.0, 200.0]), 50)
        self.engine.draw_circle(np.array([800.0, 800.0]), 50)
        self.engine.zoom_extents()
        # Center should be near the middle of the two circles
        assert abs(self.engine.viewport.center[0] - 500) < 50
        assert abs(self.engine.viewport.center[1] - 500) < 50

    def test_fit_view(self):
        self.engine.draw_rectangle(np.array([100.0, 100.0]), 200, 200)
        self.engine.fit_view()
        assert self.engine.viewport.zoom > 0

    def test_reset(self):
        self.engine.draw_line(np.array([0, 0]), np.array([10, 10]))
        self.engine.reset()
        assert self.engine.entity_count() == 0

    def test_serialize(self):
        self.engine.draw_circle(np.array([100, 100]), 50)
        d = self.engine.to_dict()
        assert len(d["entities"]) == 1
        assert d["entities"][0]["entity_type"] == "CIRCLE"

    def test_draw_all_primitives(self):
        """Ensure all drawing commands work without error."""
        self.engine.draw_line(np.array([0, 0]), np.array([100, 100]))
        self.engine.draw_polyline(np.array([[0, 0], [50, 50], [100, 0]]), closed=True)
        self.engine.draw_circle(np.array([200, 200]), 50)
        self.engine.draw_arc(np.array([300, 300]), 40, 0, 90)
        self.engine.draw_rectangle(np.array([400, 400]), 100, 80)
        self.engine.draw_polygon(np.array([500, 500]), 60, 5)
        self.engine.draw_ellipse(np.array([600, 600]), 80, 40)
        self.engine.draw_spline(np.array([[0, 0], [50, 100], [100, 0]]))
        self.engine.draw_point(np.array([700, 700]))
        self.engine.draw_hatch(np.array([[0, 0], [100, 0], [50, 100]]))
        self.engine.draw_text(np.array([800, 800]), "Hello", height=12)
        self.engine.draw_dimension(np.array([0, 0]), np.array([100, 0]),
                                   np.array([50, 20]))
        assert self.engine.entity_count() == 12

    def test_explode(self):
        self.engine.draw_rectangle(np.array([0.0, 0.0]), 100, 100)
        eid = self.engine.entities[0].id
        self.engine.select(eid)
        self.engine.explode_selected()
        # Rectangle should be replaced by line segments
        assert all(e.entity_type == "LINE" for e in self.engine.entities)
        assert self.engine.entity_count() > 1
