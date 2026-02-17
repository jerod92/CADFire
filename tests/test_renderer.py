"""Tests for the renderer."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cadfire.engine.cad_engine import CADEngine
from cadfire.renderer.rasterizer import Renderer


class TestRenderer:
    def setup_method(self):
        self.engine = CADEngine()
        self.renderer = Renderer()

    def test_render_empty(self):
        obs = self.renderer.render(self.engine)
        assert obs.shape == (128, 128, self.renderer.num_channels())
        assert obs.dtype == np.float32
        # Background should be black (0,0,0) normalized
        assert np.allclose(obs[:, :, 0:3], 0.0)

    def test_render_with_entity(self):
        self.engine.draw_circle(np.array([500.0, 500.0]), 200)
        obs = self.renderer.render(self.engine)
        # Some pixels in the viewport should be non-zero
        assert obs[:, :, 0:3].max() > 0

    def test_render_rgb_only(self):
        self.engine.draw_line(np.array([0.0, 0.0]), np.array([1000.0, 1000.0]))
        rgb = self.renderer.render_rgb_only(self.engine)
        assert rgb.shape == (128, 128, 3)
        assert rgb.dtype == np.uint8

    def test_num_channels(self):
        # 3 (viewport) + 3 (reference) + 8 (layers) + 1 (selection) = 15
        assert self.renderer.num_channels() == 15

    def test_layer_masks(self):
        self.engine.set_layer(2)
        self.engine.draw_circle(np.array([500.0, 500.0]), 200)
        obs = self.renderer.render(self.engine)
        # Layer 2 mask should have some non-zero pixels
        layer_mask = obs[:, :, 8]  # channel 6 + 2
        assert layer_mask.max() > 0

    def test_selection_mask(self):
        eid = self.engine.draw_circle(np.array([500.0, 500.0]), 200)
        self.engine.select(eid)
        obs = self.renderer.render(self.engine)
        sel_mask = obs[:, :, -1]
        assert sel_mask.max() > 0

    def test_reference_image(self):
        ref = np.ones((64, 64, 3), dtype=np.uint8) * 128
        obs = self.renderer.render(self.engine, reference_image=ref)
        # Reference channels should be non-zero
        ref_channels = obs[:, :, 3:6]
        assert ref_channels.max() > 0
