"""Tests for cadfire/training/supervised_diagnostics.py"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cadfire.utils.config import load_config, tool_to_index
from cadfire.engine.cad_engine import CADEngine
from cadfire.renderer.rasterizer import Renderer
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.training.supervised_diagnostics import (
    _heatmap_to_rgb,
    _layer_composite_rgb,
    _resize_nearest,
    _draw_cross,
    _make_label_bar,
    _make_text_panel,
    _build_state_vec,
    _decode_state_vec,
    _decode_prompt,
    _build_sample_frame,
    _gen_phase1_grid,
    _gen_phase2_samples,
    _gen_phase3_trajectories,
    generate_supervised_diagnostics,
)


# ── Image helper tests ─────────────────────────────────────────────────────────

class TestImageHelpers:
    def test_heatmap_to_rgb_shape(self):
        h = np.random.rand(32, 32).astype(np.float32)
        rgb = _heatmap_to_rgb(h)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_heatmap_to_rgb_range(self):
        h = np.zeros((10, 10), dtype=np.float32)
        rgb = _heatmap_to_rgb(h)
        assert rgb.min() >= 0 and rgb.max() <= 255

        h2 = np.ones((10, 10), dtype=np.float32)
        rgb2 = _heatmap_to_rgb(h2)
        assert rgb2.min() >= 0 and rgb2.max() <= 255

    def test_heatmap_clips_out_of_range(self):
        h = np.array([[-1.0, 2.0]], dtype=np.float32)
        rgb = _heatmap_to_rgb(h)
        assert rgb.shape == (1, 2, 3)

    def test_layer_composite_rgb_shape(self):
        config = load_config()
        n_layers = config["layers"]["max_layers"]
        C = 6 + n_layers + 1 + 4
        image = np.random.rand(32, 32, C).astype(np.float32)
        result = _layer_composite_rgb(image, n_layers)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_layer_composite_rgb_no_layers(self):
        # All layer channels zero → black composite
        config = load_config()
        n_layers = config["layers"]["max_layers"]
        C = 6 + n_layers + 1 + 4
        image = np.zeros((16, 16, C), dtype=np.float32)
        result = _layer_composite_rgb(image, n_layers)
        assert result.shape == (16, 16, 3)
        assert result.max() == 0

    def test_resize_nearest_3d(self):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out = _resize_nearest(arr, 32, 32)
        assert out.shape == (32, 32, 3)

    def test_resize_nearest_2d(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        out = _resize_nearest(arr, 16, 16)
        assert out.shape == (16, 16)

    def test_draw_cross(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        out = _draw_cross(rgb, 16, 16, color=(255, 0, 0), size=3)
        assert out.shape == (32, 32, 3)
        # Cross pixels should be red
        assert out[16, 16, 0] == 255

    def test_draw_cross_no_mutation(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        _draw_cross(rgb, 16, 16)
        # Original should be unchanged (draw_cross returns a copy)
        assert rgb[16, 16, 0] == 0


# ── Text helper tests ──────────────────────────────────────────────────────────

class TestTextHelpers:
    def test_make_label_bar_shape(self):
        bar = _make_label_bar("test", W=200, bar_h=20)
        assert bar.shape == (20, 200, 3)
        assert bar.dtype == np.uint8

    def test_make_label_bar_zero_width(self):
        # Should not raise
        bar = _make_label_bar("x", W=1, bar_h=10)
        assert bar.shape == (10, 1, 3)

    def test_make_text_panel_shape(self):
        lines = [("Hello", (255, 255, 255)), ("World", (200, 200, 0))]
        panel = _make_text_panel(lines, W=200, H=100)
        assert panel.shape == (100, 200, 3)
        assert panel.dtype == np.uint8

    def test_make_text_panel_empty_lines(self):
        panel = _make_text_panel([], W=100, H=50)
        assert panel.shape == (50, 100, 3)


# ── State vector tests ─────────────────────────────────────────────────────────

class TestStateVec:
    def test_build_state_vec_shape(self):
        config = load_config()
        engine = CADEngine(config)
        tool_idx = tool_to_index()
        vec = _build_state_vec(engine, tool_idx, config)
        assert vec.shape == (config["model"]["state_dim"],)
        assert vec.dtype == np.float32

    def test_build_state_vec_range(self):
        config = load_config()
        engine = CADEngine(config)
        tool_idx = tool_to_index()
        vec = _build_state_vec(engine, tool_idx, config)
        # All values should be in [0, 1] (normalised)
        assert vec.min() >= 0.0
        assert vec.max() <= 1.0

    def test_decode_state_vec_fields(self):
        config = load_config()
        vec = np.zeros(config["model"]["state_dim"], dtype=np.float32)
        decoded = _decode_state_vec(vec, config)
        assert "active_tool" in decoded
        assert "zoom" in decoded
        assert "viewport" in decoded
        assert "entities" in decoded
        assert "selected" in decoded
        assert "pending_pts" in decoded

    def test_decode_state_vec_roundtrip(self):
        config = load_config()
        engine = CADEngine(config)
        tool_idx = tool_to_index()
        # Add a few entities so counts are non-zero
        from cadfire.engine.geometry import CircleEntity
        engine.add_entity(CircleEntity(
            center=np.array([500.0, 500.0]), radius=80.0, color_index=1
        ))
        engine.add_entity(CircleEntity(
            center=np.array([300.0, 300.0]), radius=50.0, color_index=2
        ))
        vec = _build_state_vec(engine, tool_idx, config)
        decoded = _decode_state_vec(vec, config)
        entities = int(decoded["entities"])
        assert entities == 2


# ── Prompt decode tests ────────────────────────────────────────────────────────

class TestDecodePrompt:
    def test_decode_simple(self):
        config = load_config()
        tokenizer = BPETokenizer(
            vocab_size=config["model"]["text_vocab_size"],
            max_len=config["model"]["text_max_len"],
        )
        prompt = "Draw a circle"
        ids = np.array(tokenizer.encode_padded(prompt), dtype=np.int32)
        decoded = _decode_prompt(ids, tokenizer)
        # Should reconstruct the original text (may differ slightly in spacing)
        assert "circle" in decoded.lower()

    def test_decode_multiturn(self):
        config = load_config()
        tokenizer = BPETokenizer(
            vocab_size=config["model"]["text_vocab_size"],
            max_len=config["model"]["text_max_len"],
        )
        prompt = "Draw a circle | make it smaller"
        ids = np.array(tokenizer.encode_padded(prompt), dtype=np.int32)
        decoded = _decode_prompt(ids, tokenizer)
        # Both turns should be recoverable
        assert "circle" in decoded.lower()
        assert "smaller" in decoded.lower()

    def test_decode_pad_tokens(self):
        config = load_config()
        tokenizer = BPETokenizer(
            vocab_size=config["model"]["text_vocab_size"],
            max_len=config["model"]["text_max_len"],
        )
        # All-zero (PAD) ids should not crash
        ids = np.zeros(config["model"]["text_max_len"], dtype=np.int32)
        decoded = _decode_prompt(ids, tokenizer)
        assert isinstance(decoded, str)


# ── Frame builder tests ────────────────────────────────────────────────────────

class TestBuildSampleFrame:
    """Tests for the core frame-building function."""

    def _make_sample(self, config, seed=42):
        """Helper: produce a minimal valid sample dict."""
        H = config["canvas"]["render_height"]
        W = config["canvas"]["render_width"]
        n_layers = config["layers"]["max_layers"]
        C = 6 + n_layers + 1 + 4  # 19 channels

        rng = np.random.RandomState(seed)
        image = rng.rand(H, W, C).astype(np.float32)
        cursor_mask = np.zeros((H, W), dtype=np.float32)
        cursor_mask[H // 2, W // 2] = 1.0

        tokenizer = BPETokenizer(
            vocab_size=config["model"]["text_vocab_size"],
            max_len=config["model"]["text_max_len"],
        )
        text_ids = np.array(
            tokenizer.encode_padded("Select the circle"), dtype=np.int32
        )
        state_vec = np.zeros(config["model"]["state_dim"], dtype=np.float32)
        return image, cursor_mask, text_ids, state_vec, tokenizer

    def test_frame_shape(self):
        config = load_config()
        image, cursor_mask, text_ids, state_vec, tokenizer = self._make_sample(config)
        frame = _build_sample_frame(
            image=image,
            cursor_mask=cursor_mask,
            text_ids=text_ids,
            state_vec=state_vec,
            oracle_tool="SELECT",
            cursor_weight=1.0,
            task_name="SemanticSelectTask",
            tokenizer=tokenizer,
            config=config,
            panel_size=64,
        )
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8

    def test_frame_positive_dimensions(self):
        config = load_config()
        image, cursor_mask, text_ids, state_vec, tokenizer = self._make_sample(config)
        frame = _build_sample_frame(
            image=image,
            cursor_mask=cursor_mask,
            text_ids=text_ids,
            state_vec=state_vec,
            oracle_tool="MULTISELECT",
            cursor_weight=1.0,
            task_name="SemanticMultiSelectTask",
            tokenizer=tokenizer,
            config=config,
            panel_size=64,
        )
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0

    def test_frame_with_step_info(self):
        config = load_config()
        image, cursor_mask, text_ids, state_vec, tokenizer = self._make_sample(config)
        frame = _build_sample_frame(
            image=image,
            cursor_mask=cursor_mask,
            text_ids=text_ids,
            state_vec=state_vec,
            oracle_tool="POLYLINE",
            cursor_weight=1.0,
            task_name="Phase3-TeacherForcing",
            tokenizer=tokenizer,
            config=config,
            panel_size=64,
            step_idx=2,
            n_steps=6,
        )
        assert frame.ndim == 3

    def test_frame_empty_cursor_mask(self):
        """Zero cursor mask (tool-only action) should not crash."""
        config = load_config()
        image, _, text_ids, state_vec, tokenizer = self._make_sample(config)
        H = config["canvas"]["render_height"]
        W = config["canvas"]["render_width"]
        zero_cursor = np.zeros((H, W), dtype=np.float32)
        frame = _build_sample_frame(
            image=image,
            cursor_mask=zero_cursor,
            text_ids=text_ids,
            state_vec=state_vec,
            oracle_tool="ERASE",
            cursor_weight=0.05,
            task_name="EraseFromChatTask",
            tokenizer=tokenizer,
            config=config,
            panel_size=64,
        )
        assert frame.ndim == 3


# ── Phase generator tests ──────────────────────────────────────────────────────

class TestPhaseGenerators:
    def test_phase1_grid(self, tmp_path):
        config = load_config()
        _gen_phase1_grid(config, tmp_path, verbose=False)
        candidates = list(tmp_path.rglob("*.png")) + list(tmp_path.rglob("*.npy"))
        assert len(candidates) >= 1

    def test_phase2_samples_minimal(self, tmp_path):
        """Phase 2: 1 sample per task, small panel."""
        config = load_config()
        counts = _gen_phase2_samples(
            config=config,
            n_per_task=1,
            output_dir=tmp_path,
            seed=0,
            panel_size=64,
            verbose=False,
        )
        # Should generate at least some task outputs
        assert len(counts) > 0
        total = sum(counts.values())
        assert total > 0

    def test_phase2_output_files(self, tmp_path):
        """Phase 2: output files must exist."""
        config = load_config()
        _gen_phase2_samples(
            config=config,
            n_per_task=1,
            output_dir=tmp_path,
            seed=1,
            panel_size=64,
            verbose=False,
        )
        phase2_dir = tmp_path / "phase2"
        assert phase2_dir.exists()
        outputs = list(phase2_dir.glob("*.png")) + list(phase2_dir.glob("*.npy"))
        assert len(outputs) > 0

    def test_phase3_trajectories_minimal(self, tmp_path):
        """Phase 3: 1 trajectory, small panel."""
        config = load_config()
        saved = _gen_phase3_trajectories(
            config=config,
            n_trajectories=1,
            output_dir=tmp_path,
            seed=0,
            panel_size=64,
            verbose=False,
        )
        assert saved >= 1
        phase3_dir = tmp_path / "phase3"
        assert phase3_dir.exists()
        outputs = list(phase3_dir.glob("*.png")) + list(phase3_dir.glob("*.npy"))
        assert len(outputs) >= 1


# ── Integration tests ──────────────────────────────────────────────────────────

class TestGenerateSupervisedDiagnostics:
    def test_phase1_only(self, tmp_path):
        config = load_config()
        results = generate_supervised_diagnostics(
            config=config,
            output_dir=str(tmp_path),
            phases=(1,),
            verbose=False,
        )
        assert "phase1" in results

    def test_phase2_only(self, tmp_path):
        config = load_config()
        results = generate_supervised_diagnostics(
            config=config,
            output_dir=str(tmp_path),
            n_per_task=1,
            phases=(2,),
            panel_size=64,
            verbose=False,
        )
        assert "phase2" in results
        assert results["phase2"]["task_count"] > 0

    def test_phase3_only(self, tmp_path):
        config = load_config()
        results = generate_supervised_diagnostics(
            config=config,
            output_dir=str(tmp_path),
            n_trajectories=1,
            phases=(3,),
            panel_size=64,
            verbose=False,
        )
        assert "phase3" in results
        assert results["phase3"]["saved"] >= 1

    def test_all_phases(self, tmp_path):
        config = load_config()
        results = generate_supervised_diagnostics(
            config=config,
            output_dir=str(tmp_path),
            n_per_task=1,
            n_trajectories=1,
            phases=(1, 2, 3),
            panel_size=64,
            seed=42,
            verbose=False,
        )
        assert "phase1" in results
        assert "phase2" in results
        assert "phase3" in results

    def test_output_directory_created(self, tmp_path):
        config = load_config()
        new_dir = str(tmp_path / "deep" / "nested" / "diag")
        generate_supervised_diagnostics(
            config=config,
            output_dir=new_dir,
            n_per_task=1,
            n_trajectories=1,
            phases=(2,),
            panel_size=64,
            verbose=False,
        )
        assert Path(new_dir).exists()

    def test_reproducible_with_seed(self, tmp_path):
        """Same seed → same number of outputs."""
        config = load_config()
        r1 = generate_supervised_diagnostics(
            config=config,
            output_dir=str(tmp_path / "run1"),
            n_per_task=1,
            n_trajectories=1,
            phases=(2, 3),
            panel_size=64,
            seed=7,
            verbose=False,
        )
        r2 = generate_supervised_diagnostics(
            config=config,
            output_dir=str(tmp_path / "run2"),
            n_per_task=1,
            n_trajectories=1,
            phases=(2, 3),
            panel_size=64,
            seed=7,
            verbose=False,
        )
        # Same task count both runs
        assert r1["phase2"]["task_count"] == r2["phase2"]["task_count"]
        assert r1["phase3"]["saved"] == r2["phase3"]["saved"]
