This `CLAUDE.md` file serves as the definitive guide for the project's architecture, philosophy, and development standards. It ensures that any agent (human or AI) can maintain the integrity of the RL model and environment.

---

# CLAUDE.md: CAD-RL Project Guide

## 1. Project Vision
Building a Reinforcement Learning (RL) agent capable of professional-grade 2D CAD drafting. The agent operates in a custom environment, interpreting visual and textual prompts to execute CAD commands through a hybrid Tool/Cursor action space.

## 2. Technical Stack
- **Environment:** Custom `Numpy`-based CAD Engine (for speed and geometric precision).
- **Model:** `PyTorch` (Scratch-built Vision/Text encoders).
- **Training:** RL loop (PPO) with curriculum learning and incremental task loading.
- **Compute:** Optimized for `vast.ai` / Jupyter workflows.
- **Export:** Native DXF writer for CAD compatibility.

## 3. Project Structure

```
cadfire/
  engine/          # CAD engine: geometry primitives, entity management, undo/redo
    geometry.py    # Entity base class + all primitives (Line, Circle, Arc, etc.)
    cad_engine.py  # Stateful engine: entities, layers, selection, viewport
  renderer/        # Pure-numpy rasterizer producing multi-channel observation tensors
    rasterizer.py  # Bresenham line drawing, polygon fill, layer/selection masks
  env/             # Gym-compatible RL environment
    cad_env.py     # Observation building, action dispatch, tool execution
  model/           # PyTorch model architecture (all from scratch)
    vision_encoder.py  # ResNet-style CNN with skip connections for UNet
    text_encoder.py    # Bidirectional GRU for prompt encoding
    fusion.py          # Cross-attention fusion bridge
    action_heads.py    # Tool Head (MLP) + Cursor Head (UNet decoder)
    cad_agent.py       # Unified agent model with act/evaluate/extend
  tasks/           # Task & reward system (the primary extensibility point)
    base.py        # BaseTask ABC with reward helpers
    registry.py    # Auto-discovery registry with @register_task decorator
    draw_tasks.py  # Line, Circle, Rectangle, Polygon, Ellipse, Arc, Multi
    select_tasks.py    # Select shape, select by color, erase selection
    view_tasks.py      # Fit view, zoom to center
    modify_tasks.py    # Move, rotate, scale, copy, change layer
    trace_tasks.py     # Trace line/circle/composite from reference image
  training/        # PPO trainer, rollout buffer, checkpointing
    ppo.py         # Full PPO-Clip with curriculum learning
    pretrain_tools.py  # Supervised pre-training for text→tool classifier
    rollout.py     # GAE-based rollout buffer
    checkpoint.py  # Save/load with tool-list growth handling
  tokenizer/       # Byte-pair encoding tokenizer
    bpe.py         # Trainable BPE with CAD-aware pre-tokenization
  export/          # DXF export
    dxf_writer.py  # Pure-Python DXF R2010 writer
  utils/           # Config loader
    config.py      # Centralized config from config.json
config.json        # All hyperparameters, tool list, rendering settings
train.py           # CLI training script
train.ipynb        # Jupyter notebook for vast.ai
tests/             # 78 tests covering all modules
```

## 4. Architecture Overview

### A. Model Input (Multi-Modal)
1.  **Image Tensor:** `(H x W x C)` where C = 3 + 3 + L + 1 + 4 = 19 (for L=8 layers, H=W=256)
    - Channels 0-2: Current Viewport RGB (includes ghosting and selection highlights).
    - Channels 3-5: User-provided Raster Image (for tracing/reference).
    - Channels 6-(6+L-1): Layer Masks (Binary masks for each CAD layer).
    - Channel (6+L): Selection Mask (Binary mask of currently selected items).
    - Channel (6+L+1): X Ground Coords (tanh-scaled world x, centered on viewport).
    - Channel (6+L+2): Y Ground Coords (tanh-scaled world y, centered on viewport).
    - Channel (6+L+3): X Window Coords (linear ramp 0→1, left to right).
    - Channel (6+L+4): Y Window Coords (linear ramp 0→1, top to bottom).
2.  **Text Input:** Tokenized prompt (BPE, vocab_size=4096, max_len=128).
3.  **Vector State:** 16-dim vector: active tool, zoom, viewport center, layer, color, entity/selection counts.

### B. Model Head (Action Space)
- **Tool Space (Bridge):** MLP outputting logits over the 55-tool command set.
- **Cursor Space (UNet-Up):** UNet decoder with skip connections producing `(1, H, W)` heatmap.
    - **Single Point:** Argmax of the heatmap determines the precise coordinate.
    - **Multi-select:** If MULTISELECT is active, threshold at 0.5 for binary mask.
- **Value Head:** Scalar value estimate for PPO.
- **Param Head:** Tanh-bounded scalar for numeric parameters (angles, scale factors, etc.).

## 5. Environment & Command State
The tool list is defined in `config.json`. The model's Tool Head is dynamically sized to match. Tools include:

**Drawing:** LINE, POLYLINE, CIRCLE, ARC, RECTANGLE, POLYGON, ELLIPSE, SPLINE, POINT, HATCH, MTEXT, DTEXT, DIM_*
**Modify:** MOVE, COPY, ROTATE, SCALE, MIRROR, OFFSET, TRIM, EXTEND, FILLET, CHAMFER, ARRAY_*, EXPLODE, JOIN, BREAK, LENGTHEN
**Selection:** SELECT, MULTISELECT, DESELECT, ERASE
**Layer/Property:** LAYER_SET, LAYER_OFF/ON, LAYER_FREEZE/THAW, COLOR_SET, LINETYPE_SET, MATCHPROP
**Viewport:** ZOOM_IN/OUT, ZOOM_EXTENTS, PAN, FIT_VIEW
**Control:** UNDO, REDO, CONFIRM, CANCEL, NOOP

Multi-step commands (LINE needs 2 points, POLYLINE needs N+CONFIRM) use `engine.pending_points`.

## 6. Universal Task & Reward System
The task system is designed so **any AI agent can add a new task without modifying existing code**:

### Adding a New Task (the complete recipe)
1. Create a file `cadfire/tasks/my_task.py`
2. Inherit from `BaseTask`, use `@register_task`
3. Implement `setup(engine)` -> return `{"prompt": "...", ...}`
4. Implement `compute_reward(engine, action, step)` -> return `{"reward": float, "terminated": bool}`
5. Set `task_name`, `task_category`, `difficulty` class attributes
6. Override `generate_prompt_variants()` for lexical variation

Example:
```python
@register_task
class RotateShapeTask(BaseTask):
    task_name = "rotate_arbitrary"
    task_category = "modify"
    difficulty = 4.0

    def generate_prompt_variants(self):
        return ["Rotate the {shape} by {angle} degrees", ...]

    def setup(self, engine):
        # Create geometry, pick random params, return prompt
        ...
        return {"prompt": prompt, "target_entities": [target]}

    def compute_reward(self, engine, action, step):
        iou = self.iou_reward(engine.entities, [self._target])
        return {"reward": iou, "terminated": iou > 0.7}
```

That's it. The registry auto-discovers on `TaskRegistry.discover()`, and the training loop samples it automatically. No other files need changing.

### Reward Helpers (available on BaseTask)
- `iou_reward(entities, targets)`: Rasterized IoU comparison
- `entity_count_reward(current, target)`: Count matching
- `bbox_occupancy_reward(engine)`: Viewport fill ratio
- `selection_reward(engine, target_ids)`: F1 score for selection

## 7. Training & Vast.ai Workflow
- **Curriculum Learning:** Difficulty cap starts at 2.0 and increases every 5000 steps.
- **Checkpoint Compatibility:** `CheckpointManager` stores tool_list metadata. When tools are added, `agent.extend_tools()` grows the head preserving all existing weights.
- **Deployment:**
    1. Clone repo to Vast.ai.
    2. `pip install numpy torch` + run `train.ipynb` or `python train.py`.
    3. Training saves `checkpoints/latest.pt` and `checkpoints/diagnostics.json`.
    4. Pull updates with `git pull`: new tasks are auto-discovered, model resumes seamlessly.

## 8. Implementation Status

### Environment [DONE]
- [x] Numpy Engine: Coordinate system, Layer management, Geometry storage
- [x] Renderer: RGB + Layer masks + Selection mask + Reference image + Coordinate grids (256×256)
- [x] State Manager: Vector state + multi-channel image tensor (19 channels)

### Model Architecture [DONE]
- [x] Vision Encoder: ResNet-style with 4-scale skip connections
- [x] Text Encoder: Bidirectional GRU + BPE Tokenizer
- [x] Fusion Bridge: Cross-attention between Vision, Text, and State
- [x] UNet-Up Decoder: Skip connections for pixel-precise cursor heatmap
- [x] Tool Head: MLP with dynamic sizing for tool extension

### Initial Tasks [DONE]
- [x] Fit to View: Reward based on bounding box occupancy
- [x] Draw Primitive: Line, Circle, Rectangle, Polygon, Ellipse, Arc, Multi-primitive
- [x] Select Shape: Reward for isolating a specific entity (ambiguity-free: unique types per scene)
- [x] Select by Color: Reward for selecting all objects of a given color
- [x] Erase Selection: Reward for targeted deletion
- [x] Zoom/Pan: Reward for centering a target coordinate
- [x] Trace Line/Circle/Composite: Reward for matching reference image
- [x] Change Property: Layer assignment
- [x] Move/Rotate/Scale/Copy: Transform tasks with IoU reward

### Training Infrastructure [DONE]
- [x] PPO-Clip with GAE
- [x] Supervised pre-training for text→tool classifier (pretrain_tools.py)
- [x] Rollout buffer with mini-batch iteration
- [x] Checkpoint save/load with tool-list growth
- [x] Diagnostics JSON logging
- [x] Curriculum learning with difficulty progression
- [x] DXF export for CAD compatibility

### Tests [DONE]
- [x] 78 tests passing across engine, renderer, env, model, tasks, export, tokenizer

## 9. Development Rules
1.  **Future Proofing:** Never hard-code the number of tools or layers; always reference `config.json`.
2.  **No Pretrained Models:** All features learned from scratch in the CAD context.
3.  **Headless Env:** The model sees raw data tensors. RGB rendering is for human diagnostics only.
4.  **Task Isolation:** Each task is fully self-contained. Rewards live in the task, not the training loop.
5.  **Config-Driven:** All hyperparameters, tool lists, and rendering settings come from `config.json`.

---
*End of CLAUDE.md*
