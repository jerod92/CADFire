This `CLAUDE.md` file serves as the definitive guide for the project's architecture, philosophy, and development standards. It ensures that any agent (human or AI) can maintain the integrity of the RL model and environment.

---

# CLAUDE.md: CAD-RL Project Guide

## 1. Project Vision
Building a Reinforcement Learning (RL) agent capable of professional-grade 2D CAD drafting. The agent operates in a custom environment, interpreting visual and textual prompts to execute CAD commands through a hybrid Tool/Cursor action space.

## 2. Technical Stack
- **Environment:** Custom `Numpy`-based CAD Engine (for speed and geometric precision).
- **Model:** `PyTorch` (Scratch-built Vision/Text encoders).
- **Training:** RL loop (PPO or SAC) designed for incremental task loading.
- **Compute:** Optimized for `vast.ai` / Jupyter workflows.

## 3. Architecture Overview

### A. Model Input (Multi-Modal)
1.  **Image Tensor:** $(H \times W \times C)$
    - Channels 0-2: Current Viewport RGB (includes "ghosting" and selection highlights).
    - Channels 3-5: User-provided Raster Image (for tracing/reference).
    - Channels 6-(6+L): Layer Masks (Binary masks for each CAD layer, padded to fixed $L$).
    - Channel (6+L+1): Selection Mask (Binary mask of currently selected items).
2.  **Text Input:** Tokenized multi-step chat history (User prompts only).
3.  **Vector State:** Current active tool, zoom level, viewport coordinates, and active layer index.

### B. Model Head (Action Space)
- **Tool Space (Bridge):** A MLP head outputting a probability distribution over the CAD command set (the "Toolbox").
- **Cursor Space (UNet-Up):** A UNet-style decoder with skip-connections from the vision encoder.
    - **Single Point:** Argmax of the heatmap determines the precise coordinate.
    - **Multi-select:** If the "Select" tool is active, the heatmap acts as a binary mask; all objects whose centroids/bounds fall within the mask are selected.

## 4. Environment & Command State
The state is defined by the "Universal CAD Checklist." Every command listed in the checklist must have a corresponding logic block in the environment and a placeholder in the Model's Tool Head.

### Added Commands for RL Efficiency:
- **MULTISELECT:** Uses the UNet mask to select multiple objects.
- **FIT_VIEW:** Automatically adjusts zoom/pan to bound all existing geometry with a 5% margin.
- **UNDO / REDO:** Essential for RL recovery and exploration.

## 5. Universal Task & Reward System
To ensure compatibility and prevent training restarts when adding tasks:
1.  **Task Provider Interface:** All tasks (Trace, Draw, Edit) must inherit from a `BaseTask` class.
2.  **Reward Logic:** Rewards are encapsulated within the `Task` object, not the Training Loop. The loop simply calls `task.compute_reward(env_state, action)`.
3.  **Lexical Variation:** Generators must use templates (e.g., `"Draw a {color} {primitive}"`) to ensure the text encoder learns generalized concepts.

## 6. Training & Vast.ai Workflow
- **Compatibility:** Model checkpoints store the `tool_map` metadata. If new tools are added, the head is extended, but weights are preserved.
- **Deployment:** 
    1. Clone repo to Vast.ai.
    2. Run `train.ipynb`.
    3. The training loop periodically saves `latest.pt` and `diagnostics.json`.
    4. Updates can be pulled via `git pull`; the environment is re-initialized, but the model loads previous weights into the compatible architecture.

## 7. Initial Implementation Checklist (Milestone 1)

### Environment Skeleton
- [ ] **Numpy Engine:** Coordinate system, Layer management, and Geometry storage.
- [ ] **Renderer:** RGB generation for Viewport, Raster, and Ghosting/Selection.
- [ ] **State Manager:** Padded layer tensors and tool-state vector.

### Model Architecture
- [ ] **Vision Encoder:** CNN/ResNet (Scratch).
- [ ] **Text Encoder:** Small Transformer/GRU (Scratch) + Byte-Pair Tokenizer.
- [ ] **Fusion Bridge:** Cross-attention mechanism between Vision, Text, and State.
- [ ] **UNet-Up Decoder:** Skip connections for pixel-perfect cursor placement.

### Initial Tasks (Dataset & Reward Providers)
- [ ] **Fit to View:** Reward based on bounding box occupancy.
- [ ] **Draw Primitive:** (Line, Circle, Rect) Reward based on geometric IoU.
- [ ] **Select Shape:** Reward for isolating a specific ID.
- [ ] **Select by Color:** Reward for selecting all objects of a specific hex/index.
- [ ] **Erase Selection:** Reward for reduction of targeted entity count.
- [ ] **Zoom/Pan:** Reward for bringing a target coordinate into the center 50% of the view.
- [ ] **Trace Line:** Reward for overlap between drawn polyline and raster channel 3-5 pixels.
- [ ] **Change Property:** (Layer/Color) Reward for modifying metadata of selection.

## 8. Development Rules
1.  **Future Proofing:** Never hard-code the number of tools or layers in the model logic; always reference `config.json`.
2.  **No Pretrained Models:** Vision and Text features must be learned within the CAD context to ensure the latent space is optimized for technical drawings rather than natural photos.
3.  **Headless Env:** The model never sees the "GUI." It sees the raw data. GUI rendering is for human diagnostics only.

---
*End of CLAUDE.md*
