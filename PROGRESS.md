# CADFire Implementation Progress

**Status Key:** [x] = Implemented | [T] = Has training task | [-] = Not yet needed/low priority

## Milestone 1: Core Foundation (COMPLETE)

- [x] Numpy CAD Engine with 12 geometry types
- [x] Pure-numpy rasterizer (15-channel observation tensor)
- [x] Gym-compatible RL environment (55 tools)
- [x] ResNet Vision Encoder + GRU Text Encoder + Cross-Attention Fusion
- [x] UNet Cursor Head + MLP Tool Head (extensible)
- [x] PPO-Clip trainer with GAE and curriculum learning
- [x] Checkpoint manager with tool-list growth handling
- [x] BPE tokenizer
- [x] DXF R2010 export
- [x] 20 training tasks across 5 categories
- [x] 78 tests passing
- [x] train.py CLI + train.ipynb for vast.ai

---

## Command Status

### 1. Basic Drawing Commands
| Command | Engine | Env | Task | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **LINE** | [x] | [x] | [T] | draw_line, trace_line |
| **POLYLINE** | [x] | [x] | [-] | N-point + CONFIRM |
| **CIRCLE** | [x] | [x] | [T] | draw_circle, trace_circle |
| **ARC** | [x] | [x] | [T] | draw_arc |
| **RECTANGLE** | [x] | [x] | [T] | draw_rectangle |
| **POLYGON** | [x] | [x] | [T] | draw_polygon |
| **ELLIPSE** | [x] | [x] | [T] | draw_ellipse |
| **HATCH** | [x] | [x] | [-] | Boundary fill |
| **SPLINE** | [x] | [x] | [-] | Polynomial interp |
| **POINT** | [x] | [x] | [-] | Single click |
| **XLINE** | [-] | [-] | [-] | Low priority |
| **RAY** | [-] | [-] | [-] | Low priority |
| **DONUT** | [-] | [-] | [-] | Low priority |
| **REVCLOUD** | [-] | [-] | [-] | Low priority |
| **WIPEOUT** | [-] | [-] | [-] | Low priority |
| **REGION** | [-] | [-] | [-] | Low priority |

### 2. Modify Commands
| Command | Engine | Env | Task | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **MOVE** | [x] | [x] | [T] | move_shape |
| **COPY** | [x] | [x] | [T] | copy_shape |
| **ROTATE** | [x] | [x] | [T] | rotate_shape |
| **SCALE** | [x] | [x] | [T] | scale_shape |
| **MIRROR** | [x] | [x] | [-] | |
| **OFFSET** | [x] | [x] | [-] | Scale approx |
| **EXPLODE** | [x] | [x] | [-] | Rect/Poly -> Lines |
| **MATCHPROP** | [x] | [x] | [-] | |
| **STRETCH** | [-] | [-] | [-] | |
| **TRIM** | [-] | [-] | [-] | Needs intersection |
| **EXTEND** | [-] | [-] | [-] | Needs intersection |
| **FILLET** | [-] | [-] | [-] | Placeholder |
| **CHAMFER** | [-] | [-] | [-] | Placeholder |
| **ARRAY** | [-] | [-] | [-] | Placeholder |
| **JOIN** | [-] | [-] | [-] | Placeholder |
| **BREAK** | [-] | [-] | [-] | Placeholder |
| **LENGTHEN** | [-] | [-] | [-] | Placeholder |
| **ALIGN** | [-] | [-] | [-] | |
| **OVERKILL** | [-] | [-] | [-] | |

### 3. Annotation & Dimensioning
| Command | Engine | Env | Task | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **MTEXT** | [x] | [x] | [-] | Multiline text |
| **DTEXT** | [x] | [x] | [-] | Single-line text |
| **DIM_LINEAR** | [x] | [x] | [-] | 2-point |
| **DIM_ALIGNED** | [x] | [x] | [-] | 2-point |
| **DIM_ANGULAR** | [x] | [x] | [-] | 2-point |
| **DIM_RADIUS** | [x] | [x] | [-] | 2-point |
| **DIM_DIAMETER** | [x] | [x] | [-] | 2-point |

### 4. Layer & Object Properties
| Command | Engine | Env | Task | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **LAYER_SET** | [x] | [x] | [T] | change_layer |
| **LAYER_OFF** | [x] | [x] | [-] | |
| **LAYER_ON** | [x] | [x] | [-] | All layers |
| **LAYER_FREEZE** | [x] | [x] | [-] | |
| **LAYER_THAW** | [x] | [x] | [-] | All layers |
| **COLOR_SET** | [x] | [x] | [-] | |
| **LINETYPE_SET** | [x] | [x] | [-] | |
| **MATCHPROP** | [x] | [x] | [-] | |

### 5. Selection
| Command | Engine | Env | Task | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **SELECT** | [x] | [x] | [T] | select_shape |
| **MULTISELECT** | [x] | [x] | [T] | select_by_color |
| **DESELECT** | [x] | [x] | [-] | |
| **ERASE** | [x] | [x] | [T] | erase_selection |

### 6. Viewing & Navigation
| Command | Engine | Env | Task | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **ZOOM_IN/OUT** | [x] | [x] | [-] | |
| **ZOOM_EXTENTS** | [x] | [x] | [-] | |
| **PAN** | [x] | [x] | [-] | 2-point |
| **FIT_VIEW** | [x] | [x] | [T] | fit_view |

### 7. Control
| Command | Engine | Env | Task | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **UNDO** | [x] | [x] | [-] | State snapshot |
| **REDO** | [x] | [x] | [-] | |
| **CONFIRM** | [x] | [x] | [-] | Finalize multi-step |
| **CANCEL** | [x] | [x] | [-] | Clear pending |
| **NOOP** | [x] | [x] | [-] | |

---

## Training Tasks (20 total)

| Task | Category | Difficulty | Description |
| :--- | :--- | :---: | :--- |
| draw_line | draw | 1.0 | Draw a line between specified points |
| draw_circle | draw | 1.0 | Draw a circle at center with radius |
| draw_rectangle | draw | 1.5 | Draw a rectangle with dimensions |
| draw_polygon | draw | 2.0 | Draw a regular polygon |
| draw_ellipse | draw | 2.5 | Draw an ellipse with axes |
| draw_arc | draw | 3.0 | Draw an arc with angles |
| draw_multi_primitive | draw | 4.0 | Draw multiple shapes |
| select_shape | select | 1.5 | Select a specific entity |
| select_by_color | select | 3.0 | Select all entities of a color |
| erase_selection | modify | 2.5 | Delete a targeted entity |
| fit_view | view | 1.0 | Fit geometry to viewport |
| zoom_to_center | view | 2.0 | Navigate to a coordinate |
| move_shape | modify | 3.0 | Move entity to target position |
| rotate_shape | modify | 4.0 | Rotate entity by angle |
| scale_shape | modify | 3.5 | Scale entity by factor |
| copy_shape | modify | 3.5 | Duplicate entity at offset |
| change_layer | property | 2.0 | Change entity layer |
| trace_line | trace | 2.0 | Trace line from reference |
| trace_circle | trace | 2.5 | Trace circle from reference |
| trace_composite | trace | 5.0 | Trace multiple shapes from reference |
