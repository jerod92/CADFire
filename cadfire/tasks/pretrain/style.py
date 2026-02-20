"""
Supervised tasks for visual style properties: linetype and lineweight.

Both tools are property-setting operations — the cursor position matters very
little (these are not spatial picks).  The loss signal comes almost entirely
from the tool-head cross-entropy.

LinetypeSetTask    – LINETYPE_SET  "Make the line dashed / dotted / centre-line"
LineweightSetTask  – LINEWEIGHT_SET "Make the outline thicker / set lineweight to 2"

These teach the model:
  1. To call the right property tool when a style instruction appears.
  2. (Loosely) to aim near the target entity for context — cursor_weight is
     kept at 0.1 so the cursor loss is a soft regulariser, not the main signal.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import Entity
from cadfire.tasks.pretrain.pretrain_select_tasks import (
    SEMANTIC_SHAPES, NUM_SEMANTIC_SHAPES, _QUADRANT_CENTRES, _place_shape,
)

# ── Linetype catalogue ────────────────────────────────────────────────────────
# Maps a human-readable description to the internal linetype string that will
# be stored on the entity.  The engine currently passes these through to DXF.

LINETYPES: List[Tuple[str, str]] = [
    ("CONTINUOUS",  "continuous"),
    ("DASHED",      "dashed"),
    ("DASHED2",     "short-dashed"),
    ("DOT",         "dotted"),
    ("DASHDOT",     "dash-dot"),
    ("DASHDOT2",    "short dash-dot"),
    ("CENTER",      "centre-line"),
    ("CENTER2",     "short centre-line"),
    ("PHANTOM",     "phantom"),
    ("HIDDEN",      "hidden"),
]

_LINETYPE_NAMES       = [lt[0] for lt in LINETYPES]
_LINETYPE_DESCRIPTORS = [lt[1] for lt in LINETYPES]
NUM_LINETYPES         = len(LINETYPES)

_LINETYPE_PROMPTS = [
    "Change the linetype to {descriptor}",
    "Set the line style to {descriptor}",
    "Make the {shape} line {descriptor}",
    "Apply a {descriptor} linetype to the {shape}",
    "Switch the {shape} to a {descriptor} line",
    "Use a {descriptor} line for the {shape}",
]

# ── Lineweight catalogue ──────────────────────────────────────────────────────
# Discrete standard CAD lineweights (mm-scale, stored as float on entity).

LINEWEIGHTS: List[Tuple[float, str]] = [
    (0.13, "very thin"),
    (0.18, "thin"),
    (0.25, "fine"),
    (0.35, "normal"),
    (0.50, "medium"),
    (0.70, "thick"),
    (1.00, "bold"),
    (1.40, "extra bold"),
    (2.00, "ultra bold"),
]

_LW_VALUES      = [lw[0] for lw in LINEWEIGHTS]
_LW_DESCRIPTORS = [lw[1] for lw in LINEWEIGHTS]
NUM_LINEWEIGHTS = len(LINEWEIGHTS)

_LINEWEIGHT_PROMPTS = [
    "Make the {shape} outline {descriptor}",
    "Set the lineweight to {descriptor}",
    "Apply a {descriptor} lineweight to the {shape}",
    "Change the line thickness to {descriptor}",
    "Use a {descriptor} line weight for the {shape}",
    "Make the {shape} line {descriptor}",
    "Adjust the {shape} stroke to {descriptor}",
]


# ── Helper ────────────────────────────────────────────────────────────────────

def _single_shape_scene(
    rng: np.random.RandomState,
    engine: CADEngine,
) -> Tuple[Entity, str]:
    """Place one semantic shape at a random central position."""
    type_idx = int(rng.randint(NUM_SEMANTIC_SHAPES))
    pos_idx  = int(rng.randint(len(_QUADRANT_CENTRES)))
    cx, cy   = _QUADRANT_CENTRES[pos_idx]
    name, entity = _place_shape(rng, type_idx, cx, cy)
    engine.add_entity(entity, save_undo=False)
    return entity, name


# ═══════════════════════════════════════════════════════════════════════════════
# LinetypeSetTask
# ═══════════════════════════════════════════════════════════════════════════════

class LinetypeSetTask:
    """
    Single-step LINETYPE_SET supervised task.

    A shape is placed on the canvas and pre-selected.  The agent must call
    LINETYPE_SET to change the active (and thus entity) linetype.

    Cursor points near the entity centroid with very low weight; the main
    loss signal is the tool cross-entropy.
    """

    tool_name      = "LINETYPE_SET"
    cursor_loss_weight = 0.1

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._entity_centroid: Optional[np.ndarray] = None
        self._target_linetype: str = "CONTINUOUS"

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        entity, shape_name = _single_shape_scene(self.rng, engine)
        engine.selected_ids.add(entity.id)
        self._entity_centroid = entity.centroid()

        lt_idx = int(self.rng.randint(NUM_LINETYPES))
        self._target_linetype = _LINETYPE_NAMES[lt_idx]
        descriptor            = _LINETYPE_DESCRIPTORS[lt_idx]

        template = _LINETYPE_PROMPTS[int(self.rng.randint(len(_LINETYPE_PROMPTS)))]
        prompt = template.format(shape=shape_name, descriptor=descriptor)

        return {
            "prompt": prompt,
            "entity": entity,
            "shape_name": shape_name,
            "target_linetype": self._target_linetype,
            "descriptor": descriptor,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "LINETYPE_SET",
            "cursor_world": self._entity_centroid,
            "cursor_weight": self.cursor_loss_weight,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LineweightSetTask
# ═══════════════════════════════════════════════════════════════════════════════

class LineweightSetTask:
    """
    Single-step LINEWEIGHT_SET supervised task.

    A shape is placed on the canvas and pre-selected.  The agent must call
    LINEWEIGHT_SET to change the active lineweight.

    Cursor weight is low (0.1); the main training signal is the tool CE loss.
    The target lineweight value is stored in ``setup_info["target_lineweight"]``
    for any downstream use (e.g., param-head regression targets).
    """

    tool_name      = "LINEWEIGHT_SET"
    cursor_loss_weight = 0.1

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._entity_centroid:   Optional[np.ndarray] = None
        self._target_lineweight: float = 0.35

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        entity, shape_name = _single_shape_scene(self.rng, engine)
        engine.selected_ids.add(entity.id)
        self._entity_centroid = entity.centroid()

        lw_idx = int(self.rng.randint(NUM_LINEWEIGHTS))
        self._target_lineweight = _LW_VALUES[lw_idx]
        descriptor              = _LW_DESCRIPTORS[lw_idx]

        template = _LINEWEIGHT_PROMPTS[int(self.rng.randint(len(_LINEWEIGHT_PROMPTS)))]
        prompt = template.format(shape=shape_name, descriptor=descriptor)

        return {
            "prompt": prompt,
            "entity": entity,
            "shape_name": shape_name,
            "target_lineweight": self._target_lineweight,
            "descriptor": descriptor,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "LINEWEIGHT_SET",
            "cursor_world": self._entity_centroid,
            "cursor_weight": self.cursor_loss_weight,
        }
