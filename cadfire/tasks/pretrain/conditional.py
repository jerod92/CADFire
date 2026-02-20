"""
Conditional-reasoning supervised tasks.

These tasks inject logical connectives (IF, UNLESS, EXCEPT, ONLY, AND, OR)
into the natural-language prompt so the model learns to evaluate a scene
condition before committing to an action.

Single-step tasks (usable in Phase 2 pretrain_semantic dataset):
  IfSelectTask          – "If there are any circles, select one."
                          Oracle = SELECT when condition is TRUE, NOOP when FALSE.
  UnlessColorTask       – "Select the {shape}, unless it is {color}."
                          Oracle = SELECT when shape is NOT that color, else NOOP.
  ExceptEraseTask       – "Erase everything except the {shape}."
                          All non-protected entities are pre-selected; oracle = ERASE.
  OnlyColorSelectTask   – "Select only the {color} shapes."
                          Oracle = SELECT at the first matching entity's centroid.
  OrColorSelectTask     – "Select all {color1} or {color2} shapes."
                          Oracle = MULTISELECT at all matching centroids.

Multi-step trajectory (usable in Phase 3 teacher-forcing):
  AndSelectTrajectory   – "Select the {shape1} and the {shape2}."
                          Step 1: SELECT shape1.  Step 2: MULTISELECT shape2.
                          Returns a list[dict] compatible with TeacherForcingDataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.engine.geometry import Entity
from cadfire.tasks.pretrain.pretrain_select_tasks import (
    SEMANTIC_SHAPES, NUM_SEMANTIC_SHAPES, _QUADRANT_CENTRES, _place_shape,
)

# ── Colour palette (mirrors config.json) ──────────────────────────────────────

_COLOR_NAMES = ["white", "red", "yellow", "green", "cyan", "blue", "magenta", "gray"]
_NUM_COLORS  = len(_COLOR_NAMES)


# ── Shared scene builder ───────────────────────────────────────────────────────

def _build_quadrant_scene(
    rng: np.random.RandomState,
    engine: CADEngine,
    n: int = 4,
) -> Tuple[List[Entity], List[str]]:
    """
    Place *n* distinct semantic shapes at random quadrant positions.

    Returns (entities, shape_names).
    """
    positions = list(rng.choice(len(_QUADRANT_CENTRES), size=min(n, len(_QUADRANT_CENTRES)),
                                replace=False))
    type_indices = list(rng.choice(NUM_SEMANTIC_SHAPES, size=min(n, len(_QUADRANT_CENTRES)),
                                   replace=False))
    entities, names = [], []
    for pos_idx, type_idx in zip(positions, type_indices):
        cx, cy = _QUADRANT_CENTRES[pos_idx]
        name, entity = _place_shape(rng, type_idx, cx, cy)
        engine.add_entity(entity, save_undo=False)
        entities.append(entity)
        names.append(name)
    return entities, names


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IfSelectTask
# ═══════════════════════════════════════════════════════════════════════════════

_IF_TRUE_PROMPTS = [
    "If there are any {shape}s, select one",
    "Select a {shape} if one exists on the canvas",
    "If you see a {shape}, click on it to select it",
    "Should there be a {shape} present, select it",
    "Only if a {shape} is visible, select it",
]
_IF_FALSE_PROMPTS = [
    "If there are any {shape}s, select one",   # same prompt — scene has none
    "Select a {shape} if one exists on the canvas",
    "If you see a {shape}, click on it to select it",
]


class IfSelectTask:
    """
    Conditional SELECT: agent must evaluate whether a named shape is present.

    TRUE  (70 %): scene contains the named shape → oracle SELECT at its centroid.
    FALSE (30 %): scene does NOT contain it      → oracle NOOP (cursor ignored).
    """

    tool_name = "SELECT"   # or NOOP — depends on condition
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None, true_ratio: float = 0.7):
        self.rng = np.random.RandomState(seed)
        self.true_ratio = true_ratio
        self._oracle_tool: str = "NOOP"
        self._oracle_cursor: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        condition_true = self.rng.rand() < self.true_ratio

        # Always place 3 distractor shapes
        n_quad = len(_QUADRANT_CENTRES)
        all_type_indices = list(self.rng.choice(NUM_SEMANTIC_SHAPES,
                                                size=n_quad, replace=False))

        if condition_true:
            # The target type is one of the placed shapes
            target_type_idx = all_type_indices[0]
        else:
            # The target type is deliberately NOT placed
            target_type_idx = all_type_indices[0]
            # Replace it with a different shape in the scene
            alt_pool = [i for i in range(NUM_SEMANTIC_SHAPES)
                        if i != target_type_idx]
            all_type_indices[0] = int(self.rng.choice(alt_pool))

        target_name, _ = SEMANTIC_SHAPES[target_type_idx]
        target_entity: Optional[Entity] = None

        for i, (cx, cy) in enumerate(_QUADRANT_CENTRES):
            name, entity = _place_shape(self.rng, all_type_indices[i], cx, cy)
            engine.add_entity(entity, save_undo=False)
            if condition_true and i == 0:
                target_entity = entity

        if condition_true:
            self._oracle_tool   = "SELECT"
            self._oracle_cursor = target_entity.centroid()
            prompts = _IF_TRUE_PROMPTS
        else:
            self._oracle_tool   = "NOOP"
            self._oracle_cursor = np.array([500.0, 500.0])  # ignored
            prompts = _IF_FALSE_PROMPTS

        template = prompts[int(self.rng.randint(len(prompts)))]
        prompt = template.format(shape=target_name)

        return {
            "prompt": prompt,
            "condition_true": condition_true,
            "target_name": target_name,
            "target_entity": target_entity,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": self._oracle_tool,
            "cursor_world": self._oracle_cursor,
            "cursor_weight": self.cursor_loss_weight if self._oracle_tool != "NOOP" else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UnlessColorTask
# ═══════════════════════════════════════════════════════════════════════════════

_UNLESS_PROMPTS = [
    "Select the {shape}, unless it is {color}",
    "Click the {shape} to select it, but not if it is {color}",
    "Select the {shape} — skip it if the color is {color}",
    "If the {shape} is not {color}, select it",
    "Choose the {shape}, except when it appears {color}",
]


class UnlessColorTask:
    """
    Conditional SELECT: the agent must check an entity's color before acting.

    The scene has one entity.  The prompt specifies an excluded color.
      Entity IS the excluded color  → oracle NOOP  (condition triggered).
      Entity is NOT that color      → oracle SELECT at entity centroid.
    """

    tool_name = "SELECT"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._oracle_tool: str = "SELECT"
        self._oracle_cursor: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        type_idx = int(self.rng.randint(NUM_SEMANTIC_SHAPES))
        cx = float(self.rng.uniform(350, 650))
        cy = float(self.rng.uniform(350, 650))
        name, entity = _place_shape(self.rng, type_idx, cx, cy)

        # Assign a concrete color to the entity
        entity_color_idx = int(self.rng.randint(_NUM_COLORS))
        entity.color_index = entity_color_idx
        engine.add_entity(entity, save_undo=False)

        # Pick a forbidden color (50/50: matches entity vs does not)
        if self.rng.rand() < 0.5:
            forbidden_color_idx = entity_color_idx        # condition fires → NOOP
        else:
            pool = [i for i in range(_NUM_COLORS) if i != entity_color_idx]
            forbidden_color_idx = int(self.rng.choice(pool))  # condition ok → SELECT

        condition_fired = (entity_color_idx == forbidden_color_idx)
        forbidden_color_name = _COLOR_NAMES[forbidden_color_idx]

        if condition_fired:
            self._oracle_tool   = "NOOP"
            self._oracle_cursor = np.array([500.0, 500.0])
        else:
            self._oracle_tool   = "SELECT"
            self._oracle_cursor = entity.centroid()

        template = _UNLESS_PROMPTS[int(self.rng.randint(len(_UNLESS_PROMPTS)))]
        prompt = template.format(shape=name, color=forbidden_color_name)

        return {
            "prompt": prompt,
            "entity": entity,
            "shape_name": name,
            "forbidden_color": forbidden_color_name,
            "condition_fired": condition_fired,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": self._oracle_tool,
            "cursor_world": self._oracle_cursor,
            "cursor_weight": self.cursor_loss_weight if self._oracle_tool != "NOOP" else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ExceptEraseTask
# ═══════════════════════════════════════════════════════════════════════════════

_EXCEPT_PROMPTS = [
    "Erase everything except the {shape}",
    "Delete all shapes but leave the {shape}",
    "Remove everything on the canvas except for the {shape}",
    "Keep only the {shape} and erase the rest",
    "Erase all shapes except the {shape}",
    "Clear the canvas but preserve the {shape}",
]


class ExceptEraseTask:
    """
    Conditional ERASE: delete everything on the canvas EXCEPT one protected shape.

    The protected entity is not pre-selected; all others are.
    Oracle = ERASE (cursor weight very low since any click on the selection works).
    """

    tool_name = "ERASE"
    cursor_loss_weight = 0.05  # tool prediction is what matters; cursor is secondary

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._erase_cursor: Optional[np.ndarray] = None

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        n = int(self.rng.randint(3, 6))   # 3-5 shapes total
        entities, names = _build_quadrant_scene(self.rng, engine, n=n)

        # Choose one to protect
        protect_idx = int(self.rng.randint(len(entities)))
        protected = entities[protect_idx]
        protected_name = names[protect_idx]

        # Pre-select everything else
        for i, entity in enumerate(entities):
            if i != protect_idx:
                engine.selected_ids.add(entity.id)

        # Cursor at centroid of any selected entity (arbitrary; just confirms erase)
        selected = [e for i, e in enumerate(entities) if i != protect_idx]
        self._erase_cursor = selected[0].centroid() if selected else np.array([500.0, 500.0])

        template = _EXCEPT_PROMPTS[int(self.rng.randint(len(_EXCEPT_PROMPTS)))]
        prompt = template.format(shape=protected_name)

        return {
            "prompt": prompt,
            "protected_entity": protected,
            "protected_name": protected_name,
            "all_entities": entities,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "ERASE",
            "cursor_world": self._erase_cursor,
            "cursor_weight": self.cursor_loss_weight,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OnlyColorSelectTask
# ═══════════════════════════════════════════════════════════════════════════════

_ONLY_PROMPTS = [
    "Select only the {color} shapes",
    "Select the {color} ones only",
    "Choose all {color} objects and nothing else",
    "Pick out the {color} shapes — ignore the rest",
    "Select nothing but the {color} shapes",
    "Find and select the {color} shapes exclusively",
]


class OnlyColorSelectTask:
    """
    Filtered SELECT: agent must select a shape of the named color only.

    Scene has 3-5 entities with mixed colors.  At least one matches the
    target color.  Oracle = MULTISELECT at all matching centroids.
    """

    tool_name = "MULTISELECT"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target_centroids: List[np.ndarray] = []

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        n = int(self.rng.randint(3, 6))
        entities, names = _build_quadrant_scene(self.rng, engine, n=n)

        # Pick a target color
        target_color_idx = int(self.rng.randint(_NUM_COLORS))

        # Assign colors: at least one entity gets the target color
        color_indices = [int(self.rng.randint(_NUM_COLORS)) for _ in entities]
        force_idx = int(self.rng.randint(len(entities)))
        color_indices[force_idx] = target_color_idx
        for entity, cidx in zip(entities, color_indices):
            entity.color_index = cidx

        # Collect matching entities
        matching = [e for e, cidx in zip(entities, color_indices)
                    if cidx == target_color_idx]
        self._target_centroids = [e.centroid() for e in matching]

        target_color_name = _COLOR_NAMES[target_color_idx]
        template = _ONLY_PROMPTS[int(self.rng.randint(len(_ONLY_PROMPTS)))]
        prompt = template.format(color=target_color_name)

        return {
            "prompt": prompt,
            "target_color": target_color_name,
            "target_entities": matching,
            "all_entities": entities,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "MULTISELECT",
            "cursor_world": self._target_centroids,
            "cursor_weight": self.cursor_loss_weight,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OrColorSelectTask
# ═══════════════════════════════════════════════════════════════════════════════

_OR_PROMPTS = [
    "Select all {color1} or {color2} shapes",
    "Choose every shape that is {color1} or {color2}",
    "Select shapes that are either {color1} or {color2}",
    "Click on any {color1} or {color2} objects",
    "Grab all shapes that appear {color1} or {color2}",
]


class OrColorSelectTask:
    """
    Union SELECT: agent selects all shapes matching either of two named colors.

    Oracle = MULTISELECT at all matching centroids.
    """

    tool_name = "MULTISELECT"
    cursor_loss_weight = 1.0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._target_centroids: List[np.ndarray] = []

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        n = int(self.rng.randint(4, 6))
        entities, names = _build_quadrant_scene(self.rng, engine, n=n)

        # Pick two distinct target colors
        c1, c2 = map(int, self.rng.choice(_NUM_COLORS, size=2, replace=False))

        # Assign colors ensuring at least one entity per target color
        color_indices = [int(self.rng.randint(_NUM_COLORS)) for _ in entities]
        color_indices[0] = c1
        if len(entities) > 1:
            color_indices[1] = c2
        for entity, cidx in zip(entities, color_indices):
            entity.color_index = cidx

        matching = [e for e, cidx in zip(entities, color_indices)
                    if cidx in (c1, c2)]
        self._target_centroids = [e.centroid() for e in matching]

        c1_name, c2_name = _COLOR_NAMES[c1], _COLOR_NAMES[c2]
        template = _OR_PROMPTS[int(self.rng.randint(len(_OR_PROMPTS)))]
        prompt = template.format(color1=c1_name, color2=c2_name)

        return {
            "prompt": prompt,
            "color1": c1_name,
            "color2": c2_name,
            "target_entities": matching,
        }

    def oracle_action(self, engine: CADEngine, setup_info: Dict) -> Dict:
        return {
            "tool": "MULTISELECT",
            "cursor_world": self._target_centroids,
            "cursor_weight": self.cursor_loss_weight,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AndSelectTrajectory  (multi-step, Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

_AND_PROMPTS = [
    "Select the {shape1} and the {shape2}",
    "Click on both the {shape1} and the {shape2}",
    "Select the {shape1} as well as the {shape2}",
    "Choose the {shape1} and also the {shape2}",
    "Pick the {shape1} and {shape2} together",
    "I need both the {shape1} and the {shape2} selected",
]


class AndSelectTrajectory:
    """
    Two-step SELECT trajectory: "Select the {shape1} and the {shape2}."

    Step 1: SELECT  at shape1 centroid.
    Step 2: MULTISELECT at shape2 centroid  (adds to existing selection).

    Intended for the Phase-3 TeacherForcingDataset.  Call
    ``generate_steps(engine)`` to obtain a list of two step-dicts compatible
    with the Phase-3 training loop.

    The step dicts here do NOT include rendered images / text_ids — those are
    assembled by the caller (pretrain_teacher.py) using ``oracle_to_cursor_mask``
    and the tokenizer, matching the pattern of other short-trajectory builders.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self._shape1_entity: Optional[Entity] = None
        self._shape1_name: str = ""
        self._shape2_entity: Optional[Entity] = None
        self._shape2_name: str = ""

    def setup(self, engine: CADEngine) -> Dict[str, Any]:
        """Place two distinct shapes and record them."""
        type_indices = list(self.rng.choice(NUM_SEMANTIC_SHAPES, size=2, replace=False))
        positions = list(self.rng.choice(len(_QUADRANT_CENTRES), size=2, replace=False))

        cx1, cy1 = _QUADRANT_CENTRES[positions[0]]
        self._shape1_name, self._shape1_entity = _place_shape(
            self.rng, type_indices[0], cx1, cy1
        )
        engine.add_entity(self._shape1_entity, save_undo=False)

        cx2, cy2 = _QUADRANT_CENTRES[positions[1]]
        self._shape2_name, self._shape2_entity = _place_shape(
            self.rng, type_indices[1], cx2, cy2
        )
        engine.add_entity(self._shape2_entity, save_undo=False)

        template = _AND_PROMPTS[int(self.rng.randint(len(_AND_PROMPTS)))]
        prompt = template.format(shape1=self._shape1_name, shape2=self._shape2_name)

        return {
            "prompt": prompt,
            "shape1_entity": self._shape1_entity,
            "shape1_name": self._shape1_name,
            "shape2_entity": self._shape2_entity,
            "shape2_name": self._shape2_name,
        }

    def oracle_step1(self) -> Dict[str, Any]:
        """First oracle: SELECT shape1."""
        return {
            "tool": "SELECT",
            "cursor_world": self._shape1_entity.centroid(),
            "cursor_weight": 1.0,
        }

    def oracle_step2(self) -> Dict[str, Any]:
        """Second oracle: MULTISELECT shape2 (extends selection)."""
        return {
            "tool": "MULTISELECT",
            "cursor_world": self._shape2_entity.centroid(),
            "cursor_weight": 1.0,
        }
