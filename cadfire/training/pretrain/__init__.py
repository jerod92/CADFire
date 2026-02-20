"""
Pretraining sub-package: supervised warm-start runners for Phases 1–3.

  tools    – Phase 1: text→tool cross-entropy pretraining
  semantic – Phase 2: single-step semantic cursor pretraining
  teacher  – Phase 3: teacher-forced multi-step trajectory pretraining
  cursor   – Cursor-specific auxiliary pretraining utilities
"""

from cadfire.training.pretrain.tools import pretrain_tool_classifier
from cadfire.training.pretrain.semantic import pretrain_semantic_cursor
from cadfire.training.pretrain.teacher import pretrain_teacher_forcing

__all__ = [
    "pretrain_tool_classifier",
    "pretrain_semantic_cursor",
    "pretrain_teacher_forcing",
]
