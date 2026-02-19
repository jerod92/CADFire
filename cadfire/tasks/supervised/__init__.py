"""
Supervised pretraining tasks for CADFire.

These tasks are designed exclusively for supervised (behavioural-cloning)
pretraining phases.  Unlike PPO tasks they are NOT registered with the task
registry and do NOT appear in the RL curriculum.

Each task:
  1. Places the agent in a *mid-task* state (the hard part is already done).
  2. Exposes an ``oracle_action(engine, setup_info)`` that returns the single
     correct next step: ``{"tool": str, "cursor_world": np.ndarray|None,
                           "cursor_weight": float}``.
  3. The caller (pretrain_semantic.py / pretrain_teacher.py) renders the
     observation, converts the oracle to pixel-space, and trains with CE +
     focal-BCE losses.

Folder layout
─────────────
  select.py        – SemanticSelectTask, SemanticMultiSelectTask
  delete.py        – DeleteObjectTask         (ERASE)
  pan.py           – PanTask (up/down/left/right)
  zoom.py          – ZoomInTask, ZoomOutTask
  hatch.py         – HatchObjectTask          (HATCH)
  trace_next.py    – TraceNextPointTask       (POLYLINE – next vertex)
  copy_paste.py    – CopyObjectTask           (COPY)
  move.py          – MoveObjectTask           (MOVE)
  rotate.py        – RotateObjectTask         (ROTATE)
  polygon_trace.py – PolygonTraceTask         (multi-step, used in Phase 3)
"""
