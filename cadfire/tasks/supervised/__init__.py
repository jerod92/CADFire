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
Single-step tasks (original 11):
  select.py         – SemanticSelectTask, SemanticMultiSelectTask
  delete.py         – DeleteObjectTask          (ERASE)
  pan.py            – PanTask (up/down/left/right)
  zoom.py           – ZoomInTask, ZoomOutTask
  hatch.py          – HatchObjectTask           (HATCH)
  trace_next.py     – TraceNextPointTask        (POLYLINE – next vertex)
  copy_paste.py     – CopyObjectTask            (COPY)
  move.py           – MoveObjectTask            (MOVE)
  rotate.py         – RotateObjectTask          (ROTATE)

New single-step transform tasks:
  transform_extra.py – ScaleObjectTask          (SCALE)
                     – MirrorObjectTask         (MIRROR)
                     – OffsetTask               (OFFSET)

Multi-turn chat tasks (prompt = "<turn 1> | <turn 2>"):
  multiturn.py      – ScaleFromChatTask         (SCALE)
                    – MoveFromChatTask          (MOVE)
                    – RotateFromChatTask        (ROTATE)
                    – EraseFromChatTask         (ERASE)
                    – ChangeColorFromChatTask   (COLOR_SET)
                    – CopyFromChatTask          (COPY)
  These teach the model to interpret conversational context: the entity
  from turn 1 already exists on the canvas and is pre-selected; the agent
  must satisfy turn 2.

Multi-step trajectory (Phase 3 teacher-forcing):
  polygon_trace.py  – PolygonTraceTask          (multi-step, used in Phase 3)
"""
