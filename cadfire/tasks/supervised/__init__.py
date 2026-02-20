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
  move.py           – MoveObjectTask            (MOVE – marker-driven)
                    – PrepositionalMoveTask     (MOVE – directional & relational)
                    – prepositional_move_step() (helper for Phase-3 builders)
  rotate.py         – RotateObjectTask          (ROTATE)

New single-step transform tasks:
  transform_extra.py – ScaleObjectTask          (SCALE)
                     – MirrorObjectTask         (MIRROR)
                     – OffsetTask               (OFFSET)

Geometric-editing tasks:
  geometry_ops.py    – FilletTask               (FILLET  – round corner, cursor at vertex)
                     – ChamferTask              (CHAMFER – bevel corner, cursor at vertex)
                     – TrimTask                 (TRIM    – cursor on segment to remove)

Array tasks:
  array_ops.py       – ArrayRectTask            (ARRAY_RECT  – cursor at far-corner of grid)
                     – ArrayPolarTask           (ARRAY_POLAR – cursor at rotation centre)

Conditional-reasoning tasks (IF / UNLESS / EXCEPT / ONLY / OR / AND):
  conditional.py    – IfSelectTask              (SELECT or NOOP based on IF condition)
                    – UnlessColorTask           (SELECT or NOOP based on UNLESS color)
                    – ExceptEraseTask           (ERASE all except one protected shape)
                    – OnlyColorSelectTask       (MULTISELECT only matching-color shapes)
                    – OrColorSelectTask         (MULTISELECT color1 OR color2 shapes)
                    – AndSelectTrajectory       (2-step SELECT+MULTISELECT, Phase 3)

Style / property tasks:
  style.py          – LinetypeSetTask           (LINETYPE_SET)
                    – LineweightSetTask         (LINEWEIGHT_SET)

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

Phase-3 trajectory builders in pretrain_teacher.py use:
  • PolygonTraceTask          (70 % of trajectories, 4–9 steps)
  • _build_select_then_erase  (2 steps: SELECT → ERASE)
  • _build_select_then_rotate (2 steps: SELECT → ROTATE)
  • _build_select_then_copy   (2 steps: SELECT → COPY)
  • _build_select_then_move   (2 steps: SELECT → MOVE prepositional)
  • _build_and_select         (2 steps: SELECT → MULTISELECT "and" query)
"""
