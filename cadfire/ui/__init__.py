"""
CADFire interactive demo UI.

Launch with::

    python scripts/demo.py
    # or programmatically:
    from cadfire.ui import launch_demo
    launch_demo()

The demo is a full Tkinter-based application that is simultaneously:

  • A **human CAD tool**: mouse + keyboard for drawing, editing, viewport.
  • An **AI viewer**: load a trained checkpoint, type a prompt, watch the
    model execute actions step-by-step.
  • A **training-wheels panel**: toggle which tool categories the AI is
    allowed to use (e.g. disable Pan/Zoom to focus the agent on drawing).

Dependencies: tkinter (stdlib) + optional Pillow for faster canvas rendering.
"""

from cadfire.ui.app import CADFireApp


def launch_demo(
    checkpoint_dir: str = "model_saves",
    config_path: str | None = None,
    display_scale: int = 3,
) -> None:
    """
    Launch the interactive demo window.

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        config_path   : Path to config.json (defaults to project root).
        display_scale : Integer scale factor for the canvas (3 → 768×768).
    """
    import tkinter as tk
    root = tk.Tk()
    app = CADFireApp(
        root,
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        display_scale=display_scale,
    )
    root.mainloop()
