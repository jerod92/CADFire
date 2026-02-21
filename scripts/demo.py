#!/usr/bin/env python3
"""
CADFire Interactive Demo Launcher

Usage
─────
  python scripts/demo.py
  python scripts/demo.py --checkpoint model_saves/
  python scripts/demo.py --scale 2          # smaller window (512x512 canvas)
  python scripts/demo.py --scale 4          # bigger window (1024x1024 canvas)

Requires: numpy, torch (optional – UI works without a loaded model),
          tkinter (stdlib), pillow (optional – faster canvas rendering).

Without a model loaded the demo acts as a standalone human CAD tool:
draw shapes with the mouse, export to DXF, undo/redo, etc.

Load a trained checkpoint via File → Load Model… or --checkpoint, then
type a prompt and press "Run Step" (or enable Auto) to watch the model act.

Use the Training Wheels panel to restrict which tool categories the model
is allowed to use — handy for curriculum evaluation.
"""

import argparse
import sys
import os

# Ensure the project root is on sys.path when running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="CADFire interactive demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="model_saves",
        help="Path to model_saves/ directory (default: model_saves/)",
    )
    parser.add_argument(
        "--scale", "-s",
        type=int, default=3,
        help="Canvas display scale factor (default: 3 → 768×768 canvas)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json (default: project root config.json)",
    )
    args = parser.parse_args()

    try:
        import tkinter as tk
    except ImportError:
        print("ERROR: tkinter is not available.\n"
              "Install it via your system package manager, e.g.:\n"
              "  sudo apt install python3-tk", file=sys.stderr)
        sys.exit(1)

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Note: Pillow not found – canvas will use PPM fallback (slightly slower).\n"
              "      Install with: pip install pillow")

    from cadfire.ui import launch_demo

    print("Starting CADFire demo…")
    print(f"  Canvas scale : {args.scale}× ({256*args.scale}×{256*args.scale} px)")
    print(f"  Checkpoint   : {args.checkpoint}")
    print("  Close the window or press Ctrl+C to exit.\n")

    launch_demo(
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        display_scale=args.scale,
    )


if __name__ == "__main__":
    main()
