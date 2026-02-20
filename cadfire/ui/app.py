"""
CADFireApp: main Tkinter window for the interactive demo.

Layout
──────
┌─────────────────────────────────────────────────────┐
│  Menu: File | Edit | View | Help                    │
├──────────┬──────────────────────┬───────────────────┤
│ Toolbar  │  Canvas  (scaled)    │  Right Panel      │
│ (left)   │                      │  ├ AI / Prompt    │
│          │                      │  ├ Training Wheels│
│          │                      │  ├ Layers         │
│          │                      │  └ Properties     │
├──────────┴──────────────────────┴───────────────────┤
│  Status bar: tool | world XY | entities | zoom      │
└─────────────────────────────────────────────────────┘

Human drawing
─────────────
Click on the canvas to feed world-space coordinates to the active tool.
Multi-step tools (LINE, CIRCLE …) collect clicks until they have enough
points, then commit.  POLYLINE / SPLINE / HATCH accumulate until Enter.

Keyboard shortcuts
──────────────────
  L  LINE    C  CIRCLE   R  RECT   A  ARC   P  POLYGON   E  ELLIPSE
  .  POINT   S  SELECT   X  ERASE  M  MOVE  O  COPY      V  FIT_VIEW
  Z  UNDO    Y  REDO     Esc  CANCEL   Enter  CONFIRM
  +/-  ZOOM   Middle-drag  PAN   Scroll  ZOOM

Training Wheels
───────────────
Category checkboxes mask the AI's tool logits before sampling.
Human toolbar always shows all tools regardless of the mask.
"""

from __future__ import annotations

import base64
import math
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cadfire.engine.cad_engine import CADEngine
from cadfire.renderer.rasterizer import Renderer
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_list, tool_to_index, index_to_tool, num_tools
from cadfire.utils.draw_utils import screen_to_world, world_to_screen
from cadfire.utils.color_utils import index_to_hex, DEFAULT_PALETTE, readable_text_color
from cadfire.ui.controller import AIController, TOOL_CATEGORIES

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── Constants ────────────────────────────────────────────────────────────────

# Keyboard key → tool name
TOOL_KEYS: Dict[str, str] = {
    "l": "LINE",     "c": "CIRCLE",    "r": "RECTANGLE", "a": "ARC",
    "p": "POLYGON",  "e": "ELLIPSE",   ".": "POINT",
    "s": "SELECT",   "x": "ERASE",     "m": "MOVE",      "o": "COPY",
    "v": "FIT_VIEW", "z": "UNDO",      "y": "REDO",
}

# How many world-space clicks each tool needs before it fires.
# 0 = execute immediately (no cursor), -1 = accumulate until CONFIRM
TOOL_CLICK_REQS: Dict[str, int] = {
    "LINE": 2, "RECTANGLE": 2, "CIRCLE": 2, "ELLIPSE": 2, "POLYGON": 2,
    "ARC": 3,  "MOVE": 2,     "COPY": 2,   "MIRROR": 2,
    "DIM_LINEAR": 2, "DIM_ALIGNED": 2,
    "POINT": 1, "SELECT": 1,  "MTEXT": 1,  "DTEXT": 1,
    "ROTATE": 1, "SCALE": 1,  "MATCHPROP": 1,
    "ERASE": 0, "DESELECT": 0, "ZOOM_IN": 0, "ZOOM_OUT": 0,
    "FIT_VIEW": 0, "ZOOM_EXTENTS": 0, "UNDO": 0, "REDO": 0,
    "CONFIRM": 0, "CANCEL": 0, "NOOP": 0, "EXPLODE": 0, "OFFSET": 0,
    "POLYLINE": -1, "SPLINE": -1, "HATCH": -1,
}

TOOL_CATEGORY_ICONS: Dict[str, str] = {
    "Drawing":    "✏",  "Annotation": "⊞", "Modify":     "⚙",
    "Arrays":     "▦",  "Selection":  "◻", "Properties": "#",
    "View":       "⊕",  "Control":    "↩",
}

# Dark theme colours
CANVAS_BG       = "#1a1a2e"
PANEL_BG        = "#16213e"
TOOL_BG         = "#0f3460"
TOOL_ACTIVE_BG  = "#e94560"
TOOL_ACTIVE_FG  = "#ffffff"
TOOL_NORMAL_FG  = "#a8c0d6"
STATUS_BG       = "#0d0d1a"
STATUS_FG       = "#7fbbdd"
GHOST_COLOR     = "#4488ff"


# ── Image conversion ─────────────────────────────────────────────────────────

def _rgb_to_photoimage(rgb: np.ndarray, scale: int) -> tk.PhotoImage:
    """
    Convert (H, W, 3) uint8 numpy array → Tkinter PhotoImage.
    Uses PIL when available; falls back to PPM base64 encoding otherwise.
    """
    if scale != 1:
        rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    if _HAS_PIL:
        return ImageTk.PhotoImage(Image.fromarray(rgb, mode="RGB"))
    H, W = rgb.shape[:2]
    ppm = f"P6\n{W} {H}\n255\n".encode() + rgb.astype(np.uint8).tobytes()
    return tk.PhotoImage(data=base64.b64encode(ppm).decode("ascii"))


# ── Application ──────────────────────────────────────────────────────────────

class CADFireApp:
    """
    Full interactive CADFire demo.  Instantiate, then call ``root.mainloop()``.
    """

    def __init__(self, root: tk.Tk, checkpoint_dir: str = "model_saves",
                 config_path: str | None = None, display_scale: int = 3):
        self.root = root
        self.root.title("CADFire — Interactive Demo")
        self.root.configure(bg=PANEL_BG)

        # Core objects
        self.config   = load_config(config_path) if config_path else load_config()
        self.engine   = CADEngine(self.config)
        self.renderer = Renderer(self.config)
        self.tokenizer = BPETokenizer(
            vocab_size=self.config["model"]["text_vocab_size"],
            max_len=self.config["model"]["text_max_len"],
        )
        self._tool_to_idx = tool_to_index()
        self._idx_to_tool = index_to_tool()
        self._tool_list   = tool_list()

        # Display
        self.display_scale = display_scale
        rw = self.config["canvas"]["render_width"]
        rh = self.config["canvas"]["render_height"]
        self.canvas_px_w = rw * display_scale
        self.canvas_px_h = rh * display_scale
        self._photo: Optional[tk.PhotoImage] = None

        # Human drawing state
        self.active_tool      = "LINE"
        self._pending_clicks: List[np.ndarray] = []
        self._last_mouse_world: Optional[np.ndarray] = None
        self._polygon_sides   = 6
        self._ghost_ids: List[int] = []

        # AI state
        self.ai_ctrl          = AIController(self._tool_list)
        self._auto_ai         = False
        self._ai_delay_ms     = 500
        self._ai_steps        = 0
        self._checkpoint_dir  = checkpoint_dir

        # Pan state
        self._pan_start_screen: Optional[np.ndarray] = None
        self._pan_start_center: Optional[np.ndarray] = None

        self._build_menu()
        self._build_layout()
        self._bind_events()
        self._update_display()
        self._update_status()

    # ─────────────────────────────────────────────────────── Menu ──────

    def _build_menu(self):
        m = tk.Menu(self.root, bg=PANEL_BG, fg=TOOL_NORMAL_FG,
                    activebackground=TOOL_ACTIVE_BG, tearoff=False)
        self.root.config(menu=m)

        def _add(parent, label, cmd):
            parent.add_command(label=label, command=cmd)

        f = tk.Menu(m, tearoff=False, bg=PANEL_BG, fg=TOOL_NORMAL_FG)
        m.add_cascade(label="File", menu=f)
        _add(f, "New",          self._cmd_new)
        _add(f, "Load Model…",  self._cmd_load_model)
        f.add_separator()
        _add(f, "Export DXF…",  self._cmd_export_dxf)
        f.add_separator()
        _add(f, "Quit",         self.root.quit)

        e = tk.Menu(m, tearoff=False, bg=PANEL_BG, fg=TOOL_NORMAL_FG)
        m.add_cascade(label="Edit", menu=e)
        _add(e, "Undo  Ctrl+Z",     self._cmd_undo)
        _add(e, "Redo  Ctrl+Y",     self._cmd_redo)
        e.add_separator()
        _add(e, "Deselect All",     self._cmd_deselect)
        _add(e, "Erase Selected",   self._cmd_erase)

        v = tk.Menu(m, tearoff=False, bg=PANEL_BG, fg=TOOL_NORMAL_FG)
        m.add_cascade(label="View", menu=v)
        _add(v, "Fit View  V",  self._cmd_fit_view)
        _add(v, "Zoom In   +",  self._cmd_zoom_in)
        _add(v, "Zoom Out  -",  self._cmd_zoom_out)

        h = tk.Menu(m, tearoff=False, bg=PANEL_BG, fg=TOOL_NORMAL_FG)
        m.add_cascade(label="Help", menu=h)
        _add(h, "Keyboard Shortcuts", self._show_help)

    # ─────────────────────────────────────────────────────── Layout ────

    def _build_layout(self):
        top = tk.Frame(self.root, bg=PANEL_BG)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._build_toolbar(top)
        self._build_canvas(top)
        self._build_right_panel(top)
        self._build_status_bar()

    # ── Left toolbar ────────────────────────────────────────────────────

    def _build_toolbar(self, parent):
        frame = tk.Frame(parent, bg=TOOL_BG, width=118, relief=tk.RIDGE, bd=1)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0), pady=2)
        frame.pack_propagate(False)

        tk.Label(frame, text="TOOLS", bg=TOOL_BG, fg=TOOL_NORMAL_FG,
                 font=("Helvetica", 8, "bold")).pack(pady=(5, 1))

        self._tool_btns: Dict[str, tk.Button] = {}

        for cat, tools in TOOL_CATEGORIES.items():
            icon = TOOL_CATEGORY_ICONS.get(cat, "")
            tk.Label(frame, text=f"{icon} {cat}", bg=TOOL_BG,
                     fg="#4477aa", font=("Helvetica", 7, "bold"),
                     anchor=tk.W, padx=5).pack(fill=tk.X, pady=(3, 0))
            for t in tools:
                b = tk.Button(
                    frame, text=t.replace("_", " "),
                    font=("Courier", 7), width=13,
                    bg=TOOL_BG, fg=TOOL_NORMAL_FG,
                    activebackground=TOOL_ACTIVE_BG, activeforeground=TOOL_ACTIVE_FG,
                    relief=tk.FLAT, bd=0, cursor="hand2",
                    command=lambda n=t: self._select_tool(n),
                )
                b.pack(fill=tk.X, padx=3, pady=1)
                self._tool_btns[t] = b

        self._refresh_toolbar()

    # ── Centre canvas ───────────────────────────────────────────────────

    def _build_canvas(self, parent):
        box = tk.Frame(parent, bg=CANVAS_BG, relief=tk.SUNKEN, bd=2)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.canvas = tk.Canvas(box, width=self.canvas_px_w, height=self.canvas_px_h,
                                bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(expand=True)

    # ── Right notebook ──────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, width=256, relief=tk.RIDGE, bd=1)
        frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=2)
        frame.pack_propagate(False)

        nb = ttk.Notebook(frame)
        nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._build_ai_tab(nb)
        self._build_wheels_tab(nb)
        self._build_layers_tab(nb)
        self._build_props_tab(nb)

    # ── AI tab ──────────────────────────────────────────────────────────

    def _build_ai_tab(self, nb):
        tab = tk.Frame(nb, bg=PANEL_BG)
        nb.add(tab, text=" AI ")

        self._ai_info_lbl = tk.Label(tab, text=self.ai_ctrl.model_info,
            bg=PANEL_BG, fg="#aaddaa", font=("Courier", 8),
            wraplength=238, justify=tk.LEFT, anchor=tk.W)
        self._ai_info_lbl.pack(fill=tk.X, padx=6, pady=(6, 2))

        tk.Button(tab, text="Load Model…", bg=TOOL_BG, fg=TOOL_NORMAL_FG,
                  font=("Helvetica", 9), relief=tk.FLAT,
                  command=self._cmd_load_model).pack(fill=tk.X, padx=6, pady=2)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=4)

        tk.Label(tab, text="Prompt:", bg=PANEL_BG, fg=TOOL_NORMAL_FG,
                 font=("Helvetica", 9)).pack(anchor=tk.W, padx=6)
        self._prompt_var = tk.StringVar()
        e = tk.Entry(tab, textvariable=self._prompt_var,
                     bg="#1a2a3a", fg="#ddeeff", insertbackground="white",
                     font=("Helvetica", 9), relief=tk.FLAT, bd=2)
        e.pack(fill=tk.X, padx=6, pady=2)
        e.bind("<Return>", lambda _: self._ai_step())

        row = tk.Frame(tab, bg=PANEL_BG)
        row.pack(fill=tk.X, padx=6, pady=4)
        tk.Button(row, text="▶ Run Step", bg="#2255aa", fg="white",
                  font=("Helvetica", 9, "bold"), relief=tk.FLAT,
                  command=self._ai_step).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        self._auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, text="Auto", variable=self._auto_var,
                       command=self._toggle_auto,
                       bg=PANEL_BG, fg=TOOL_NORMAL_FG, selectcolor=TOOL_BG,
                       activebackground=PANEL_BG).pack(side=tk.LEFT)

        spd = tk.Frame(tab, bg=PANEL_BG)
        spd.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(spd, text="Speed:", bg=PANEL_BG, fg=TOOL_NORMAL_FG,
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        self._spd_var = tk.IntVar(value=self._ai_delay_ms)
        tk.Scale(spd, from_=100, to=3000, orient=tk.HORIZONTAL,
                 variable=self._spd_var, length=140, bg=PANEL_BG, fg=TOOL_NORMAL_FG,
                 troughcolor=TOOL_BG, showvalue=False,
                 command=lambda v: setattr(self, "_ai_delay_ms", int(v))
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._det_var = tk.BooleanVar(value=False)
        tk.Checkbutton(tab, text="Deterministic (argmax)", variable=self._det_var,
                       bg=PANEL_BG, fg=TOOL_NORMAL_FG, selectcolor=TOOL_BG,
                       activebackground=PANEL_BG).pack(anchor=tk.W, padx=6)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=4)

        tk.Label(tab, text="Last AI Action:", bg=PANEL_BG, fg=TOOL_NORMAL_FG,
                 font=("Helvetica", 8, "bold")).pack(anchor=tk.W, padx=6)
        self._ai_action_lbl = tk.Label(tab, text="—", bg=PANEL_BG, fg="#88ccff",
                                       font=("Courier", 9), justify=tk.LEFT,
                                       anchor=tk.W, wraplength=238)
        self._ai_action_lbl.pack(fill=tk.X, padx=6, pady=2)
        self._ai_step_lbl = tk.Label(tab, text="Steps: 0", bg=PANEL_BG,
                                     fg=STATUS_FG, font=("Courier", 8))
        self._ai_step_lbl.pack(anchor=tk.W, padx=6)

        tk.Button(tab, text="Reset Canvas", bg="#551111", fg="white",
                  font=("Helvetica", 9), relief=tk.FLAT,
                  command=self._cmd_new).pack(fill=tk.X, padx=6, pady=(8, 2))

    # ── Training Wheels tab ─────────────────────────────────────────────

    def _build_wheels_tab(self, nb):
        tab = tk.Frame(nb, bg=PANEL_BG)
        nb.add(tab, text=" Wheels ")

        tk.Label(tab, text="Training Wheels\nEnable / disable tool categories\nfor the AI:",
                 bg=PANEL_BG, fg=TOOL_NORMAL_FG, font=("Helvetica", 9),
                 justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(8, 4))

        self._tw_vars: Dict[str, tk.BooleanVar] = {}
        for cat in TOOL_CATEGORIES:
            var = tk.BooleanVar(value=True)
            self._tw_vars[cat] = var
            row = tk.Frame(tab, bg=PANEL_BG)
            row.pack(fill=tk.X, padx=8, pady=1)
            icon = TOOL_CATEGORY_ICONS.get(cat, "")
            tk.Checkbutton(row, text=f"{icon}  {cat}",
                           variable=var,
                           command=lambda c=cat, v=var: self.ai_ctrl.tool_mask.set_category(c, v.get()),
                           bg=PANEL_BG, fg=TOOL_NORMAL_FG, selectcolor=TOOL_BG,
                           activebackground=PANEL_BG,
                           font=("Helvetica", 9)).pack(side=tk.LEFT)
            tk.Label(row, text=f"({len(TOOL_CATEGORIES[cat])})",
                     bg=PANEL_BG, fg="#446688", font=("Helvetica", 8)
                     ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        presets = tk.Frame(tab, bg=PANEL_BG)
        presets.pack(fill=tk.X, padx=8)

        def _preset_btn(text, cmd):
            tk.Button(presets, text=text, bg=TOOL_BG, fg=TOOL_NORMAL_FG,
                      font=("Helvetica", 8), relief=tk.FLAT, command=cmd,
                      ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=1)

        _preset_btn("All",          lambda: self._tw_set_all(True))
        _preset_btn("Drawing only", self._tw_drawing_only)
        _preset_btn("No Pan/Zoom",  self._tw_no_pan_zoom)

    # ── Layers tab ──────────────────────────────────────────────────────

    def _build_layers_tab(self, nb):
        tab = tk.Frame(nb, bg=PANEL_BG)
        nb.add(tab, text=" Layers ")

        lf = tk.Frame(tab, bg=PANEL_BG)
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        sb = tk.Scrollbar(lf, orient=tk.VERTICAL)
        self._layer_lb = tk.Listbox(lf, yscrollcommand=sb.set,
                                    bg="#0a1520", fg=TOOL_NORMAL_FG,
                                    selectbackground=TOOL_ACTIVE_BG,
                                    font=("Courier", 9), relief=tk.FLAT,
                                    bd=0, activestyle="none")
        sb.config(command=self._layer_lb.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._layer_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._layer_lb.bind("<<ListboxSelect>>", self._on_layer_sel)

        tk.Button(tab, text="Toggle Visibility", bg=TOOL_BG, fg=TOOL_NORMAL_FG,
                  relief=tk.FLAT, command=self._cmd_layer_toggle,
                  font=("Helvetica", 8)).pack(fill=tk.X, padx=4, pady=2)
        self._refresh_layers()

    # ── Properties tab ──────────────────────────────────────────────────

    def _build_props_tab(self, nb):
        tab = tk.Frame(nb, bg=PANEL_BG)
        nb.add(tab, text=" Props ")

        tk.Label(tab, text="Active Color:", bg=PANEL_BG, fg=TOOL_NORMAL_FG,
                 font=("Helvetica", 9)).pack(anchor=tk.W, padx=8, pady=(8, 2))
        cf = tk.Frame(tab, bg=PANEL_BG)
        cf.pack(padx=8, pady=2)
        for i, rgb in enumerate(DEFAULT_PALETTE):
            hex_c = index_to_hex(i)
            text_c = "#000000" if sum(rgb) > 382 else "#ffffff"
            lbl = tk.Label(cf, text=str(i), width=3, bg=hex_c, fg=text_c,
                           font=("Courier", 8, "bold"), relief=tk.RAISED, cursor="hand2")
            lbl.grid(row=i // 4, column=i % 4, padx=2, pady=2)
            lbl.bind("<Button-1>", lambda _, idx=i: self._set_color(idx))

        self._color_lbl = tk.Label(tab, text="Color 0 — White",
                                   bg=index_to_hex(0), fg="black",
                                   font=("Helvetica", 9))
        self._color_lbl.pack(fill=tk.X, padx=8, pady=4)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        sf = tk.Frame(tab, bg=PANEL_BG)
        sf.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(sf, text="Polygon sides:", bg=PANEL_BG, fg=TOOL_NORMAL_FG,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._sides_var = tk.IntVar(value=self._polygon_sides)
        tk.Spinbox(sf, from_=3, to=24, textvariable=self._sides_var, width=4,
                   bg="#1a2a3a", fg="#ddeeff",
                   command=lambda: setattr(self, "_polygon_sides",
                                           max(3, min(24, self._sides_var.get())))
                   ).pack(side=tk.LEFT, padx=4)

    # ─────────────────────────────────────────────────── Status bar ────

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=STATUS_BG, height=22)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        def _lbl(text, **kw):
            l = tk.Label(bar, text=text, bg=STATUS_BG, fg=STATUS_FG,
                         font=("Courier", 9), padx=8, **kw)
            return l

        self._st_tool    = _lbl("Tool: LINE",   fg=TOOL_ACTIVE_BG)
        self._st_tool.pack(side=tk.LEFT)
        self._st_coords  = _lbl("X: —   Y: —")
        self._st_coords.pack(side=tk.LEFT)
        self._st_ents    = _lbl("Ents: 0")
        self._st_ents.pack(side=tk.LEFT)
        self._st_sel     = _lbl("Sel: 0")
        self._st_sel.pack(side=tk.LEFT)
        self._st_pending = _lbl("", fg="#ffcc44")
        self._st_pending.pack(side=tk.LEFT)
        self._st_zoom    = _lbl("Zoom: 1.0×")
        self._st_zoom.pack(side=tk.RIGHT)

    # ─────────────────────────────────────────────────── Events ────────

    def _bind_events(self):
        c = self.canvas
        c.bind("<Button-1>",   self._on_click)
        c.bind("<Motion>",     self._on_move)
        c.bind("<Button-2>",   self._on_pan_start)
        c.bind("<B2-Motion>",  self._on_pan_drag)
        c.bind("<MouseWheel>", self._on_scroll)
        c.bind("<Button-4>",   lambda _: self._do_zoom(True))
        c.bind("<Button-5>",   lambda _: self._do_zoom(False))
        self.root.bind("<Key>",       self._on_key)
        self.root.bind("<Return>",    lambda _: self._exec("CONFIRM", None))
        self.root.bind("<Escape>",    lambda _: self._exec("CANCEL",  None))
        self.root.bind("<Control-z>", lambda _: self._cmd_undo())
        self.root.bind("<Control-y>", lambda _: self._cmd_redo())
        self.root.bind("<plus>",      lambda _: self._cmd_zoom_in())
        self.root.bind("<equal>",     lambda _: self._cmd_zoom_in())
        self.root.bind("<minus>",     lambda _: self._cmd_zoom_out())

    # ─────────────────────────────────────────────── Canvas helpers ────

    def _s2w(self, sx: float, sy: float) -> np.ndarray:
        """Screen pixel → world coordinate."""
        cs  = self.config["canvas"]["render_width"] * self.display_scale
        ws  = self.config["canvas"]["world_width"]
        return screen_to_world(np.array([sx, sy]),
                               self.engine.viewport.center,
                               self.engine.viewport.zoom, ws, cs)

    def _w2s(self, wx: float, wy: float) -> Tuple[float, float]:
        """World coordinate → screen pixel."""
        cs  = self.config["canvas"]["render_width"] * self.display_scale
        ws  = self.config["canvas"]["world_width"]
        p = world_to_screen(np.array([[wx, wy]]),
                            self.engine.viewport.center,
                            self.engine.viewport.zoom, ws, cs)[0]
        return float(p[0]), float(p[1])

    # ─────────────────────────────────────────────── Mouse handlers ────

    def _on_click(self, ev):
        self._handle_click(self._s2w(ev.x, ev.y))

    def _on_move(self, ev):
        w = self._s2w(ev.x, ev.y)
        self._last_mouse_world = w
        self._draw_ghost(ev.x, ev.y)
        self._st_coords.config(text=f"X: {w[0]:8.1f}   Y: {w[1]:8.1f}")

    def _on_pan_start(self, ev):
        self._pan_start_screen = np.array([ev.x, ev.y], dtype=np.float64)
        self._pan_start_center = self.engine.viewport.center.copy()

    def _on_pan_drag(self, ev):
        if self._pan_start_screen is None:
            return
        cs  = self.config["canvas"]["render_width"] * self.display_scale
        vis = self.config["canvas"]["world_width"] / self.engine.viewport.zoom
        dx  = -(ev.x - self._pan_start_screen[0]) / cs * vis
        dy  =  (ev.y - self._pan_start_screen[1]) / cs * vis
        self.engine.viewport.center = self._pan_start_center + np.array([dx, dy])
        self._update_display()

    def _on_scroll(self, ev):
        self._do_zoom(ev.delta > 0)

    def _do_zoom(self, zoom_in: bool):
        if zoom_in:
            self.engine.zoom_in()
        else:
            self.engine.zoom_out()
        self._update_display()
        self._update_status()

    def _on_key(self, ev):
        ch = ev.char.lower()
        if ch in TOOL_KEYS:
            t = TOOL_KEYS[ch]
            if TOOL_CLICK_REQS.get(t, 1) == 0:
                self._exec(t, None)
            else:
                self._select_tool(t)

    # ──────────────────────────────────────────────── Ghost preview ────

    def _draw_ghost(self, sx: float, sy: float):
        for gid in self._ghost_ids:
            self.canvas.delete(gid)
        self._ghost_ids.clear()

        if not self._pending_clicks:
            return

        fp = self._pending_clicks[0]
        fx, fy = self._w2s(fp[0], fp[1])

        t = self.active_tool
        if t in ("LINE", "RECTANGLE", "MOVE", "COPY", "MIRROR"):
            self._ghost_ids.append(self.canvas.create_line(
                fx, fy, sx, sy, fill=GHOST_COLOR, dash=(4, 4), width=1))
        elif t == "CIRCLE":
            r = math.hypot(sx - fx, sy - fy)
            self._ghost_ids.append(self.canvas.create_oval(
                fx-r, fy-r, fx+r, fy+r, outline=GHOST_COLOR, dash=(4,4), width=1))
        elif t in ("POLYLINE", "SPLINE", "HATCH") and len(self._pending_clicks) >= 1:
            coords = []
            for p in self._pending_clicks:
                px, py = self._w2s(p[0], p[1])
                coords += [px, py]
            coords += [sx, sy]
            if len(coords) >= 4:
                self._ghost_ids.append(self.canvas.create_line(
                    *coords, fill=GHOST_COLOR, dash=(4,4), width=1))

        # Dot at each collected click
        for p in self._pending_clicks:
            px, py = self._w2s(p[0], p[1])
            self._ghost_ids.append(self.canvas.create_oval(
                px-4, py-4, px+4, py+4, fill=GHOST_COLOR, outline="white", width=1))

    # ───────────────────────────────────────── Human tool execution ────

    def _select_tool(self, name: str):
        if name != self.active_tool:
            self._pending_clicks.clear()
            for gid in self._ghost_ids:
                self.canvas.delete(gid)
            self._ghost_ids.clear()
        self.active_tool = name
        self._refresh_toolbar()
        self._update_status()

    def _handle_click(self, world: np.ndarray):
        req = TOOL_CLICK_REQS.get(self.active_tool, 1)
        if req == 0:
            self._exec(self.active_tool, None)
            return
        self._pending_clicks.append(world.copy())
        if req == -1:          # accumulate – update display only
            self._update_display()
            return
        if len(self._pending_clicks) >= req:
            pts = self._pending_clicks.copy()
            self._pending_clicks.clear()
            for gid in self._ghost_ids:
                self.canvas.delete(gid)
            self._ghost_ids.clear()
            self._commit_clicks(self.active_tool, pts)

    def _commit_clicks(self, tool: str, pts: List[np.ndarray]):
        """Convert collected world points into cursor heatmaps and execute."""
        W = self.config["canvas"]["render_width"]
        H = self.config["canvas"]["render_height"]
        eng = self.engine

        def _to_px(wp):
            ndc = eng.viewport.world_to_ndc(wp.reshape(1, 2))[0]
            px = max(0, min(W-1, int(ndc[0] * W)))
            py = max(0, min(H-1, int((1.0 - ndc[1]) * H)))
            return px, py

        for pt in pts:
            hm = np.zeros((H, W), dtype=np.float32)
            px, py = _to_px(pt)
            hm[py, px] = 1.0
            self._exec(tool, hm)

    def _exec(self, tool: str, cursor: np.ndarray | None, param: float = 0.0):
        """
        Dispatch one tool action to the engine.
        Mirrors CADEnv._execute_tool so human and AI share the same path.
        """
        eng = self.engine
        W   = self.config["canvas"]["render_width"]
        H   = self.config["canvas"]["render_height"]

        # Decode cursor
        cursor_world = None
        cursor_mask  = None
        if cursor is not None:
            if tool == "MULTISELECT":
                cursor_mask = (cursor > 0.5).astype(np.float32)
            else:
                flat = int(np.argmax(cursor))
                py, px = divmod(flat, W)
                ndc = np.array([[px / W, 1.0 - py / H]])
                cursor_world = eng.viewport.ndc_to_world(ndc)[0]

        # ── Drawing ──────────────────────────────────────────────────
        if tool == "LINE" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world); eng.active_tool = "LINE"
            else:
                eng.draw_line(eng.pending_points.pop(0), cursor_world)
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "POLYLINE" and cursor_world is not None:
            eng.pending_points.append(cursor_world); eng.active_tool = "POLYLINE"

        elif tool == "CIRCLE" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world); eng.active_tool = "CIRCLE"
            else:
                c = eng.pending_points.pop(0)
                eng.draw_circle(c, max(float(np.linalg.norm(cursor_world - c)), 1.0))
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "ARC" and cursor_world is not None:
            eng.pending_points.append(cursor_world)
            if len(eng.pending_points) >= 3:
                c  = eng.pending_points[0]
                r  = max(float(np.linalg.norm(eng.pending_points[1] - c)), 1.0)
                sa = math.degrees(math.atan2(eng.pending_points[1][1]-c[1],
                                             eng.pending_points[1][0]-c[0]))
                ea = math.degrees(math.atan2(eng.pending_points[2][1]-c[1],
                                             eng.pending_points[2][0]-c[0]))
                eng.draw_arc(c, r, sa, ea)
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "RECTANGLE" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world); eng.active_tool = "RECTANGLE"
            else:
                p1 = eng.pending_points.pop(0)
                corner = np.minimum(p1, cursor_world)
                sz = np.abs(cursor_world - p1)
                eng.draw_rectangle(corner, max(sz[0], 1.0), max(sz[1], 1.0))
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "POLYGON" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world); eng.active_tool = "POLYGON"
            else:
                c = eng.pending_points.pop(0)
                eng.draw_polygon(c, max(float(np.linalg.norm(cursor_world - c)), 1.0),
                                 self._polygon_sides)
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "ELLIPSE" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world); eng.active_tool = "ELLIPSE"
            else:
                c = eng.pending_points.pop(0)
                d = cursor_world - c
                eng.draw_ellipse(c, max(abs(d[0]), 1.0), max(abs(d[1]), 1.0))
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "SPLINE" and cursor_world is not None:
            eng.pending_points.append(cursor_world); eng.active_tool = "SPLINE"

        elif tool == "POINT" and cursor_world is not None:
            eng.draw_point(cursor_world)

        elif tool == "HATCH" and cursor_world is not None:
            eng.pending_points.append(cursor_world); eng.active_tool = "HATCH"

        elif tool == "MTEXT" and cursor_world is not None:
            t = simpledialog.askstring("Text", "Enter text:", parent=self.root) or "Text"
            eng.draw_text(cursor_world, t, height=20.0, multiline=True)

        elif tool == "DTEXT" and cursor_world is not None:
            t = simpledialog.askstring("Text", "Enter text:", parent=self.root) or "Text"
            eng.draw_text(cursor_world, t, height=15.0)

        # ── Modify ───────────────────────────────────────────────────
        elif tool == "MOVE" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world)
            else:
                base = eng.pending_points.pop(0)
                eng.move_selected(*(cursor_world - base).tolist())
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "COPY" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world)
            else:
                base = eng.pending_points.pop(0)
                eng.copy_selected(*(cursor_world - base).tolist())
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "ROTATE" and cursor_world is not None:
            angle = simpledialog.askfloat("Rotate", "Angle (degrees):",
                                          parent=self.root) or 90.0
            eng.rotate_selected(angle, cursor_world)

        elif tool == "SCALE" and cursor_world is not None:
            factor = simpledialog.askfloat("Scale", "Scale factor:",
                                           parent=self.root) or 2.0
            eng.scale_selected(factor, cursor_world)

        elif tool == "MIRROR" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world)
            else:
                p1 = eng.pending_points.pop(0)
                eng.mirror_selected(p1, cursor_world)
                eng.pending_points.clear(); eng.active_tool = "NOOP"

        elif tool == "OFFSET":
            d = simpledialog.askfloat("Offset", "Distance:", parent=self.root) or 10.0
            eng.offset_selected(d)

        elif tool == "ERASE":
            eng.erase_selected()

        elif tool == "EXPLODE":
            eng.explode_selected()

        elif tool == "MATCHPROP" and cursor_world is not None:
            hit = eng.select_at_point(cursor_world)
            if hit:
                eng.matchprop(hit)

        # ── Selection ────────────────────────────────────────────────
        elif tool == "SELECT" and cursor_world is not None:
            eng.select_at_point(cursor_world, tolerance=20.0)

        elif tool == "MULTISELECT" and cursor_mask is not None:
            eng.select_in_region(cursor_mask, W, H)

        elif tool == "DESELECT":
            eng.deselect_all()

        # ── Layer / Properties ───────────────────────────────────────
        elif tool == "LAYER_SET":
            eng.set_layer(max(0, min(int(param), len(eng.layers)-1)))
        elif tool == "LAYER_OFF":
            eng.layer_off(max(0, min(int(param), len(eng.layers)-1)))
        elif tool == "LAYER_ON":
            eng.layer_on(max(0, min(int(param), len(eng.layers)-1)))
        elif tool == "COLOR_SET":
            eng.active_color = max(0, min(int(param), 7))

        # ── Viewport ─────────────────────────────────────────────────
        elif tool == "ZOOM_IN":
            eng.zoom_in()
        elif tool == "ZOOM_OUT":
            eng.zoom_out()
        elif tool == "ZOOM_EXTENTS":
            eng.zoom_extents()
        elif tool == "FIT_VIEW":
            eng.fit_view()
        elif tool == "PAN" and cursor_world is not None:
            if not eng.pending_points:
                eng.pending_points.append(cursor_world)
            else:
                base = eng.pending_points.pop(0)
                eng.viewport.center -= cursor_world - base
                eng.pending_points.clear()

        # ── Undo / Redo ──────────────────────────────────────────────
        elif tool == "UNDO":
            eng.undo()
        elif tool == "REDO":
            eng.redo()

        # ── Multi-step commit / cancel ───────────────────────────────
        elif tool == "CONFIRM":
            at = eng.active_tool
            if at == "POLYLINE" and len(eng.pending_points) >= 2:
                eng.draw_polyline(np.array(eng.pending_points))
            elif at == "SPLINE" and len(eng.pending_points) >= 2:
                eng.draw_spline(np.array(eng.pending_points))
            elif at == "HATCH" and len(eng.pending_points) >= 3:
                eng.draw_hatch(np.array(eng.pending_points))
            eng.pending_points.clear(); eng.active_tool = "NOOP"
            self._pending_clicks.clear()

        elif tool == "CANCEL":
            eng.pending_points.clear()
            eng.ghost_entities.clear()
            eng.active_tool = "NOOP"
            self._pending_clicks.clear()
            for gid in self._ghost_ids:
                self.canvas.delete(gid)
            self._ghost_ids.clear()

        self._update_display()
        self._update_status()

    # ───────────────────────────────────────────────── AI execution ────

    def _build_obs(self) -> Dict[str, np.ndarray]:
        image    = self.renderer.render(self.engine)
        max_len  = self.config["model"]["text_max_len"]
        prompt   = self._prompt_var.get()
        ids      = (np.array(self.tokenizer.encode_padded(prompt), dtype=np.int32)
                    if prompt else np.zeros(max_len, dtype=np.int32))
        return {"image": image, "text_ids": ids, "state_vec": self._build_state()}

    def _build_state(self) -> np.ndarray:
        eng  = self.engine
        cfg  = self.config
        n    = num_tools()
        dim  = cfg["model"]["state_dim"]
        w, h = cfg["canvas"]["world_width"], cfg["canvas"]["world_height"]
        vec  = np.zeros(dim, dtype=np.float32)
        vec[0] = self._tool_to_idx.get(eng.active_tool, 0) / max(n, 1)
        vec[1] = float(np.log1p(eng.viewport.zoom)) / 5.0
        vec[2] = eng.viewport.center[0] / w
        vec[3] = eng.viewport.center[1] / h
        vec[4] = eng.active_layer / max(len(eng.layers), 1)
        vec[5] = eng.active_color / 8.0
        vec[6] = min(len(eng.entities), 100) / 100.0
        vec[7] = min(len(eng.selected_ids), 50) / 50.0
        vec[8] = min(len(eng.pending_points), 10) / 10.0
        return vec

    def _ai_step(self):
        if not self.ai_ctrl.has_model:
            messagebox.showinfo("No model",
                                "Load a checkpoint first (File → Load Model…).")
            return
        self.ai_ctrl.step_async(self._build_obs(),
                                callback=self._on_ai_result,
                                deterministic=self._det_var.get())

    def _on_ai_result(self, result):
        self.root.after(0, lambda: self._apply_ai(result))

    def _apply_ai(self, result):
        if result is None or "error" in result:
            err = (result or {}).get("error", "None")
            self._ai_action_lbl.config(text=f"Error: {err}", fg="#ff6666")
            return
        tool_id  = result["tool_id"]
        cursor   = result["cursor"]
        param    = result.get("param", 0.0)
        entropy  = result.get("tool_entropy", 0.0)
        name     = self._idx_to_tool.get(tool_id, "NOOP")

        self._ai_steps += 1
        self._ai_step_lbl.config(text=f"Steps: {self._ai_steps}")
        self._ai_action_lbl.config(
            text=f"{name}\n  param={param:.3f}  H={entropy:.2f}", fg="#88ccff")

        self._exec(name, cursor, param)

        if self._auto_ai:
            self.root.after(self._ai_delay_ms, self._ai_step)

    def _toggle_auto(self):
        self._auto_ai = self._auto_var.get()
        if self._auto_ai:
            self._ai_step()

    # ─────────────────────────────────────────────────── Display ───────

    def _update_display(self):
        obs = self.renderer.render(self.engine)
        rgb = (obs[:, :, 0:3] * 255).clip(0, 255).astype(np.uint8)
        photo = _rgb_to_photoimage(rgb, self.display_scale)
        self._photo = photo
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def _update_status(self):
        eng = self.engine
        self._st_tool.config(text=f"Tool: {self.active_tool}")
        self._st_ents.config(text=f"Ents: {eng.entity_count()}")
        self._st_sel.config(text=f"Sel: {eng.selected_count()}")
        self._st_zoom.config(text=f"Zoom: {eng.viewport.zoom:.2f}×")
        n = len(self._pending_clicks) + len(eng.pending_points)
        self._st_pending.config(text=f"[{n} pts]" if n else "")

    def _refresh_toolbar(self):
        for name, btn in self._tool_btns.items():
            if name == self.active_tool:
                btn.config(bg=TOOL_ACTIVE_BG, fg=TOOL_ACTIVE_FG, relief=tk.RIDGE)
            else:
                btn.config(bg=TOOL_BG, fg=TOOL_NORMAL_FG, relief=tk.FLAT)

    def _refresh_layers(self):
        self._layer_lb.delete(0, tk.END)
        for i, layer in enumerate(self.engine.layers):
            vis = "👁" if layer.visible else "  "
            frz = "❄" if layer.frozen  else " "
            act = "►" if i == self.engine.active_layer else " "
            self._layer_lb.insert(tk.END, f"{act}{vis}{frz} {i}: {layer.name}")

    # ─────────────────────────────────────────────── UI callbacks ──────

    def _on_layer_sel(self, _=None):
        sel = self._layer_lb.curselection()
        if sel:
            self.engine.set_layer(sel[0])
            self._update_status()

    def _cmd_layer_toggle(self):
        sel = self._layer_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        if self.engine.layers[idx].visible:
            self.engine.layer_off(idx)
        else:
            self.engine.layer_on(idx)
        self._refresh_layers()
        self._update_display()

    def _set_color(self, idx: int):
        self.engine.active_color = idx
        names = {0:"White",1:"Red",2:"Yellow",3:"Green",4:"Cyan",
                 5:"Blue",6:"Magenta",7:"Gray"}
        self._color_lbl.config(text=f"Color {idx} — {names.get(idx,'')}",
                               bg=index_to_hex(idx))

    def _tw_set_all(self, on: bool):
        for cat, var in self._tw_vars.items():
            var.set(on)
            self.ai_ctrl.tool_mask.set_category(cat, on)

    def _tw_drawing_only(self):
        for cat, var in self._tw_vars.items():
            on = cat in ("Drawing", "Control")
            var.set(on); self.ai_ctrl.tool_mask.set_category(cat, on)

    def _tw_no_pan_zoom(self):
        for cat, var in self._tw_vars.items():
            on = (cat != "View")
            var.set(on); self.ai_ctrl.tool_mask.set_category(cat, on)

    # ─────────────────────────────────────────────── Menu commands ─────

    def _cmd_new(self):
        self.engine.reset()
        self._pending_clicks.clear()
        for gid in self._ghost_ids:
            self.canvas.delete(gid)
        self._ghost_ids.clear()
        self._ai_steps = 0
        self._ai_step_lbl.config(text="Steps: 0")
        self._ai_action_lbl.config(text="—")
        self._update_display(); self._update_status(); self._refresh_layers()

    def _cmd_load_model(self):
        path = filedialog.askdirectory(title="Select model_saves/ directory",
                                       initialdir=self._checkpoint_dir)
        if not path:
            return
        self._checkpoint_dir = path
        ok = self.ai_ctrl.load_model(path, self.config)
        self._ai_info_lbl.config(text=self.ai_ctrl.model_info,
                                 fg="#aaddaa" if ok else "#dd6666")

    def _cmd_export_dxf(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")])
        if not path:
            return
        try:
            from cadfire.export.dxf_writer import DXFWriter
            DXFWriter().write(self.engine.entities, path)
            messagebox.showinfo("Export", f"Saved:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _cmd_undo(self):
        self.engine.undo(); self._update_display(); self._update_status()

    def _cmd_redo(self):
        self.engine.redo(); self._update_display(); self._update_status()

    def _cmd_deselect(self):
        self.engine.deselect_all(); self._update_display(); self._update_status()

    def _cmd_erase(self):
        self.engine.erase_selected(); self._update_display(); self._update_status()

    def _cmd_fit_view(self):
        self.engine.fit_view(); self._update_display(); self._update_status()

    def _cmd_zoom_in(self):
        self.engine.zoom_in(); self._update_display(); self._update_status()

    def _cmd_zoom_out(self):
        self.engine.zoom_out(); self._update_display(); self._update_status()

    def _show_help(self):
        messagebox.showinfo("Keyboard Shortcuts",
            "L  Line            C  Circle\n"
            "R  Rectangle       A  Arc\n"
            "P  Polygon         E  Ellipse\n"
            ".  Point           S  Select\n"
            "X  Erase           M  Move\n"
            "O  Copy            V  Fit View\n"
            "Z  Undo            Y  Redo\n"
            "+  Zoom In         -  Zoom Out\n"
            "Enter  Confirm     Esc  Cancel\n\n"
            "Mouse\n"
            "Left-click  place point\n"
            "Mid-drag    pan\n"
            "Scroll      zoom\n")
