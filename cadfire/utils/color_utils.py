"""
Color utility functions for CADFire.

Provides helpers for working with the project's CAD color palette,
converting between formats (index, RGB tuple, hex string), and
computing derived colors (contrast, lighter/darker variants).

The canonical palette is defined in config.json and loaded by
:func:`load_palette`, but the default 8-color palette is also
available as a module-level constant so these helpers work without
a config file.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Default 8-color CAD palette mirroring config.json
# Index 0 = White (default drawing color), 7 = Gray
DEFAULT_PALETTE: List[Tuple[int, int, int]] = [
    (255, 255, 255),  # 0  White
    (255,  60,  60),  # 1  Red
    (255, 220,  50),  # 2  Yellow
    ( 60, 200,  80),  # 3  Green
    ( 60, 210, 210),  # 4  Cyan
    ( 80, 120, 255),  # 5  Blue
    (200,  80, 200),  # 6  Magenta
    (150, 150, 150),  # 7  Gray
]

CAD_COLOR_NAMES: Dict[int, str] = {
    0: "White",
    1: "Red",
    2: "Yellow",
    3: "Green",
    4: "Cyan",
    5: "Blue",
    6: "Magenta",
    7: "Gray",
}

BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)       # Black canvas
SELECTION_COLOR: Tuple[int, int, int] = (0, 180, 255)     # Selection highlight


# ── Palette Loading ──────────────────────────────────────────────────────────

def load_palette(config: dict | None = None) -> List[Tuple[int, int, int]]:
    """
    Return the palette as a list of (R, G, B) uint8 tuples.

    Falls back to :data:`DEFAULT_PALETTE` if config is not provided.
    """
    if config is None:
        return list(DEFAULT_PALETTE)
    raw = config.get("colors", {}).get("palette", DEFAULT_PALETTE)
    return [tuple(int(v) for v in c) for c in raw]  # type: ignore[return-value]


# ── Format Conversions ───────────────────────────────────────────────────────

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an (R, G, B) tuple to a '#RRGGBB' hex string."""
    r, g, b = (max(0, min(255, v)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert a '#RRGGBB' or 'RRGGBB' hex string to an (R, G, B) tuple."""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def index_to_rgb(
    color_index: int,
    palette: List[Tuple[int, int, int]] | None = None,
) -> Tuple[int, int, int]:
    """Return the (R, G, B) tuple for a palette color index."""
    pal = palette if palette is not None else DEFAULT_PALETTE
    return pal[color_index % len(pal)]


def index_to_hex(
    color_index: int,
    palette: List[Tuple[int, int, int]] | None = None,
) -> str:
    """Return the '#RRGGBB' hex string for a palette color index."""
    return rgb_to_hex(index_to_rgb(color_index, palette))


def rgb_to_float(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert uint8 RGB to float RGB in [0, 1]."""
    return rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0


def float_to_rgb(rgb_float: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert float RGB in [0, 1] to uint8 RGB."""
    return (
        max(0, min(255, round(rgb_float[0] * 255))),
        max(0, min(255, round(rgb_float[1] * 255))),
        max(0, min(255, round(rgb_float[2] * 255))),
    )


# ── Color Manipulation ───────────────────────────────────────────────────────

def lighten(rgb: Tuple[int, int, int], factor: float = 0.3) -> Tuple[int, int, int]:
    """Lighten a color towards white by *factor* (0 = unchanged, 1 = white)."""
    return tuple(
        min(255, round(v + (255 - v) * factor)) for v in rgb
    )  # type: ignore[return-value]


def darken(rgb: Tuple[int, int, int], factor: float = 0.3) -> Tuple[int, int, int]:
    """Darken a color towards black by *factor* (0 = unchanged, 1 = black)."""
    return tuple(
        max(0, round(v * (1.0 - factor))) for v in rgb
    )  # type: ignore[return-value]


def with_alpha(rgb: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int, int]:
    """Append an alpha channel (0.0–1.0) to an RGB tuple → RGBA."""
    return (*rgb, max(0, min(255, round(alpha * 255))))  # type: ignore[return-value]


def blend(
    fg: Tuple[int, int, int],
    bg: Tuple[int, int, int],
    alpha: float,
) -> Tuple[int, int, int]:
    """Alpha-blend *fg* over *bg*. alpha=1 → pure fg, alpha=0 → pure bg."""
    a = max(0.0, min(1.0, alpha))
    return tuple(
        max(0, min(255, round(fg[i] * a + bg[i] * (1.0 - a)))) for i in range(3)
    )  # type: ignore[return-value]


# ── Contrast / Legibility ────────────────────────────────────────────────────

def luminance(rgb: Tuple[int, int, int]) -> float:
    """
    Relative luminance of an sRGB colour (WCAG 2.1 formula).

    Returns a value in [0, 1] where 0 = black and 1 = white.
    """
    def linearise(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * linearise(r) + 0.7152 * linearise(g) + 0.0722 * linearise(b)


def contrast_ratio(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """WCAG contrast ratio between two colours. 1 = identical, 21 = maximum."""
    l1, l2 = luminance(c1), luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def readable_text_color(
    bg: Tuple[int, int, int],
    dark: Tuple[int, int, int] = (0, 0, 0),
    light: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[int, int, int]:
    """
    Choose either *dark* or *light* as text color to maximise legibility
    on *bg* according to WCAG contrast ratio.
    """
    if contrast_ratio(light, bg) >= contrast_ratio(dark, bg):
        return light
    return dark


# ── Nearest Palette Color ────────────────────────────────────────────────────

def nearest_palette_index(
    rgb: Tuple[int, int, int],
    palette: List[Tuple[int, int, int]] | None = None,
) -> int:
    """
    Return the palette index whose color is closest to *rgb* in RGB space.

    Useful for snapping an arbitrary picked color back to the CAD palette.
    """
    pal = np.asarray(palette if palette is not None else DEFAULT_PALETTE, dtype=np.float32)
    query = np.asarray(rgb, dtype=np.float32)
    dists = np.linalg.norm(pal - query, axis=1)
    return int(np.argmin(dists))


# ── Palette as numpy array ────────────────────────────────────────────────────

def palette_as_array(
    palette: List[Tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Return the palette as a (N, 3) uint8 numpy array."""
    pal = palette if palette is not None else DEFAULT_PALETTE
    return np.array(pal, dtype=np.uint8)
