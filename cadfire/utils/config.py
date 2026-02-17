"""Centralized config loader. Every module references this instead of hard-coding values."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


_CONFIG_CACHE: Dict[str, Any] = {}


def _find_config() -> Path:
    """Walk up from this file to find config.json at repo root."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config.json not found in any parent directory")


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load and cache config.json. Optionally override path."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE and path is None:
        return _CONFIG_CACHE
    config_path = Path(path) if path else _find_config()
    with open(config_path) as f:
        _CONFIG_CACHE = json.load(f)
    return _CONFIG_CACHE


def get(section: str, key: str | None = None) -> Any:
    """Shorthand: get('canvas', 'world_width') or get('tools')."""
    cfg = load_config()
    val = cfg[section]
    if key is not None:
        val = val[key]
    return val


def tool_list() -> List[str]:
    """Return ordered tool list from config."""
    return load_config()["tools"]


def tool_to_index() -> Dict[str, int]:
    """Return tool name -> index mapping."""
    return {name: i for i, name in enumerate(tool_list())}


def index_to_tool() -> Dict[int, str]:
    """Return index -> tool name mapping."""
    return {i: name for i, name in enumerate(tool_list())}


def num_tools() -> int:
    return len(tool_list())


def reload():
    """Force re-read from disk (useful after hot-updating config)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = {}
    load_config()
