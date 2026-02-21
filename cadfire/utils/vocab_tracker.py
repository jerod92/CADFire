"""
Vocabulary Tracker — meta-module for semantic learning-space analysis.

Scans every task file in ``cadfire/tasks/`` (and supervised sub-tasks),
extracts all text prompts / prompt templates, tokenises them, and builds a
vocabulary registry that records:

  • Which words/tokens appear in prompts and how often.
  • Which CAD concepts are well-represented vs. rarely covered.
  • Template-level breakdown by task category.

Typical usage
─────────────
    from cadfire.utils.vocab_tracker import VocabTracker

    tracker = VocabTracker()
    report  = tracker.build()           # dict of stats + word counts
    tracker.save("vocab_report.json")   # write to disk

    # Quick print summary:
    tracker.print_summary()

    # Find gaps vs. a reference CAD word list:
    gaps = tracker.find_gaps(reference_words=["extrude", "dimension", "hatch"])

CLI
───
    python -m cadfire.utils.vocab_tracker [--output vocab_report.json]

Design
──────
The tracker operates purely on *static* source code — it never instantiates
any task class, so it is safe to run at any time without the CAD engine.
It collects prompts from two sources:

  1. ``generate_prompt_variants(self)`` method return-value literals found by
     parsing the AST of each task file.
  2. Module-level list constants whose name ends in ``_PROMPTS`` (e.g.
     ``_TRACE_PROMPTS``, ``MULTITURN_PROMPTS``).

All found strings are lowercased, split on whitespace and punctuation, and
counted.  Format placeholders (``{x1}``, ``{angle}``) are stripped so that
structural scaffold words are separated from CAD-semantics words.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ── CAD reference vocabulary for gap analysis ────────────────────────────────

# A curated set of CAD-domain terms we would ideally want the model to
# encounter during training.  The tracker will flag any of these that appear
# fewer than MIN_GAP_COUNT times across all prompts.
_CAD_REFERENCE_VOCAB: Set[str] = {
    # Primitives
    "line", "circle", "arc", "rectangle", "polygon", "ellipse", "polyline",
    "spline", "point", "hatch", "text", "dimension",
    # Modifications
    "move", "copy", "rotate", "scale", "mirror", "offset", "trim", "extend",
    "fillet", "chamfer", "array", "explode", "join", "break", "lengthen",
    # Selection
    "select", "deselect", "erase", "delete",
    # Layers / properties
    "layer", "color", "colour", "linetype", "lineweight", "property",
    # Viewport
    "zoom", "pan", "fit", "view", "center",
    # Attributes
    "red", "blue", "green", "yellow", "cyan", "magenta", "white", "gray",
    "grey", "black",
    # Spatial / geometric descriptors
    "center", "radius", "diameter", "angle", "width", "height", "length",
    "start", "end", "from", "to", "at", "by", "around",
    # Trace / reference
    "trace", "outline", "follow", "match", "reference",
    # Multi-step
    "draw", "create", "place", "add", "make", "connect", "close",
    # Numbers & units (as words)
    "degrees", "units", "pixels",
}

_STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "in", "on", "for", "with",
    "and", "or", "use", "it", "its", "is", "are", "that", "this",
    "all", "some", "into", "as", "be",
}

MIN_GAP_COUNT = 3  # report if a reference term appears fewer than this many times


# ── AST helpers ──────────────────────────────────────────────────────────────

def _extract_string_list_from_ast(node: ast.AST) -> List[str]:
    """Recursively extract all string literals from an AST node (list/return)."""
    results: List[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        results.append(node.value)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            results.extend(_extract_string_list_from_ast(elt))
    elif isinstance(node, ast.Return) and node.value is not None:
        results.extend(_extract_string_list_from_ast(node.value))
    return results


def _scrape_file(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Parse a Python source file and return:
      - module_prompts  : strings from module-level ``*_PROMPTS`` lists
      - method_prompts  : {method_name: [strings]} from generate_prompt_variants
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return [], {}

    module_prompts: List[str] = []
    # Using a list of lists instead of a dict to avoid overwriting methods of the same name in different classes
    all_method_prompts: List[List[str]] = []

    for node in ast.walk(tree):
        # Module-level assignment: FOO_PROMPTS = [...]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_PROMPTS"):
                    module_prompts.extend(_extract_string_list_from_ast(node.value))

        # Method: generate_prompt_variants
        if isinstance(node, ast.FunctionDef) and node.name == "generate_prompt_variants":
            strings: List[str] = []
            # We only want to extract from the body statements (like the ast.Return) 
            # to avoid walk and extracting the same list multiple times.
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    strings.extend(_extract_string_list_from_ast(stmt.value))
            if strings:
                all_method_prompts.append(strings)

    # For compatibility with caller, convert back to a dummy dict
    method_prompts = {f"func_{i}": p for i, p in enumerate(all_method_prompts)}
    return module_prompts, method_prompts


# ── Tokenisation ─────────────────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(r"\{[^}]*\}")  # strip {x1}, {angle}, etc.
_NON_ALPHA_RE   = re.compile(r"[^a-z0-9\s]")


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip placeholders and punctuation, split into tokens."""
    has_color = "{color" in text.lower() or "{colour" in text.lower()
    text = _PLACEHOLDER_RE.sub(" ", text).lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    if has_color:
        tokens.extend([
            "red", "blue", "green", "yellow", "cyan", "magenta", 
            "white", "black", "gray", "grey", "color", "colour"
        ])
    return tokens


# ── VocabTracker ─────────────────────────────────────────────────────────────

class VocabTracker:
    """
    Scans all task source files and builds a vocabulary registry.

    Attributes
    ----------
    tasks_root : Path
        Root directory containing task modules (default: cadfire/tasks/).
    prompts_by_file : Dict[str, List[str]]
        Raw prompt strings keyed by relative file path.
    word_counts : Counter
        Global token frequency across all prompts.
    category_word_counts : Dict[str, Counter]
        Per-category token frequency (key = file stem or directory name).
    total_prompts : int
        Total number of distinct prompt template strings found.
    """

    def __init__(self, tasks_root: Optional[Path] = None):
        if tasks_root is None:
            # Locate cadfire/tasks/ relative to this file
            here = Path(__file__).resolve().parent          # cadfire/utils/
            tasks_root = here.parent / "tasks"
        self.tasks_root: Path = Path(tasks_root)

        self.prompts_by_file: Dict[str, List[str]] = {}
        self.word_counts: Counter = Counter()
        self.category_word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.total_prompts: int = 0
        self._built: bool = False

    # ── scanning ──────────────────────────────────────────────────────────

    def _collect_task_files(self) -> List[Path]:
        """Return all .py files under tasks_root (recursive)."""
        return sorted(self.tasks_root.rglob("*.py"))

    def _category_for(self, path: Path) -> str:
        """Derive a readable category label from the file path."""
        parts = path.relative_to(self.tasks_root).parts
        if len(parts) == 1:
            # Top-level file: strip suffix and _tasks
            stem = parts[0].replace("_tasks.py", "").replace(".py", "")
            return stem
        else:
            # Sub-directory: use directory name
            return parts[0]

    def build(self) -> Dict[str, Any]:
        """
        Scan all task files and build the vocabulary registry.

        Returns
        -------
        dict with keys:
          total_prompts         : int
          unique_tokens         : int
          top_tokens            : List[(token, count)]
          bottom_tokens         : List[(token, count)]  — least common
          category_summary      : Dict[str, int]         — prompt counts per category
          full_word_counts      : Dict[str, int]
          prompts_by_file       : Dict[str, List[str]]
          reference_gap_report  : Dict[str, int]         — under-represented terms
        """
        py_files = self._collect_task_files()

        for path in py_files:
            rel = str(path.relative_to(self.tasks_root))
            module_p, method_p = _scrape_file(path)
            all_prompts = module_p + [s for lst in method_p.values() for s in lst]
            if not all_prompts:
                continue

            self.prompts_by_file[rel] = all_prompts
            self.total_prompts += len(all_prompts)
            cat = self._category_for(path)

            for prompt in all_prompts:
                tokens = _tokenise(prompt)
                self.word_counts.update(tokens)
                self.category_word_counts[cat].update(tokens)

        self._built = True
        return self._make_report()

    # ── analysis ──────────────────────────────────────────────────────────

    def find_gaps(self, reference_words: Optional[List[str]] = None,
                  min_count: int = MIN_GAP_COUNT) -> Dict[str, int]:
        """
        Return reference words that appear fewer than ``min_count`` times.

        Parameters
        ----------
        reference_words :
            List of words to check.  Defaults to ``_CAD_REFERENCE_VOCAB``.
        min_count :
            Alert threshold.

        Returns
        -------
        Dict mapping under-represented word → its actual count (may be 0).
        """
        if not self._built:
            self.build()
        ref = set(reference_words) if reference_words else _CAD_REFERENCE_VOCAB
        return {
            word: self.word_counts.get(word, 0)
            for word in sorted(ref)
            if self.word_counts.get(word, 0) < min_count
        }

    def category_summary(self) -> Dict[str, int]:
        """Return total prompt count per category."""
        return {
            cat: sum(self.prompts_by_file.get(f, [])
                     and len(self.prompts_by_file[f]) or 0
                     for f in self.prompts_by_file
                     if self._category_for(self.tasks_root / f) == cat)
            for cat in self.category_word_counts
        }

    def _make_report(self) -> Dict[str, Any]:
        total_unique = len(self.word_counts)
        top50    = self.word_counts.most_common(50)
        bottom50 = self.word_counts.most_common()[-50:]

        # Prompt count per file
        file_prompt_counts = {f: len(ps) for f, ps in self.prompts_by_file.items()}

        # Per-category prompt totals
        cat_prompt_counts: Dict[str, int] = defaultdict(int)
        for f, ps in self.prompts_by_file.items():
            cat = self._category_for(self.tasks_root / f)
            cat_prompt_counts[cat] += len(ps)

        gaps = self.find_gaps()

        return {
            "total_prompts":        self.total_prompts,
            "unique_tokens":        total_unique,
            "top_tokens":           top50,
            "bottom_tokens":        bottom50,
            "category_summary":     dict(cat_prompt_counts),
            "file_prompt_counts":   file_prompt_counts,
            "full_word_counts":     dict(self.word_counts),
            "prompts_by_file":      self.prompts_by_file,
            "reference_gap_report": gaps,
        }

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self, output_path: str = "vocab_report.json") -> Path:
        """Save the full report as JSON."""
        if not self._built:
            self.build()
        report = self._make_report()
        # word_counts are Counter: serialize as sorted list for readability
        report["top_tokens"]    = [[w, c] for w, c in report["top_tokens"]]
        report["bottom_tokens"] = [[w, c] for w, c in report["bottom_tokens"]]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return out

    def print_summary(self) -> None:
        """Print a concise human-readable vocabulary summary."""
        if not self._built:
            self.build()

        print("\n══════════════════════════════════════════════")
        print("  CADFire Vocabulary Tracker  –  Summary")
        print("══════════════════════════════════════════════")
        print(f"  Total prompt templates : {self.total_prompts}")
        print(f"  Unique tokens          : {len(self.word_counts)}")
        print()

        print("  Prompts by category:")
        cat_totals: Dict[str, int] = defaultdict(int)
        for f, ps in self.prompts_by_file.items():
            cat = self._category_for(self.tasks_root / f)
            cat_totals[cat] += len(ps)
        for cat, n in sorted(cat_totals.items(), key=lambda x: -x[1]):
            print(f"    {cat:<25} {n:>4} templates")

        print()
        print("  Top-20 tokens:")
        for word, cnt in self.word_counts.most_common(20):
            bar = "█" * min(cnt, 40)
            print(f"    {word:<20} {cnt:>5}  {bar}")

        gaps = self.find_gaps()
        if gaps:
            print()
            print(f"  ⚠  Semantic gaps (< {MIN_GAP_COUNT} occurrences in reference CAD vocab):")
            for word, cnt in sorted(gaps.items(), key=lambda x: x[1]):
                status = "MISSING" if cnt == 0 else f"{cnt}×"
                print(f"    {word:<20} {status}")
        else:
            print("\n  ✓  No gaps found in reference CAD vocabulary.")

        print("══════════════════════════════════════════════\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan CADFire task prompts and report vocabulary coverage."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="vocab_report.json",
        help="Path to write the JSON report (default: vocab_report.json)"
    )
    parser.add_argument(
        "--tasks-root", type=str, default=None,
        help="Override path to cadfire/tasks/ directory"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Skip the printed summary (just write the JSON report)"
    )
    args = parser.parse_args()

    root = Path(args.tasks_root) if args.tasks_root else None
    tracker = VocabTracker(tasks_root=root)
    tracker.build()

    if not args.quiet:
        tracker.print_summary()

    out = tracker.save(args.output)
    print(f"[vocab_tracker] Full report written to: {out}")


if __name__ == "__main__":
    main()
