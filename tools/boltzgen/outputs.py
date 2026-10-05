"""Read what BoltzGen leaves behind.

BoltzGen runs a whole pipeline, not a single step: it designs backbones,
inverse folds them into sequences, refolds those against the target, scores
them and ranks the survivors. So its output tree is deep and its shape depends
on which steps ran.

    output/
      intermediate_designs/                 backbones
      intermediate_designs_inverse_folded/  sequences, refolds, metrics
      final_ranked_designs/
        final_<budget>_designs/             what you actually want
        final_designs_metrics_<budget>.csv

This reads the final ranked designs when they exist and falls back to whatever
structures were produced, so a run stopped after an early step still returns
something useful rather than an error.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

FINAL_DIR = "final_ranked_designs"
# Holds the backbone as designed, before it was refolded against the target.
BEFORE_REFOLDING = "before_refolding"
STRUCTURE_SUFFIXES = (".cif", ".pdb")


def describe_tree(root: Path, limit: int = 30) -> str:
    """A compact listing of what is under `root`, for error messages."""
    if not root.is_dir():
        return f"{root} does not exist"
    found = sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )
    if not found:
        return f"{root} is empty"
    shown = found[:limit]
    extra = f" (and {len(found) - limit} more)" if len(found) > limit else ""
    return f"{root} contains {shown}{extra}"


def read_metrics(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-design metrics, keyed by whatever the CSV calls each design.

    BoltzGen writes several metric tables across the pipeline. The final one is
    preferred; earlier ones fill in when a run stopped short.
    """
    tables = sorted(out_dir.rglob("*metrics*.csv"))
    preferred = [p for p in tables if "final" in p.name] or tables
    metrics: dict[str, dict[str, Any]] = {}

    for table in preferred:
        try:
            rows = list(csv.DictReader(table.read_text().splitlines()))
        except (OSError, csv.Error):
            continue
        for row in rows:
            # The identifying column is not consistently named across tables.
            key = next(
                (row[c] for c in ("design", "name", "design_name", "id", "sample")
                 if row.get(c)),
                None,
            )
            if not key:
                continue
            converted: dict[str, Any] = {}
            for column, value in row.items():
                try:
                    converted[column] = float(value)
                except (TypeError, ValueError):
                    converted[column] = value
            metrics[Path(str(key)).stem] = converted
    return metrics


def read_designs(out_dir: Path) -> list[dict[str, Any]]:
    """Collect the designs a run produced, best first where ranked.

    Returns:
        One entry per design with its local path, whether it came from the
        final ranked set, and any metrics found for it.

    Raises:
        FileNotFoundError: listing what is actually on disk.
    """
    def collect(root: Path) -> list[Path]:
        """Structures directly under `root`, ignoring pre-refolding copies.

        BoltzGen keeps each design twice, once as the raw backbone under
        `before_refolding` and once refolded against the target. The refolded
        one is the result; the other is an intermediate of the same design and
        would otherwise be returned as if it were a second candidate.
        """
        found: list[Path] = []
        for suffix in STRUCTURE_SUFFIXES:
            found.extend(
                p for p in sorted(root.glob(f"*{suffix}"))
                if BEFORE_REFOLDING not in p.parts
            )
        return found

    structures: list[Path] = []
    final = False

    # Prefer the finished shortlist, which is nested one level down and named
    # for the budget that produced it.
    final_root = out_dir / FINAL_DIR
    if final_root.is_dir():
        for candidate in sorted(final_root.glob("final_*_designs")):
            structures.extend(collect(candidate))
        if not structures:
            for candidate in sorted(final_root.glob("intermediate_ranked_*_designs")):
                structures.extend(collect(candidate))
        final = bool(structures)

    if not structures:
        for suffix in STRUCTURE_SUFFIXES:
            structures.extend(
                p for p in sorted(out_dir.rglob(f"*{suffix}"))
                if BEFORE_REFOLDING not in p.parts
            )

    if not structures:
        raise FileNotFoundError(
            f"BoltzGen produced no structures under {out_dir}. "
            f"{describe_tree(out_dir)}"
        )

    metrics = read_metrics(out_dir)
    designs: list[dict[str, Any]] = []
    for path in structures:
        entry: dict[str, Any] = {
            "structure": path,
            "path": path.relative_to(out_dir).as_posix(),
            "name": path.stem,
            "final": final,
        }
        entry.update(metrics.get(path.stem, {}))
        designs.append(entry)

    def rank(entry: dict[str, Any]) -> float:
        for key in ("rank", "score", "iptm", "confidence"):
            value = entry.get(key)
            if isinstance(value, float):
                # A rank column counts upward, so invert it to sort with the
                # score columns, which count the other way.
                return -value if key == "rank" else value
        return float("-inf")

    return sorted(designs, key=rank, reverse=True)
