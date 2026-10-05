"""Read what Protenix leaves in its output directory.

Protenix organises results by sample name and random seed, and the exact
nesting has changed between releases. So rather than hard-coding a path, this
searches the output tree for structures and pairs each with the confidence file
sitting nearest to it.

That choice is deliberate. Assuming a documented layout is what previously cost
a GPU run on another tool here: the prediction succeeded and only the reader
failed. Searching costs nothing at this scale and degrades into a clear report
of what is actually on disk instead of a crash.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STRUCTURE_PATTERNS = ("*.cif", "*.pdb")
# Confidence files are named with some variation across releases.
CONFIDENCE_HINTS = ("confidence", "summary")
# The per-atom confidence dump, holding the PAE matrix and per-atom pLDDT.
# Written only when a run asks for atom confidence.
FULL_DATA_HINT = "full_data"

# Scores worth surfacing when present, whatever the release calls its file.
SCORES = (
    "ptm", "iptm", "plddt", "complex_plddt", "gpde", "ranking_score",
    "chain_ptm", "chain_iptm", "chain_pair_iptm", "chain_plddt",
)

_SEED = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)
_SAMPLE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)


def describe_tree(root: Path, limit: int = 40) -> str:
    """A compact listing of what is under `root`, for error messages."""
    if not root.is_dir():
        return f"{root} does not exist"
    found = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    if not found:
        return f"{root} is empty"
    shown = found[:limit]
    suffix = f" (and {len(found) - limit} more)" if len(found) > limit else ""
    return f"{root} contains {shown}{suffix}"


def _index(text: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def _nearest(structure: Path, hints: tuple[str, ...]) -> Path | None:
    """The json matching one structure, chosen by sample index.

    Protenix names a structure `<job>_sample_0.cif` and its confidence files
    `<job>_summary_confidence_sample_0.json` and `<job>_full_data_sample_0.json`,
    so they share no stem and only the sample index ties them together.

    An ambiguous match returns None rather than the first candidate. Guessing
    would attach one sample's confidence to another's structure, which reads as
    a plausible result and is the kind of mismatch that is never noticed.
    """
    candidates = [
        p for p in structure.parent.glob("*.json")
        if any(hint in p.name.lower() for hint in hints)
    ]
    if not candidates:
        return None

    wanted = _index(structure.name, _SAMPLE)
    if wanted is not None:
        same = [p for p in candidates if _index(p.name, _SAMPLE) == wanted]
        if len(same) == 1:
            return same[0]
        if same:
            return None

    # No sample index anywhere, which is the single-prediction layout.
    return candidates[0] if len(candidates) == 1 else None


def _nearest_confidence(structure: Path) -> Path | None:
    """The summary confidence file belonging to one structure."""
    return _nearest(structure, CONFIDENCE_HINTS)


def _nearest_full_data(structure: Path) -> Path | None:
    """The per-atom confidence dump belonging to one structure, if written."""
    return _nearest(structure, (FULL_DATA_HINT,))


def read_predictions(out_dir: Path) -> list[dict[str, Any]]:
    """Collect every predicted structure under an output directory.

    Returns:
        One entry per structure with its local path, the seed and sample
        indices parsed from its path when present, and any confidence scores
        found alongside. Sorted by ranking score where available, best first.

    Raises:
        FileNotFoundError: listing what is actually on disk.
    """
    structures: list[Path] = []
    for pattern in STRUCTURE_PATTERNS:
        structures.extend(sorted(out_dir.rglob(pattern)))
    # An error directory holds failure notes, not results.
    structures = [p for p in structures if "ERR" not in p.parts]

    if not structures:
        raise FileNotFoundError(
            f"Protenix wrote no structures under {out_dir}. "
            f"{describe_tree(out_dir)}"
        )

    results: list[dict[str, Any]] = []
    for structure in structures:
        relative = structure.relative_to(out_dir).as_posix()
        entry: dict[str, Any] = {
            "structure": structure,
            "path": relative,
            "format": structure.suffix.lstrip("."),
            "seed": _index(relative, _SEED),
            "sample": _index(relative, _SAMPLE),
        }
        confidence = _nearest_confidence(structure)
        if confidence:
            entry["confidence_file"] = confidence
            try:
                data = json.loads(confidence.read_text())
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                entry.update({k: data[k] for k in SCORES if k in data})
        # Holds the PAE matrix and per-atom pLDDT, which
        # `tools.protenix.companions` turns into the shared companion files.
        entry["full_data_file"] = _nearest_full_data(structure)
        results.append(entry)

    def rank(entry: dict[str, Any]) -> float:
        for key in ("ranking_score", "iptm", "ptm", "plddt"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return float("-inf")

    return sorted(results, key=rank, reverse=True)
