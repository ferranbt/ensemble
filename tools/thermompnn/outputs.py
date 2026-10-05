"""Read the site-saturation table ThermoMPNN writes.

For an input `target.pdb` it writes `ThermoMPNN_inference_target.csv` into the
output directory, one row per candidate substitution:

    ,Model,Dataset,ddG_pred,position,wildtype,mutation

`position` is 0-indexed into the chain that was scanned, so it is converted
here to the 1-indexed numbering everything else in this repo uses.

**On the sign of ddG: positive means destabilising.**
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

COLUMNS = ("ddG_pred", "position", "wildtype", "mutation")


def report_path(out_dir: Path, pdb_path: Path) -> Path:
    """Where ThermoMPNN puts its predictions, given the structure it scanned."""
    return out_dir / f"ThermoMPNN_inference_{pdb_path.stem}.csv"


def parse_predictions(path: Path) -> list[dict[str, Any]]:
    """Parse the predictions CSV into one record per substitution.

    Returns:
        Records with `position` (1-indexed), `wild_type`, `mutant`, `ddG` and
        the `mutation` string in `A123K` form.

    Raises:
        ValueError: if the file has no usable rows, naming the columns found.
    """
    rows = list(csv.DictReader(path.read_text().splitlines()))
    if not rows:
        raise ValueError(f"{path} holds no rows")

    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"{path} is missing column(s) {missing}. It has: "
            f"{sorted(k for k in rows[0] if k)}"
        )

    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            # Reported 0-indexed within the scanned chain.
            position = int(float(row["position"])) + 1
            ddg = float(row["ddG_pred"])
        except (TypeError, ValueError):
            continue
        wild_type, mutant = row["wildtype"], row["mutation"]
        if not wild_type or not mutant or wild_type == mutant:
            continue
        records.append({
            "position": position,
            "wild_type": wild_type,
            "mutant": mutant,
            "mutation": f"{wild_type}{position}{mutant}",
            "ddG": ddg,
        })

    if not records:
        raise ValueError(f"{path} has rows but none could be read as predictions")
    return records


def summarise(
    records: list[dict[str, Any]], top_k: int = 5, descending: bool = False
) -> list[dict[str, Any]]:
    """Group substitutions by position, keeping the best tolerated first.

    Args:
        records: Output of `parse_predictions`.
        top_k: Substitutions to keep per position.
        descending: Sort by ddG high to low, surfacing the most destabilising
            substitutions. The default is ascending, which surfaces the ones a
            protein will tolerate, since those are what you would act on.

    Returns:
        One entry per position with its wild-type residue, the retained
        substitutions, and the range of predicted effects at that position.
        A wide range means the position is sensitive to what sits there.
    """
    by_position: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_position.setdefault(record["position"], []).append(record)

    summary: list[dict[str, Any]] = []
    for position in sorted(by_position):
        entries = sorted(
            by_position[position], key=lambda r: r["ddG"], reverse=descending
        )
        values = [e["ddG"] for e in entries]
        summary.append({
            "position": position,
            "wild_type": entries[0]["wild_type"],
            "ddG_min": min(values),
            "ddG_max": max(values),
            "ddG_range": max(values) - min(values),
            "top": [
                {"mutation": e["mutation"], "mutant": e["mutant"], "ddG": e["ddG"]}
                for e in entries[:top_k]
            ],
        })
    return summary
