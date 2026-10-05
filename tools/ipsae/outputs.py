"""Read the tables ipsae.py writes.

For a structure `model_0.pdb` scored at cutoffs 10 and 15, the script writes
three files beside it:

    model_0_10_15.txt         one row per chain pair, per scoring type
    model_0_10_15_byres.txt   per-residue detail
    model_0_10_15.pml         a PyMOL script highlighting the interface

The chain-pair table is whitespace delimited with a single header line. Rows
come in scoring types: `asym` for each direction separately and `max` for the
better of the two, which is the one to rank on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Columns worth returning as numbers. The rest stay as text.
NUMERIC = {
    "PAE", "Dist", "ipSAE", "ipSAE_d0chn", "ipSAE_d0dom", "ipTM_af",
    "ipTM_d0chn", "pDockQ", "pDockQ2", "LIS", "n0res", "n0chn", "n0dom",
    "d0res", "d0chn", "d0dom", "nres1", "nres2", "dist1", "dist2",
}
INTEGER = {"n0res", "n0chn", "n0dom", "nres1", "nres2", "dist1", "dist2"}


def report_paths(structure: Path, pae_cutoff: float, dist_cutoff: float) -> dict[str, Path]:
    """Where ipsae.py puts its output, given the structure it scored.

    The script names outputs after the structure with the two cutoffs appended
    as zero-padded integers, and writes them into the structure's directory.
    """
    stem = structure.with_suffix("")
    suffix = f"{int(pae_cutoff):02d}_{int(dist_cutoff):02d}"
    return {
        "summary": stem.with_name(f"{stem.name}_{suffix}.txt"),
        "by_residue": stem.with_name(f"{stem.name}_{suffix}_byres.txt"),
        "pymol": stem.with_name(f"{stem.name}_{suffix}.pml"),
    }


def _convert(column: str, value: str) -> Any:
    if column not in NUMERIC:
        return value
    try:
        return int(value) if column in INTEGER else float(value)
    except ValueError:
        return value


def parse_summary(path: Path) -> list[dict[str, Any]]:
    """Parse the chain-pair table into rows.

    Returns:
        One dict per row, keyed by the table's own column names. Rows with
        `Type` of `max` carry the symmetric score for a chain pair; rows with
        `asym` carry one direction each.
    """
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []

    for line in lines:
        fields = line.split()
        if fields and fields[0] == "Chn1":
            header = fields
            continue
        if header is None:
            continue
        # Trailing model names can contain spaces; fold any excess into the
        # last column rather than dropping the row.
        if len(fields) > len(header):
            fields = fields[: len(header) - 1] + [" ".join(fields[len(header) - 1:])]
        if len(fields) != len(header):
            continue
        rows.append({c: _convert(c, v) for c, v in zip(header, fields)})

    if not rows:
        raise ValueError(
            f"No score rows found in {path}. The file holds "
            f"{len(lines)} non-empty line(s)."
        )
    return rows


def best_interfaces(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The symmetric score for each chain pair, best interface first.

    `ipSAE` is the headline number. It runs 0 to 1 and says how confidently the
    interface is predicted, not how tightly the chains bind.
    """
    symmetric = [row for row in rows if str(row.get("Type", "")).lower() == "max"]
    return sorted(
        symmetric or rows,
        key=lambda row: row.get("ipSAE", float("-inf")),
        reverse=True,
    )
