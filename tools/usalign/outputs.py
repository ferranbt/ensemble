"""Read US-align's tabular output.

With `-outfmt 2` US-align writes a header line beginning with `#` and one row
per alignment:

    #PDBchain1  PDBchain2  TM1  TM2  RMSD  ID1  ID2  IDali  L1  L2  Lali

Columns are looked up by name from that header rather than by position, since
a change in their column order would otherwise silently swap two numbers that
are both plausible.

**TM1 and TM2 are not interchangeable.** Each is normalised by the length of a
different structure, so for structures of different lengths they disagree, and
which one answers your question depends on which structure you consider the
reference. A short fragment matching part of a large protein scores high
against the fragment's length and low against the protein's. Both are reported.
"""

from __future__ import annotations

from typing import Any

# Above this, two structures are generally taken to share a fold; below it they
# do not. The threshold is from the TM-score literature, not chosen here.
SAME_FOLD = 0.5

NUMERIC = ("TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali")


def parse(text: str) -> list[dict[str, Any]]:
    """Every alignment US-align reported, as dictionaries.

    Raises:
        ValueError: when no header row is present, which means the output is
            not `-outfmt 2` and the columns are unknown.
    """
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # Only the first header is the column list; US-align repeats it
            # when several alignments are run together.
            if header is None:
                header = stripped.lstrip("#").split()
            continue
        if header is None:
            raise ValueError(
                "US-align produced rows with no header, so the columns cannot "
                "be identified. This output is not from `-outfmt 2`."
            )
        fields = stripped.split("\t") if "\t" in stripped else stripped.split()
        row: dict[str, Any] = dict(zip(header, fields))
        for key in NUMERIC:
            if key in row:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    row[key] = None
        rows.append(row)

    if header is None:
        raise ValueError("US-align produced no output to read")
    return rows


def summarise(row: dict[str, Any]) -> dict[str, Any]:
    """One alignment under names that say what they mean."""
    tm1, tm2 = row.get("TM1"), row.get("TM2")
    scores = [t for t in (tm1, tm2) if isinstance(t, float)]
    return {
        "tm_score_reference": tm2,
        "tm_score_subject": tm1,
        # The conservative reading: a match is only as good as its worse
        # normalisation, which is what stops a fragment scoring well against a
        # whole protein.
        "tm_score": min(scores) if scores else None,
        "same_fold": bool(scores) and min(scores) >= SAME_FOLD,
        "rmsd": row.get("RMSD"),
        "aligned_residues": row.get("Lali"),
        "sequence_identity": row.get("IDali"),
        "reference_length": row.get("L2"),
        "subject_length": row.get("L1"),
    }
