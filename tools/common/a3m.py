"""Reading and retargeting alignment files.

An a3m alignment begins with the query it was built for, and everything after
is aligned to that first row. A predictor handed an alignment whose query is
not the sequence it is folding does not complain: it folds, and the result is
quietly much worse. Measured on a real protein here, one conservative
substitution at the terminal residue took a prediction from pLDDT 0.97 to 0.46,
purely because the alignment still described the parent.

That matters because reusing a parent's alignment across its point variants is
both standard and worth doing, since a variant differs by a handful of residues
and searching again for each one is waste. It only works if the query row is
updated to the variant, which is what `retarget` does. Substitutions do not
change length, so every other row still lines up column for column.

`tools.common.msa` decides *where* an alignment comes from; this handles the
file once it is on disk.
"""

from __future__ import annotations

from pathlib import Path

# How much of the query a replacement must still match. A point-variant series
# is nearly identical; anything far below this is a different protein, and
# silently retargeting onto it would fabricate an alignment describing nothing.
MIN_IDENTITY = 0.8


def read_query(path: Path) -> str:
    """The sequence an alignment was built for: its first row.

    Raises:
        ValueError: if the file holds no sequence after its first header.
    """
    query = None
    seen_header = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if seen_header:
                break
            seen_header = True
            continue
        if seen_header:
            query = stripped
            break
    if not query:
        raise ValueError(f"{path} has no query sequence after its first header")
    return query


def identity(left: str, right: str) -> float:
    """Fraction of positions two equal-length sequences share."""
    if not left or len(left) != len(right):
        return 0.0
    same = sum(1 for a, b in zip(left, right) if a == b)
    return same / len(left)


def retarget(path: Path, sequence: str) -> bool:
    """Make an alignment describe `sequence`, rewriting the file in place.

    Returns True when the query row was replaced, False when it already
    matched and nothing was written.

    Raises:
        ValueError: when the alignment cannot describe this sequence, either
            because the lengths differ, which means more than substitutions
            separate them, or because too little of it matches, which means it
            is an alignment of something else. Retargeting either would invent
            an alignment rather than reuse one.
    """
    query = read_query(path)
    if query == sequence:
        return False

    if len(query) != len(sequence):
        raise ValueError(
            f"{path.name} was built for a {len(query)}-residue query but this "
            f"chain is {len(sequence)} residues. An alignment can be reused "
            f"across substitutions, which preserve length, not across "
            f"insertions or deletions."
        )

    shared = identity(query, sequence)
    if shared < MIN_IDENTITY:
        raise ValueError(
            f"{path.name} matches this chain at only {shared:.0%} of positions. "
            f"It is an alignment of a different protein, and pointing it at "
            f"this one would describe homologues that are not its own."
        )

    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        lines[index] = sequence
        break
    path.write_text("\n".join(lines) + "\n")
    return True
