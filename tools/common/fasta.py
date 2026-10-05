"""Minimal FASTA reading and writing, shared across tools."""

from __future__ import annotations

from typing import Iterable, NamedTuple


class FastaRecord(NamedTuple):
    header: str
    sequence: str


def parse(text: str) -> list[FastaRecord]:
    """Parse FASTA text. Multi-line sequences are joined; blank lines ignored."""
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is not None:
            records.append(FastaRecord(header, "".join(chunks)))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
        elif header is not None:
            chunks.append(line)
    flush()
    return records


def format(records: Iterable[tuple[str, str]]) -> str:
    """Render (header, sequence) pairs as FASTA text."""
    return "".join(f">{header}\n{sequence}\n" for header, sequence in records)
