"""Minimal PDB reading, shared by tools that need chains and coordinates.

Only what the tools actually use: which residues exist, which chain each
belongs to, and where their atoms are. Anything richer belongs in a real
structure library, which is a dependency none of these containers currently
carry.

Residues come back in file order. That matters: matrices produced alongside a
structure, such as a predicted aligned error matrix, are indexed in the same
order, so the two line up position by position.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Residue:
    """One residue, identified the way a PDB file identifies it."""

    chain: str
    number: int
    name: str
    # Insertion codes distinguish residues sharing a number, common in
    # antibody numbering schemes.
    insertion: str = ""
    atoms: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.insertion)

    def atom(self, name: str) -> tuple[float, float, float] | None:
        return self.atoms.get(name)


def read_pdb(path: Path, include_hetatm: bool = False) -> list[Residue]:
    """Read residues from a PDB file, in the order they appear.

    Args:
        path: The structure to read.
        include_hetatm: Also read HETATM records, which hold ligands, ions and
            waters. Off by default, since those are not part of a protein chain
            and would misalign per-residue arrays.

    Raises:
        ValueError: if the file holds no atoms at all.
    """
    residues: "OrderedDict[tuple[str, int, str], Residue]" = OrderedDict()
    prefixes = ("ATOM",) if not include_hetatm else ("ATOM", "HETATM")

    for line in path.read_text().splitlines():
        if not line.startswith(prefixes):
            continue
        try:
            number = int(line[22:26])
            coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            # Malformed or truncated coordinate line; skip rather than guess.
            continue

        # Alternate locations: keep the first conformer only, so each atom is
        # counted once.
        altloc = line[16:17].strip()
        if altloc not in ("", "A"):
            continue

        key = (line[21], number, line[26:27].strip())
        if key not in residues:
            residues[key] = Residue(
                chain=key[0], number=number, name=line[17:20].strip(),
                insertion=key[2],
            )
        residues[key].atoms.setdefault(line[12:16].strip(), coords)

    if not residues:
        raise ValueError(f"No atom records found in {path}")
    return list(residues.values())


def chain_labels(residues: list[Residue]) -> list[str]:
    """The chain of each residue, aligned to a per-residue matrix."""
    return [residue.chain for residue in residues]


def chains(residues: list[Residue]) -> "OrderedDict[str, list[int]]":
    """Chain ids to their residue numbers, in file order."""
    grouped: "OrderedDict[str, list[int]]" = OrderedDict()
    for residue in residues:
        grouped.setdefault(residue.chain, []).append(residue.number)
    return grouped
