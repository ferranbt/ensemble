"""Superposition and RMSD over paired alpha carbons.

Implementation for `tools.rmsd.app.compare_structures`, which carries the
documentation of what the numbers mean and when to use them.

Two implementation choices are worth stating here because neither is obvious
from the code:

**Residues are paired by position in the chain, not by sequence.** gemmi will
pair them by sequence alignment, which handles a truncated chain, but it is the
wrong tool for a design loop: comparing a poly-glycine RFdiffusion backbone
against a prediction carrying a real designed sequence gives two sequences that
share nothing, so the alignment produces confident nonsense. Position `i`
corresponds to position `i` by construction. The cost is that unequal chains
are rejected rather than aligned, which is the right trade: a length mismatch
here is a bug in the caller, not something to paper over.

**Both mmCIF and PDB are read directly.** gemmi handles either, so nothing is
converted down to PDB first; that conversion is lossy in exactly the field this
module keys on, the chain name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Alpha carbons only. Every published version of this filter is a CA RMSD, and
# side chains are not comparable anyway when the sequence was redesigned.
ATOM = "CA"

# Any altloc. Predictors emit one conformation, but a crystal structure used as
# a reference target often does not.
ALTLOC = "*"


@dataclass(frozen=True)
class ChainPair:
    """One chain in the reference and the chain it corresponds to."""

    reference: str
    subject: str

    def __str__(self) -> str:
        return (
            self.reference
            if self.reference == self.subject
            else f"{self.reference}={self.subject}"
        )


def parse_chains(spec: str) -> list[ChainPair]:
    """Read a space-separated chain specification, e.g. `"A B=C"`.

    >>> [str(p) for p in parse_chains("A B=C")]
    ['A', 'B=C']
    """
    pairs = []
    for token in spec.split():
        reference, separator, subject = token.partition("=")
        if not reference or (separator and not subject):
            raise ValueError(
                f"Cannot read chain {token!r}. Expected a name like 'A', or "
                f"'A=C' when the reference and subject name it differently."
            )
        pairs.append(ChainPair(reference, subject or reference))
    return pairs


def read_model(path: Path):
    """The first model of a structure, ready to measure.

    Alternative conformations are dropped so a residue contributes one CA
    rather than several, which would otherwise inflate the atom count and pair
    the wrong positions together.
    """
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    if not len(structure):
        raise ValueError(f"{path} contains no model")
    return structure[0]


def ca_positions(model, chain_name: str) -> list:
    """The alpha carbon of every polymer residue in one chain, in file order.

    Only the polymer is walked, so ligands, ions and waters cannot enter the
    measurement. A residue with no CA is skipped rather than failing, which
    covers a non-standard residue in a reference structure.
    """
    for chain in model:
        if chain.name != chain_name:
            continue
        positions = []
        for residue in chain.get_polymer():
            atom = residue.find_atom(ATOM, ALTLOC)
            if atom is not None:
                positions.append(atom.pos)
        if not positions:
            raise ValueError(
                f"Chain {chain_name!r} has no {ATOM} atoms in its polymer, so "
                f"there is nothing to compare."
            )
        return positions

    available = ", ".join(chain.name for chain in model) or "none"
    raise KeyError(f"No chain {chain_name!r} in this structure. It has: {available}.")


def paired_positions(reference, subject, pairs: list[ChainPair]) -> tuple[list, list]:
    """Corresponding alpha carbons across several chains, paired by position.

    Raises:
        ValueError: when a chain has a different number of residues in the two
            structures. Pairing them anyway would shift every residue after the
            difference and report a plausible-looking number that means nothing.
    """
    fixed: list = []
    movable: list = []
    for pair in pairs:
        left = ca_positions(reference, pair.reference)
        right = ca_positions(subject, pair.subject)
        if len(left) != len(right):
            raise ValueError(
                f"Chain {pair} has {len(left)} residues in the reference and "
                f"{len(right)} in the subject. Residues are paired by "
                f"position, so unequal chains cannot be compared; trim them to "
                f"the same span first."
            )
        fixed.extend(left)
        movable.extend(right)
    return fixed, movable


def rmsd(fixed: list, movable: list, transform=None) -> float:
    """Root-mean-square distance between paired positions.

    With `transform`, positions in `movable` are moved by it first. That is
    what separates a complex RMSD from an ordinary one: the fit comes from one
    set of chains and is applied to another, rather than being re-optimised for
    the chains being measured.
    """
    import gemmi

    total = 0.0
    for a, b in zip(fixed, movable):
        if transform is not None:
            moved = transform.apply(b)
            b = gemmi.Position(moved.x, moved.y, moved.z)
        total += a.dist(b) ** 2
    return math.sqrt(total / len(fixed)) if fixed else 0.0


def compare(
    reference: Path,
    subject: Path,
    align: str = "",
    measure: str = "",
) -> dict[str, Any]:
    """Superpose on `align`, measure `measure`. See `app.compare_structures`."""
    import gemmi

    align_pairs = parse_chains(align)
    measure_pairs = parse_chains(measure)
    if not align_pairs and not measure_pairs:
        raise ValueError(
            "Name at least one chain in `align` or `measure`. For a binder "
            "filter, align on the target and measure the binder."
        )
    # Either argument alone is a complete request: aligning without measuring
    # compares those chains to themselves, and measuring without aligning
    # superposes on what it measures.
    align_pairs = align_pairs or measure_pairs
    measure_pairs = measure_pairs or align_pairs

    reference_model = read_model(reference)
    subject_model = read_model(subject)

    align_fixed, align_movable = paired_positions(
        reference_model, subject_model, align_pairs
    )
    superposition = gemmi.superpose_positions(align_fixed, align_movable)

    measure_fixed, measure_movable = paired_positions(
        reference_model, subject_model, measure_pairs
    )

    return {
        # The published filter's second term: measured in the alignment's frame.
        "rmsd": rmsd(measure_fixed, measure_movable, superposition.transform),
        # How well the frame itself fits. A large value invalidates `rmsd`.
        "align_rmsd": superposition.rmsd,
        # Fold accuracy of the measured chains, placement ignored.
        "measure_rmsd": gemmi.superpose_positions(
            measure_fixed, measure_movable
        ).rmsd,
        "align_chains": [str(p) for p in align_pairs],
        "measure_chains": [str(p) for p in measure_pairs],
        "align_atoms": len(align_fixed),
        "measure_atoms": len(measure_fixed),
    }
