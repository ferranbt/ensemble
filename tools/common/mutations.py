"""Parse and apply mutations in the standard `A123K` notation.

A mutation names the wild-type residue, its 1-indexed position, and the
replacement: `A123K` means "alanine at position 123 becomes lysine". Several
mutations combine into one variant, joined by `:` as ProteinGym and most deep
mutational scanning datasets write them, or by `,`.

Everything here is pure. The wild-type letter is not decoration: it is checked
against the sequence, which catches the off-by-one and wrong-sequence mistakes
that otherwise produce plausible-looking nonsense.
"""

from __future__ import annotations

import re
from typing import Iterable, NamedTuple

# The 20 standard amino acids. ESM can emit others, but they are not
# substitutions anyone means to make.
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")
SEPARATORS = re.compile(r"[:,;+\s]+")


class Mutation(NamedTuple):
    """One substitution. `position` is 1-indexed, as written."""

    wt: str
    position: int
    mt: str

    def __str__(self) -> str:
        return f"{self.wt}{self.position}{self.mt}"

    @property
    def index(self) -> int:
        """The 0-indexed offset into a sequence string."""
        return self.position - 1


def parse(text: str) -> list[Mutation]:
    """Parse one variant, which may hold several mutations.

    >>> parse("A123K:T50A")
    [Mutation(wt='A', position=123, mt='K'), Mutation(wt='T', position=50, mt='A')]
    """
    mutations: list[Mutation] = []
    for token in SEPARATORS.split(text.strip().upper()):
        if not token:
            continue
        match = MUTATION.match(token)
        if not match:
            raise ValueError(
                f"Cannot read mutation {token!r}. Expected the wild-type "
                f"residue, its 1-indexed position and the replacement, like "
                f"'A123K'."
            )
        wt, position, mt = match.group(1), int(match.group(2)), match.group(3)
        for residue, role in ((wt, "wild-type"), (mt, "replacement")):
            if residue not in RESIDUES:
                raise ValueError(
                    f"{token!r} has {residue!r} as its {role} residue, which is "
                    f"not one of the 20 standard amino acids"
                )
        if position < 1:
            raise ValueError(f"{token!r} has a position below 1")
        mutations.append(Mutation(wt, position, mt))

    if not mutations:
        raise ValueError(f"No mutations found in {text!r}")

    positions = [m.position for m in mutations]
    if len(set(positions)) != len(positions):
        repeated = sorted({p for p in positions if positions.count(p) > 1})
        raise ValueError(
            f"Variant {text!r} mutates position(s) {repeated} more than once"
        )
    return mutations


def validate(mutations: Iterable[Mutation], sequence: str) -> None:
    """Check each mutation against the sequence it claims to describe.

    Raises:
        ValueError: naming the position, what was expected and what is there.
            A mismatch usually means the numbering is offset or the sequence is
            not the one the variants were called against.
    """
    for mutation in mutations:
        if mutation.index >= len(sequence):
            raise ValueError(
                f"{mutation} is beyond the sequence, which is "
                f"{len(sequence)} residues long"
            )
        found = sequence[mutation.index]
        if found != mutation.wt:
            raise ValueError(
                f"{mutation} expects {mutation.wt} at position "
                f"{mutation.position}, but the sequence has {found}. The "
                f"numbering may be offset, or this may be the wrong sequence."
            )


def apply(sequence: str, mutations: Iterable[Mutation]) -> str:
    """Return the sequence with every mutation applied."""
    mutations = list(mutations)
    validate(mutations, sequence)
    residues = list(sequence)
    for mutation in mutations:
        residues[mutation.index] = mutation.mt
    return "".join(residues)


def clean_sequence(sequence: str) -> str:
    """Normalise a sequence, rejecting anything ESM cannot score."""
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("The sequence is empty")
    unknown = sorted(set(cleaned) - set(RESIDUES))
    if unknown:
        raise ValueError(
            f"Sequence contains {''.join(unknown)}, which is not a standard "
            f"amino acid. Split chains on '/' and remove gaps first."
        )
    return cleaned
