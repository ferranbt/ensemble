"""A chain of a complex, and everything that belongs to it.

Predictors take a complex one chain at a time, and each chain carries more than
a sequence: where its alignment comes from, and once resolved, where that
alignment sits on disk. Keeping those together means a tool that enriches a
chain, as the alignment search does, hands the same structure onward with no
second mapping to keep in step.

The type does the validating, so there is no separate check to remember to
call. Constructing a `Chain` with a sequence that is not one fails there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# The 20 standard amino acids plus X for an unknown residue.
RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")
CHAIN_ID = re.compile(r"^[A-Za-z0-9]{1,4}$")


@dataclass
class Chain:
    """One polypeptide of a complex.

    Attributes:
        sequence: One letter per residue. Upper-cased and checked on creation.
        msa: Where this chain's alignment comes from. Empty runs single
            sequence, an `s3://` URI names a precomputed alignment, and an
            http(s) URL names a server to query. A de novo binder wants the
            empty default, since it has no homologues to find.
        alignment: The alignment on local disk, once fetched. Filled in by the
            tool at run time; not something a caller supplies.
    """

    sequence: str
    msa: str = ""
    alignment: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self.sequence = "".join(str(self.sequence).split()).upper()
        if not self.sequence:
            raise ValueError("A chain needs a sequence")
        bad = sorted(set(self.sequence) - RESIDUES)
        if bad:
            raise ValueError(
                f"Sequence contains {''.join(bad)}, which is not an amino "
                f"acid. Split multi-chain sequences on '/' first."
            )

    def as_dict(self) -> dict[str, str]:
        """The wire form, for returning a chain to a caller or a later step."""
        spec = {"sequence": self.sequence}
        if self.msa:
            spec["msa"] = self.msa
        return spec


def parse(chains: dict[str, dict | str]) -> dict[str, Chain]:
    """Build chains from the wire form a workflow or caller supplies.

    A bare string is taken as the sequence, so the common case stays short:

    >>> parse({"A": "MKT"})
    {'A': Chain(sequence='MKT', msa='')}
    >>> parse({"A": {"sequence": "MKT", "msa": "s3://bucket/a.a3m"}})
    {'A': Chain(sequence='MKT', msa='s3://bucket/a.a3m')}

    Raises:
        ValueError: naming the chain, since an error deep in a mapping is
            otherwise hard to place.
    """
    if not chains:
        raise ValueError('Pass at least one chain, as {"A": "MKT..."}')

    parsed: dict[str, Chain] = {}
    for chain_id, spec in chains.items():
        if not CHAIN_ID.match(str(chain_id)):
            raise ValueError(
                f"Chain id {chain_id!r} must be 1-4 alphanumeric characters"
            )
        try:
            parsed[str(chain_id)] = (
                Chain(sequence=spec) if isinstance(spec, str)
                else Chain(**spec)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Chain {chain_id}: {exc}") from exc
    return parsed


def as_dict(chains: dict[str, Chain]) -> dict[str, dict[str, str]]:
    """The wire form of a whole complex, to return or pass to a later step."""
    return {chain_id: chain.as_dict() for chain_id, chain in chains.items()}
