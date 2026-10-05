"""Chain and sequence handling shared by the structure predictors.

Every predictor here takes the same thing, a complex described as chain ids
mapped to one-letter sequences, and rejects the same mistakes. The most common
of those is handing over a ProteinMPNN sequence with its chains still joined by
'/', which reads as a single chain containing an impossible residue.
"""

from __future__ import annotations

import re

# The 20 standard amino acids plus X for an unknown residue, which predictors
# accept even though it carries no information.
RESIDUES = set("ACDEFGHIKLMNPQRSTVWYX")
CHAIN_ID = re.compile(r"^[A-Za-z0-9]{1,4}$")


def validate_chains(chains: dict[str, str]) -> None:
    """Reject chain ids and sequences a predictor would fail on.

    Raises:
        ValueError: naming the chain and what is wrong with it.
    """
    if not chains:
        raise ValueError("Pass at least one chain, as {chain_id: sequence}")
    for chain_id, sequence in chains.items():
        if not CHAIN_ID.match(chain_id):
            raise ValueError(
                f"Chain id {chain_id!r} must be 1-4 alphanumeric characters"
            )
        if not sequence:
            raise ValueError(f"Chain {chain_id} has an empty sequence")
        bad = sorted(set(sequence.upper()) - RESIDUES)
        if bad:
            raise ValueError(
                f"Chain {chain_id} has characters that are not amino acids: "
                f"{''.join(bad)}. Split multi-chain sequences on '/' first."
            )


def split_chains(sequence: str, chain_ids: str) -> dict[str, str]:
    """Turn a ProteinMPNN sequence into named chains.

    ProteinMPNN joins a complex's chains with '/'. Chain ids are given in the
    same order, space separated.

    >>> split_chains("MKT/GHN", "A B")
    {'A': 'MKT', 'B': 'GHN'}
    """
    parts = [p for p in sequence.split("/") if p]
    ids = chain_ids.split()
    if len(parts) != len(ids):
        raise ValueError(
            f"Sequence has {len(parts)} chain(s) but {len(ids)} id(s) were "
            f"given ({chain_ids!r}). They must correspond in order."
        )
    return dict(zip(ids, parts))
