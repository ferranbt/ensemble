"""Candidate binders that carry sequences, and the parts shared to read them.

BoltzGen and ProteinMPNN return the same thing: candidate binders with
sequences, ranked. They arrive there differently, BoltzGen generating a
backbone and sequencing it in one pipeline and ProteinMPNN sequencing a
backbone it was handed, but the result a later stage reads is the same, so it
is worth agreeing on:

    design          index, so ordering and selection read the same way
    structure_uri   the structure this candidate is
    binder_chain    which chain is the binder, rather than a guessed letter
    chain_lengths   residues per chain
    sequences       {chain: sequence}

`sequences` as a map is what lets a candidate go straight into a predictor's
`chains` argument; ProteinMPNN's own output is one `/`-joined string, which
cannot. `binder_chain` is what lets it go into a scorer or another designer;
BoltzGen never reported it, so a step after BoltzGen had to hardcode a chain
letter and hope.

RFdiffusion is deliberately not part of this. It returns backbones with no
sequences, which is a different result, and it ranks nothing, so it has no
meaningful `best`. It returns its own dict and borrows only the helpers below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# gemmi reports three-letter residue names; the chain type speaks one-letter.
STANDARD = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Chemically modified residues, read as the amino acid they are a form of.
# MSE is the one that matters: selenomethionine is used to phase crystal
# structures and appears in a large share of the PDB, so reading it as unknown
# turns an ordinary methionine into an X and breaks every sequence tool
# downstream, which is exactly what it did the first time this ran.
MODIFIED = {
    "MSE": "M",  # selenomethionine
    "SEC": "C",  # selenocysteine
    "CSO": "C", "CME": "C", "CSD": "C", "OCS": "C",  # oxidised cysteines
    "MLY": "K", "KCX": "K", "LLP": "K",  # modified lysines
    "HYP": "P",  # hydroxyproline
    "SEP": "S", "TPO": "T", "PTR": "Y",  # phosphorylated residues
    "TYS": "Y",  # sulfotyrosine
    "PCA": "Q",  # pyroglutamate
}

THREE_TO_ONE = {**STANDARD, **MODIFIED}


def chain_lengths(structure: Path) -> dict[str, int]:
    """Residues per polymer chain, for any structure format gemmi reads."""
    import gemmi

    parsed = gemmi.read_structure(str(structure))
    parsed.setup_entities()
    if not len(parsed):
        raise ValueError(f"{structure} contains no model")
    return {chain.name: len(chain.get_polymer()) for chain in parsed[0]}


def binder_chain(
    lengths: dict[str, int], target_lengths: dict[str, int]
) -> str | None:
    """Which chain of a design is the newly generated one.

    Generators letter their own output, so this compares against the target's
    chains rather than assuming a convention: the chain that is new, or failing
    that the one whose length changed.

    Returns None when it cannot be told apart. A caller should treat that as
    unknown rather than guessing, since designing or scoring the wrong chain
    yields a plausible number for the wrong question.
    """
    novel = [c for c in lengths if c not in target_lengths]
    if len(novel) == 1:
        return novel[0]
    changed = [c for c, n in lengths.items() if target_lengths.get(c) != n]
    return changed[0] if len(changed) == 1 else None


def read_sequences(structure: Path) -> dict[str, str]:
    """Every chain's one-letter sequence, read from a structure.

    A residue outside the standard twenty becomes X, which is what the chain
    type accepts for an unknown.
    """
    import gemmi

    parsed = gemmi.read_structure(str(structure))
    parsed.setup_entities()
    if not len(parsed):
        raise ValueError(f"{structure} contains no model")

    return {
        chain.name: "".join(
            THREE_TO_ONE.get(residue.name.strip().upper(), "X")
            for residue in chain.get_polymer()
        )
        for chain in parsed[0]
    }


def split_sequences(sequence: str, chain_ids: list[str]) -> dict[str, str]:
    """ProteinMPNN's `/`-joined sequence mapped onto the chains it designed.

    Its fasta carries only the designed chains, in the order they were named,
    so the mapping is exact rather than inferred.

    Raises:
        ValueError: when the counts disagree, which would otherwise assign one
            chain's sequence to another and produce a valid-looking design of
            the wrong protein.
    """
    parts = [p for p in str(sequence).split("/") if p]
    if len(parts) != len(chain_ids):
        raise ValueError(
            f"ProteinMPNN returned {len(parts)} chain sequence(s) but "
            f"{len(chain_ids)} chain(s) were designed ({', '.join(chain_ids)}). "
            f"Refusing to guess which belongs to which."
        )
    return dict(zip(chain_ids, parts))


def envelope(
    *,
    name: str,
    designs: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """The result shared by the tools that return sequenced candidates.

    Args:
        designs: One entry per candidate, sorted best first. Both tools rank
            their own output, so `best` means something here.
        extra: Tool-specific fields, kept alongside the standard ones.
    """
    return {
        "name": name,
        "best": designs[0] if designs else None,
        "designs": designs,
        **extra,
    }


def summary_lines(result: dict[str, Any]) -> list[str]:
    """One line per candidate, for a command-line run."""
    lines = []
    for design in result.get("designs") or []:
        parts = [f"design {design.get('design', 0)}"]
        if design.get("binder_chain"):
            parts.append(f"binder={design['binder_chain']}")
        binder = (design.get("sequences") or {}).get(design.get("binder_chain"))
        if binder:
            parts.append(f"seq={binder[:24]}{'...' if len(binder) > 24 else ''}")
        for key in ("score", "global_score", "seq_recovery"):
            value = design.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.4f}")
        if design.get("structure_uri"):
            parts.append(str(design["structure_uri"]))
        lines.append("  ".join(parts))
    return lines or ["no designs produced"]
