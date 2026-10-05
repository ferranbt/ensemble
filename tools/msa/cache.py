"""Content-addressed caching for alignment searches.

An alignment depends only on the query sequence and the search settings, so the
same target searched twice gives the same answer. That matters here more than
it usually would: a campaign folds one target hundreds of times, and every one
of those would otherwise queue a fresh search against a shared public server.

Keying by content means the second call is a file read. It also makes us a
better citizen of a service we do not run.

Pure functions, no network and no Modal.
"""

from __future__ import annotations

import hashlib
import re

# Alignments are cheap to keep and expensive to fetch, so nothing expires them.
CACHE_DIR = "/cache/msa"

RESIDUES = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def normalise(sequence: str) -> str:
    """Strip whitespace and upper-case, so formatting never splits the cache."""
    cleaned = re.sub(r"\s+", "", sequence).upper()
    if not cleaned:
        raise ValueError("The sequence is empty")
    unknown = sorted(set(cleaned) - RESIDUES)
    if unknown:
        raise ValueError(
            f"Sequence contains {''.join(unknown)}, which is not an amino "
            f"acid. Split chains on '/' first."
        )
    return cleaned


def mode(use_env: bool = True, use_filter: bool = True, pair: bool = False,
         pairing_strategy: str = "greedy") -> str:
    """The server's mode string, mirroring how the client builds it.

    Recorded in the cache key because two modes give different alignments for
    the same sequence.
    """
    if pair:
        if pairing_strategy not in ("greedy", "complete"):
            raise ValueError(
                f"pairing_strategy must be 'greedy' or 'complete', "
                f"got {pairing_strategy!r}"
            )
        base = f"pair{pairing_strategy}"
        return f"{base}-env" if use_env else base
    if use_filter:
        return "env" if use_env else "all"
    return "env-nofilter" if use_env else "nofilter"


def key(sequence: str, search_mode: str) -> str:
    """A stable cache key for one sequence under one mode."""
    digest = hashlib.sha256(
        f"{search_mode}\n{normalise(sequence)}".encode()
    ).hexdigest()
    return f"{search_mode}-{digest[:32]}"


def unique(sequences: dict[str, str]) -> dict[str, list[str]]:
    """Group chains by their normalised sequence.

    A homodimer is one search, not two. The returned mapping is
    `{sequence: [chain ids]}` so results can be fanned back out to every chain
    that shares a sequence.
    """
    grouped: dict[str, list[str]] = {}
    for chain_id, sequence in sequences.items():
        grouped.setdefault(normalise(sequence), []).append(chain_id)
    return grouped
