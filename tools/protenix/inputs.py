"""Build the JSON job description Protenix predicts from.

Protenix takes a JSON array of jobs, even for a single prediction. Each job
names itself and lists its chains:

    [{"name": "complex",
      "sequences": [{"proteinChain": {"sequence": "MKT...", "count": 1,
                                      "id": ["A"]}}]}]

Multiple sequence alignments are optional and supplied as file paths. Leaving
them out runs single sequence, which is less accurate and sends nothing
anywhere. That is the same trade Boltz makes, so the two tools can be pointed
at the same problem and compared.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.common.sequences import (  # noqa: F401
    RESIDUES,
    split_chains,
    validate_chains,
)


def write_job(
    path: Path,
    chains: dict[str, str],
    name: str = "complex",
    msa_paths: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Write the input JSON for one prediction, returning the path written.

    Args:
        path: Where to write it.
        chains: Chain id to one-letter sequence.
        name: Job name. Protenix names its output directory after this.
        msa_paths: Optional precomputed alignments per chain, as
            `{"A": {"pairedMsaPath": ..., "unpairedMsaPath": ...}}`. Omitted
            chains run in single sequence mode.
    """
    validate_chains(chains)
    msa_paths = msa_paths or {}

    entities = []
    for chain_id, sequence in chains.items():
        chain: dict[str, object] = {
            "sequence": sequence.upper(),
            "count": 1,
            "id": [chain_id],
        }
        chain.update(msa_paths.get(chain_id, {}))
        entities.append({"proteinChain": chain})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"name": name, "sequences": entities}], indent=2))
    return path
