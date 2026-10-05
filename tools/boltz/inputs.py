"""Build the YAML job description Boltz predicts from.

Boltz takes one YAML file per prediction, listing each chain as an entity:

    version: 1
    sequences:
      - protein:
          id: A
          sequence: NMYSYKKIGNKYIVSINNHT...
          msa: /tmp/msa_x/A.a3m          # a precomputed alignment
      - protein:
          id: B
          sequence: GWSTELEKHREELKEFLKKE...
          msa: empty                     # single sequence

The `msa` field is read three ways, which is exactly the three settings a
`Chain` can carry: a local `.a3m` path uses that alignment, the literal
`empty` runs single sequence, and **omitting the field entirely** tells Boltz
to query the server passed as `--use_msa_server`.
"""

from __future__ import annotations

from pathlib import Path

from tools.common.chains import Chain
from tools.common.sequences import split_chains  # noqa: F401


def write_job(path: Path, chains: dict[str, Chain]) -> Path:
    """Write the YAML for one prediction, returning the path written.

    Args:
        path: Where to write. Boltz names its outputs after this file's stem.
        chains: The complex. A chain with `alignment` set uses that file; one
            with an `msa` server keeps the field absent so Boltz queries it;
            anything else runs single sequence.
    """
    import yaml

    if not chains:
        raise ValueError("Pass at least one chain")

    entities = []
    for chain_id, chain in chains.items():
        protein: dict[str, object] = {
            "id": chain_id,
            "sequence": chain.sequence,
        }
        if chain.alignment is not None:
            protein["msa"] = str(chain.alignment)
        elif not chain.msa:
            protein["msa"] = "empty"
        entities.append({"protein": protein})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"version": 1, "sequences": entities}, sort_keys=False)
    )
    return path
