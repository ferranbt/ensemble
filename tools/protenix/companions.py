"""Convert Protenix's confidence output into the shared companion shape.

Protenix predicts everything the interface score needs and writes it in its own
layout: one `*_full_data_sample_*.json` holding a per-token PAE matrix and a
per-atom pLDDT, and one `*_summary_confidence_sample_*.json` holding chain-pair
ipTM as a matrix. The convention in `tools.common.predictions` asks for a
per-token pLDDT array and ipTM nested by chain index, because that is what the
scoring script reads.

Converting here rather than in the consumer is what makes predictors
substitutable: a workflow stage can swap Boltz for Protenix without any
downstream stage knowing, and there is one reader instead of one per producer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.common import predictions

# Keys inside Protenix's full_data json.
PAE = "token_pair_pae"
ATOM_PLDDT = "atom_plddt"
ATOM_TO_TOKEN = "atom_to_token_idx"


def token_plddt(atom_plddt, atom_to_token_idx) -> Any:
    """Per-atom pLDDT averaged into one value per token.

    Protenix scores every atom; the PAE matrix and the scoring script are both
    indexed by token. `atom_to_token_idx` is the mapping Protenix already
    provides, so this is its own aggregation rather than a guess about which
    atom stands for a residue.
    """
    import numpy as np

    values = np.asarray(atom_plddt, dtype="float64")
    tokens = np.asarray(atom_to_token_idx, dtype="int64")
    if values.shape != tokens.shape:
        raise ValueError(
            f"Protenix reported {values.size} atom pLDDT values but "
            f"{tokens.size} atom-to-token indices. They index the same atoms, "
            f"so they cannot differ."
        )

    count = np.bincount(tokens)
    total = np.bincount(tokens, weights=values)
    # A token with no atoms cannot happen in Protenix's own output, but a
    # divide-by-zero here would produce a NaN that reads as a real score.
    return np.divide(
        total, count, out=np.zeros_like(total), where=count > 0
    ).astype("float32")


def pair_chains_iptm(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Protenix's chain-pair ipTM matrix as the nested form ipsae.py reads.

    Boltz writes `{"0": {"1": 0.93}}`, indices as strings in chain order.
    Protenix writes a square matrix. Same numbers, different container.
    """
    matrix = summary.get("chain_pair_iptm") or []
    return {
        str(i): {str(j): float(value) for j, value in enumerate(row)}
        for i, row in enumerate(matrix)
    }


def confidence(summary: dict[str, Any]) -> dict[str, Any]:
    """Protenix's summary under the shared key names.

    Protenix's own keys are kept alongside the translated ones, so nothing is
    lost for a caller that wants the native values.
    """
    translated: dict[str, Any] = dict(summary)
    translated["pair_chains_iptm"] = pair_chains_iptm(summary)
    if "plddt" in summary:
        translated["complex_plddt"] = summary["plddt"]
    if "ranking_score" in summary:
        translated["confidence_score"] = summary["ranking_score"]
    return translated


def write(
    structure_name: str,
    work_dir: Path,
    full_data: Path | None,
    summary: Path | None,
) -> dict[str, Path]:
    """Write whichever companions Protenix's output supports.

    Args:
        structure_name: Filename the structure will be stored under, which the
            companion names are derived from.
        work_dir: Where to write them.
        full_data: Protenix's `*_full_data_sample_*.json`, present only when
            the run asked for atom confidence.
        summary: Protenix's `*_summary_confidence_sample_*.json`.

    Returns:
        Companion kind to local path, holding only those that could be built.
    """
    import numpy as np

    written: dict[str, Path] = {}

    def target(kind: str) -> Path:
        return work_dir / predictions.companion_name(structure_name, kind)

    if full_data and full_data.is_file():
        data = json.loads(full_data.read_text())

        if PAE in data:
            path = target("pae")
            # Already per-token, which is the shape the convention asks for.
            np.savez_compressed(path, pae=np.asarray(data[PAE], dtype="float32"))
            written["pae"] = path

        if ATOM_PLDDT in data and ATOM_TO_TOKEN in data:
            path = target("plddt")
            # Left on Protenix's 0-1 scale; the reader rescales if it needs to.
            np.savez_compressed(
                path, plddt=token_plddt(data[ATOM_PLDDT], data[ATOM_TO_TOKEN])
            )
            written["plddt"] = path

    if summary and summary.is_file():
        path = target("confidence")
        path.write_text(json.dumps(confidence(json.loads(summary.read_text()))))
        written["confidence"] = path

    return written
