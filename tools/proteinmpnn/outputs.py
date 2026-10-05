"""Read back what ProteinMPNN writes for each inference mode.

The runner has no structured output mode. results are reported as files 
under the output folder: designs as FASTA with metrics packed into the 
header line as `key=value` text, and scores and probabilities as compressed
numpy archives. These helpers turn each into plain Python so the Modal
functions can return JSON-serialisable results.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from tools.common import fasta
from tools.proteinmpnn.inputs import ALPHABET

# Header fields ProteinMPNN writes as integers rather than floats.
_INT_FIELDS = {"sample", "seed"}
# Header fields written with Python's repr of a list, e.g. "['A', 'B']".
_LIST_FIELDS = {"fixed_chains", "designed_chains"}


def _split_fields(header: str) -> list[str]:
    """Split a header on commas that separate fields.

    Chain lists are written with Python's repr, so a header can contain
    `designed_chains=['A', 'B']` where the comma inside the brackets does not
    start a new field. Splitting naively corrupts every field after it.
    """
    fields: list[str] = []
    depth = 0
    current: list[str] = []
    for char in header:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
    fields.append("".join(current))
    return [f.strip() for f in fields if f.strip()]


def _coerce(key: str, value: str) -> Any:
    """Convert one header value to the type its field implies."""
    if key in _LIST_FIELDS:
        try:
            return list(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return value
    try:
        return int(value) if key in _INT_FIELDS else float(value)
    except ValueError:
        return value


def parse_designs(path: Path) -> list[dict[str, Any]]:
    """Parse a designs FASTA into records, native sequence first.

    ProteinMPNN writes one entry for the input sequence with a header like
    `name, score=..., global_score=..., designed_chains=[...]`, then one entry
    per sample with `T=..., sample=..., score=..., seq_recovery=...`. Chains
    within a sequence are separated by `/`.

    Score is mean negative log likelihood over the designed positions only, and
    global score covers every position. Lower is better for both.
    """
    records: list[dict[str, Any]] = []
    for entry in fasta.parse(path.read_text()):
        record: dict[str, Any] = {
            "sequence": entry.sequence,
            "chains": entry.sequence.split("/"),
            "header": entry.header,
        }
        for field in _split_fields(entry.header):
            key, sep, value = field.partition("=")
            if not sep:
                record.setdefault("name", key.strip())
                continue
            record[key.strip()] = _coerce(key.strip(), value.strip())
        record["is_native"] = "sample" not in record
        records.append(record)
    return records


def read_scores(path: Path) -> dict[str, Any]:
    """Read one `score_only` archive.

    Each entry of `score` is the sequence scored under an independently sampled
    autoregressive decoding order, so the spread across entries reflects the
    model's order sensitivity, not experimental error.

    Only `score` and `global_score` are guaranteed present. Archives written by
    ProteinMPNN before the scored sequence was added to the output carry
    neither `seq_str` nor `S`, so `sequence` is None for those.
    """
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        scores = np.asarray(data["score"], dtype=float).ravel()
        global_scores = np.asarray(data["global_score"], dtype=float).ravel()
        sequence = str(data["seq_str"]) if "seq_str" in data.files else None

    return {
        "sequence": sequence,
        "score": float(scores.mean()),
        "score_std": float(scores.std()),
        "global_score": float(global_scores.mean()),
        "global_score_std": float(global_scores.std()),
        "num_decoding_orders": int(scores.size),
    }


def read_probabilities(
    path: Path,
    top_k: int = 5,
    include_matrix: bool = False,
) -> dict[str, Any]:
    """Read a conditional or unconditional probability archive.

    The archive holds log probabilities shaped (batch, length, 21) over
    `ALPHABET`, averaged here across the batch of decoding orders. Only
    designable positions are returned.

    Args:
        path: The `.npz` written by the runner.
        top_k: How many amino acids to report per position, highest first.
        include_matrix: Also return the full 21-way distribution per position.
            Off by default because it is large and rarely what a caller wants.

    Returns:
        A dict with one entry per designable position: its index, the residue
        present in the input, the most likely amino acids, and the
        distribution's entropy in bits as a measure of model uncertainty.
    """
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        log_p = np.asarray(data["log_p"], dtype=np.float64)
        native = np.asarray(data["S"], dtype=int)
        design_mask = np.asarray(data["design_mask"], dtype=float)

    # Average the per-decoding-order distributions in probability space.
    probs = np.exp(log_p).mean(axis=0)
    probs /= probs.sum(axis=-1, keepdims=True)

    length = min(probs.shape[0], native.shape[0], design_mask.shape[0])
    probs = probs[:length]
    order = np.argsort(-probs, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(probs * np.where(probs > 0, np.log2(probs), 0.0)).sum(axis=-1)

    k = max(1, min(top_k, len(ALPHABET)))
    positions: list[dict[str, Any]] = []
    for i in range(length):
        if design_mask[i] <= 0:
            continue
        entry: dict[str, Any] = {
            "position": i,
            "native_aa": ALPHABET[native[i]] if native[i] < len(ALPHABET) else "X",
            "entropy_bits": float(entropy[i]),
            "top": [
                {"aa": ALPHABET[j], "probability": float(probs[i, j])}
                for j in order[i, :k]
            ],
        }
        if include_matrix:
            entry["probabilities"] = {
                aa: float(probs[i, j]) for j, aa in enumerate(ALPHABET)
            }
        positions.append(entry)

    return {"alphabet": ALPHABET, "positions": positions}
