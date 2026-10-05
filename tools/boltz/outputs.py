"""Read what Boltz writes for a prediction.

Boltz lays its results out as:

    out_dir/boltz_results_<job>/predictions/<job>/<job>_model_0.pdb
                                                 /confidence_<job>_model_0.json
                                                 /pae_<job>_model_0.npz

with one structure and one confidence file per diffusion sample. Every score
in the confidence file runs 0 to 1, higher meaning more confident.

Note the `boltz_results_<job>` level. Boltz's own documentation omits it, so
`find_job_dir` searches rather than trusting the layout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_MODEL_INDEX = re.compile(r"_model_(\d+)\.")

# Copied through verbatim when present. `iptm` is the interface score, and the
# one that matters for whether a binder is predicted to actually bind.
SCORES = (
    "confidence_score",
    "ptm",
    "iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    "ligand_iptm",
    "protein_iptm",
)


def _describe(path: Path) -> str:
    """What actually exists at or above `path`, for an error message.

    Walks up to the nearest directory that exists and lists it. Listing the
    missing directory itself would raise, which is what previously masked a
    wrong-path bug behind a `FileNotFoundError` from the error handler.
    """
    probe = path
    while not probe.is_dir() and probe != probe.parent:
        probe = probe.parent
    try:
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in probe.iterdir())
    except OSError as exc:  # unreadable, or raced away underneath us
        return f"could not list {probe}: {exc}"
    return f"nearest existing directory {probe} contains {entries or ['nothing']}"


def find_job_dir(out_dir: Path, name: str) -> Path:
    """Locate the predictions directory for one job under a Boltz output root.

    Tries the documented nesting first, then falls back to searching, so that a
    change in Boltz's layout degrades into a slower lookup rather than a crash.

    Raises:
        FileNotFoundError: with a listing of what is actually on disk, when
            there is no single obvious match. An *empty* predictions directory
            almost always means Boltz failed to build an alignment and skipped
            the record: it prints "Failed to process ... Skipping" and exits 0,
            so nothing before this point reports a problem.
    """
    expected = out_dir / f"boltz_results_{name}" / "predictions" / name
    if expected.is_dir():
        return expected

    candidates = sorted(
        p for p in out_dir.glob("boltz_results_*/predictions/*") if p.is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        named = [p for p in candidates if p.name == name]
        if len(named) == 1:
            return named[0]
        raise FileNotFoundError(
            f"Several Boltz predictions under {out_dir} and none named {name!r}: "
            f"{[str(p) for p in candidates]}"
        )

    raise FileNotFoundError(
        f"Boltz wrote no predictions for {name!r} under {out_dir}. "
        f"{_describe(expected)}. Boltz exits 0 after skipping a record it "
        f"could not process, so the cause is in its own log: an alignment "
        f"that could not be built is by far the most common one, and it is "
        f"reported there as 'Failed to process ... Skipping'."
    )


def model_index(path: Path) -> int:
    """The diffusion sample number encoded in a Boltz output filename."""
    match = _MODEL_INDEX.search(path.name)
    if not match:
        raise ValueError(f"Not a Boltz model file: {path.name}")
    return int(match.group(1))


def read_predictions(job_dir: Path, structure_suffix: str = ".pdb") -> list[dict[str, Any]]:
    """Collect every model in a prediction directory, most confident first.

    Args:
        job_dir: `out_dir/predictions/<job>`.
        structure_suffix: `.pdb` or `.cif`, matching the requested format.

    Returns:
        One entry per diffusion sample with its `model` index, `structure` path,
        the confidence scores flattened in, and `chain_pair_iptm`, the interface
        score for each pair of chains. For a two-chain binder job that pair
        score is the number to rank designs on.
    """
    structures = sorted(job_dir.glob(f"*_model_*{structure_suffix}"), key=model_index)
    if not structures:
        raise FileNotFoundError(
            f"Boltz wrote no {structure_suffix} models into {job_dir}. "
            f"{_describe(job_dir)}"
        )

    models: list[dict[str, Any]] = []
    for structure in structures:
        index = model_index(structure)
        entry: dict[str, Any] = {"model": index, "structure": structure}

        confidence = job_dir / f"confidence_{structure.stem}.json"
        entry["confidence"] = confidence if confidence.exists() else None
        if confidence.exists():
            data = json.loads(confidence.read_text())
            entry.update({k: data[k] for k in SCORES if k in data})
            entry["chain_ptm"] = data.get("chains_ptm", {})
            entry["chain_pair_iptm"] = data.get("pair_chains_iptm", {})

        # The aligned error matrix, needed to score the interface properly.
        # Boltz names it after the model, beside the structure.
        pae = job_dir / f"pae_{structure.stem}.npz"
        entry["pae"] = pae if pae.exists() else None

        # The per-residue confidence array. ipsae.py finds it by replacing
        # "pae" with "plddt" in the aligned error path, and silently falls back
        # to zeros when it is absent, which pins pDockQ to its floor constant
        # rather than failing. So it has to travel alongside the PAE.
        plddt = job_dir / f"plddt_{structure.stem}.npz"
        entry["plddt"] = plddt if plddt.exists() else None
        models.append(entry)

    return sorted(
        models, key=lambda m: m.get("confidence_score", float("-inf")), reverse=True
    )


