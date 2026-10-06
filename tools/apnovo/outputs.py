"""Read what a campaign wrote, across all four stages.

    pipeline_index.json          stages completed
    metadata/designs.json        index of generated designs
    01_generation/<prefix>.cif   plus _motif.cif and .fa
    02_resequence/<prefix>_seq<NN>_reseq.cif
    03_folded/<prefix>_<state>_folded_seed<N>.cif  plus _confidences_
    04_eval/<prefix>_evaluation.json, evaluation_summary.csv

Prefixes change between stages: generation names a design `<job>_<nnnn>`, and
resequencing appends `_seq<NN>` per sampled sequence. That longer prefix is
what stages three and four use, so a campaign's results are keyed by the
resequenced prefix, not the generated one.

Designs come from each stage's own index or filenames rather than a glob for
`*.cif`, since `<prefix>_motif.cif` sits beside `<prefix>.cif`.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

INDEX = "metadata/designs.json"
PIPELINE_INDEX = "pipeline_index.json"
SUMMARY = "04_eval/evaluation_summary.csv"

METADATA_SUFFIX = "_metadata.json"
MOTIF_SUFFIX = "_motif.cif"
RESEQ_SUFFIX = "_reseq.cif"
EVALUATION_SUFFIX = "_evaluation.json"

GENERATION = "01_generation"
RESEQUENCE = "02_resequence"
FOLDED = "03_folded"
EVAL = "04_eval"

# `<prefix>_<state>_folded_seed<N>.cif`. Both halves may contain underscores,
# so the split between them is resolved against the known prefixes rather than
# guessed from the name.
_FOLDED = re.compile(r"^(?P<stem>.+)_folded_seed(?P<seed>\d+)\.cif$")


def read_index(out_dir: Path) -> list[str]:
    """Generated design prefixes, in the order the generator recorded them.

    Raises:
        FileNotFoundError: if the index is absent, meaning generation wrote
            nothing.
    """
    index = out_dir / INDEX
    if not index.is_file():
        raise FileNotFoundError(
            f"AP Novo wrote no design index at {index}. The generator reports "
            f"its designs there, so this means generation produced nothing "
            f"rather than that the designs are elsewhere."
        )
    return _prefixes(json.loads(index.read_text()))


def _prefixes(index: Any) -> list[str]:
    """The design prefixes in the index, which carries more than those."""
    entries = index.get("designs", index) if isinstance(index, dict) else index
    if not isinstance(entries, list):
        raise ValueError(
            f"Expected a list of designs in the index, got {type(entries).__name__}."
        )

    prefixes = []
    for entry in entries:
        prefix = entry.get("prefix") if isinstance(entry, dict) else entry
        if not prefix:
            raise ValueError(f"Design index entry has no prefix: {entry!r}")
        prefixes.append(str(prefix))
    return prefixes


def stages_completed(out_dir: Path) -> list[str]:
    """Which stages the campaign record says finished.

    Empty when the record is missing: a campaign stopped early still leaves
    usable designs behind.
    """
    index = out_dir / PIPELINE_INDEX
    if not index.is_file():
        # Not fatal: the designs are read from their own stage directories.
        # Printed rather than ignored because an empty stage list otherwise
        # looks the same as a campaign that ran nothing.
        print(
            f"[apnovo] no campaign record at {index}; "
            f"{out_dir} holds {sorted(p.name for p in out_dir.iterdir())}",
            flush=True,
        )
        return []
    record = json.loads(index.read_text())
    # `stages_run` is what upstream writes; the others are tolerated in case
    # that name changes, since nothing here depends on the list being present.
    done = (
        record.get("stages_run")
        or record.get("completed_stages")
        or record.get("stages")
        or []
    )
    if isinstance(done, dict):
        return [name for name, finished in done.items() if finished]
    return [str(name) for name in done]


def read_generated(out_dir: Path) -> list[dict[str, Any]]:
    """The backbones generation produced, one entry per design."""
    stage = out_dir / GENERATION
    meta_dir = out_dir / "metadata"

    designs = []
    for prefix in read_index(out_dir):
        structure = stage / f"{prefix}.cif"
        if not structure.is_file():
            raise FileNotFoundError(
                f"The design index lists {prefix!r} but {structure} is "
                f"missing, so that design did not finish."
            )

        motif = stage / f"{prefix}{MOTIF_SUFFIX}"
        fasta = stage / f"{prefix}.fa"
        pdb = stage / f"{prefix}.pdb"
        metadata = meta_dir / f"{prefix}{METADATA_SUFFIX}"

        designs.append({
            "prefix": prefix,
            "structure": structure,
            "pdb": pdb if pdb.is_file() else None,
            "motif": motif if motif.is_file() else None,
            "sequence": read_sequence(fasta) if fasta.is_file() else "",
            "metadata": json.loads(metadata.read_text())
            if metadata.is_file()
            else {},
        })
    return designs


def read_resequenced(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Resequenced designs, grouped under the generated prefix they came from.

    Empty when resequencing was off, which is a setting rather than a failure.
    """
    stage = out_dir / RESEQUENCE
    if not stage.is_dir():
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for structure in sorted(stage.glob(f"*{RESEQ_SUFFIX}")):
        prefix = structure.name[: -len(RESEQ_SUFFIX)]
        parent = prefix.rsplit("_seq", 1)[0]
        fasta = stage / f"{prefix}.fa"
        grouped.setdefault(parent, []).append({
            "prefix": prefix,
            "structure": structure,
            "sequence": read_sequence(fasta) if fasta.is_file() else "",
        })
    return grouped


def read_folded(
    out_dir: Path, prefixes: Iterable[str]
) -> dict[str, list[dict[str, Any]]]:
    """Predictions, grouped by design prefix, one entry per state and seed.

    Args:
        out_dir: The campaign root.
        prefixes: The design prefixes folding ran on. Needed because a state
            name may contain underscores, so where the prefix ends cannot be
            read off the filename.
    """
    stage = out_dir / FOLDED
    if not stage.is_dir():
        return {}

    known = sorted(prefixes, key=len, reverse=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(stage.glob("*_folded_seed*.cif")):
        matched = _FOLDED.match(path.name)
        if not matched:
            continue
        stem, seed = matched["stem"], int(matched["seed"])
        # Longest first, so `kemp_0000_seq00` wins over `kemp_0000`.
        prefix = next((p for p in known if stem.startswith(f"{p}_")), "")
        if not prefix:
            continue
        state = stem[len(prefix) + 1:]
        confidences = stage / f"{stem}_confidences_seed{seed}.json"
        grouped.setdefault(prefix, []).append({
            "state": state,
            "seed": seed,
            "structure": path,
            "confidences": json.loads(confidences.read_text())
            if confidences.is_file()
            else {},
        })
    return grouped


def read_evaluations(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-design metrics, keyed by design prefix."""
    stage = out_dir / EVAL
    if not stage.is_dir():
        return {}

    evaluations = {}
    for path in sorted(stage.glob(f"*{EVALUATION_SUFFIX}")):
        prefix = path.name[: -len(EVALUATION_SUFFIX)]
        record = json.loads(path.read_text())
        evaluations[prefix] = record.get("metrics", record)
    return evaluations


def read_summary(out_dir: Path) -> list[dict[str, str]]:
    """The campaign summary table, as written.

    Rows stay strings and nothing is ranked here: which column to rank on
    differs per evaluation suite.
    """
    summary = out_dir / SUMMARY
    if not summary.is_file():
        return []
    with summary.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_sequence(fasta: Path) -> str:
    """The single sequence in a one-record FASTA, without header or newlines."""
    lines = [
        line.strip()
        for line in fasta.read_text().splitlines()
        if line.strip() and not line.startswith(">")
    ]
    return "".join(lines)


def fixed_residues(metadata: dict[str, Any]) -> list[str]:
    """Motif positions on the design: the catalytic residues, held still."""
    value = metadata.get("fixed_residues_flag") or metadata.get("fixed_residues")
    if not value:
        return []
    if isinstance(value, str):
        return [part for part in value.replace(",", " ").split() if part]
    return [str(part) for part in value]


def ranking_confidence(folded: list[dict[str, Any]]) -> float | None:
    """The best `ranking_confidence` across a design's predictions.

    A summary, not the answer: which state matters depends on the question, so
    the per-state numbers are returned alongside it.
    """
    scores = [
        entry["confidences"].get("ranking_confidence")
        for entry in folded
        if isinstance(entry.get("confidences"), dict)
    ]
    usable = [score for score in scores if isinstance(score, (int, float))]
    return max(usable) if usable else None
