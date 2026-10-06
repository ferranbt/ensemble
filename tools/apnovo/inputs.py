"""Build the JSON manifest AP Novo runs a campaign from.

AP Novo takes no design arguments on the command line: the whole campaign is
one manifest file. Paths inside it resolve against the manifest's own
directory, so it is written beside the motif and `input_file` is a bare name.

`resequence.enabled` has no default upstream, so it is always written.
`folding.inputs` and `folding.states` are left out, because upstream derives
the first from `resequence.enabled` and the second from the motif.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_DIR = "/opt/alphaprotein-novo"
RUNNER = f"{REPO_DIR}/run_pipeline.py"

# LigandMPNN resequences from its own environment: its pinned torch and numpy
# cannot share one with JAX and AlphaFold 3.
LIGANDMPNN_DIR = "/opt/LigandMPNN"
LIGANDMPNN_PYTHON = "/opt/ligandmpnn-venv/bin/python"

CACHE_DIR = "/cache/apnovo"
GENERATOR_DIR = f"{CACHE_DIR}/apnovo_generator"

GENERATOR_WEIGHTS = "generator.bin.zst"
GENERATOR_URL = f"https://storage.googleapis.com/alphaprotein_novo/{GENERATOR_WEIGHTS}"

# `af3_la` is fine-tuned for the covalent intermediates AP Novo designs around,
# which is why it rather than stock AlphaFold 3 is the default.
AF3_MODELS = {
    "af3_la": (
        f"{CACHE_DIR}/af3_la",
        "af3_leaving_atom.bin.zst",
        "https://storage.googleapis.com/alphafold3/af3_leaving_atom.bin.zst",
    ),
    "af3": (
        f"{CACHE_DIR}/af3",
        "af3.bin.zst",
        "https://storage.googleapis.com/alphafold3/af3.bin.zst",
    ),
}

SUITES = (
    "kemp_eliminase",
    "serine_esterase",
    "dehp_esterase",
    "carbene_transfer",
    "nitrene_transfer",
)

STAGES = ("generation", "resequence", "folding", "evaluation")


def design_job(
    *,
    name: str,
    input_file: str,
    motif_str: str,
    num_designs: int,
    motif_atoms: str = "",
    is_author_naming: bool = True,
    num_sampling_steps: int = 1000,
    seq_length: str = "",
    unindexed_motif_residues: str = "",
    reseq_residues: str = "",
    seed_start: int = 0,
    partial_diffusion_input_file: str = "",
    partial_diffusion_num_steps: int = 0,
) -> dict[str, Any]:
    """One entry of the manifest's `designs` list.

    Optional fields are omitted rather than nulled, since upstream validates
    them and treats an absent field differently from a null one.
    `partial_diffusion_num_steps` of 0 means no partial diffusion, which keeps
    every argument a plain scalar.

    Raises:
        ValueError: if unindexed conditioning is combined with a motif string
            that also pins residues, or if only one half of a partial
            diffusion run is given.
    """
    job: dict[str, Any] = {
        "name": name,
        "input_file": input_file,
        "motif_str": motif_str,
        "is_author_naming": is_author_naming,
        "num_designs": num_designs,
        "num_sampling_steps": num_sampling_steps,
        "seed_start": seed_start,
    }

    if motif_atoms:
        job["motif_atoms"] = motif_atoms
    if seq_length:
        job["seq_length"] = seq_length
    if reseq_residues:
        job["reseq_residues"] = reseq_residues

    if unindexed_motif_residues:
        if _fixes_residues(motif_str):
            raise ValueError(
                f"unindexed_motif_residues={unindexed_motif_residues!r} asks "
                f"the model to place the motif, but motif_str={motif_str!r} "
                f"already pins residues to positions. Give a motif string "
                f"with a single designable element and ligand chains only, "
                f"such as '10-250/B1'."
            )
        job["unindexed_motif_residues"] = unindexed_motif_residues

    if bool(partial_diffusion_input_file) != bool(partial_diffusion_num_steps):
        raise ValueError(
            "Partial diffusion needs both a starting structure and a step "
            "count; got "
            f"partial_diffusion_input_file={partial_diffusion_input_file!r} "
            f"and partial_diffusion_num_steps={partial_diffusion_num_steps!r}."
        )
    if partial_diffusion_input_file:
        job["partial_diffusion_input_file"] = partial_diffusion_input_file
        job["partial_diffusion_num_steps"] = partial_diffusion_num_steps

    return job


def _fixes_residues(motif_str: str) -> bool:
    """Whether the protein chain pins any input residue to a position.

    Only the first chain is read; later ones are ligands, always fixed.
    """
    for segment in motif_str.split("/")[0].replace("|", ",").split(","):
        segment = segment.strip()
        if not segment or segment == "{}":
            continue
        # `A1` names an input chain and is fixed; `5-10` is a designable length.
        if segment[0].isalpha():
            return True
    return False


def manifest(
    *,
    job: dict[str, Any],
    resequence: bool,
    resequence_temperature: float = 0.1,
    sequences_per_design: int = 1,
    use_side_chain_context: bool = True,
    folding_seeds: list[int] | None = None,
    folding_states: list[dict[str, Any]] | None = None,
    suite: str = "",
    reference_cif: str = "",
    model_dir: str = GENERATOR_DIR,
) -> dict[str, Any]:
    """The whole campaign: one design job and what later stages do with it.

    Raises:
        ValueError: if `suite` is not one upstream ships, or if the folding
            seeds repeat, which upstream rejects only after generation has run.
    """
    if suite and suite not in SUITES:
        raise ValueError(
            f"Unknown evaluation suite {suite!r}. Upstream ships "
            f"{', '.join(SUITES)}. Leave it empty for the generic metrics."
        )

    seeds = list(folding_seeds) if folding_seeds else [0]
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"folding_seeds lists a seed more than once: {seeds}. Each seed "
            f"predicts every state once, so a repeat is duplicated work."
        )

    document: dict[str, Any] = {
        "settings": {"model_dir": model_dir},
        "designs": [job],
        "resequence": {
            "enabled": resequence,
            "temperature": resequence_temperature,
            "num_sequences": sequences_per_design,
            "use_side_chain_context": use_side_chain_context,
        },
        "folding": {"seeds": seeds},
    }

    if folding_states is not None:
        document["folding"]["states"] = folding_states

    evaluation = {}
    if suite:
        evaluation["suite"] = suite
    if reference_cif:
        evaluation["reference_cif"] = reference_cif
    if evaluation:
        document["evaluation"] = evaluation

    return document


def write_manifest(path: Path, document: dict[str, Any]) -> Path:
    """Write a manifest, returning the path. Must sit beside the motif file."""
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def pipeline_args(
    *,
    manifest_path: Path,
    output_dir: Path,
    generator_dir: str,
    af3_dir: str,
    resequence: bool,
    resume: bool = False,
    only_stage: str = "",
) -> list[str]:
    """The command line for one campaign.

    `--resume` is off by default although upstream defaults it on: a Modal call
    starts with an empty directory, so there is never earlier work to resume,
    and leaving it on would silently skip designs if one were reused. The
    LigandMPNN flags go in only when resequencing is on, which is exactly when
    upstream requires them.

    Raises:
        ValueError: if `only_stage` is not one of the four stage names.
    """
    if only_stage and only_stage not in STAGES:
        raise ValueError(
            f"only_stage={only_stage!r} is not a stage. Upstream runs "
            f"{', '.join(STAGES)}."
        )

    args = [
        f"--manifest={manifest_path}",
        f"--output_dir={output_dir}",
        f"--apn_model_dir={generator_dir}",
        f"--af3_model_dir={af3_dir}",
        f"--resume={'true' if resume else 'false'}",
    ]
    if resequence:
        args += [
            f"--ligandmpnn_dir={LIGANDMPNN_DIR}",
            f"--ligandmpnn_python={LIGANDMPNN_PYTHON}",
        ]
    if only_stage:
        args.append(f"--only_stage={only_stage}")
    return args
