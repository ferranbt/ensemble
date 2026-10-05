"""BoltzGen (https://github.com/HannesStark/boltzgen) as a Modal GPU action.

A generative binder design model from the Boltz family. Unlike the other tools
here it is a whole pipeline rather than a step: one invocation designs
backbones, inverse folds them into sequences, refolds the complexes, scores
them and returns a ranked shortlist.

That makes it an alternative to the RFdiffusion, ProteinMPNN and Boltz chain
already in this repo rather than an addition to it. Worth having both, because
this one reports far higher success rates than RFdiffusion while the separate
chain gives you control over each step.

    modal run tools/boltzgen/app.py::design_binders \\
        --target-uri s3://bucket/target.pdb --target-chain A \\
        --hotspots '["50", "54", "57"]' --binder-length 80-140
"""

from __future__ import annotations

from typing import Any

import modal

from tools.boltzgen import inputs, outputs
from tools.common import design as design_spec
from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool

BOLTZGEN_VERSION = "0.3.2"

# The project's reference hardware.
GPU = "A100-80GB"
TIMEOUT = 2 * 60 * 60
CACHE = "boltzgen-cache"
CACHE_DIR = "/cache/boltzgen"

app = app_for("boltzgen")

image = with_local_sources(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        *STORAGE_PACKAGES, "pyyaml", f"boltzgen=={BOLTZGEN_VERSION}"
    )
    .env({"HF_HOME": CACHE_DIR})
)


@tool(
    "boltzgen/design_binders",
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    cache=CACHE,
    cache_path=CACHE_DIR,
)
def design_binders(
    rt,
    target_uri: str,
    target_chain: str = "A",
    hotspots: list[str] | None = None,
    binder_length: str = "80-140",
    num_designs: int = 4,
    name: str = "binder",
    protocol: str = "protein-anything",
    candidates: int = 8,
    alpha: float = 0.001,
) -> dict[str, Any]:
    """Design binders against a target, end to end.

    Args:
        target_uri: `s3://bucket/key` of the target structure, `.pdb` or `.cif`.
        target_chain: Which chain to bind.
        hotspots: Target residues to aim at, in the numbering written in the
            file. These are translated to BoltzGen's 1-based canonical indexing
            for you, which is not the same thing and is easy to get wrong.
        binder_length: Residues to generate, as `"100"` or `"80-140"`.
        num_designs: How many ranked designs come back. This is BoltzGen's
            `budget`; `candidates` is how many it generates to pick them from.
        name: Identifier for the run.
        protocol: What kind of binder. `protein-anything` is the general case;
            `nanobody-anything` and `antibody-anything` are specialised.
            BoltzGen-specific.
        candidates: Designs generated before filtering, from which
            `num_designs` are kept. Must be at least `num_designs`.
        alpha: Diversity against quality, from 0 to 1. Lower favours quality.

    Returns:
        Sequenced candidates in the shape described in `tools.common.design`,
        the same one ProteinMPNN returns, ranked best first. They carry
        sequences, having been inverse folded and refolded inside the pipeline,
        so they go straight to folding or scoring with no sequence design step.
    """
    if protocol not in inputs.PROTOCOLS:
        raise ValueError(
            f"protocol must be one of {', '.join(inputs.PROTOCOLS)}, "
            f"got {protocol!r}"
        )
    if num_designs > candidates:
        raise ValueError(
            f"num_designs ({num_designs}) cannot exceed candidates "
            f"({candidates}): there would be nothing to rank."
        )

    target = rt.fetch(target_uri, "target")
    # Paths inside the spec resolve against its own directory, so both files
    # live together.
    spec = inputs.write_spec(
        rt.work / f"{name}.yaml",
        target_pdb=target,
        target_chain=target_chain,
        binder_length=binder_length,
        hotspots=[h for h in (hotspots or [])],
    )

    out_dir = rt.work / "out"
    run([
        "boltzgen", "run", str(spec),
        "--output", str(out_dir),
        "--protocol", protocol,
        "--num_designs", candidates,
        "--budget", num_designs,
        "--alpha", alpha,
        "--cache", CACHE_DIR,
    ])
    rt.save_cache()

    # Which chain the binder ended up as is decided by BoltzGen, so it is read
    # back by comparing against the target rather than assumed. Without it a
    # later step has no way to name the binder except by guessing a letter.
    target_lengths = design_spec.chain_lengths(target)

    designs = []
    for index, raw in enumerate(outputs.read_designs(out_dir)):
        structure = raw.pop("structure")
        lengths = design_spec.chain_lengths(structure)
        sequences = design_spec.read_sequences(structure)
        designs.append({
            "design": index,
            "structure_uri": rt.storage.upload(
                structure, rt.prefix(f"design_{index}{structure.suffix}")
            ),
            "binder_chain": design_spec.binder_chain(lengths, target_lengths),
            "chain_lengths": lengths,
            "sequences": sequences,
            # BoltzGen's own name and ranking metrics for this design.
            **raw,
        })

    # The metrics tables explain the ranking, so keep them with the designs.
    reports: dict[str, str] = {}
    for table in sorted(out_dir.rglob("*metrics*.csv")):
        reports[table.name] = rt.storage.upload(table, rt.prefix(table.name))

    return design_spec.envelope(
        name=name,
        designs=designs,
        boltzgen_version=BOLTZGEN_VERSION,
        target_uri=target_uri,
        target_chain=target_chain,
        hotspots=list(hotspots or []),
        binder_length=binder_length,
        protocol=protocol,
        reports=reports,
    )
