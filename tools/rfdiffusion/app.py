"""RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) as a Modal action.

One action, `design_binder_backbone`: given a target structure, generate new
protein backbones docked against it.

Checkpoints are 461 MB each and are fetched on demand into a Modal Volume.

    modal run tools/rfdiffusion/app.py::design_binder_backbone \
        --target-uri s3://bucket/target.pdb --target-chain A \
        --hotspots '["A30", "A33", "A34"]' --binder-length 70-100
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from tools.common import convert
from tools.common import design as design_spec
from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool
from tools.rfdiffusion import inputs, outputs

REPO_URL = "https://github.com/RosettaCommons/RFdiffusion.git"
REPO_COMMIT = "86507b6538f51fce57b5a72477165f03999ed7ae"

# torch 1.12 predates the numpy 2 ABI break and cannot use it.
NUMPY_PIN = "numpy<2"

# CUDA 11.6 predates Hopper, so an H100 would fail to run these kernels.
GPU = "A10G"
TIMEOUT = 60 * 60

CACHE = "rfdiffusion-cache"
CACHE_DIR = inputs.CACHE_DIR

app = app_for("rfdiffusion")

image = with_local_sources(
    # The project's Dockerfile uses Python 3.9, but Modal's standalone builds
    # start at 3.10. Both torch 1.12.1+cu116 and dgl 1.0.2+cu116 publish cp310
    # wheels, so the pinned stack is unaffected.
    modal.Image.from_registry(
        "nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04", add_python="3.10"
    )
    .apt_install("git", "wget")
    .pip_install(
        "torch==1.12.1+cu116",
        extra_index_url="https://download.pytorch.org/whl/cu116",
    )
    .pip_install(
        "dgl==1.0.2+cu116",
        find_links="https://data.dgl.ai/wheels/cu116/repo.html",
    )
    .pip_install(
        "e3nn==0.3.3",
        "wandb==0.12.0",
        "pynvml==11.0.0",
        "decorator==5.1.0",
        "hydra-core==1.3.2",
        "pyrsistent==0.19.3",
        *STORAGE_PACKAGES,
    )
    .run_commands(
        f"git clone {REPO_URL} {inputs.REPO_DIR}",
        f"cd {inputs.REPO_DIR} && git checkout {REPO_COMMIT}",
        f"pip install --no-cache-dir -r {inputs.REPO_DIR}/env/SE3Transformer/requirements.txt",
        f"pip install --no-cache-dir {inputs.REPO_DIR}/env/SE3Transformer",
        f"pip install --no-cache-dir --no-deps {inputs.REPO_DIR}",
        # Last, and deliberately so. torch 1.12 is compiled against the numpy 1
        # ABI, and any numpy 2 in the image makes every tensor-to-array call
        # fail with "Numpy is not available". Several dependencies above pull
        # numpy transitively, so this pin has to come after all of them.
        f"pip install --no-cache-dir '{NUMPY_PIN}'",
    )
)


OPTIONS = dict(
    image=image, gpu=GPU, timeout=TIMEOUT, cache=CACHE, cache_path=CACHE_DIR
)


def _fetch_checkpoint(rt, name: str) -> str:
    """Download one checkpoint into the Volume unless it is already there."""
    models = Path(inputs.MODELS_DIR)
    models.mkdir(parents=True, exist_ok=True)
    target = models / f"{name}_ckpt.pt"
    if not target.exists():
        print(f"fetching {name} checkpoint, about 461 MB", flush=True)
        run(["wget", "-q", "-O", str(target), inputs.checkpoint_url(name)])
        rt.save_cache()
    return str(target)


@tool("rfdiffusion/design_binder_backbone", **OPTIONS)
def design_binder_backbone(
    rt,
    target_uri: str,
    target_chain: str = "A",
    hotspots: list[str] | None = None,
    binder_length: str = "70-100",
    num_designs: int = 4,
    name: str = "binder",
    target_residues: str = "",
    checkpoint: str = "Complex_base",
    noise_scale: float = 0.0,
    diffuser_steps: int = 50,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Generate backbones for a binder against a fixed target.

    Args:
        target_uri: `s3://bucket/key` of the target PDB.
        target_chain: Which chain of that structure to bind. Every other chain
            is dropped from the job.
        hotspots: Target residues the binder should contact, as `["A30","A33"]`
            or bare numbers on the target chain. This is the main steering
            control. Without it the model picks where to bind, which is rarely
            what you want.
        binder_length: How many residues to generate, as `"80"` or `"70-100"`.
            A range samples a different length per design.
        num_designs: How many backbones come back.
        name: Identifier for the run, used to name outputs.
        target_residues: Residue range of the target to keep, e.g. `"1-100"`.
            Empty means the whole chain, read from the file. RFdiffusion-specific.
        checkpoint: Weights to use. `Complex_base` is the recommended one for
            binders; `Complex_beta` gives more varied topologies.
        noise_scale: Noise added during denoising, applied to both translations
            and rotations. Zero is recommended for binders, since less noise
            produces more designable backbones. Raise it toward 1 for diversity.
        diffuser_steps: Denoising steps. Fewer is faster and lower quality.
        deterministic: Make a run reproducible. RFdiffusion takes no seed of
            its own; it derives one from the design index, so this is on or off
            rather than a number.

    Returns:
        `designs`, one entry per backbone with its S3 URI, chain lengths and
        which chain is the generated binder. These are backbones with no
        sequence, and nothing ranks them, so there is no `best`: feed a
        design's `structure_uri` to ProteinMPNN as `design_chains`, using its
        `binder_chain`, to give it a sequence.
    """
    source = convert.fetch_structure(target_uri, "pdb", rt.work, stem="target")
    target = source.path
    target_chain = convert.resolve_chain(target_chain, source)

    contigs = inputs.binder_contigs(
        inputs.chain_spec(target, target_chain, target_residues), binder_length
    )
    resolved_hotspots = inputs.normalise_hotspots(hotspots, target_chain)
    target_lengths = {
        c: len(r) for c, r in inputs.read_chains(target).items() if c == target_chain
    }

    out_dir = rt.work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        ["python", inputs.RUNNER,
         *inputs.inference_args(
             input_pdb=target,
             output_prefix=out_dir / name,
             contigs=contigs,
             num_designs=num_designs,
             checkpoint_path=_fetch_checkpoint(rt, checkpoint),
             hotspots=resolved_hotspots,
             noise_scale=noise_scale,
             diffuser_steps=diffuser_steps,
             deterministic=deterministic,
         )],
        cwd=inputs.REPO_DIR,
    )

    designs = []
    for raw in outputs.read_designs(out_dir, name):
        lengths = raw["chain_lengths"]
        designs.append({
            "design": raw["design"],
            "structure_uri": rt.storage.upload(
                raw["structure"], rt.prefix(f"design_{raw['design']}.pdb")
            ),
            "binder_chain": design_spec.binder_chain(lengths, target_lengths),
            "chain_lengths": lengths,
            "residues": raw["residues"],
        })

    return {
        "name": name,
        "target_uri": target_uri,
        "target_chain": target_chain,
        "contigs": contigs,
        "hotspots": resolved_hotspots,
        "binder_length": binder_length,
        "checkpoint": checkpoint,
        "designs": designs,
    }


@tool("rfdiffusion/design_monomer", **OPTIONS)
def design_monomer(
    rt,
    length: str = "80-120",
    num_designs: int = 8,
    name: str = "scaffold",
    checkpoint: str = "Base",
    noise_scale: float = 1.0,
    diffuser_steps: int = 50,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Generate protein backbones out of nothing, with no target.

    A backbone is not yet a protein: most shapes that look like one have no
    sequence that folds into them. Which of these are real is settled by
    designing sequences for them and predicting those sequences back, not here.

    Args:
        length: Residues to generate, as `"100"` or a range that varies per
            design.
        num_designs: How many backbones to generate.
        name: Identifier for the run, used to name outputs.
        checkpoint: Weights to use. `Base` is the unconditional model; the
            `Complex_` checkpoints are for building against a target and are
            the wrong ones here.
        noise_scale: Noise during denoising. Unlike binder design, where zero
            noise gives more designable backbones, the default of 1 is right
            for generating a varied set: a library of near-identical folds is
            not a library.
        diffuser_steps: Denoising steps. Fewer is faster and lower quality.
        deterministic: Make a run reproducible. RFdiffusion takes no seed of
            its own, deriving one from the design index instead.

    Returns:
        `designs`, one entry per backbone with its S3 URI and length. There is
        no `best`: nothing here ranks them, and whether a backbone is any good
        is a question for the sequence design and refolding that follow.
    """
    contigs = inputs.monomer_contigs(length)

    out_dir = rt.work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    placeholder = Path(inputs.EXAMPLE_PDB)
    if not placeholder.is_file():
        raise FileNotFoundError(
            f"RFdiffusion parses an input structure even when generating "
            f"unconditionally, and its own example is missing at {placeholder}. "
            f"The image clones the repository, so this means the clone or its "
            f"layout changed."
        )

    run(
        ["python", inputs.RUNNER,
         *inputs.inference_args(
             # Parsed and then ignored: the contig names no chain, so nothing
             # from this structure reaches the design.
             input_pdb=placeholder,
             output_prefix=out_dir / name,
             contigs=contigs,
             num_designs=num_designs,
             checkpoint_path=_fetch_checkpoint(rt, checkpoint),
             hotspots=[],
             noise_scale=noise_scale,
             diffuser_steps=diffuser_steps,
             deterministic=deterministic,
         )],
        cwd=inputs.REPO_DIR,
    )

    designs = []
    for raw in outputs.read_designs(out_dir, name):
        lengths = raw["chain_lengths"]
        designs.append({
            "design": raw["design"],
            "structure_uri": rt.storage.upload(
                raw["structure"], rt.prefix(f"scaffold_{raw['design']}.pdb")
            ),
            # A monomer has one chain, so there is nothing to identify: the
            # chain to design is the only one there is.
            "chain": next(iter(lengths), "A"),
            "chain_lengths": lengths,
            "residues": raw["residues"],
        })

    return {
        "name": name,
        "contigs": contigs,
        "length": length,
        "checkpoint": checkpoint,
        "designs": designs,
    }
