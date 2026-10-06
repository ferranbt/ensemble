"""AlphaProtein Novo (https://github.com/google-deepmind/alphaprotein-novo).

One action, `design_enzyme`: given a 3D catalytic motif, run the whole AP Novo
campaign and return scored designs. This is motif scaffolding, not binder
design. You supply the arrangement of sidechain and ligand atoms representing
the transition state of a reaction, and the pipeline generates proteins that
hold it, redesigns their sequences, refolds them and scores how well the
catalytic geometry survived.

All four upstream stages run here: diffusion generation, LigandMPNN
resequencing, AlphaFold 3 folding, and the enzyme metric suites.

Weights total about 1.6 GB and download on first use into a Modal Volume.

    modal deploy tools/apnovo/app.py

    modal run tools/apnovo/app.py::design_enzyme \\
        --motif-uri s3://bucket/kemp_motif.cif \\
        --motif-str 'A1,A2|10-100,{},2-80,{},10-100/B1' \\
        --motif-atoms 'A1:OE2,OE1,CD,CG A2:OG,CB,CA' \\
        --suite kemp_eliminase --num-designs 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from tools.apnovo import inputs, outputs
from tools.common import convert
from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool

REPO_URL = "https://github.com/google-deepmind/alphaprotein-novo.git"
LIGANDMPNN_URL = "https://github.com/dauparas/LigandMPNN.git"

# From git, not PyPI: the `alphafold3` name on PyPI belongs to an unrelated
# third-party reimplementation.
AF3_URL = "https://github.com/google-deepmind/alphafold3.git"

# AP Novo needs 3.12; LigandMPNN pins torch 2.2.1 and numpy 1.23, which cannot
# share an environment with JAX and AlphaFold 3, so it gets its own on 3.11.
PYTHON_VERSION = "3.12"
LIGANDMPNN_PYTHON_VERSION = "3.11"

GPU = "A100"
TIMEOUT = 8 * 60 * 60

CACHE = "apnovo-cache"

app = app_for("apnovo")

image = with_local_sources(
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python=PYTHON_VERSION
    )
    .apt_install("git", "wget", "zstd", "build-essential", "cmake", "zlib1g-dev")
    .pip_install("jax[cuda12]>=0.4.30", "uv")
    # CC and CXX are named because something in the build environment points
    # CXX at clang++, which is absent here and from AlphaFold 3's supported
    # setup; CMake then fails naming the variable rather than the compiler.
    # Parallelism is capped because this wheel builds abseil, libcifpp and dssp
    # through FetchContent, and at full width the builder ran out of memory.
    .run_commands(
        f"CC=gcc CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=4 MAKEFLAGS=-j4 "
        f"pip install --no-cache-dir git+{AF3_URL}"
    )
    .run_commands(
        f"git clone {REPO_URL} {inputs.REPO_DIR}",
        f"CC=gcc CXX=g++ pip install --no-cache-dir {inputs.REPO_DIR}",
        # Compiles the chemical component dictionary into the pickle
        # AlphaFold 3 reads. Needed by generation, not only folding:
        # `structure_features` builds a `Ccd()` at module scope and sits on
        # `run_pipeline.py`'s import path, so without it the run fails during
        # import rather than when it meets a ligand.
        "build_data",
    )
    .run_commands(
        f"git clone {LIGANDMPNN_URL} {inputs.LIGANDMPNN_DIR}",
        f"cd {inputs.LIGANDMPNN_DIR} && bash get_model_params.sh ./model_params",
        f"uv venv --python {LIGANDMPNN_PYTHON_VERSION} /opt/ligandmpnn-venv",
        f"VIRTUAL_ENV=/opt/ligandmpnn-venv uv pip install "
        f"-r {inputs.LIGANDMPNN_DIR}/requirements.txt",
        # ProDy imports pkg_resources, removed in setuptools 82.
        "VIRTUAL_ENV=/opt/ligandmpnn-venv uv pip install 'setuptools<82'",
    )
    .pip_install(*STORAGE_PACKAGES)
)


def _fetch(rt, directory: str, filename: str, url: str, size: str) -> str:
    """Download one weight file into the Volume unless it is already there."""
    models = Path(directory)
    models.mkdir(parents=True, exist_ok=True)
    target = models / filename
    if not target.exists():
        print(f"fetching {filename}, about {size}", flush=True)
        run(["wget", "-q", "-O", str(target), url])
        rt.save_cache()
    return str(models)


@tool(
    "apnovo/design_enzyme",
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    cache=CACHE,
    cache_path=inputs.CACHE_DIR,
)
def design_enzyme(
    rt,
    motif_uri: str,
    motif_str: str,
    num_designs: int = 2,
    motif_atoms: str = "",
    name: str = "design",
    is_author_naming: bool = True,
    num_sampling_steps: int = 1000,
    seq_length: str = "",
    unindexed_motif_residues: str = "",
    reseq_residues: str = "",
    seed_start: int = 0,
    partial_diffusion_uri: str = "",
    partial_diffusion_num_steps: int = 0,
    resequence: bool = True,
    resequence_temperature: float = 0.1,
    sequences_per_design: int = 1,
    use_side_chain_context: bool = True,
    folding_seeds: list[int] | None = None,
    folding_states: list[dict[str, Any]] | None = None,
    af3_model: str = "af3_la",
    suite: str = "",
    reference_uri: str = "",
) -> dict[str, Any]:
    """Run an AP Novo enzyme design campaign end to end.

    Args:
        motif_uri: `s3://bucket/key` of the motif CIF: sidechain and ligand
            atoms in the arrangement the reaction needs, usually a transition
            state or covalent intermediate.
        motif_str: Which parts of that file are the motif and how they sit in
            the design, as chains delimited by `/`. The first chain is the
            protein, a comma-separated run of fixed input residues (`A1`,
            `A5-6`) and designable lengths (`10-40`). Later chains are single
            ligand residues, always fixed. Wrapping fixed residues in `|` and
            marking their slots with `{}` samples their order.
        num_designs: Backbones to generate.
        motif_atoms: Atoms of each motif residue to hold fixed, as
            `"A1:NE2,ND1 A2:OD1"`. Naming atoms lets the model build the rest
            of the sidechain. Empty fixes every atom.
        name: Identifier for the run. Designs are named `<name>_0000` onward.
        is_author_naming: Whether `motif_str` uses author residue numbers, as
            PyMOL and the literature do, rather than PDB internal numbering.
            Wrong here scaffolds the wrong residues without failing.
        num_sampling_steps: Reverse diffusion steps. The model was trained at
            1000; lower is for smoke tests.
        seq_length: Total residues of the designed chain, as `"150"` or
            `"120-160"`. Empty lets the designable ranges sample freely.
        unindexed_motif_residues: Motif residues to place without fixing their
            sequence positions, as `"A1,A2,A3"`. Requires a `motif_str` with
            one designable element and ligand chains only, such as
            `"10-250/B1"`. Empty means indexed conditioning, which upstream
            reports as slightly better in practice.
        reseq_residues: Motif residues whose amino-acid identity may change,
            holding backbone position but not type.
        seed_start: First RNG seed; designs take consecutive seeds from it.
        partial_diffusion_uri: `s3://bucket/key` of an existing design to
            diversify instead of building from nothing. Needs
            `partial_diffusion_num_steps`.
        partial_diffusion_num_steps: How many of the 1000 steps to unroll from
            that structure. Roughly 600 keeps TM-score near 95 to the parent
            and 900 near 80. Zero means no partial diffusion.
        resequence: Redesign each backbone's sequence with LigandMPNN before
            folding. Off folds the co-generated sequence instead, which is a
            different experiment rather than a cheaper one.
        resequence_temperature: LigandMPNN sampling temperature.
        sequences_per_design: Sequences sampled per backbone. Above 1, each
            becomes its own design from folding onward.
        use_side_chain_context: Pass fixed-residue sidechain coordinates to
            LigandMPNN. True improves results; False reproduces the paper.
        folding_seeds: AlphaFold 3 seeds, as `[0, 1, 2]`. Every state is
            predicted once per seed, so this multiplies the folding cost.
            Defaults to one seed.
        folding_states: Structures to predict per design, as
            `[{"name": "complex", "ligands": [{"id": "B", "ccd_code": "6NT"}]}]`.
            Omitted, upstream derives the apo and ligand-bound states from the
            motif, which is what the upstream examples rely on.
        af3_model: `af3_la` or `af3`. The leaving-atom weights are the default
            because they are fine-tuned for covalent intermediates.
        suite: Enzyme metric suite: `kemp_eliminase`, `serine_esterase`,
            `dehp_esterase`, `carbene_transfer` or `nitrene_transfer`. Empty
            runs the generic metrics only.
        reference_uri: `s3://bucket/key` of a structure to score against.
            Empty scores each design against its own recorded motif, which is
            usually what you want.

    Returns:
        `designs`, one entry per final design, carrying its predicted
        structures per state and seed, its confidences, and its metrics.
        `summary` is the campaign table as written, and `stages` says which
        stages finished. There is no `best`: which metric decides a design
        depends on the suite, so the ranking is left to the caller.
    """
    motif = convert.fetch_structure(motif_uri, "cif", rt.work, stem="motif")
    if motif.renamed_chains:
        raise ValueError(
            f"Converting {motif_uri} to mmCIF renamed chains "
            f"{motif.renamed_chains}, so motif_str={motif_str!r} would name "
            f"chains the pipeline cannot find. Supply the motif as mmCIF, or "
            f"rewrite the motif string against the new names."
        )

    if af3_model not in inputs.AF3_MODELS:
        raise ValueError(
            f"af3_model={af3_model!r} is not a choice. Use one of "
            f"{', '.join(inputs.AF3_MODELS)}."
        )

    # The manifest resolves its paths against its own directory, so it is
    # written beside the motif.
    job_dir = motif.path.parent
    job: dict[str, Any] = dict(
        name=name,
        input_file=motif.path.name,
        motif_str=motif_str,
        num_designs=num_designs,
        motif_atoms=motif_atoms,
        is_author_naming=is_author_naming,
        num_sampling_steps=num_sampling_steps,
        seq_length=seq_length,
        unindexed_motif_residues=unindexed_motif_residues,
        reseq_residues=reseq_residues,
        seed_start=seed_start,
    )

    if partial_diffusion_uri:
        parent = convert.fetch_structure(
            partial_diffusion_uri, "cif", job_dir, stem="parent"
        )
        job["partial_diffusion_input_file"] = parent.path.name
        job["partial_diffusion_num_steps"] = partial_diffusion_num_steps
    elif partial_diffusion_num_steps:
        job["partial_diffusion_num_steps"] = partial_diffusion_num_steps

    reference = ""
    if reference_uri:
        reference = convert.fetch_structure(
            reference_uri, "cif", job_dir, stem="reference"
        ).path.name

    document = inputs.manifest(
        job=inputs.design_job(**job),
        resequence=resequence,
        resequence_temperature=resequence_temperature,
        sequences_per_design=sequences_per_design,
        use_side_chain_context=use_side_chain_context,
        folding_seeds=folding_seeds,
        folding_states=folding_states,
        suite=suite,
        reference_cif=reference,
    )
    manifest_path = inputs.write_manifest(job_dir / "manifest.json", document)

    out_dir = rt.work / "campaign"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator_dir = _fetch(
        rt,
        inputs.GENERATOR_DIR,
        inputs.GENERATOR_WEIGHTS,
        inputs.GENERATOR_URL,
        "550 MB",
    )
    af3_dir = _fetch(rt, *inputs.AF3_MODELS[af3_model], "1 GB")

    run(
        ["python", inputs.RUNNER,
         *inputs.pipeline_args(
             manifest_path=manifest_path,
             output_dir=out_dir,
             generator_dir=generator_dir,
             af3_dir=af3_dir,
             resequence=resequence,
         )],
        cwd=inputs.REPO_DIR,
    )

    return _collect(rt, out_dir, name=name, motif_uri=motif_uri,
                    motif_str=motif_str, suite=suite, af3_model=af3_model)


def _collect(
    rt, out_dir: Path, *, name: str, motif_uri: str, motif_str: str,
    suite: str, af3_model: str,
) -> dict[str, Any]:
    """Upload every artifact and assemble the result."""
    generated = outputs.read_generated(out_dir)
    resequenced = outputs.read_resequenced(out_dir)
    evaluations = outputs.read_evaluations(out_dir)

    # Folding keys off the resequenced prefix when resequencing ran, and off
    # the generated one otherwise.
    folded = outputs.read_folded(
        out_dir,
        [variant["prefix"] for variants in resequenced.values()
         for variant in variants]
        or [backbone["prefix"] for backbone in generated],
    )

    designs = []
    for backbone in generated:
        parent = backbone["prefix"]
        entry: dict[str, Any] = {
            "design": parent,
            "generated_uri": rt.storage.upload(
                backbone["structure"], rt.prefix(f"{parent}.cif")
            ),
            "generated_sequence": backbone["sequence"],
            "fixed_residues": outputs.fixed_residues(backbone["metadata"]),
        }
        if backbone["motif"] is not None:
            entry["motif_uri"] = rt.storage.upload(
                backbone["motif"], rt.prefix(f"{parent}{outputs.MOTIF_SUFFIX}")
            )

        # Resequencing turns one backbone into several designs, each with its
        # own folds and metrics; without it the backbone is the design.
        variants = resequenced.get(parent) or [{"prefix": parent, "structure": None,
                                                "sequence": backbone["sequence"]}]
        entry["designs"] = [
            _variant(rt, variant, folded, evaluations) for variant in variants
        ]
        designs.append(entry)

    return {
        "name": name,
        "motif_uri": motif_uri,
        "motif_str": motif_str,
        "suite": suite,
        "af3_model": af3_model,
        "stages": outputs.stages_completed(out_dir),
        "designs": designs,
        "summary": outputs.read_summary(out_dir),
    }


def _variant(
    rt, variant: dict[str, Any],
    folded: dict[str, list[dict[str, Any]]],
    evaluations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One final design: its sequence, its predictions and its metrics."""
    prefix = variant["prefix"]
    predictions = folded.get(prefix, [])

    entry: dict[str, Any] = {
        "design": prefix,
        "sequence": variant["sequence"],
        "length": len(variant["sequence"]),
        "metrics": evaluations.get(prefix, {}),
        "ranking_confidence": outputs.ranking_confidence(predictions),
        "folded": [
            {
                "state": prediction["state"],
                "seed": prediction["seed"],
                "structure_uri": rt.storage.upload(
                    prediction["structure"],
                    rt.prefix(
                        f"{prefix}_{prediction['state']}"
                        f"_seed{prediction['seed']}.cif"
                    ),
                ),
                "confidences": prediction["confidences"],
            }
            for prediction in predictions
        ],
    }
    if variant["structure"] is not None:
        entry["structure_uri"] = rt.storage.upload(
            variant["structure"], rt.prefix(f"{prefix}{outputs.RESEQ_SUFFIX}")
        )
    return entry
