"""ProteinMPNN (https://github.com/dauparas/ProteinMPNN) as Modal GPU actions.

Three actions, split by what they return rather than by what constrains them.
Every way of restricting a design, which chains are designed, which positions
are frozen, which are tied, and which amino acids are biased or forbidden, is a
parameter of `design_sequences` rather than a separate entry point.

    design_sequences       sample sequences for a backbone
    score_sequences        score existing sequences against a backbone
    residue_probabilities  per-position amino acid distributions

Binder design is the `design_chains` parameter of `design_sequences`: name
only the binder chain and every other chain is held fixed as context. Note it
names the chain to change, the opposite of a generator's `target_chain`.

Deploy, then call from anywhere:

    modal deploy tools/proteinmpnn/app.py

    fn = modal.Function.from_name("proteinmpnn", "design_sequences")
    fn.remote(structure_uri="s3://bucket/structures/complex.pdb", design_chains="B")

Or run one directly:

    modal run tools/proteinmpnn/app.py::design_sequences \\
        --structure-uri s3://bucket/complex.pdb --design-chains B
"""

from __future__ import annotations

from typing import Any

from tools.common import convert, fasta
from tools.common import design as design_spec
from tools.common.images import torch_image, with_local_sources
from tools.common.shell import batches_for, run
from tools.common.tool import app_for, tool
from tools.proteinmpnn import inputs, outputs
from tools.proteinmpnn.inputs import Constraints

REPO_URL = "https://github.com/dauparas/ProteinMPNN.git"
REPO_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"

GPU = "T4"
TIMEOUT = 30 * 60

app = app_for("proteinmpnn")

image = with_local_sources(
    torch_image().run_commands(
        f"git clone {REPO_URL} {inputs.REPO_DIR}",
        f"cd {inputs.REPO_DIR} && git checkout {REPO_COMMIT}",
    )
)

OPTIONS = dict(image=image, gpu=GPU, timeout=TIMEOUT)


def _constraints(
    chains: str,
    fixed_positions: str,
    tied_positions: str,
    homooligomer: bool,
    omit_aas: str,
    bias_aas: dict[str, float] | None,
) -> Constraints:
    return Constraints(
        chains=chains,
        fixed_positions=fixed_positions,
        tied_positions=tied_positions,
        homooligomer=homooligomer,
        omit_aas=omit_aas,
        bias_aas=bias_aas or {},
    )


@tool("proteinmpnn/design_sequences", **OPTIONS)
def design_sequences(
    rt,
    structure_uri: str,
    design_chains: str = "A",
    num_designs: int = 8,
    temperature: float = 0.1,
    seed: int = 37,
    model_name: str = "v_48_020",
    fixed_positions: str = "",
    tied_positions: str = "",
    homooligomer: bool = False,
    omit_aas: str = "X",
    bias_aas: dict[str, float] | None = None,
    soluble: bool = False,
    ca_only: bool = False,
    backbone_noise: float = 0.0,
    batch_size: int = 8,
    name: str = "input",
) -> dict[str, Any]:
    """Sample sequences for a backbone.

    Args:
        structure_uri: `s3://bucket/key` of the input PDB.
        design_chains: Space-separated chain IDs to **design**, e.g. "A" or
            "A B". Chains present in the structure but not named here are held
            fixed and act as context. For binder design this is the binder
            chain, which is the opposite of a generator's `target_chain`: pass
            a design's `binder_chain`, never its `target_chain`.
        num_designs: How many sequences to sample.
        temperature: Sampling temperature. Values from 0.1 to 0.3 are typical;
            higher gives more diversity at the cost of lower confidence.
        seed: Random seed. Pass a nonzero value for reproducible runs.
        model_name: Weight checkpoint. The trailing number is training backbone
            noise in hundredths of an Angstrom, so v_48_002 suits crystal
            structures and v_48_020, the default, suits designed or predicted
            backbones.
        fixed_positions: Positions to hold at their input identity, as
            comma-separated groups matching the order of `design_chains`. Use
            this to preserve interface hotspots.
        tied_positions: Positions forced to share an identity across chains.
        homooligomer: Tie every position across all chains.
        omit_aas: Amino acids never sampled, e.g. "CX" to exclude cysteine.
        bias_aas: Per amino acid log-odds added before sampling, e.g.
            {"D": 1.39, "E": 1.39} to favour polar residues.
        soluble: Use weights trained on soluble proteins only.
        ca_only: Use alpha-carbon-only weights, for backbones without full atoms.
        backbone_noise: Gaussian noise in Angstroms added to backbone
            coordinates, which trades fidelity for robustness to imperfect
            input geometry.
        batch_size: Sequences per forward pass. Reduce on GPU memory errors.
        name: Identifier for this structure, used in outputs.

    Returns:
        Sequenced candidates in the shape described in `tools.common.design`,
        the same one BoltzGen returns, sorted best first. Score is mean
        negative log likelihood over designed positions, so lower is better.
        The backbone is unchanged, so every design shares the input
        `structure_uri`; what varies is `sequences`, which describes the whole
        complex, the designed chains carrying their new sequence and the fixed
        ones their input identity. That goes straight into a predictor's
        `chains` argument with nothing else to look up.
    """
    # ProteinMPNN expecps PDB files
    source = convert.fetch_structure(structure_uri, "pdb", rt.work)
    structure = source.path
    design_chains = convert.resolve_chains(design_chains, source)
    prepared = inputs.prepare(
        rt.work, structure, name,
        _constraints(design_chains, fixed_positions, tied_positions,
                     homooligomer, omit_aas, bias_aas),
        ca_only=ca_only,
    )

    run(
        ["python", inputs.RUNNER, *prepared.args,
         *inputs.model_args(model_name, soluble, ca_only),
         "--num_seq_per_target", num_designs,
         "--batch_size", batches_for(num_designs, batch_size),
         "--sampling_temp", temperature,
         "--backbone_noise", backbone_noise,
         "--seed", seed]
    )

    fasta_path = prepared.out_dir / "seqs" / f"{name}.fa"
    records = outputs.parse_designs(fasta_path)
    designed = design_chains.split()
    chain_lengths = design_spec.chain_lengths(structure)
    # The fixed chains at their input identity, so each design describes the
    # whole complex rather than only the part that changed. That is what makes
    # it foldable without a second lookup for the target's sequence, and it
    # keeps the target at the length the backbone actually has: a generator may
    # have truncated it, and folding the full-length target instead would break
    # a self-consistency comparison, which pairs residues by position.
    native = design_spec.read_sequences(structure)

    designs = []
    ranked = sorted(
        (r for r in records if not r["is_native"]),
        key=lambda r: r.get("score", float("inf")),
    )
    for index, raw in enumerate(ranked):
        sequence = raw.pop("sequence")
        designs.append({
            "design": index,
            # The backbone is what was handed in; only the sequence is new.
            "structure_uri": structure_uri,
            "binder_chain": designed[0] if len(designed) == 1 else None,
            "chain_lengths": chain_lengths,
            "sequences": {
                **native,
                **design_spec.split_sequences(sequence, designed),
            },
            **{k: v for k, v in raw.items() if k != "is_native"},
        })

    native = next((r for r in records if r["is_native"]), None)
    return design_spec.envelope(
        name=name,
        designs=designs,
        structure_uri=structure_uri,
        design_chains=designed,
        model_name=model_name,
        designs_uri=rt.storage.upload(fasta_path, rt.prefix("designs.fa")),
        native=native,
    )


@tool("proteinmpnn/score_sequences", **OPTIONS)
def score_sequences(
    rt,
    structure_uri: str,
    sequences: list[str] | None = None,
    chains: str = "A",
    num_decoding_orders: int = 10,
    seed: int = 37,
    model_name: str = "v_48_020",
    fixed_positions: str = "",
    omit_aas: str = "X",
    soluble: bool = False,
    ca_only: bool = False,
    batch_size: int = 8,
    name: str = "input",
) -> dict[str, Any]:
    """Score sequences against a backbone without generating new ones.

    Much cheaper than re-sampling, so this is the way to rank or filter
    candidates that came from elsewhere, including designs from an earlier
    `design_sequences` call.

    This action writes no artifact. Everything the runner computes is returned
    inline, so there is nothing worth storing.

    Args:
        structure_uri: `s3://bucket/key` of the input PDB.
        sequences: Sequences to score. Each must cover the structure's chains
            in alphabetical chain order, and may separate them with "/", which
            is ignored. Omit to score only the structure's own sequence.
        chains: Space-separated chain IDs treated as designed. Score is
            averaged over these positions; global score covers all positions.
        num_decoding_orders: How many random autoregressive decoding orders to
            average each score over. More gives a steadier estimate.
        seed: Random seed.
        model_name: Weight checkpoint, as in `design_sequences`.
        fixed_positions: Positions excluded from the per-chain score.
        omit_aas: Amino acids masked during scoring.
        soluble: Use weights trained on soluble proteins only.
        ca_only: Use alpha-carbon-only weights.
        batch_size: Decoding orders per forward pass.
        name: Identifier for this structure.

    Returns:
        `native`, the structure's own sequence scored; and `scored`, one entry
        per supplied sequence with mean score, its standard deviation across
        decoding orders, and the global score. Lower scores are better.
    """
    source = convert.fetch_structure(structure_uri, "pdb", rt.work)
    structure = source.path
    chains = convert.resolve_chains(chains, source)
    prepared = inputs.prepare(
        rt.work, structure, name,
        _constraints(chains, fixed_positions, "", False, omit_aas, None),
        ca_only=ca_only,
    )

    cmd = [
        "python", inputs.RUNNER, *prepared.args,
        *inputs.model_args(model_name, soluble, ca_only),
        "--score_only", 1,
        "--num_seq_per_target", num_decoding_orders,
        "--batch_size", batches_for(num_decoding_orders, batch_size),
        "--seed", seed,
    ]

    sequences = list(sequences or [])
    if sequences:
        query = rt.work / "query.fa"
        query.write_text(
            fasta.format((f"query_{i}", s) for i, s in enumerate(sequences, 1))
        )
        cmd += ["--path_to_fasta", str(query)]

    run(cmd)

    score_dir = prepared.out_dir / "score_only"
    result: dict[str, Any] = {
        "name": name,
        "designed_chains": chains.split(),
        "model_name": model_name,
        "structure_uri": structure_uri,
        "native": outputs.read_scores(score_dir / f"{name}_pdb.npz"),
        "scored": [],
    }
    # The runner numbers FASTA outputs from 1, in the order supplied.
    for i, sequence in enumerate(sequences, start=1):
        entry = outputs.read_scores(score_dir / f"{name}_fasta_{i}.npz")
        entry["input_sequence"] = sequence
        result["scored"].append(entry)
    return result


@tool("proteinmpnn/residue_probabilities", **OPTIONS)
def residue_probabilities(
    rt,
    structure_uri: str,
    chains: str = "A",
    mode: str = "conditional",
    num_samples: int = 10,
    seed: int = 37,
    model_name: str = "v_48_020",
    fixed_positions: str = "",
    omit_aas: str = "X",
    soluble: bool = False,
    ca_only: bool = False,
    batch_size: int = 8,
    top_k: int = 5,
    include_matrix: bool = False,
    name: str = "input",
) -> dict[str, Any]:
    """Per-position amino acid distributions for a backbone.

    Use this to choose hotspots, to see which positions the model is confident
    about, and to find positions worth leaving free in a later design run.

    Args:
        structure_uri: `s3://bucket/key` of the input PDB.
        chains: Space-separated chain IDs to report on. Others stay fixed as
            context and are excluded from the output.
        mode: Which distribution to compute.
            `conditional` gives p(residue | backbone and the rest of the
            sequence), the best guide to a single substitution in an otherwise
            intact sequence.
            `backbone` gives p(residue | backbone) with sequence context
            withheld, which is what a design run effectively sees.
            `unconditional` is the same quantity in one forward pass, cheaper
            and coarser.
        num_samples: Decoding orders to average over. Ignored by
            `unconditional`, which is deterministic.
        seed: Random seed.
        model_name: Weight checkpoint, as in `design_sequences`.
        fixed_positions: Positions excluded from the output.
        omit_aas: Amino acids masked before computing the distribution.
        soluble: Use weights trained on soluble proteins only.
        ca_only: Use alpha-carbon-only weights.
        batch_size: Decoding orders per forward pass.
        top_k: Amino acids reported per position, most likely first.
        include_matrix: Also return the full 21-way distribution per position.
        name: Identifier for this structure.

    Returns:
        `positions`, one entry per designable position holding the input
        residue, the most likely amino acids with probabilities, and entropy in
        bits. High entropy means the model considers many residues plausible.
        `probabilities_uri` points at the archive holding the full 21-way
        distribution for every position, which the summary drops.
    """
    modes = {
        "conditional": (["--conditional_probs_only", "1"], "conditional_probs_only"),
        "backbone": (
            ["--conditional_probs_only", "1", "--conditional_probs_only_backbone", "1"],
            "conditional_probs_only",
        ),
        "unconditional": (["--unconditional_probs_only", "1"], "unconditional_probs_only"),
    }
    if mode not in modes:
        raise ValueError(f"mode must be one of {', '.join(modes)}, got {mode!r}")
    mode_args, subdir = modes[mode]

    source = convert.fetch_structure(structure_uri, "pdb", rt.work)
    structure = source.path
    chains = convert.resolve_chains(chains, source)
    prepared = inputs.prepare(
        rt.work, structure, name,
        _constraints(chains, fixed_positions, "", False, omit_aas, None),
        ca_only=ca_only,
    )

    run(
        ["python", inputs.RUNNER, *prepared.args,
         *inputs.model_args(model_name, soluble, ca_only), *mode_args,
         "--num_seq_per_target", num_samples,
         "--batch_size", batches_for(num_samples, batch_size),
         "--seed", seed]
    )

    archive = prepared.out_dir / subdir / f"{name}.npz"
    result = outputs.read_probabilities(
        archive, top_k=top_k, include_matrix=include_matrix
    )
    result.update({
        "name": name,
        "mode": mode,
        "designed_chains": chains.split(),
        "model_name": model_name,
        "structure_uri": structure_uri,
        "probabilities_uri": rt.storage.upload(
            archive, rt.prefix("probabilities.npz")
        ),
    })
    return result

