"""Boltz-2 (https://github.com/jwohlwend/boltz) as a Modal GPU action.

One action, `predict_complex`: given the sequences of a complex, predict its
structure and report how confident the model is that those chains fold and
interact as given. In a binder loop this is the filter. ProteinMPNN will hand
you many sequences and rank them by its own likelihood, which says nothing
about whether they bind; the interface score here does.

Model weights are several gigabytes and are downloaded on first use into a
Modal Volume, so only the first call pays for that.

    modal deploy tools/boltz/app.py

    fn = modal.Function.from_name("boltz", "predict_complex")
    fn.remote(chains={"A": "MKT...", "B": "GHN..."})

Or from the command line:

    modal run tools/boltz/app.py::predict_complex \\
        --chains '{"A": "MKT...", "B": "GHN..."}'
"""

from __future__ import annotations

from typing import Any

from tools.boltz import inputs, outputs
from tools.common import a3m, folding
from tools.common import chains as chain_spec
from tools.common import msa as msa_setting
from tools.common import predictions
from tools.common.images import base_image, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool

BOLTZ_VERSION = "2.2.1"

# Boltz-2 is a large model; a T4's 16GB is not enough headroom for complexes.
GPU = "A10G"
TIMEOUT = 60 * 60

# Weights, the chemical component dictionary and the molecule set land here on
# first use and are shared by every later call.
CACHE = "boltz-cache"
CACHE_DIR = "/cache/boltz"

app = app_for("boltz")

image = with_local_sources(
    base_image()
    .pip_install(f"boltz[cuda]=={BOLTZ_VERSION}")
    .env({"BOLTZ_CACHE": CACHE_DIR})
)


@tool(
    "boltz/predict_complex",
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    cache=CACHE,
    cache_path=CACHE_DIR,
)
def predict_complex(
    rt,
    chains: dict[str, dict | str],
    samples: int = 1,
    seed: int = 0,
    name: str = "complex",
    default_msa: str = "",
    recycling_steps: int = 3,
    use_potentials: bool = False,
) -> dict[str, Any]:
    """Predict the structure of a complex and score its interface.

    Args:
        chains: The complex, one entry per chain. A bare string is the
            sequence; a mapping may also carry that chain's `msa`:

                {"A": {"sequence": <target>, "msa": "s3://bucket/t.a3m"},
                 "B": <binder sequence>}

            An alignment is the largest accuracy lever here and the only
            setting that sends a sequence anywhere. An `s3://` value uses one
            already searched, an http(s) URL has Boltz query that server, and
            empty runs single sequence.

            Mixing per chain is supported and tested: a target with an
            alignment beside a binder with none predicts fine, which is the
            binder case. Sharing one alignment across chains with *different*
            sequences is the trap: it predicts without complaint but degrades,
            measurably. Barnase-barstar scored ipSAE 0.94 with each chain
            aligned and 0.68 when barstar was handed barnase's alignment.
            Identical sequences sharing an alignment is fine and correct.
        samples: How many structures to sample. More gives a better sense of
            whether the prediction is stable, at linear cost.
        seed: Random seed.
        name: Identifier for the job; Boltz names its outputs after it.
        default_msa: Applied to every chain that names no `msa` of its own; a
            per-chain value always wins. `colabfold` has Boltz fetch alignments
            itself, which is what a complex assembled by a generator needs,
            since no earlier step holds one to pass along.

            Boltz searches every chain it is given this way, including a de
            novo binder that has no homologues to find, and caches nothing
            between calls, so the same target is searched again per design. At
            a handful of designs that is a few extra queries; at campaign scale
            search once with `msa/search_msa`, which caches by sequence, and
            pass the resulting `s3://` alignments per chain instead.
        recycling_steps: Refinement passes through the model. Boltz-specific.
        use_potentials: Apply inference-time potentials for physical
            plausibility. Slower, and usually worth it for a final check
            rather than for bulk filtering. Boltz-specific.

    Returns:
        The standard predictor result described in `tools.common.folding`:
        `models` sorted most confident first, each with `structure_uri`, the
        companion URIs and the shared confidence scores, and `best` as the
        first of those. Interface confidence, `iptm`, is the number to rank
        binder designs on. Everything runs 0 to 1, higher is better.
    """
    complex_chains = chain_spec.parse(chains)
    if default_msa:
        for chain in complex_chains.values():
            if not chain.msa:
                chain.msa = default_msa
    # Fetch any alignment that was searched earlier, so Boltz reads a file
    # rather than re-querying a server for something already known.
    for chain_id, chain in complex_chains.items():
        if msa_setting.kind(chain.msa) == msa_setting.ARTIFACT:
            chain.alignment = rt.storage.download(
                chain.msa, rt.work / "msa" / f"{chain_id}.a3m"
            )
            # An alignment built for a slightly different sequence, as when a
            # series of point variants reuses its parent's, still describes the
            # parent in its first row. Boltz folds it without complaining and
            # the prediction is far worse: measured here, one substitution took
            # a real protein from pLDDT 0.97 to 0.46.
            if a3m.retarget(chain.alignment, chain.sequence):
                print(
                    f"chain {chain_id}: alignment retargeted from its original "
                    f"query to this chain's sequence",
                    flush=True,
                )
    folding.shared_alignment_warning(complex_chains)
    server = folding.one_server(complex_chains, "Boltz")

    job = inputs.write_job(rt.work / f"{name}.yaml", complex_chains)

    out_dir = rt.work / "out"
    command = [
        "boltz", "predict", str(job),
        "--out_dir", str(out_dir),
        "--cache", CACHE_DIR,
        "--output_format", "pdb",
        "--diffusion_samples", samples,
        "--recycling_steps", recycling_steps,
        "--seed", seed,
        "--accelerator", "gpu",
        "--devices", 1,
    ]
    # Only chains that named a server need one; chains holding an alignment
    # already have their file referenced in the job.
    if server:
        command += ["--use_msa_server", "--msa_server_url", server]
    if use_potentials:
        command.append("--use_potentials")

    run(command)
    # Persist the weights this call may have just downloaded.
    rt.save_cache()

    raw_models = outputs.read_predictions(outputs.find_job_dir(out_dir, name))

    chain_ids = list(complex_chains)
    models: list[dict[str, Any]] = []
    for raw in raw_models:
        model: dict[str, Any] = {"model": raw["model"]}
        model.update(
            predictions.publish(
                structure=raw["structure"],
                companion_files={k: raw.get(k) for k in predictions.COMPANIONS},
                prefix=rt.prefix(),
                stem=f"model_{raw['model']}",
            )
        )
        model.update(folding.scores(raw, chain_ids))
        if len(chain_ids) == 2:
            model["interface_iptm"] = folding.interface(model, *chain_ids)
        models.append(model)

    return folding.envelope(
        name=name,
        predictor="boltz",
        version=BOLTZ_VERSION,
        complex_chains=complex_chains,
        models=models,
        seed=seed,
    )
