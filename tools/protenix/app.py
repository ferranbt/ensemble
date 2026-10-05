"""Protenix (https://github.com/bytedance/Protenix) as a Modal GPU action.

An open reproduction of AlphaFold3 from ByteDance. One action,
`predict_complex`, with the same shape as the Boltz one: give it the sequences
of a complex, get back a predicted structure and confidence scores.

Having both matters for a binder pipeline. No single in-silico metric reliably
separates real binders from decoys, and the standing advice is to combine
orthogonal signals. Two independently trained predictors agreeing on an
interface is a stronger signal than either alone, and disagreement is itself
informative.

Weights download on first use into a Modal Volume, so only the first call pays.

    modal run tools/protenix/app.py::predict_complex \\
        --chains '{"A": "MKT...", "B": "GHN..."}'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from tools.common import chains as chain_spec
from tools.common import folding
from tools.common import msa as msa_setting
from tools.common import predictions
from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool
from tools.protenix import companions, inputs, outputs

PROTENIX_VERSION = "2.0.0"
# Checkpoint names the CLI accepts via -n.
DEFAULT_MODEL = "protenix_base_default_v1.0.0"

GPU = "A10G"
TIMEOUT = 60 * 60
CACHE = "protenix-cache"
CACHE_DIR = "/cache/protenix"

def msa_args(msa: str, precomputed: bool = False) -> list[str]:
    """Runner flags selecting where evolutionary context comes from.

    With precomputed alignments the runner must use them rather than search,
    so no server mode is named: the paths are in the job's own JSON.
    """
    if precomputed:
        return ["--use_msa", "true"]
    mode = msa_setting.protenix_mode(msa)
    if not mode:
        return ["--use_msa", "false"]
    return ["--use_msa", "true", "--msa_server_mode", mode]


def job_alignment(complex_chains: dict[str, chain_spec.Chain]) -> str:
    """The one alignment server Protenix may search for a whole job.

    Chains carry their own `msa` in the shared input contract, but Protenix
    searches once for the complex, so any chains naming a server have to agree.
    Precomputed `s3://` alignments are not servers and are handled by
    `alignment_paths` instead.
    """
    named = sorted({
        chain.msa for chain in complex_chains.values()
        if chain.msa and msa_setting.kind(chain.msa) != msa_setting.ARTIFACT
    })
    if len(named) > 1:
        raise ValueError(
            f"Protenix searches one alignment for the whole job, but this "
            f"complex names {len(named)}: {named}. Give every chain the same "
            f"setting, or use Boltz, which takes one per chain."
        )
    return named[0] if named else ""


def alignment_paths(
    rt, complex_chains: dict[str, chain_spec.Chain]
) -> dict[str, dict[str, str]]:
    """A local alignment file for every chain, for a job using precomputed ones.

    Protenix reads an alignment per chain as `unpairedMsaPath`, so a chain that
    has one searched earlier never needs the public server again. Its own
    client asks that server for several databases at once and fails the whole
    job when any of them is unavailable, which is a real failure mode rather
    than a hypothetical: it is what stopped this tool working.

    A chain with no alignment gets a file holding just its own sequence, which
    is what "no homologues" means and is the right answer for a de novo binder.
    Writing one for every chain keeps the job in a single mode rather than
    mixing precomputed and searched, which is not a combination Protenix
    documents.
    """
    directory = rt.work / "msa"
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, dict[str, str]] = {}
    for chain_id, chain in complex_chains.items():
        if msa_setting.kind(chain.msa) == msa_setting.ARTIFACT:
            local = rt.storage.download(chain.msa, directory / f"{chain_id}.a3m")
        else:
            local = directory / f"{chain_id}.a3m"
            local.write_text(f">{chain_id}\n{chain.sequence}\n")
        paths[chain_id] = {"unpairedMsaPath": str(local)}
    return paths


def uses_precomputed(complex_chains: dict[str, chain_spec.Chain]) -> bool:
    """Whether any chain brought an alignment of its own."""
    return any(
        msa_setting.kind(chain.msa) == msa_setting.ARTIFACT
        for chain in complex_chains.values()
    )


app = app_for("protenix")

# Protenix compiles a custom layer-norm CUDA kernel when its model module is
# imported. That is not behind a flag, so it needs the full CUDA toolkit and
# nvcc, which the PyPI torch wheel does not carry: the wheel bundles only the
# runtime. Hence a devel base image rather than debian_slim.
image = with_local_sources(
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    # ninja drives torch's just-in-time extension build.
    .pip_install("ninja", *STORAGE_PACKAGES)
    .pip_install(f"protenix=={PROTENIX_VERSION}")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)
# Protenix downloads weights and reference data under the home directory.
# Pointing HOME at the cache is done at run time, not in the image: setting it
# as an image variable makes the build write there, and Modal refuses to mount
# a volume over a path that is not empty.


@tool(
    "protenix/predict_complex",
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
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Predict the structure of a complex with Protenix.

    Args:
        chains: The complex, one entry per chain. A bare string is the
            sequence; a mapping may also carry that chain's `msa`. See
            `tools.common.chains`.

            A precomputed `s3://` alignment is read directly, per chain, and is
            the reliable option: Protenix's own client asks the public server
            for several databases at once and fails the whole job if any is
            unavailable. Search once with `msa/search_msa`, which caches by
            sequence, and pass the result.

            Asking Protenix to search instead means naming a service rather
            than an address, so that form takes `colabfold` or the ColabFold
            URL, and one setting covers the whole job. Mixing the two is not
            supported: if any chain brings its own alignment, every chain uses
            a file, and a chain without one is given its own sequence alone,
            which is what a de novo binder should have anyway.
        samples: How many structures to sample.
        seed: Random seed.
        name: Job name. Protenix names its outputs after it.
        model: Checkpoint name passed to the CLI. Protenix-specific.

    Returns:
        The standard predictor result described in `tools.common.folding`.
        `msa` records what was used, since that changes both the accuracy and
        whether the sequences left the container.
    """
    import os

    complex_chains = chain_spec.parse(chains)
    folding.shared_alignment_warning(complex_chains)
    msa = job_alignment(complex_chains)

    # Send Protenix's downloads into the mounted volume. The subprocess
    # inherits this, so weights survive between calls.
    os.environ["HOME"] = CACHE_DIR
    os.environ.setdefault("PROTENIX_DATA_ROOT_DIR", CACHE_DIR)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    precomputed = uses_precomputed(complex_chains)
    job = inputs.write_job(
        rt.work / f"{name}.json",
        {c: chain.sequence for c, chain in complex_chains.items()},
        name=name,
        msa_paths=alignment_paths(rt, complex_chains) if precomputed else None,
    )
    out_dir = rt.work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    run([
        "protenix", "pred",
        "--input", str(job),
        "--out_dir", str(out_dir),
        "--model_name", model,
        "--seeds", seed,
        "--sample", samples,
        "--use_default_params", "true",
        # Without this Protenix writes only its summary, and the PAE matrix
        # and per-atom pLDDT never reach disk, leaving nothing to score an
        # interface from.
        "--need_atom_confidence", "true",
        *msa_args(msa, precomputed),
        # The default kernels are custom CUDA extensions compiled on first use,
        # which needs the full toolkit and nvcc. The PyPI torch wheel ships
        # only the CUDA runtime, so ask for the pure PyTorch implementations
        # instead. Slower, and it keeps the image small.
        "--trimul_kernel", "torch",
        "--triatt_kernel", "torch",
        "--enable_fusion", "false",
    ])
    # Persist whatever weights this call downloaded.
    rt.save_cache()

    raw_models = outputs.read_predictions(out_dir)
    chain_ids = list(complex_chains)
    models: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_models):
        structure = raw["structure"]
        stem = f"model_{index}"
        # Normalised to the shared companion shape so ipSAE, and anything else
        # downstream, reads Protenix exactly as it reads Boltz.
        written = companions.write(
            structure_name=f"{stem}{structure.suffix}",
            work_dir=rt.work,
            full_data=raw.get("full_data_file"),
            summary=raw.get("confidence_file"),
        )
        entry: dict[str, Any] = {
            "model": index,
            # Which seed and sample this came from, which Protenix encodes in
            # its output paths and nothing else records.
            "seed": raw.get("seed"),
            "sample": raw.get("sample"),
        }
        entry.update(
            predictions.publish(
                structure=structure,
                companion_files=written,
                prefix=rt.prefix(),
                stem=stem,
            )
        )
        entry.update(folding.scores(raw, chain_ids))
        if len(chain_ids) == 2:
            entry["interface_iptm"] = folding.interface(entry, *chain_ids)
        models.append(entry)

    return folding.envelope(
        name=name,
        predictor="protenix",
        version=PROTENIX_VERSION,
        complex_chains=complex_chains,
        models=models,
        seed=seed,
        checkpoint=model,
        msa=msa_setting.normalise(msa),
    )

