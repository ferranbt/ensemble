"""ESMFold2 (https://github.com/Biohub/esm) as a Modal GPU action.

A structure predictor pairing ESM Cambrian embeddings with a diffusion head.
Despite the shared name it does a different job from `tools.esm`: that scores
sequences and never produces coordinates, this predicts structures. It belongs
beside Boltz and Protenix, not beside the language model.

Two properties make it the best fit of the three predictors here:

Single sequence is its primary mode rather than a degraded one. We run every
predictor without alignments, so nothing leaves the container, and the others
take a real accuracy penalty for that. This one is built for it.

It is strongest where you are headed. On antibody-antigen complexes from a
single sequence it reports a higher DockQ pass rate than AlphaFold3, which is
exactly the affinity maturation case.

Weights are ungated on Hugging Face and cached in a Modal Volume, so only the
first call downloads them.

    modal run tools/esmfold2/app.py::predict_complex \\
        --chains '{"A": "MKT...", "B": "GHN..."}'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from tools.common import chains as chain_spec
from tools.common import folding, predictions
from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.tool import app_for, tool

CHECKPOINT = "biohub/ESMFold2"
ESM_VERSION = "3.4.1"

# The 6B model does not fit an A10G at full precision.
GPU = "A100-80GB"
TIMEOUT = 60 * 60
CACHE = "esmfold2-cache"
CACHE_DIR = "/cache/huggingface"

app = app_for("esmfold2")

# The package requires Python 3.12 or newer.
image = with_local_sources(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*STORAGE_PACKAGES, f"esm=={ESM_VERSION}")
    .env({"HF_HOME": CACHE_DIR})
)


@tool(
    "esmfold2/predict_complex",
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
    num_loops: int = 20,
    num_sampling_steps: int = 100,
    half_precision: bool = True,
) -> dict[str, Any]:
    """Predict the structure of a complex from sequence alone.

    Args:
        chains: The complex, one entry per chain. A bare string is the
            sequence; see `tools.common.chains`. A chain that names an `msa` is
            refused rather than ignored: this predictor takes no alignment at
            all, and quietly dropping one would report a single-sequence
            prediction as though it had used homologues.
        samples: How many structures to sample. More than one shows whether
            the prediction is stable.
        seed: Random seed.
        name: Identifier for the run.
        num_loops: Recycling passes through the trunk. More costs time and
            usually helps on harder targets. ESMFold2-specific.
        num_sampling_steps: Diffusion steps for the structure head.
            ESMFold2-specific.
        half_precision: Run under bfloat16 autocast, which is faster and uses
            less memory. Turn it off to predict at full precision.

    Returns:
        The standard predictor result described in `tools.common.folding`.
        Interface confidence, `iptm`, is the number to rank binder designs on.
        Everything runs 0 to 1, higher is better.
    """
    import contextlib

    import torch
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        EsmFold2Model,
        ProteinInput,
        StructurePredictionInput,
    )

    complex_chains = chain_spec.parse(chains)
    folding.single_sequence_only(complex_chains, "ESMFold2")

    model = EsmFold2Model.from_pretrained(CHECKPOINT, device="cuda").eval()
    # Downloading the checkpoint is the expensive part of a first call, so
    # persist it before doing anything that might fail.
    rt.save_cache()

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id=chain_id, sequence=chain.sequence)
            for chain_id, chain in complex_chains.items()
        ]
    )

    # Autocast rather than casting the model. The input builder emits float32
    # tensors, so a bfloat16 model meets them in the first linear layer and
    # dies on the dtype mismatch; autocast converts at each operation instead
    # and leaves the weights alone.
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if half_precision
        else contextlib.nullcontext()
    )
    with torch.no_grad(), precision:
        result = ESMFold2InputBuilder().fold(
            model, spi,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=samples,
            seed=seed,
        )

    structure = rt.work / f"{name}.cif"
    structure.write_text(result.complex.to_mmcif())

    def array(value: Any):
        """A tensor as numpy, whatever device and dtype it arrived on.

        The cast to float32 is not cosmetic: numpy has no bfloat16, so under
        autocast a direct conversion raises `unsupported ScalarType BFloat16`.
        """
        import numpy as np

        if hasattr(value, "detach"):
            return value.detach().to("cpu", torch.float32).numpy()
        return np.asarray(value, dtype="float32")

    def scalar(value: Any) -> float | None:
        """Confidence values come back as tensors; unwrap without assuming."""
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(array(value).mean())
            except Exception:
                return None

    # Written under the convention in `tools.common.predictions`, so ipSAE
    # finds them from the structure URI alone. Neither is documented as part of
    # the result, so both are best effort.
    import numpy as np

    companion_files: dict[str, Path | None] = {}
    for kind in ("pae", "plddt"):
        value = getattr(result, kind, None)
        if value is None or not getattr(value, "shape", None):
            continue
        archive = rt.work / predictions.companion_name(f"{name}.cif", kind)
        np.savez_compressed(archive, **{kind: array(value)})
        companion_files[kind] = archive

    chain_ids = list(complex_chains)
    # One structure per call: the fold returns a single complex whatever
    # `samples` was, so there is one model to report rather than a list.
    entry: dict[str, Any] = {"model": 0}
    entry.update(
        predictions.publish(
            structure=structure,
            companion_files=companion_files,
            prefix=rt.prefix(),
            stem=name,
        )
    )
    entry.update(
        folding.scores(
            {
                "plddt": scalar(getattr(result, "plddt", None)),
                "ptm": scalar(getattr(result, "ptm", None)),
                "iptm": scalar(getattr(result, "iptm", None)),
            },
            chain_ids,
        )
    )

    return folding.envelope(
        name=name,
        predictor="esmfold2",
        version=ESM_VERSION,
        complex_chains=complex_chains,
        models=[entry],
        seed=seed,
        checkpoint=CHECKPOINT,
    )
