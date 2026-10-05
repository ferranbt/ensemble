"""BioEmu (https://github.com/microsoft/bioemu) as a Modal GPU action.

One action, `sample_ensemble`: given a sequence, sample the structures the
protein actually populates at equilibrium rather than the single one it is most
likely to be in.

That is a different question from the one every other predictor here answers,
and it is the one they cannot. Boltz, Protenix and ESMFold2 return a structure
and a confidence; none of them can say whether a fold is rigid or marginal,
whether a loop swings, or whether a designed protein sits in one state or
flickers between two. BioEmu's authors report relative free energies within
about 1 kcal/mol of millisecond simulation and of experiment, at a fraction of
the cost.

It samples **one chain**. There is no complex and no interface here, so this
answers questions about stability and flexibility, not about binding.

    modal deploy tools/bioemu/app.py

    modal run tools/bioemu/app.py::sample_ensemble --sequence GYDPETGTWG \\
        --num-samples 100
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

from tools.bioemu import outputs
from tools.common import a3m, convert
from tools.common import msa as msa_setting
from tools.common import mutations as mut
from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.tool import app_for, tool

# The latest checkpoint, trained on an extended set of MD simulations and on
# experimental folding free energies. `bioemu-v1.1` is the one the Science
# paper used, and is what to compare against when a published number is the
# reference rather than the best available answer.
DEFAULT_MODEL = "bioemu-v1.2"

# Their own timings on this card: 1000 samples of a 100-residue protein in
# about 4 minutes, 300 residues in about 40. Memory scales with the square of
# the length, so the card matters more than the sample count.
GPU = "A100-80GB"
TIMEOUT = 2 * 60 * 60

# AlphaFold2 weights for the embeddings, about 3.5GB, plus the BioEmu
# checkpoint and the SO3 precomputations, all fetched on first use.
CACHE = "bioemu-cache"
CACHE_DIR = "/cache/bioemu"

app = app_for("bioemu")

image = with_local_sources(
    modal.Image.debian_slim(python_version="3.11")
    # zlib for mdtraj, which builds its trajectory readers from source.
    .apt_install("git", "zlib1g-dev")
    .pip_install(*STORAGE_PACKAGES)
    # bioemu samples; bioemu-benchmarks carries their fraction-of-native-
    # contacts measure and the fitted free-energy estimator that reads it, and
    # brings mdtraj at whatever version it expects. Not pinned here: pinning it
    # separately only creates a version to disagree about.
    .pip_install("bioemu[cuda]", "bioemu-benchmarks")
)


@tool(
    "bioemu/sample_ensemble",
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    cache=CACHE,
    cache_path=CACHE_DIR,
)
def sample_ensemble(
    rt,
    sequence: str,
    num_samples: int = 100,
    msa: str = "",
    reference_uri: str = "",
    reference_chain: str = "A",
    model_name: str = DEFAULT_MODEL,
    seed: int = 0,
    batch_size_100: int = 10,
    name: str = "ensemble",
) -> dict[str, Any]:
    """Sample the equilibrium ensemble of one protein chain.

    Args:
        sequence: The chain's one-letter sequence.
        num_samples: Structures to sample. The distribution is the answer here,
            so this is the resolution of that answer: a few hundred is enough
            to see whether a fold is rigid, and thousands to put a number on a
            free energy.
        msa: Where evolutionary context comes from. An `s3://` alignment is
            read directly, which is worth doing: BioEmu otherwise queries the
            public ColabFold server itself, and `msa/search_msa` caches by
            sequence. Empty lets BioEmu search.

            An alignment searched for a near-identical sequence, as when
            comparing point variants, is retargeted onto this one first, since
            its first row is the query it was built for.
        reference_uri: Optional structure to measure the ensemble against, in
            S3. Supplying it is what turns flexibility into a stability
            statement: how much of the reference's contact map the ensemble
            keeps, and the free energy that follows from it.
        reference_chain: Which chain of that structure the ensemble
            corresponds to. Only that chain is used: a crystal structure of an
            oligomer holds several copies, and BioEmu samples one.
        model_name: `bioemu-v1.2`, the default, is the latest and adds training
            on experimental folding free energies. `bioemu-v1.1` is the
            checkpoint the Science paper used, so it is what to pass when
            reproducing a published number.
        seed: Base random seed, so an ensemble can be reproduced.
        batch_size_100: Batch size for a 100-residue chain, scaled down
            automatically for longer ones. Reduce on out-of-memory.
        name: Identifier for the run.

    Returns:
        `topology_uri` and `trajectory_uri` for the ensemble itself, and
        `observables`: per-residue fluctuation, compactness and its spread,
        and, when a reference was given, the fraction of native contacts kept
        and the folding free energy from BioEmu's own estimator. See
        `tools.bioemu.outputs` for what each means.
    """
    from bioemu.sample import main as sample

    parent = mut.clean_sequence(sequence)

    # BioEmu caches AlphaFold2 weights and its own checkpoint under the home
    # directory. Pointed at the volume at run time rather than in the image:
    # setting it during the build makes the mount path non-empty and Modal
    # then refuses to mount over it.
    os.environ["HOME"] = CACHE_DIR
    os.environ.setdefault("COLABFOLD_DIR", f"{CACHE_DIR}/colabfold")
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # BioEmu takes either a sequence or a path to an a3m, and prefers the a3m:
    # it then does no searching of its own.
    query: str | Path = parent
    if msa_setting.kind(msa) == msa_setting.ARTIFACT:
        alignment = rt.storage.download(msa, rt.work / "msa" / "query.a3m")
        if a3m.retarget(alignment, parent):
            print("alignment retargeted onto this sequence", flush=True)
        query = alignment
    elif msa:
        raise ValueError(
            f"BioEmu searches with its own bundled ColabFold, so a server "
            f"cannot be named ({msa!r}). Leave `msa` empty to let it search, "
            f"or pass an s3:// alignment from msa/search_msa, which is cached."
        )

    out_dir = rt.work / "out"
    sample(
        sequence=query,
        num_samples=num_samples,
        output_dir=out_dir,
        model_name=model_name,
        batch_size_100=batch_size_100,
        base_seed=seed or None,
    )
    rt.save_cache()

    topology = out_dir / "topology.pdb"
    trajectory = out_dir / "samples.xtc"
    for produced in (topology, trajectory):
        if not produced.is_file():
            listing = sorted(p.name for p in out_dir.glob("*")) if out_dir.is_dir() else []
            raise FileNotFoundError(
                f"BioEmu wrote no {produced.name} in {out_dir}. It produced: "
                f"{listing or 'nothing'}."
            )

    ensemble = outputs.load(topology, trajectory)

    reference = None
    if reference_uri:
        local = rt.fetch(reference_uri, "reference")
        # One chain only. Their contact-map measure aligns sequences but is
        # indexed against a single chain, so handing it a trimer walks off the
        # end of the ensemble.
        single = convert.extract_chain(
            local, reference_chain, rt.work / f"reference_{reference_chain}.pdb"
        )
        reference = outputs.load_structure(single)

    return {
        "name": name,
        "model_name": model_name,
        "sequence": parent,
        "length": len(parent),
        "requested_samples": num_samples,
        "msa": msa_setting.normalise(msa),
        "seed": seed,
        "topology_uri": rt.storage.upload(topology, rt.prefix("topology.pdb")),
        "trajectory_uri": rt.storage.upload(trajectory, rt.prefix("samples.xtc")),
        "reference_uri": reference_uri or None,
        "observables": outputs.describe(ensemble, reference),
    }


