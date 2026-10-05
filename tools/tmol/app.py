"""Rosetta's energy function via tmol (https://github.com/uw-ipd/tmol).

One action, `score_energy`: a physics-based energy for a structure.

**This is the only non-neural score in the repo, and that is the point.** ipSAE,
ipTM, pLDDT, ProteinMPNN's likelihood and ThermoMPNN's ddG are all deep
learning, and mostly trained on overlapping data. When Boltz liked a design at
ipSAE 0.718 and Protenix scored the same design 0.000, no amount of further
neural opinion resolved it. An energy function computes van der Waals packing,
hydrogen bonding, electrostatics and solvation from the coordinates. It can be
wrong, but it is wrong for unrelated reasons, which is what makes agreement
with it worth something.

tmol is the Institute for Protein Design's own PyTorch implementation of the
`beta_nov2016` energy function, so this needs no PyRosetta licence.

**Side chains must be present.** The terms are largely side-chain interactions,
so a bare backbone scores meaninglessly. Run `faspr/pack_sidechains` first;
the workflow does that rather than this tool calling out to it.

    modal deploy tools/tmol/app.py

    modal run tools/tmol/app.py::score_energy \\
        --structure-uri s3://bucket/packed.pdb
"""

from __future__ import annotations

from typing import Any

import modal

from tools.common import convert, design
from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.tool import app_for, tool

# A release wheel built against one exact Python, torch and CUDA. All three
# have to match what the image installs or the compiled extension loads and
# then fails: `Could not load this library: tmol/_C...so`, with nothing said
# about versions. The wheel name is the specification, so torch is pinned to
# the version in it rather than left to resolve.
TMOL_VERSION = "0.1.55"
TORCH_VERSION = "2.11"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TMOL_WHEEL = (
    f"https://github.com/uw-ipd/tmol/releases/download/v{TMOL_VERSION}/"
    f"tmol-{TMOL_VERSION}%2Bcu128torch{TORCH_VERSION}"
    f"-cp312-cp312-manylinux_2_28_x86_64.whl"
)

GPU = "A10G"
TIMEOUT = 30 * 60

app = app_for("tmol")

image = with_local_sources(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*STORAGE_PACKAGES)
    .pip_install(f"torch=={TORCH_VERSION}.*", index_url=CUDA_INDEX)
    .pip_install(TMOL_WHEEL)
)


@tool("tmol/score_energy", image=image, gpu=GPU, timeout=TIMEOUT)
def score_energy(
    rt,
    structure_uri: str,
    relax: bool = True,
    relax_steps: int = 10,
    name: str = "energy",
) -> dict[str, Any]:
    """Score a structure with the Rosetta beta_nov2016 energy function.

    Args:
        structure_uri: `s3://bucket/key` of the structure, with side chains
            present. mmCIF is converted to PDB first.
        relax: Minimise the coordinates before scoring, by L-BFGS in Cartesian
            space. Worth leaving on: a predicted structure has small local
            geometry errors that an energy function punishes severely, so an
            unrelaxed score says more about those than about the design. The
            score before relaxing is reported too, and a large gap between
            them is itself a signal that the geometry was strained.
        relax_steps: L-BFGS iterations.
        name: Identifier for the run.

    Returns:
        `total_energy` in Rosetta Energy Units, lower being better, with
        `energy_per_residue` alongside because the total grows with size and
        so cannot be compared between proteins of different lengths.
        `pre_relax_energy` is the score before minimisation, and `terms` breaks
        the total down by weighted component, which is where an unphysical
        design usually shows itself: a large `fa_rep` means atoms are
        overlapping.

        REU are not kcal/mol. They are useful for ranking, not as a measured
        free energy.
    """
    import time

    import tmol
    import torch

    source = convert.fetch_structure(structure_uri, "pdb", rt.work, stem="input")
    structure = source.path

    began = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_stack = tmol.pose_stack_from_pdb(str(structure), device=device)
    sfxn = tmol.beta2016_score_function(device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

    def measure(module) -> tuple[float, dict[str, float]]:
        """The total and its weighted breakdown, for the current coordinates."""
        total = module(pose_stack.coords).item()
        terms: dict[str, float] = {}
        weights = sfxn.weights_tensor()
        unweighted = module.unweighted_scores(pose_stack.coords)
        for index, score_type in enumerate(sfxn.all_score_types()):
            weight = weights[index].item()
            if weight:
                terms[score_type.name] = unweighted[index, 0].item() * weight
        return total, terms

    pre_relax = None
    if relax:
        pre_relax, _ = measure(scorer)
        torch.set_grad_enabled(True)
        pose_stack.coords.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [pose_stack.coords],
            lr=0.1,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )
        for _ in range(relax_steps):
            def closure():
                optimizer.zero_grad()
                energy = scorer(pose_stack.coords)
                energy.backward()
                return energy

            optimizer.step(closure)
        pose_stack.coords.requires_grad_(False)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

    total, terms = measure(scorer)
    residues = sum(design.chain_lengths(structure).values())

    return {
        "name": name,
        "structure_uri": structure_uri,
        "converted_from": source.source_format if source.converted else None,
        "score_function": "beta_nov2016",
        "relaxed": relax,
        "residues": residues,
        "total_energy": total,
        "energy_per_residue": total / residues if residues else None,
        "pre_relax_energy": pre_relax,
        "terms": {k: round(v, 4) for k, v in sorted(terms.items())},
        "elapsed_ms": int((time.perf_counter() - began) * 1000),
    }
