"""Structure comparison as a Modal action.

Numpy over a few hundred coordinates, so this needs no GPU and takes about a
second. It is a Modal function only so a workflow can reach it by name like
every other tool.

    modal deploy tools/rmsd/app.py

    modal run tools/rmsd/app.py::compare_structures \\
        --reference-uri s3://bucket/runs/<id>/backbone.pdb \\
        --subject-uri s3://bucket/runs/<id>/model_0.pdb \\
        --align A --measure B
"""

from __future__ import annotations

from typing import Any

from tools.common.images import base_image, with_local_sources
from tools.common.tool import app_for, tool
from tools.rmsd import compare as compare_module

TIMEOUT = 15 * 60

app = app_for("rmsd")

image = with_local_sources(base_image())

@tool("rmsd/compare_structures", image=image, timeout=TIMEOUT)
def compare_structures(
    rt,
    reference_uri: str,
    subject_uri: str,
    align: str = "",
    measure: str = "",
    name: str = "comparison",
) -> dict[str, Any]:
    """Measure how far a predicted structure sits from the design it came from.

    This is the other half of the published binder filter. Interface
    confidence says the model is sure about a contact; it does not say the
    contact is the one that was designed. Bennett et al. filter on
    `pae_interaction < 10` **and** `complex_rmsd < 5A`, and the second term is
    what rejects a binder that folds confidently onto the wrong face of the
    target, or into a different fold than the backbone it was designed for.
    Confidence alone cannot see either failure, because the model is not
    wrong about what it predicted, only about what was asked for.

    Three numbers come out of one comparison, because "how far apart are these
    structures" is three different questions:

    `rmsd` superposes on the target chains and measures the binder. This is
    `complex_rmsd`, the filter's second term, and the only one that catches a
    binder that is folded correctly but docked somewhere else.

    `align_rmsd` is the fit quality on the chains that were superposed. A large
    value means the target itself moved, so `rmsd` is being measured against a
    shifted frame and should not be trusted.

    `measure_rmsd` superposes the measured chains on themselves. This is
    `binder_aligned_rmsd`: the same fold or not, ignoring placement. A design
    can score well here and badly on `rmsd`, and separating those two failures
    is the point of reporting both.

    Residues are paired by position in the chain rather than by sequence,
    because a design loop compares a poly-glycine backbone against a predicted
    designed sequence, where a sequence alignment would be meaningless. Chains
    of unequal length are therefore rejected rather than aligned.

    Args:
        reference_uri: `s3://bucket/key` of the structure that defines the
            frame: the RFdiffusion backbone, or the target's experimental
            structure. `.pdb` or `.cif`, read natively either way.
        subject_uri: `s3://bucket/key` of the structure being judged, normally
            a Boltz or Protenix prediction of a designed sequence.
        align: Chains to superpose on, space separated, e.g. `"A"`. For a
            binder filter this is the target, so the binder's displacement is
            measured rather than fitted away. A chain may be given as
            `reference=subject` when the two files name it differently, which
            happens because mmCIF-to-PDB conversion renames chains that do not
            fit one character. Empty superposes on `measure` instead, reducing
            the call to a plain fold comparison.
        measure: Chains to measure, space separated, e.g. `"B"`. For a binder
            filter this is the binder. Empty measures the aligned chains.
        name: Identifier for the run.

    Returns:
        `rmsd`, `align_rmsd` and `measure_rmsd` in Angstroms, lower being
        closer, along with the chains and atom counts each was computed over.
        Under 5A on `rmsd` is the published pass, from Bennett et al., but
        the threshold is deliberately not applied here: whether a design
        passes is the caller's decision, and a workflow's `where` clause is
        the place for it.

    Raises:
        ValueError: if no chain is named, or if a named chain has a different
            number of residues in the two structures.
        KeyError: if a named chain is absent, listing the chains that are
            present.
    """
    result = compare_module.compare(
        reference=rt.fetch(reference_uri, "reference"),
        subject=rt.fetch(subject_uri, "subject"),
        align=align,
        measure=measure,
    )
    return {
        "name": name,
        "reference_uri": reference_uri,
        "subject_uri": subject_uri,
        **result,
    }


