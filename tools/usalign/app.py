"""US-align (https://zhanggroup.org/US-align/) as a Modal action.

One action, `compare_folds`: align structures and report TM-score.

This answers a question `rmsd/compare_structures` cannot. That tool pairs
residues by position and refuses chains of unequal length, which is right for
a self-consistency check where position `i` corresponds to position `i` by
construction. It is useless for "are these the same fold", where the two
structures may differ in length, have insertions, or be numbered differently.

TM-score is length-normalised and runs 0 to 1, so unlike RMSD it is comparable
between proteins of different sizes: about 0.5 is the conventional line between
sharing a fold and not, and above 0.9 is near-identical. A 5A RMSD means
something quite different for a 50-residue protein than for a 500-residue one,
which is why RMSD alone cannot answer this.

The image build is taken from hgbrian/biomodals, which packages US-align for
Modal: the source is one C++ file, compiled static at build time.

    modal deploy tools/usalign/app.py

    modal run tools/usalign/app.py::compare_folds \\
        --reference-uri s3://bucket/a.pdb --subject-uris s3://bucket/b.pdb
"""

from __future__ import annotations

from typing import Any

from tools.common.images import base_image, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool
from tools.usalign import outputs

USALIGN_SOURCE = "https://zhanggroup.org/US-align/bin/module/USalign.cpp"
USALIGN = "/usr/local/bin/USalign"

# Structural alignment of two proteins is milliseconds of CPU.
TIMEOUT = 15 * 60

# `-ter 0` compares every chain of one structure against every chain of the
# other, which is the right default when the chain lettering is not known to
# match. `-mm 1 -ter 1` treats each structure as one oligomer instead.
DEFAULT_PARAMS = "-ter 0"

app = app_for("usalign")

image = with_local_sources(
    base_image()
    .apt_install("g++", "wget")
    .run_commands(
        f"cd /usr/local/bin && wget -q {USALIGN_SOURCE} "
        f"&& g++ -static -O3 -ffast-math -o USalign USalign.cpp "
        f"&& chmod +x USalign && rm USalign.cpp"
    )
)


@tool("usalign/compare_folds", image=image, timeout=TIMEOUT)
def compare_folds(
    rt,
    reference_uri: str,
    subject_uris: list[str] | str,
    params: str = DEFAULT_PARAMS,
    name: str = "alignment",
) -> dict[str, Any]:
    """Align one or more structures against a reference and report TM-score.

    Args:
        reference_uri: `s3://bucket/key` of the structure to compare against,
            `.pdb` or `.cif`.
        subject_uris: One or more structures to align onto it. Passing several
            in one call is cheaper than several calls and is how a candidate is
            searched against a set of known folds.
        params: US-align flags. The default `-ter 0` aligns all chains against
            all chains. `-mm 1 -ter 1` aligns oligomers as units. The output
            format is fixed, so it cannot be set here.
        name: Identifier for the run.

    Returns:
        `alignments`, one entry per subject sorted best first, each with
        `tm_score` and whether that clears the conventional same-fold line, the
        RMSD over aligned residues, how many aligned, and the sequence identity
        across them. `best` is the closest match.

        `tm_score` is the lower of the two normalisations US-align reports,
        which is the conservative reading: it stops a short fragment scoring
        well for matching part of a much larger protein.

    Raises:
        ValueError: if `params` tries to set the output format, which is fixed
            because the result is parsed by column name.
    """
    if "outfmt" in params:
        raise ValueError(
            "The output format is fixed at `-outfmt 2`, since the result is "
            "parsed by column name. Remove it from `params`."
        )

    subjects = [subject_uris] if isinstance(subject_uris, str) else list(subject_uris)
    if not subjects:
        raise ValueError("Pass at least one structure in `subject_uris`")

    reference = rt.fetch(reference_uri, "reference")

    alignments: list[dict[str, Any]] = []
    for index, uri in enumerate(subjects):
        subject = rt.fetch(uri, f"subject_{index}")
        # Subject first, reference second: US-align's TM1 is normalised by the
        # first structure and TM2 by the second, so this order makes TM2 the
        # reference's normalisation.
        text = run([
            USALIGN, str(subject), str(reference), "-outfmt", "2", *params.split(),
        ])
        rows = outputs.parse(text)
        if not rows:
            raise ValueError(
                f"US-align reported no alignment between {uri} and "
                f"{reference_uri}. Both must hold at least one chain it can read."
            )
        for row in rows:
            alignments.append({"subject_uri": uri, **outputs.summarise(row)})

    alignments.sort(
        key=lambda a: a["tm_score"] if a["tm_score"] is not None else -1.0,
        reverse=True,
    )
    return {
        "name": name,
        "reference_uri": reference_uri,
        "params": params,
        "same_fold_threshold": outputs.SAME_FOLD,
        "best": alignments[0] if alignments else None,
        "alignments": alignments,
    }
