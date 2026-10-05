"""Solvent accessibility via freesasa (https://freesasa.github.io/), as a Modal action.

One action, `measure_exposure`: how much of a structure is exposed to solvent,
in total, per residue, and residue by residue.

This exists because of a design that passed every other filter. A four-helix
bundle scored TM 0.970 against its backbone, pLDDT 0.820 and the best physics
energy of its batch, and a rendering showed four helices lying in a plane with
no core. Nothing numeric objected: helices in a plane pack their contact faces
perfectly well, there is simply not much of them buried. Area per residue is
what separates that from a globular fold, and `buried_fraction` is what
notices there is no core.

It also answers a question we previously guessed at. Hotspots for a binder run
were picked by measuring contact distances to a known partner, which works only
when a partner is known; surface residues are properly identified by
accessibility.

**Accessibility depends on context.** A chain measured on its own reports what
it would expose unbound. The same chain measured inside a complex reports less,
and the difference is the buried interface. So `chain` decides which question
is being asked, and the answer changes with it.

    modal deploy tools/sasa/app.py

    modal run tools/sasa/app.py::measure_exposure \\
        --structure-uri s3://bucket/design.pdb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.common import convert
from tools.common.images import base_image, with_local_sources
from tools.common.tool import app_for, tool

# Relative accessibility below which a residue counts as buried. 25% is the
# usual convention in the literature rather than a number chosen here, and it
# stays a parameter because conventions vary between 20% and 25%.
BURIED_RSA = 0.25

TIMEOUT = 15 * 60

app = app_for("sasa")

image = with_local_sources(base_image().pip_install("freesasa==2.2.*"))


def annotate(structure: Path, rsa: dict[tuple[str, str], float], target: Path) -> Path:
    """Write relative accessibility into the B-factor column.

    Makes a picture of the core: open it and colour by B to see which residues
    are buried, which is far easier to judge than a list of numbers.
    """
    import gemmi

    parsed = gemmi.read_structure(str(structure))
    parsed.setup_entities()
    for chain in parsed[0]:
        for residue in chain:
            value = rsa.get((chain.name, str(residue.seqid.num)))
            for atom in residue:
                # Percent rather than fraction: colour ramps in every viewer
                # assume B-factors in the tens, not in the ones.
                atom.b_iso = 100.0 * value if value is not None else 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed.write_pdb(str(target))
    return target


@tool("sasa/measure_exposure", image=image, timeout=TIMEOUT)
def measure_exposure(
    rt,
    structure_uri: str,
    chain: str = "",
    buried_rsa: float = BURIED_RSA,
    name: str = "exposure",
) -> dict[str, Any]:
    """Measure how exposed a structure is.

    Args:
        structure_uri: `s3://bucket/key` of the structure. mmCIF is converted
            to PDB first.
        chain: Measure this chain alone, extracted from the file first. Leave
            empty to measure everything present. The choice matters: a chain on
            its own reports what it would expose unbound, and the same chain
            within its complex reports less.
        buried_rsa: Relative accessibility below which a residue counts as
            buried. The 25% default is the common convention, not a number
            invented here.
        name: Identifier for the run.

    Returns:
        `total_area` in square Angstroms and `area_per_residue`, which is the
        number to compare across designs of different lengths: a compact fold
        exposes less per residue than a flat one of the same size.
        `buried_fraction` is the share of residues below `buried_rsa`, so a
        fold with no core reports close to zero.

        `residues` carries each residue's relative accessibility, which is what
        picks surface positions for redesign, and `annotated_uri` is the same
        structure with those values in the B-factor column for viewing.
    """
    import freesasa

    source = convert.fetch_structure(structure_uri, "pdb", rt.work, stem="input")
    structure = source.path
    if chain:
        resolved = convert.resolve_chain(chain, source)
        structure = convert.extract_chain(
            structure, resolved, rt.work / f"chain_{resolved}.pdb"
        )

    result = freesasa.calc(freesasa.Structure(str(structure)))
    areas = result.residueAreas()

    residues: list[dict[str, Any]] = []
    rsa_by_key: dict[tuple[str, str], float] = {}
    for chain_id, per_residue in areas.items():
        for number, area in per_residue.items():
            # Relative accessibility needs a reference maximum for that residue
            # type, which freesasa carries for the standard twenty and not for
            # anything else.
            relative = area.relativeTotal if area.hasRelativeAreas else None
            residues.append({
                "chain": chain_id,
                "position": int(number) if str(number).lstrip("-").isdigit() else number,
                "residue": area.residueType,
                "area": round(area.total, 2),
                "rsa": round(relative, 4) if relative is not None else None,
            })
            if relative is not None:
                rsa_by_key[(chain_id, str(number))] = relative

    residues.sort(key=lambda r: (r["chain"], str(r["position"]).zfill(6)))
    with_rsa = [r for r in residues if r["rsa"] is not None]
    buried = [r for r in with_rsa if r["rsa"] < buried_rsa]
    total = result.totalArea()

    annotated = annotate(structure, rsa_by_key, rt.work / f"{name}_rsa.pdb")

    return {
        "name": name,
        "structure_uri": structure_uri,
        "chain": chain or None,
        "buried_rsa": buried_rsa,
        "residues_measured": len(residues),
        "total_area": round(total, 2),
        "area_per_residue": round(total / len(residues), 2) if residues else None,
        "buried_fraction": round(len(buried) / len(with_rsa), 4) if with_rsa else None,
        "buried_residues": len(buried),
        "residues": residues,
        "annotated_uri": rt.storage.upload(annotated, rt.prefix(annotated.name)),
    }
