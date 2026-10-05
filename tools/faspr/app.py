"""FASPR (https://github.com/tommyhuangthu/FASPR) as a Modal action.

One action, `pack_sidechains`: given a backbone, place the side chains.

Everything upstream of here works in backbones and sequences. RFdiffusion
produces coordinates for the main chain only; ProteinMPNN assigns residue
identities without placing their atoms. A structure in that state cannot be
scored by anything physical, because the interactions a physics function
measures are largely between side chains, and it cannot be docked against
because there is nothing there to dock.

Fast: one C++ binary, a rotamer library, and no GPU.

It can also introduce mutations while packing. That is a cheaper way to look at
a point variant than refolding it: the backbone is assumed unchanged and only
the side chains move, which is the right assumption for a conservative
substitution and the wrong one for anything that reshapes the fold.

    modal deploy tools/faspr/app.py

    modal run tools/faspr/app.py::pack_sidechains \\
        --structure-uri s3://bucket/design.pdb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.common import convert, design
from tools.common.images import base_image, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool

REPO_URL = "https://github.com/tommyhuangthu/FASPR.git"
REPO_COMMIT = "0d55732fd6307f373018c6bddd842291c355c5f7"
REPO_DIR = "/opt/FASPR"

TIMEOUT = 15 * 60

app = app_for("faspr")

image = with_local_sources(
    base_image()
    .apt_install("build-essential", "g++")
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {REPO_COMMIT}",
        f"cd {REPO_DIR} && g++ -O3 --fast-math -o FASPR src/*.cpp",
        f"chmod +x {REPO_DIR}/FASPR",
    )
)


def renumber_atoms(path: Path) -> Path:
    """Rewrite ATOM serial numbers sequentially.

    FASPR's own numbering restarts in a way PyMOL misreads, so a packed
    structure renders as a fragment. Taken from hgbrian/biomodals, which hit
    the same thing.
    """
    serial = 1
    lines = []
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("ATOM"):
            lines.append(f"ATOM{serial:7d}{line[11:]}")
            serial += 1
        elif line.startswith("TER"):
            lines.append(f"TER{serial:8d}{line[11:]}")
            serial += 1
        else:
            lines.append(line)
    path.write_text("".join(lines))
    return path


@tool("faspr/pack_sidechains", image=image, timeout=TIMEOUT)
def pack_sidechains(
    rt,
    structure_uri: str,
    sequence: str = "",
    name: str = "packed",
) -> dict[str, Any]:
    """Place side chains on a backbone, optionally changing its sequence.

    Args:
        structure_uri: `s3://bucket/key` of the structure. mmCIF is converted
            to PDB first, which FASPR requires. Its main chain must be
            complete; a model missing backbone atoms is rejected by FASPR
            rather than guessed at.
        sequence: Optional one-letter sequence to pack instead of the
            structure's own, which is how a mutation is introduced. It must be
            exactly as long as the structure has residues, and it is checked
            against that before FASPR runs, because a length mismatch there
            produces a silently shifted structure.
        name: Identifier for the run.

    Returns:
        `structure_uri` of the packed structure, and the sequence it was
        packed with.

    Raises:
        ValueError: if `sequence` does not match the structure's residue count.
    """
    source = convert.fetch_structure(structure_uri, "pdb", rt.work, stem="input")
    structure = source.path

    command = [f"{REPO_DIR}/FASPR", "-i", str(structure)]
    packed = rt.work / f"{name}.pdb"
    command += ["-o", str(packed)]

    requested = "".join(str(sequence).split()).upper()
    if requested:
        residues = sum(design.chain_lengths(structure).values())
        if len(requested) != residues:
            raise ValueError(
                f"The sequence is {len(requested)} residues but the structure "
                f"has {residues}. FASPR maps them position by position, so a "
                f"mismatch packs the wrong residue at every position after it."
            )
        target = rt.work / "sequence.txt"
        target.write_text(requested)
        command += ["-s", str(target)]

    # Run from the repository: the rotamer library is found relative to the
    # working directory, and FASPR fails without explaining why elsewhere.
    run(command, cwd=REPO_DIR)
    renumber_atoms(packed)

    return {
        "name": name,
        "input_uri": structure_uri,
        "converted_from": source.source_format if source.converted else None,
        "sequence": requested or None,
        "chain_lengths": design.chain_lengths(packed),
        "structure_uri": rt.storage.upload(packed, rt.prefix(f"{name}.pdb")),
    }
