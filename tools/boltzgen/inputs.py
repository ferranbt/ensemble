"""Build the YAML design specification BoltzGen takes.

A spec names the target to bind, which of its residues to aim at, and how long
the designed binder should be:

    entities:
      - file:
          path: target.pdb
          include:
            - chain: {id: A, res_index: "1..143"}
      - protein: {id: B, sequence: "80..140"}
    constraints:
      - binding_types:
          - chain: {id: A, binding: "8,12,15", not_binding: all}

**The indexing trap.** BoltzGen counts residues from 1 in canonical mmCIF
order, not by the numbering written in the file. A crystal structure whose
chain A starts at residue 43 and has gaps will therefore disagree with its own
author numbering by an offset that changes along the chain. Hotspots given in
author numbering would silently land on the wrong residues, and the run would
succeed while aiming at the wrong face of the protein.

So this module reads the structure and translates. Pass hotspots in the
numbering you can see in the file, and `author_to_label` converts them.
"""

from __future__ import annotations

import re
from pathlib import Path

from tools.common import structure

RANGE = re.compile(r"^(\d+)-(\d+)$")

PROTOCOLS = (
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
)


def author_to_label(pdb_path: Path, chain: str) -> dict[int, int]:
    """Map the residue numbers written in a file to BoltzGen's 1-based index.

    Residues are counted in file order within the chain, starting at 1, so gaps
    and non-standard starting numbers are absorbed.

    Raises:
        ValueError: if the chain is not in the structure.
    """
    residues = structure.read_pdb(pdb_path)
    numbers = [r.number for r in residues if r.chain == chain]
    if not numbers:
        present = sorted({r.chain for r in residues})
        raise ValueError(
            f"Chain {chain!r} is not in the structure. It has: "
            f"{', '.join(present)}"
        )
    return {number: index for index, number in enumerate(numbers, start=1)}


def translate_hotspots(
    pdb_path: Path, chain: str, hotspots: list[int | str]
) -> list[int]:
    """Convert hotspots from the file's own numbering to BoltzGen's.

    Raises:
        ValueError: naming any hotspot that is not present in the chain, since
            silently dropping one would weaken the design without saying so.
    """
    mapping = author_to_label(pdb_path, chain)
    translated: list[int] = []
    missing: list[int] = []
    for hotspot in hotspots:
        number = int(str(hotspot).strip().lstrip(chain))
        if number in mapping:
            translated.append(mapping[number])
        else:
            missing.append(number)
    if missing:
        known = sorted(mapping)
        raise ValueError(
            f"Chain {chain} has no residue(s) {missing}. Its numbering runs "
            f"{known[0]} to {known[-1]} with {len(known)} residues present."
        )
    return sorted(translated)


def normalise_length(binder_length: str) -> str:
    """Accept `80` or `80-140`, return BoltzGen's `low..high` form."""
    text = str(binder_length).strip()
    if text.isdigit():
        return f"{text}..{text}"
    match = RANGE.match(text)
    if not match:
        raise ValueError(
            f"binder_length {binder_length!r} must be a number like '80' or a "
            f"range like '80-140'"
        )
    low, high = (int(n) for n in match.groups())
    if low > high:
        raise ValueError(f"binder_length {text!r} has its bounds reversed")
    if low < 1:
        raise ValueError(f"binder_length {text!r} must be at least 1")
    return f"{low}..{high}"


def write_spec(
    path: Path,
    target_pdb: Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    binder_length: str = "80-140",
    hotspots: list[int | str] | None = None,
) -> Path:
    """Write the design specification, returning the path written.

    Args:
        path: Where to write the YAML. Relative paths inside it resolve against
            this file's directory, so the target is referenced by name and must
            sit alongside.
        target_pdb: The target structure, already in the same directory.
        target_chain: Which chain of it to bind.
        binder_chain: Chain id to give the designed binder.
        binder_length: How many residues to generate.
        hotspots: Target residues to aim at, in the numbering written in the
            file. Converted for you. Without hotspots the model chooses where
            to bind, which is rarely what you want.
    """
    import yaml

    entities: list[dict] = [
        {"file": {"path": target_pdb.name,
                  "include": [{"chain": {"id": target_chain}}]}},
        {"protein": {"id": binder_chain,
                     "sequence": normalise_length(binder_length)}},
    ]

    spec: dict = {"entities": entities}
    if hotspots:
        binding = translate_hotspots(target_pdb, target_chain, hotspots)
        spec["constraints"] = [{
            "binding_types": [{
                "chain": {
                    "id": target_chain,
                    "binding": ",".join(str(i) for i in binding),
                    "not_binding": "all",
                }
            }]
        }]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path
