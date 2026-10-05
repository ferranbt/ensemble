"""Read the backbones RFdiffusion writes.

For an output prefix `out/design`, each design produces:

    out/design_0.pdb    the backbone, with every designed residue as glycine
    out/design_0.trb    a pickle of run metadata and residue mappings
    out/traj/...        the denoising trajectory, which we ignore

The PDB carries no real sequence. Designed positions are glycine placeholders,
which is exactly what ProteinMPNN expects to be handed next.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tools.rfdiffusion.inputs import read_chains

_DESIGN_INDEX = re.compile(r"_(\d+)\.pdb$")


def design_index(path: Path) -> int:
    """The design number encoded in an output filename."""
    match = _DESIGN_INDEX.search(path.name)
    if not match:
        raise ValueError(f"Not an RFdiffusion design file: {path.name}")
    return int(match.group(1))


def read_designs(out_dir: Path, prefix: str) -> list[dict[str, Any]]:
    """Collect the backbones for one run, in design order.

    Args:
        out_dir: Directory holding the outputs.
        prefix: The output prefix's basename, so siblings are not picked up.

    Returns:
        One entry per design with its index, local path, and the chains it
        contains with their lengths. Which chain is the new binder is reported
        rather than assumed, since RFdiffusion decides the lettering itself.
    """
    designs = sorted(out_dir.glob(f"{prefix}_*.pdb"), key=design_index)
    if not designs:
        found = sorted(p.name for p in out_dir.iterdir()) if out_dir.is_dir() else []
        raise FileNotFoundError(
            f"RFdiffusion wrote no designs matching {prefix}_*.pdb in "
            f"{out_dir}. Found: {found or 'nothing'}"
        )

    results: list[dict[str, Any]] = []
    for path in designs:
        chains = read_chains(path)
        results.append({
            "design": design_index(path),
            "structure": path,
            "chain_lengths": {c: len(r) for c, r in chains.items()},
            "residues": sum(len(r) for r in chains.values()),
            "trajectory": (path.with_suffix(".trb") if path.with_suffix(".trb").exists()
                           else None),
        })
    return results


