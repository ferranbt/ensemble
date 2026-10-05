"""Build the arguments RFdiffusion's inference script takes.

RFdiffusion is configured through Hydra overrides, and the one that matters is
`contigmap.contigs`, a compact string describing what to keep from the input
structure and what to generate. For binder design it reads:

    [B1-100/0 70-100]

meaning "keep chain B residues 1 to 100 of the input, start a new chain, then
generate between 70 and 100 new residues". The `/0 ` is the chain break, and
the space after the zero is significant.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from tools.common import structure

REPO_DIR = "/opt/RFdiffusion"
RUNNER = f"{REPO_DIR}/scripts/run_inference.py"
CACHE_DIR = "/cache/rfdiffusion"
MODELS_DIR = f"{CACHE_DIR}/models"

# Every checkpoint the project publishes, with the path component its URL uses.
# Each is about 461 MB, so they are fetched individually on demand rather than
# all at once.
CHECKPOINTS = {
    "Base": "6f5902ac237024bdd0c176cb93063dc4",
    "Complex_base": "e29311f6f1bf1af907f9ef9f44b8328b",
    "Complex_Fold_base": "60f09a193fb5e5ccdc4980417708dbab",
    "InpaintSeq": "74f51cfb8b440f50d70878e05361d8f0",
    "InpaintSeq_Fold": "76d00716416567174cdb7ca96e208296",
    "ActiveSite": "5532d2e1f3a4738decd58b19d633b3c3",
    "Base_epoch8": "12fc204edeae5b57713c5ad7dcb97d39",
    "Complex_beta": "f572d396fae9206628714fb2ce00f72e",
}

WEIGHTS_URL = "http://files.ipd.uw.edu/pub/RFdiffusion"

RANGE = re.compile(r"^(\d+)-(\d+)$")
HOTSPOT = re.compile(r"^([A-Za-z]?)(\d+)$")


def checkpoint_url(name: str) -> str:
    """Where to fetch one checkpoint from."""
    if name not in CHECKPOINTS:
        raise ValueError(
            f"Unknown checkpoint {name!r}. Choose from {', '.join(CHECKPOINTS)}"
        )
    return f"{WEIGHTS_URL}/{CHECKPOINTS[name]}/{name}_ckpt.pt"


def read_chains(pdb_path: Path) -> "OrderedDict[str, list[int]]":
    """Chain ids to their residue numbers, in the order the file lists them.

    Thin wrapper over the shared reader, kept because callers here and in
    `outputs` reach it by this name. The shared one additionally handles
    alternate locations and insertion codes, which this module used to ignore.
    """
    return structure.chains(structure.read_pdb(pdb_path))


def chain_segments(
    pdb_path: Path, chain: str, residue_range: str = ""
) -> list[tuple[int, int]]:
    """Contiguous runs of residues actually present in one chain.

    Crystal structures have gaps where residues were not resolved, so a chain
    numbered 43 to 185 may hold fewer than 143 residues. RFdiffusion asserts
    that every residue named in a contig exists, and fails with
    `('A', 44) is not in pdb file!` otherwise, so ranges must be built from the
    residues that are really there rather than from the first and last.

    Args:
        residue_range: Optional `start-end` to clip to, e.g. `"50-120"`.
    """
    chains = read_chains(pdb_path)
    if chain not in chains:
        raise ValueError(
            f"Chain {chain!r} is not in the structure. It has: "
            f"{', '.join(chains)}"
        )

    residues = sorted(chains[chain])
    if residue_range.strip():
        match = RANGE.match(residue_range.strip())
        if not match:
            raise ValueError(
                f"target_residues {residue_range!r} must look like '50-120'"
            )
        low, high = (int(n) for n in match.groups())
        residues = [r for r in residues if low <= r <= high]
        if not residues:
            raise ValueError(
                f"Chain {chain} has no residues in range {residue_range}"
            )

    segments: list[tuple[int, int]] = []
    start = previous = residues[0]
    for residue in residues[1:]:
        if residue != previous + 1:
            segments.append((start, previous))
            start = residue
        previous = residue
    segments.append((start, previous))
    return segments


def chain_spec(pdb_path: Path, chain: str, residue_range: str = "") -> str:
    """The kept-target half of a contig, skipping unresolved gaps.

    Returns something like `A43-43/A45-185`, one segment per contiguous run.
    """
    segments = chain_segments(pdb_path, chain, residue_range)
    return "/".join(f"{chain}{lo}-{hi}" for lo, hi in segments)


def normalise_length(binder_length: str) -> str:
    """Accept `80` or `70-100`, always return a range RFdiffusion accepts."""
    text = str(binder_length).strip()
    if text.isdigit():
        # Checked here too, not only on the range form: a bare "0" would
        # otherwise become the contig "0-0" and ask for a chain of nothing.
        if int(text) < 1:
            raise ValueError(f"binder_length {text!r} must be at least 1")
        return f"{text}-{text}"
    if RANGE.match(text):
        low, high = (int(n) for n in text.split("-"))
        if low > high:
            raise ValueError(f"binder_length {text!r} has its bounds reversed")
        if low < 1:
            raise ValueError(f"binder_length {text!r} must be at least 1")
        return text
    raise ValueError(
        f"binder_length {binder_length!r} must be a number like '80' or a "
        f"range like '70-100'"
    )


def binder_contigs(target_spec: str, binder_length: str) -> str:
    """The contig string for designing one new chain against a fixed target.

    Args:
        target_spec: The kept part of the input, e.g. `B1-100`. Several
            segments may be joined with `/`.
        binder_length: How long the generated chain should be.

    >>> binder_contigs("B1-100", "70-100")
    '[B1-100/0 70-100]'
    """
    if not target_spec.strip():
        raise ValueError("target_spec must name the chain and residues to keep")
    return f"[{target_spec}/0 {normalise_length(binder_length)}]"


# RFdiffusion parses an input structure even for unconditional generation,
# where the contig names no chain and nothing from that structure is used. Its
# config defaults to an example shipped in the repository, which is not on disk
# when the package is pip-installed, so the run dies on a missing file. The
# clone is there, so this points at the same example deliberately.
EXAMPLE_PDB = f"{REPO_DIR}/examples/input_pdbs/1qys.pdb"


def monomer_contigs(length: str) -> str:
    """The contig string for generating one chain out of nothing.

    No input structure and nothing held fixed, so the contig is only a length.
    This is what a scaffold library is built from: a fold has to be shown to
    exist before anything can be grafted into it.

    >>> monomer_contigs("100")
    '[100-100]'
    >>> monomer_contigs("80-120")
    '[80-120]'
    """
    return f"[{normalise_length(length)}]"


def normalise_hotspots(hotspots: list[str] | None, chain: str) -> list[str]:
    """Prefix bare residue numbers with the target chain, and validate.

    Hotspots tell RFdiffusion where on the target to aim. They are given as
    `A30` or, for convenience, as `30` when they are on the target chain.
    """
    result: list[str] = []
    for raw in hotspots or []:
        text = str(raw).strip()
        match = HOTSPOT.match(text)
        if not match:
            raise ValueError(
                f"Hotspot {raw!r} must be a residue number, optionally with a "
                f"chain, like 'A30' or '30'"
            )
        prefix, residue = match.groups()
        result.append(f"{(prefix or chain).upper()}{residue}")
    return result


def inference_args(
    input_pdb: Path | None,
    output_prefix: Path,
    contigs: str,
    num_designs: int,
    checkpoint_path: str,
    hotspots: list[str],
    noise_scale: float,
    diffuser_steps: int,
    deterministic: bool = False,
) -> list[str]:
    """Assemble the Hydra overrides for one inference run.

    Hydra parses these as `key=value`, and values containing brackets or commas
    must not be shell-split, which is why they are passed as single arguments.

    `input_pdb` is None for unconditional generation, where there is nothing to
    build around and the override is left out entirely.
    """
    if num_designs < 1:
        raise ValueError("num_designs must be at least 1")
    args = [
        *([f"inference.input_pdb={input_pdb}"] if input_pdb else []),
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
        f"inference.ckpt_override_path={checkpoint_path}",
        f"contigmap.contigs={contigs}",
        f"denoiser.noise_scale_ca={noise_scale}",
        f"denoiser.noise_scale_frame={noise_scale}",
        f"diffuser.T={diffuser_steps}",
        f"inference.deterministic={deterministic}",
    ]
    if hotspots:
        args.append(f"ppi.hotspot_res=[{','.join(hotspots)}]")
    return args
