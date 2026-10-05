"""Normalise a structure into the format a tool can actually read.

The pipeline is split by format. BoltzGen, Protenix and ESMFold2 emit mmCIF;
ProteinMPNN, ThermoMPNN and RFdiffusion read PDB only. So a design from a newer
generator cannot reach an older designer without a conversion step, which cuts
the loop exactly where a design would come back for redesign.

`ensure_format` is meant to be called by the consuming tool on whatever
arrived, so no caller has to know which tool wants which format. When the
format already matches it returns the original path untouched, which is the
common case.

**Why this returns a chain mapping.** mmCIF to PDB is lossy. PDB is a
fixed-column format: chain names get one character and residue numbers four. A
structure with chains named `polymer_1` or `AAA` therefore gets renamed on the
way down. Every tool here takes a chain argument, so a silent rename would mean
designing a different chain than the caller asked for, with no error. Pass the
caller's chain through `resolve_chain` and that cannot happen.

gemmi is imported inside the functions so this module stays importable without
it, matching how the output parsers treat numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FORMATS = ("pdb", "cif")

_SUFFIXES = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
}


@dataclass
class Converted:
    """The structure a tool should read, and what changed to get there."""

    path: Path
    format: str
    converted: bool = False
    source_format: str | None = None
    renamed_chains: dict[str, str] = field(default_factory=dict)

    @property
    def chains(self) -> list[str]:
        """Chain names as they appear in `path`."""
        return list(read_chain_names(self.path))


def detect_format(path: str | Path) -> str:
    """The structure format implied by a file's extension.

    Raises:
        ValueError: for an extension that names no known format. Guessing here
            would hand the next tool a file it silently misreads.
    """
    suffix = Path(str(path)).suffix.lower()
    if suffix not in _SUFFIXES:
        raise ValueError(
            f"Cannot tell the structure format of {path!r}. Expected one of "
            f"{', '.join(sorted(_SUFFIXES))}."
        )
    return _SUFFIXES[suffix]


def read_chain_names(path: Path) -> list[str]:
    """Chain names in a structure, in file order."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    return [chain.name for chain in structure[0]] if len(structure) else []


def convert_file(source: Path, target: Path) -> dict[str, str]:
    """Convert one structure on disk, returning any chain renames.

    Returns:
        A mapping from the source's chain name to the target's, holding only
        the chains whose name had to change. Empty when nothing was renamed,
        which is the usual outcome for names that already fit PDB.
    """
    import gemmi

    target_format = detect_format(target)
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    before = [chain.name for chain in structure[0]] if len(structure) else []

    target.parent.mkdir(parents=True, exist_ok=True)
    if target_format == "pdb":
        # PDB allows one character per chain, so anything longer is rewritten.
        structure.shorten_chain_names()
        after = [chain.name for chain in structure[0]] if len(structure) else []
        structure.write_pdb(str(target))
    else:
        after = before
        # mmCIF is written through a document rather than directly, and chain
        # names have no width limit there, so nothing is renamed going up.
        structure.make_mmcif_document().write_file(str(target))

    return {b: a for b, a in zip(before, after) if b != a}


def ensure_format(path: Path, required: str, work_dir: Path | None = None) -> Converted:
    """Return the structure in `required` format, converting only if needed.

    Args:
        path: The structure as it arrived.
        required: `pdb` or `cif`.
        work_dir: Where to write a conversion. Defaults to beside the input.

    Returns:
        A `Converted` whose `path` is safe to hand to the tool. When no
        conversion was needed, that is the original path and `converted` is
        False.
    """
    if required not in FORMATS:
        raise ValueError(
            f"`required` must be one of {', '.join(FORMATS)}, got {required!r}"
        )

    current = detect_format(path)
    if current == required:
        return Converted(path=path, format=required, source_format=current)

    directory = work_dir or path.parent
    target = directory / f"{path.stem}.{required}"
    if target == path:
        target = directory / f"{path.stem}_converted.{required}"

    print(f"convert {current} -> {required}: {path.name} -> {target.name}", flush=True)
    renamed = convert_file(path, target)
    if renamed:
        print(f"  chains renamed to fit {required}: {renamed}", flush=True)

    return Converted(
        path=target,
        format=required,
        converted=True,
        source_format=current,
        renamed_chains=renamed,
    )


def fetch_structure(
    structure_uri: str,
    required: str,
    work_dir: Path,
    stem: str = "input",
) -> Converted:
    """Download a stored structure and return it in the format a tool needs.

    The extension is taken from the object's key rather than assumed, because
    saving an mmCIF under a `.pdb` name makes it unreadable by the very parsers
    that name was meant to satisfy, and the failure looks like a corrupt file.

    Args:
        structure_uri: `s3://bucket/key` of the structure.
        required: `pdb` or `cif`, whatever the calling tool reads.
        work_dir: This call's scratch directory.
        stem: Basename for the downloaded file.
    """
    from tools.common import storage

    source_format = detect_format(storage.parse_uri(structure_uri).key)
    downloaded = storage.download(
        structure_uri, work_dir / f"{stem}.{source_format}"
    )
    return ensure_format(downloaded, required, work_dir)


def extract_chain(source: Path, chain: str, target: Path) -> Path:
    """Write one chain of a structure to its own file.

    Needed whenever a tool expects a single chain and is handed an assembly.
    A crystal structure of a trimer holds three copies, and a tool comparing
    it against one chain either fails on the atom count or, worse, aligns the
    wrong copy.

    Raises:
        KeyError: naming the chains that are present, when the requested one
            is not among them.
    """
    import gemmi

    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    if not len(structure):
        raise ValueError(f"{source} contains no model")

    names = [c.name for c in structure[0]]
    if chain not in names:
        raise KeyError(
            f"No chain {chain!r} in {source.name}. It has: "
            f"{', '.join(names) or 'none'}."
        )

    for name in names:
        if name != chain:
            structure[0].remove_chain(name)
    structure.setup_entities()

    target.parent.mkdir(parents=True, exist_ok=True)
    if detect_format(target) == "pdb":
        structure.write_pdb(str(target))
    else:
        structure.make_mmcif_document().write_file(str(target))
    return target


def resolve_chain(chain: str, converted: Converted) -> str:
    """Translate a caller's chain id through any rename that conversion forced.

    Returns the chain unchanged when nothing was renamed, which is almost
    always. Use this before passing a chain argument to the underlying tool.
    """
    return converted.renamed_chains.get(chain, chain)


def resolve_chains(chains: str, converted: Converted) -> str:
    """Translate a space-separated list of chain ids, as ProteinMPNN takes."""
    return " ".join(resolve_chain(c, converted) for c in chains.split())
