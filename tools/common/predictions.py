"""The files that accompany a predicted structure, and how they are named.

A structure prediction is not one artifact. Boltz writes four per model: the
structure, the aligned error matrix, the per-residue confidence array, and a
confidence summary. A tool that scores an interface needs most of them.

Passing them as separate URIs means four things that must stay in step, and a
consumer that receives three correct ones and one stale one has no way to
notice. So they travel as **one reference plus a convention**: a caller hands
over the structure, and the rest are derived from its name.

    s3://bucket/runs/<id>/model_0.pdb
    s3://bucket/runs/<id>/pae_model_0.npz
    s3://bucket/runs/<id>/plddt_model_0.npz
    s3://bucket/runs/<id>/confidence_model_0.json

The convention is not ours: `ipsae.py` locates the confidence array by
substituting "plddt" for "pae" in the path it is given, so a producer that
names things differently silently loses that file, and the score that depends
on it collapses to a constant rather than failing. Defining the naming in one
place is what keeps the producer and the consumer honest.

**The contract covers contents, not just names**, because predictors are meant
to be interchangeable within a workflow stage. Swapping Boltz for Protenix must
not require editing the stages downstream, so every producer normalises to one
shape and no consumer branches on which tool made the file:

    pae_<stem>.npz          key `pae`: per-token matrix, Angstroms
    plddt_<stem>.npz        key `plddt`: per-token array, 0-1 or 0-100
    confidence_<stem>.json  `pair_chains_iptm` nested by chain index as
                            strings, plus `iptm`, `ptm`, `complex_plddt` and
                            `confidence_score` as scalars

That shape is Boltz's, adopted as the common one because it is what the
scoring script reads natively. Producers whose own output differs convert on
the way out: Protenix emits a per-atom pLDDT and a chain-pair ipTM matrix, and
`tools.protenix.companions` folds both into the shape above. Doing it in the
producer keeps the conversion next to the tool that knows its own format, and
leaves exactly one reader for everything downstream.
"""

from __future__ import annotations

from pathlib import Path

from tools.common import storage

# kind -> (filename prefix, extension)
COMPANIONS = {
    "pae": ("pae_", ".npz"),
    "plddt": ("plddt_", ".npz"),
    "confidence": ("confidence_", ".json"),
}


def companion_name(structure: str | Path, kind: str) -> str:
    """The filename of one companion of a predicted structure.

    >>> companion_name("model_0.pdb", "pae")
    'pae_model_0.npz'
    """
    if kind not in COMPANIONS:
        raise ValueError(
            f"Unknown companion {kind!r}. Known: {', '.join(COMPANIONS)}"
        )
    prefix, extension = COMPANIONS[kind]
    return f"{prefix}{Path(str(structure)).stem}{extension}"


def companions(structure_uri: str) -> dict[str, str]:
    """The URIs of every companion of a stored structure, by convention.

    Returns every kind whether or not the object exists, so a caller can
    report which are missing rather than guessing at names.
    """
    parsed = storage.parse_uri(structure_uri)
    directory = parsed.key.rsplit("/", 1)[0] if "/" in parsed.key else ""
    name = Path(parsed.key).name

    def uri(kind: str) -> str:
        filename = companion_name(name, kind)
        key = f"{directory}/{filename}" if directory else filename
        return f"s3://{parsed.bucket}/{key}"

    return {kind: uri(kind) for kind in COMPANIONS}


def publish(
    structure: Path,
    companion_files: dict[str, Path | None],
    prefix,
    stem: str,
) -> dict[str, str | None]:
    """Store a structure and its companions, named by the convention.

    Args:
        structure: The predicted structure on local disk.
        companion_files: Companion kind to local path. A kind the predictor did
            not produce may be absent or None.
        prefix: The run's `S3Uri` prefix.
        stem: Basename to store the structure under, without an extension.

    Returns:
        `structure_uri` plus `<kind>_uri` for every kind, None where the
        predictor produced nothing. Every kind is always present as a key, so a
        consumer can tell "not produced" from "not looked for".
    """
    structure_name = f"{stem}{structure.suffix}"
    stored: dict[str, str | None] = {
        "structure_uri": storage.upload(structure, prefix.join(structure_name))
    }
    for kind in COMPANIONS:
        local = companion_files.get(kind)
        stored[f"{kind}_uri"] = (
            storage.upload(local, prefix.join(companion_name(structure_name, kind)))
            if local
            else None
        )
    return stored


def fetch(structure_uri: str, work_dir: Path, stem: str = "model") -> dict[str, Path]:
    """Download a structure and whichever companions exist beside it.

    Files land under names that follow the same convention locally, because
    the scoring script derives one path from another and would not find them
    otherwise.

    Returns:
        A mapping of `"structure"` plus each companion kind that was present,
        to its local path. A missing companion is simply absent, since not
        every predictor writes all of them.
    """
    suffix = Path(storage.parse_uri(structure_uri).key).suffix or ".pdb"
    structure = storage.download(structure_uri, work_dir / f"{stem}{suffix}")
    found: dict[str, Path] = {"structure": structure}

    for kind, uri in companions(structure_uri).items():
        if not storage.exists(uri):
            continue
        found[kind] = storage.download(
            uri, work_dir / companion_name(structure, kind)
        )
    return found
