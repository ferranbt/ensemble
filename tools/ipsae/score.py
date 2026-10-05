"""Interface scoring with ipsae.py (https://github.com/DunbrackLab/IPSAE).

`score_interface` is an ordinary Python function, not a Modal action. The work
is numpy on one matrix, so it needs no GPU and no container, and it can run
wherever the calling process runs, given a runtime to reach storage through:

    from tools.common.records import Runtime

    score.score_interface(Runtime.create("ipsae"), structure_uri)

It still speaks S3 URIs like every other tool, so results stay addressable and
it composes with the rest of the pipeline.

**Why this metric.** Boltz reports ipTM, which averages predicted aligned error
over every cross-chain residue pair. Disordered tails and domains that never
touch are included, so they drag the score down even when the binding site
itself is predicted confidently. ipSAE counts only pairs the model is confident
about and sizes its normalisation to those. A meta-analysis of several thousand
experimentally tested binders found it separates binders from non-binders
better than ipTM.

The same run also reports pDockQ, pDockQ2 and LIS, giving several orthogonal
views rather than one number to overfit to. All run 0 to 1, higher meaning a
more confidently predicted interface. None of them predicts affinity.

The upstream script is a command-line program, not a library: it reads
`sys.argv` and opens its output files at import time. So it is fetched once at
a pinned commit and run as a subprocess rather than imported.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from tools.common import convert, predictions
from tools.common.shell import run
from tools.ipsae import outputs

COMMIT = "6174cf9e71cb1bd660cc805856a18c4871a6dec3"
SCRIPT_URL = (
    f"https://raw.githubusercontent.com/DunbrackLab/IPSAE/{COMMIT}/ipsae.py"
)

CACHE_ENV = "FRIDAY_IPSAE_PATH"
DEFAULT_CACHE = Path.home() / ".cache" / "friday" / "ipsae"

# The reference tool's documented example cutoffs, in Angstroms.
PAE_CUTOFF = 10.0
DIST_CUTOFF = 15.0

STRUCTURE_SUFFIXES = (".pdb", ".cif")


def ensure_script(cache_dir: Path | None = None) -> Path:
    """The ipsae.py script, downloading it once if it is not already cached.

    Set `FRIDAY_IPSAE_PATH` to point at your own copy and skip the download
    entirely, which is what you want in an image that vendors it.
    """
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"{CACHE_ENV} points at {path}, which is not a file")
        return path

    directory = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    script = directory / f"ipsae_{COMMIT[:8]}.py"
    if not script.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"fetching ipsae.py at {COMMIT[:8]}", flush=True)
        with urlopen(SCRIPT_URL) as response:
            body = response.read()
        # Written via a temporary name so an interrupted download cannot leave
        # a truncated script that later runs look cached.
        temporary = script.with_suffix(".partial")
        temporary.write_bytes(body)
        temporary.replace(script)
    return script


def score_interface(
    rt,
    structure_uri: str,
    pae_cutoff: float = PAE_CUTOFF,
    dist_cutoff: float = DIST_CUTOFF,
    name: str = "interface",
    publish: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Score how confidently a predicted complex's interfaces are predicted.

    Args:
        structure_uri: `s3://bucket/key` of the predicted complex, `.pdb` or
            `.cif`. The aligned error matrix, per-residue confidence array and
            confidence summary are found beside it by the naming convention in
            `tools.common.predictions`, so this is the only reference needed.
        pae_cutoff: Aligned error below which a residue pair counts as
            confidently placed, in Angstroms.
        dist_cutoff: Distance below which residues count as contacting, used by
            the pDockQ scores.
        name: Identifier for the run.
        publish: Upload the score tables to S3 and return their URIs. Turn this
            off to run somewhere without write credentials.
        cache_dir: Where to keep the downloaded script.

    Returns:
        `interfaces`, one entry per chain pair sorted best first, each carrying
        ipSAE alongside ipTM, pDockQ, pDockQ2 and LIS. `best` is the top entry.

    Raises:
        ValueError: if the file extensions are wrong, which matters because the
            upstream script decides how to read its inputs from them.
    """
    suffix = Path(rt.storage.parse_uri(structure_uri).key).suffix.lower()
    if suffix not in STRUCTURE_SUFFIXES:
        raise ValueError(
            f"The structure must be one of {', '.join(STRUCTURE_SUFFIXES)}, "
            f"got {suffix!r}. ipsae.py chooses how to read its inputs from "
            f"their extensions."
        )
    script = ensure_script(cache_dir)

    # One reference in, companions derived. The aligned error matrix, the
    # per-residue confidence array and the confidence summary all sit beside
    # the structure under a shared naming convention, and the script itself
    # derives one path from another, so they must keep those names locally too.
    files = predictions.fetch(structure_uri, rt.work, stem=name)
    structure = files["structure"]
    pae = files.get("pae")
    if pae is None:
        raise FileNotFoundError(
            f"No aligned error matrix beside {structure_uri}. Expected "
            f"{predictions.companions(structure_uri)['pae']}, which is what "
            f"the interface score is computed from."
        )
    if "plddt" not in files:
        print(
            "WARNING: no per-residue confidence array found beside the "
            "structure. ipSAE still computes, but pDockQ and pDockQ2 fall "
            "back to zeros internally and report their floor constants "
            "rather than failing.",
            flush=True,
        )

    # An interface is between two chains, so a monomer has none. The script
    # would write a header and no rows, which is indistinguishable from a
    # parse failure downstream, and ranking designs on an absent score is
    # worse than being told the question was wrong.
    chains = convert.read_chain_names(structure)
    if len(chains) < 2:
        raise ValueError(
            f"{structure_uri} has {len(chains)} chain(s) ({', '.join(chains) or 'none'}), "
            f"so there is no interface to score. ipSAE compares pairs of "
            f"chains; predict the complex with both the target and the binder "
            f"present."
        )

    # The script takes the aligned error file first, then the structure.
    run(["python", str(script), str(pae), str(structure), pae_cutoff, dist_cutoff])

    reports = outputs.report_paths(structure, pae_cutoff, dist_cutoff)
    interfaces = outputs.best_interfaces(outputs.parse_summary(reports["summary"]))

    published: dict[str, str] = {}
    if publish:
        published = {
            key: rt.storage.upload(path, rt.prefix(path.name))
            for key, path in reports.items()
            if path.exists()
        }

    return {
        "name": name,
        "structure_uri": structure_uri,
        "companions": sorted(k for k in files if k != "structure"),
        "pae_cutoff": pae_cutoff,
        "dist_cutoff": dist_cutoff,
        "ipsae_commit": COMMIT,
        "best": interfaces[0] if interfaces else None,
        "interfaces": interfaces,
        "reports": published,
    }
