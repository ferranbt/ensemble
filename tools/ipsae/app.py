"""ipSAE interface scoring, as a Modal action.

The scoring itself lives in `tools.ipsae.score` and stays an ordinary function,
which is how it should be: it is numpy over one matrix and needs no GPU. This
module only gives it a Modal address, so a workflow can reach it with
`Function.from_name("ipsae", "score_interface")` like any other tool.

The upstream script is baked into the image at a pinned commit rather than
fetched per call, which is what `FRIDAY_IPSAE_PATH` exists for. A container is
ephemeral, so a runtime download would repeat on every invocation.

    modal deploy tools/ipsae/app.py

    modal run tools/ipsae/app.py::score_interface \\
        --structure-uri s3://bucket/runs/<id>/model_0.pdb

The structure is the only reference needed. The aligned error matrix and the
confidence array are found beside it by name; see `tools.common.predictions`.
"""

from __future__ import annotations

from typing import Any

from tools.common.images import base_image, with_local_sources
from tools.common.tool import app_for, tool
from tools.ipsae import score

REPO_URL = "https://github.com/DunbrackLab/IPSAE.git"
REPO_DIR = "/opt/IPSAE"

# Numpy over one matrix, so this is the cheapest action in the repo: no GPU,
# and a few seconds per complex.
TIMEOUT = 15 * 60

app = app_for("ipsae")

image = with_local_sources(
    base_image()
    .pip_install("numpy==1.26.4")
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {score.COMMIT}",
    )
    # Tells `score.ensure_script` to use the baked copy instead of downloading.
    .env({score.CACHE_ENV: f"{REPO_DIR}/ipsae.py"})
)


@tool("ipsae/score_interface", image=image, timeout=TIMEOUT)
def score_interface(
    rt,
    structure_uri: str,
    pae_cutoff: float = score.PAE_CUTOFF,
    dist_cutoff: float = score.DIST_CUTOFF,
    name: str = "interface",
) -> dict[str, Any]:
    """Score how confidently a predicted complex's interfaces are predicted.

    A thin pass-through to `tools.ipsae.score.score_interface`, which carries
    the argument documentation.

    Requires a complex of two or more chains; a monomer is rejected rather than
    scored as zero, since a zero from a monomer and a zero from a failed
    interface are indistinguishable once they reach a ranking.
    """
    return score.score_interface(
        rt,
        structure_uri=structure_uri,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
        name=name,
    )
