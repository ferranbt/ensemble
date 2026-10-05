"""ThermoMPNN (https://github.com/Kuhlman-Lab/ThermoMPNN) as a Modal GPU action.

Predicts how each possible point mutation changes a protein's folding
stability. One action, `scan_stability`, which scores every substitution at
every position of a chain in a single pass.

This is the third view on the same question, and the three are independent:

    ESM-2                     what evolution favours, from sequence
    ProteinMPNN probabilities what the backbone tolerates, from structure
    ThermoMPNN                what keeps the protein folded

It matters here because affinity maturation routinely trades stability for
binding. A variant that binds better but no longer folds is a dead end, and
this is the cheapest way to catch that before committing to a design.

Despite the name it is not ProteinMPNN with different weights. It wraps
ProteinMPNN's encoder, frozen, and adds a trained stability head, so it is a
separate model with its own checkpoint that happens to contain the other.

    modal run tools/thermompnn/app.py::scan_stability \\
        --structure-uri s3://bucket/target.pdb --chain A
"""

from __future__ import annotations

from typing import Any

from tools.common import convert
from tools.common.images import torch_image, with_local_sources
from tools.common.shell import run
from tools.common.tool import app_for, tool
from tools.thermompnn import outputs

REPO_URL = "https://github.com/Kuhlman-Lab/ThermoMPNN.git"
REPO_COMMIT = "2b04fd370e399911b1fa5848112cc9013f084110"
REPO_DIR = "/opt/ThermoMPNN"
SCRIPT = f"{REPO_DIR}/analysis/custom_inference.py"

# The frozen ProteinMPNN encoder ThermoMPNN is built on.
PROTEINMPNN_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
PROTEINMPNN_WEIGHTS_URL = (
    f"https://raw.githubusercontent.com/dauparas/ProteinMPNN/"
    f"{PROTEINMPNN_COMMIT}/vanilla_model_weights/v_48_020.pt"
)

# A small model on top of a frozen encoder, so this is the cheapest GPU tool
# in the repo. Weights ship inside the repository, so nothing downloads.
GPU = "T4"
TIMEOUT = 30 * 60

app = app_for("thermompnn")

image = with_local_sources(
    torch_image()
    .apt_install("curl")
    .pip_install(
        "pytorch-lightning==2.4.0",
        "omegaconf==2.3.0",
        "pandas==2.2.2",
        "biopython==1.84",
        "tqdm",
        "joblib",
        "wandb",
    )
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {REPO_COMMIT}",
        # ThermoMPNN loads ProteinMPNN's encoder weights, which its own
        # repository does not carry. Its shipped config points at the authors'
        # cluster filesystem, so both the file and the path need supplying.
        f"mkdir -p {REPO_DIR}/vanilla_model_weights",
        f"curl -sSL -o {REPO_DIR}/vanilla_model_weights/v_48_020.pt"
        f" {PROTEINMPNN_WEIGHTS_URL}",
        # Repoint the one config entry that matters at inference. The rest of
        # that file is training dataset locations, unused here.
        f"sed -i 's|thermompnn_dir:.*|thermompnn_dir: \"{REPO_DIR}\"|'"
        f" {REPO_DIR}/local.yaml",
    )
    .env({"WANDB_MODE": "disabled"})
)


@tool("thermompnn/scan_stability", image=image, gpu=GPU, timeout=TIMEOUT)
def scan_stability(
    rt,
    structure_uri: str,
    chain: str = "A",
    top_k: int = 5,
    descending: bool = False,
    name: str = "stability",
) -> dict[str, Any]:
    """Score every point mutation in one chain for its effect on stability.

    Args:
        structure_uri: `s3://bucket/key` of the input PDB.
        chain: Which chain to scan. Only one chain is scanned per call.
        top_k: Substitutions to report per position, largest effect first.
        descending: Sort each position's substitutions high to low, showing
            the most destabilising first. The default is ascending, showing
            the best tolerated first.
        name: Identifier for the run.

    Returns:
        `positions`, one entry per residue with its wild-type identity, the
        substitutions with the largest predicted effect, and the spread of
        effects at that position. A wide spread marks a position where the
        choice of residue matters. `predictions_uri` points at the full table.

        A positive `ddG` means destabilising, confirmed empirically against
        mutations of known effect. So low values mark substitutions the fold
        will tolerate, and that is what a maturation campaign should choose
        from once a mutation has passed the other filters.
    """
    source = convert.fetch_structure(structure_uri, "pdb", rt.work, stem=name)
    structure = source.path
    chain = convert.resolve_chain(chain, source)

    out_dir = rt.work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run from the repository root: the script resolves its config and bundled
    # checkpoint relative to its own location, and imports sibling modules.
    run(
        ["python", SCRIPT,
         "--pdb", str(structure),
         "--chain", chain,
         "--out_dir", str(out_dir)],
        cwd=REPO_DIR,
    )

    report = outputs.report_path(out_dir, structure)
    if not report.is_file():
        produced = sorted(p.name for p in out_dir.iterdir()) if out_dir.is_dir() else []
        raise FileNotFoundError(
            f"ThermoMPNN wrote no predictions at {report}. "
            f"The output directory holds: {produced or 'nothing'}"
        )

    records = outputs.parse_predictions(report)
    summary = outputs.summarise(records, top_k=top_k, descending=descending)

    return {
        "name": name,
        "chain": chain,
        "structure_uri": structure_uri,
        "converted_from": source.source_format if source.converted else None,
        "num_positions": len(summary),
        "num_mutations": len(records),
        "predictions_uri": rt.storage.upload(
            report, rt.prefix("stability_scan.csv")
        ),
        "positions": summary,
    }

