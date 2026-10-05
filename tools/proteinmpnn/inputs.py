"""Build the JSONL inputs ProteinMPNN's runner expects.

ProteinMPNN takes its structure and every constraint as separate JSONL
dictionaries produced by scripts in the repo's `helper_scripts/` directory.
This module wraps those scripts so the Modal functions share one code path for
preparing a run, whatever inference mode follows.

Position strings follow the helper scripts' own convention: residue positions
are 1-indexed within each chain, groups are comma-separated, and groups appear
in the same order as the chain list. For chains "A B", "1 2 3, 7 8" constrains
A1-A3 and B7-B8, while "1 2 3," constrains only chain A.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tools.common.shell import run

REPO_DIR = "/opt/ProteinMPNN"
HELPERS = f"{REPO_DIR}/helper_scripts"
RUNNER = f"{REPO_DIR}/protein_mpnn_run.py"

# ProteinMPNN's fixed output ordering for the 21-way per-residue distribution.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

MODEL_NAMES = ("v_48_002", "v_48_010", "v_48_020", "v_48_030")
CA_MODEL_NAMES = ("v_48_002", "v_48_010", "v_48_020")


@dataclass
class Constraints:
    """Everything that narrows what ProteinMPNN may put at each position.

    Attributes:
        chains: Space-separated chain IDs to design, e.g. "A" or "A B". Chains
            in the structure but absent here stay fixed and act as context.
            This is the mechanism behind binder design: name only the binder
            chain and the target is held fixed.
        fixed_positions: Positions to hold at their input identity, in the
            convention described in this module's docstring. Used to preserve
            interface hotspots or a scaffold core.
        tied_positions: Positions forced to share an identity across chains,
            for symmetric or repeat designs. Mutually exclusive with
            `homooligomer`.
        homooligomer: Tie every position across all chains, for homo-oligomeric
            designs. Mutually exclusive with `tied_positions`.
        omit_aas: Amino acids never sampled anywhere, e.g. "CX" to exclude
            cysteine. "X" alone is the default and excludes only the unknown
            residue token.
        bias_aas: Per amino acid log-odds added before sampling, e.g.
            {"D": 1.39, "E": 1.39} to favour polar residues on a surface.
            Positive values make a residue more likely.
    """

    chains: str = "A"
    fixed_positions: str = ""
    tied_positions: str = ""
    homooligomer: bool = False
    omit_aas: str = "X"
    bias_aas: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.chains.strip():
            raise ValueError("`chains` must name at least one chain to design")
        if self.tied_positions.strip() and self.homooligomer:
            raise ValueError(
                "Pass either `tied_positions` or `homooligomer`, not both: "
                "homooligomer ties every position across all chains already"
            )
        unknown = sorted(set(self.bias_aas) - set(ALPHABET))
        if unknown:
            raise ValueError(f"`bias_aas` has non-amino-acid keys: {unknown}")


@dataclass
class PreparedRun:
    """Paths produced by `prepare`, plus the runner flags they map to."""

    work: Path
    out_dir: Path
    name: str
    args: list[str]


def prepare(
    work: Path,
    pdb_path: Path,
    name: str,
    constraints: Constraints,
    ca_only: bool = False,
) -> PreparedRun:
    """Stage a PDB and its constraint dictionaries, returning runner arguments.

    Args:
        work: Scratch directory for this invocation.
        pdb_path: The input structure on local disk.
        name: Identifier for the structure; ProteinMPNN names outputs after it.
        constraints: Which chains and positions may change.
        ca_only: Parse an alpha-carbon-only backbone.

    Returns:
        A `PreparedRun` whose `args` are the input-related flags for the runner.
    """
    constraints.validate()

    # The parser reads a whole directory, so the structure is copied under the
    # name the outputs should carry rather than read from wherever it landed.
    pdb_dir = work / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdb_path, pdb_dir / f"{name}.pdb")

    parsed = work / "parsed.jsonl"
    assigned = work / "assigned.jsonl"
    out_dir = work / "out"

    run(
        ["python", f"{HELPERS}/parse_multiple_chains.py",
         "--input_path", pdb_dir, "--output_path", parsed]
        + (["--ca_only"] if ca_only else [])
    )
    run(
        ["python", f"{HELPERS}/assign_fixed_chains.py",
         "--input_path", parsed, "--output_path", assigned,
         "--chain_list", constraints.chains]
    )

    args = [
        "--jsonl_path", str(parsed),
        "--chain_id_jsonl", str(assigned),
        "--out_folder", str(out_dir),
        "--omit_AAs", constraints.omit_aas,
    ]

    if constraints.fixed_positions.strip():
        fixed = work / "fixed_positions.jsonl"
        run(
            ["python", f"{HELPERS}/make_fixed_positions_dict.py",
             "--input_path", parsed, "--output_path", fixed,
             "--chain_list", constraints.chains,
             "--position_list", constraints.fixed_positions]
        )
        args += ["--fixed_positions_jsonl", str(fixed)]

    if constraints.tied_positions.strip() or constraints.homooligomer:
        tied = work / "tied_positions.jsonl"
        cmd = ["python", f"{HELPERS}/make_tied_positions_dict.py",
               "--input_path", parsed, "--output_path", tied]
        if constraints.homooligomer:
            cmd += ["--homooligomer", "1"]
        else:
            cmd += ["--chain_list", constraints.chains,
                    "--position_list", constraints.tied_positions]
        run(cmd)
        args += ["--tied_positions_jsonl", str(tied)]

    if constraints.bias_aas:
        bias = work / "bias_aas.jsonl"
        residues = sorted(constraints.bias_aas)
        run(
            ["python", f"{HELPERS}/make_bias_AA.py",
             "--output_path", bias,
             "--AA_list", " ".join(residues),
             "--bias_list", " ".join(str(constraints.bias_aas[r]) for r in residues)]
        )
        args += ["--bias_AA_jsonl", str(bias)]

    return PreparedRun(work=work, out_dir=out_dir, name=name, args=args)


def model_args(model_name: str, soluble: bool, ca_only: bool) -> list[str]:
    """Runner flags selecting a weight checkpoint, validating the combination."""
    available = CA_MODEL_NAMES if ca_only else MODEL_NAMES
    if model_name not in available:
        raise ValueError(
            f"model_name {model_name!r} is not available "
            f"{'for CA-only models ' if ca_only else ''}"
            f"(choose from {', '.join(available)})"
        )
    if soluble and ca_only:
        raise ValueError("There are no CA-only soluble weights; pick one or the other")

    args = ["--model_name", model_name]
    if soluble:
        args.append("--use_soluble_model")
    if ca_only:
        args.append("--ca_only")
    return args
