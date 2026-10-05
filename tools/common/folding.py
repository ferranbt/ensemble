"""The contract every structure predictor here answers to.

Boltz, Protenix and ESMFold2 do the same job and disagree about how to say so.
Each names its sample count differently, takes its alignment somewhere else,
reports its confidence under its own keys and on its own scale, and wraps the
whole thing in its own envelope. A workflow stage that names one of them
therefore cannot be pointed at another without rewriting the stage and every
stage downstream, which defeats the point of having orthogonal predictors at
all: the reason to run two is to compare them.

So the common parts are standardised and the rest is left alone:

    predict_complex(chains, samples, seed, name, **predictor specific)

`chains` is the wire form of `tools.common.chains`, one entry per chain, each
carrying its own `msa`. `samples` is how many structures to produce and `seed`
seeds them. Anything genuinely particular to one predictor keeps its own name
and stays in that predictor's signature: Boltz's `recycling_steps`, Protenix's
`model`, ESMFold2's `num_loops`. Standardising those would mean inventing a
shared meaning that does not exist.

The result is the same shape from all three:

    run_id, name, predictor, version, chains, chain_lengths
    models: best first, each with structure_uri, the companion URIs, and
            whichever of SCORES the predictor reported
    best:   models[0]

Scores are normalised to one name and one scale, because ranking designs across
predictors is the whole reason they are here. Protenix reports pLDDT out of 100
and Boltz out of 1; ordering a mixed set by the raw number would sort by which
tool produced it. Pair scores are re-keyed by chain id, since every predictor
indexes them positionally and a caller thinks in chain names.
"""

from __future__ import annotations

from typing import Any

from tools.common import chains as chain_spec
from tools.common import msa as msa_setting

# The confidence scores a predictor reports, on 0 to 1, higher being better.
# `chain_pair_iptm` is nested by chain id. A predictor reports the subset it
# has; nothing is invented to fill a gap.
SCORES = (
    "confidence_score",
    "ptm",
    "iptm",
    "complex_plddt",
    "chain_pair_iptm",
)

# Standard name -> the keys a predictor might use for it, in preference order.
ALIASES = {
    "confidence_score": ("confidence_score", "ranking_score"),
    "ptm": ("ptm",),
    "iptm": ("iptm",),
    "complex_plddt": ("complex_plddt", "plddt"),
}


def fraction(value: Any) -> float | None:
    """A confidence score on 0 to 1, whichever scale it arrived on.

    Predictors report pLDDT both ways and the difference is invisible in a
    number like 0.95 against 95.0 until something sorts on it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number / 100.0 if number > 1.0 else number


def pair_scores(raw: Any, chain_ids: list[str]) -> dict[str, dict[str, float]]:
    """Chain-pair scores re-keyed from position to chain id.

    Every predictor writes these positionally, nested by index as strings or
    as a square matrix. A caller has chain ids, not indices, so this converts
    once here rather than at each use.

    >>> pair_scores({"0": {"1": 0.9}}, ["A", "B"])
    {'A': {'B': 0.9}}
    >>> pair_scores([[1.0, 0.9], [0.9, 1.0]], ["A", "B"])
    {'A': {'A': 1.0, 'B': 0.9}, 'B': {'A': 0.9, 'B': 1.0}}
    """
    def name(index: Any) -> str | None:
        try:
            return chain_ids[int(index)]
        except (TypeError, ValueError, IndexError):
            return None

    rows: dict[str, dict[str, float]] = {}
    items = enumerate(raw) if isinstance(raw, list) else (raw or {}).items()
    for outer, row in items:
        first = name(outer)
        if first is None:
            continue
        inner = enumerate(row) if isinstance(row, list) else (row or {}).items()
        pairs = {}
        for key, value in inner:
            second = name(key)
            score = fraction(value)
            if second is not None and score is not None:
                pairs[second] = score
        if pairs:
            rows[first] = pairs
    return rows


def scores(raw: dict[str, Any], chain_ids: list[str]) -> dict[str, Any]:
    """Whichever of `SCORES` a predictor reported, under the standard names.

    A score the predictor did not report is left out rather than defaulted,
    because a missing confidence and a confidence of zero mean opposite things
    and only one of them should pass a filter.
    """
    found: dict[str, Any] = {}
    for standard, candidates in ALIASES.items():
        for key in candidates:
            if key in raw:
                value = fraction(raw[key])
                if value is not None:
                    found[standard] = value
                break

    for key in ("chain_pair_iptm", "pair_chains_iptm"):
        if key in raw:
            pairs = pair_scores(raw[key], chain_ids)
            if pairs:
                found["chain_pair_iptm"] = pairs
            break
    return found


def interface(model: dict[str, Any], first: str, second: str) -> float | None:
    """The predicted interface score between two chains, if it was reported.

    For a two-chain binder job this is the number to rank designs on. Looks up
    both orders, since a predictor may record only one triangle.
    """
    pairs = model.get("chain_pair_iptm") or {}
    for a, b in ((first, second), (second, first)):
        row = pairs.get(a)
        if isinstance(row, dict) and row.get(b) is not None:
            return float(row[b])
    return None


def single_sequence_only(complex_chains: dict[str, chain_spec.Chain], predictor: str) -> None:
    """Refuse an alignment for a predictor that cannot use one.

    ESMFold2 takes no alignment at all. Accepting the argument and ignoring it
    would make a workflow that specifies one look like it ran with it, and the
    accuracy difference is large enough to change a conclusion.
    """
    named = sorted(
        chain_id for chain_id, chain in complex_chains.items() if chain.msa
    )
    if named:
        raise ValueError(
            f"{predictor} predicts from single sequence and cannot use an "
            f"alignment, but chain(s) {', '.join(named)} name one. Drop the "
            f"`msa` to run it, or use a predictor that takes alignments."
        )


def one_server(complex_chains: dict[str, chain_spec.Chain], predictor: str) -> str:
    """The single alignment server this job may query, if any.

    The value comes back resolved to an address. `msa_setting.servers` expects
    an already-resolved job, so passing raw chain settings hands a predictor
    the bare word `colabfold` as a URL, which fails inside its MSA client and
    surfaces as an empty predictions directory rather than an error about a
    URL.

    Raises:
        ValueError: when chains name different servers. Predictors take one
            server per job, and silently picking one would search the wrong
            database for every other chain.
    """
    servers = msa_setting.servers(
        {c: msa_setting.normalise(chain.msa) for c, chain in complex_chains.items()}
    )
    if len(servers) > 1:
        raise ValueError(
            f"{predictor} queries one alignment server per job, but this "
            f"complex names {len(servers)}: {servers}. Search them separately "
            f"and pass the resulting s3:// alignments instead."
        )
    return servers[0] if servers else ""


def shared_alignment_warning(complex_chains: dict[str, chain_spec.Chain]) -> None:
    """Warn when chains with different sequences share one alignment.

    It can only be the homologues of one of them, so the others are predicted
    from another protein's evolutionary signal. Nothing fails; the interface
    score just drops. Measured on barnase-barstar, ipSAE 0.94 with each chain
    aligned against 0.68 with one shared. Identical sequences sharing an
    alignment is correct, so that is not what this looks for.

    Only precomputed alignments count. Chains naming the same *server* are
    fine: each is searched separately with its own sequence.
    """
    artifacts = {
        c.msa for c in complex_chains.values()
        if c.msa and msa_setting.kind(c.msa) == msa_setting.ARTIFACT
    }
    for uri in artifacts:
        holders = {
            c: chain for c, chain in complex_chains.items() if chain.msa == uri
        }
        if len({chain.sequence for chain in holders.values()}) > 1:
            print(
                f"WARNING: chains {', '.join(sorted(holders))} have different "
                f"sequences but share the alignment {uri}, so it matches at "
                f"most one of them. Give each its own, or search a paired "
                f"alignment (pair=True), unless approximating on purpose.",
                flush=True,
            )


def summary_lines(result: dict[str, Any]) -> list[str]:
    """One line per model, for a command-line run.

    Reads only the standard fields, so it renders any predictor's result.
    """
    lines = []
    for model in result.get("models") or []:
        parts = [f"model {model.get('model', 0)}"]
        for key, label in (
            ("confidence_score", "confidence"),
            ("iptm", "iptm"),
            ("ptm", "ptm"),
            ("complex_plddt", "plddt"),
            ("interface_iptm", "interface"),
        ):
            value = model.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={value:.3f}")
        parts.append(str(model.get("structure_uri", "")))
        lines.append("  ".join(parts))
    return lines or ["no models predicted"]


def envelope(
    *,
    name: str,
    predictor: str,
    version: str,
    complex_chains: dict[str, chain_spec.Chain],
    models: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """The standard predictor result.

    Args:
        models: One entry per predicted structure, already sorted best first.
        extra: Predictor-specific fields, kept alongside the standard ones.
    """
    return {
        "name": name,
        "predictor": predictor,
        "version": version,
        "chains": chain_spec.as_dict(complex_chains),
        "chain_lengths": {
            c: len(chain.sequence) for c, chain in complex_chains.items()
        },
        "best": models[0] if models else None,
        "models": models,
        **extra,
    }
