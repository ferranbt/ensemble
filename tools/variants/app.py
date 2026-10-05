"""Propose stabilising variants from several mutation scans, as a Modal action.

Pure arithmetic over a few thousand numbers, so no GPU and about a second. It
is a Modal function only so a workflow can reach it by name like every other
tool.

    modal deploy tools/variants/app.py
"""

from __future__ import annotations

from typing import Any

from tools.common import mutations as mut
from tools.common.images import base_image, with_local_sources
from tools.common.tool import app_for, tool
from tools.variants import consensus

TIMEOUT = 15 * 60

app = app_for("variants")

image = with_local_sources(base_image())


@tool("variants/propose_variants", image=image, timeout=TIMEOUT)
def propose_variants(
    rt,
    sequence: str,
    scans: list[dict] | None = None,
    min_sources: int = 2,
    top_per_source: int = 50,
    max_mutations: int = 6,
    separation: int = consensus.MIN_SEPARATION,
    name: str = "variants",
) -> dict[str, Any]:
    """Combine mutation scans into variants worth testing.

    Any single predictor ranked over every possible substitution puts nonsense
    near the top, so this keeps only those that at least `min_sources` methods
    like for independent reasons: that the fold tolerates it, that evolution
    uses it there, and that the local structure wants it.

    Args:
        sequence: The parent sequence every variant is built from. Every scan
            must describe this sequence; the wild-type residue each reports is
            checked against it, so a scan of something else is rejected rather
            than silently mutating the wrong residues.
        scans: One entry per mutation scan, each a mapping of:

            positions: That scan's `positions`, as its tool returned them.
            type: What kind of evidence it carries, which is also how its
                columns are read: `stability` for a ddG (negated, since
                positive destabilises), `likelihood` for a log likelihood
                ratio, `structural` for a per-residue probability.
            label: What to call this source in the result. Defaults to `type`.
                Nothing here names a tool, so the caller decides whether a
                scan appears as `esm2` or as `likelihood`.
            keys: Optional column names, for a tool that answers the same
                question under different ones, as
                `{wild_type: wt, mutant: to, score: delta_g}`.
            higher_is_better: Optional, overriding the type's own direction.

            Two scans of the same type are allowed as long as their labels
            differ, which is how two independent stability predictors can both
            count towards agreement.
        min_sources: How many scans must independently rank a substitution
            highly. Two is the useful default; three is very strict, and one is
            just the single-predictor ranking this exists to avoid.
        top_per_source: How many of each scan's best substitutions to consider.
            Agreement is measured on rank rather than against a ddG or
            likelihood cutoff, because such a cutoff would be a number invented
            here: the published pipelines that use them earned theirs against
            experimental data we do not have. This is a shortlist size, not a
            claim about what counts as stabilising.
        max_mutations: Ceiling on how many mutations the largest stacked
            variant carries.
        separation: Minimum residues between two mutations stacked into one
            variant. Closer substitutions interact, so their separately
            predicted effects stop being additive.
        name: Identifier for the run.

    Returns:
        `variants`, each with its `mutations` in `A123K:T50A` notation and the
        full `sequence` to fold, plus `agreed`, every substitution that passed
        with the per-source scores behind it. Variants come as singles, one per
        mutation, and as cumulative stacks of the best-ranked ones, so a later
        step can see both whether a mutation helps alone and whether the
        benefits add up.

        The first entry is the unchanged parent, with no mutations and
        `is_parent` set. Whatever scores these has to score the parent the same
        way for any of the numbers to mean anything.

    Raises:
        ValueError: if no scan could be lined up with `sequence`, or if none
            produced a substitution that enough sources agree on.
    """
    parent = mut.clean_sequence(sequence)

    sources = consensus.sources(scans, parent)

    if len(sources) < min_sources:
        raise ValueError(
            f"{min_sources} scans must agree, but only {len(sources)} were "
            f"supplied ({', '.join(sources) or 'none'})."
        )

    agreed = consensus.agree(sources, min_sources, top_per_source)
    if not agreed:
        raise ValueError(
            f"No substitution appeared in the top {top_per_source} of "
            f"{min_sources} of the {len(sources)} scans. Widen "
            f"top_per_source, lower min_sources, or scan more positions."
        )

    chosen = consensus.spread(agreed, separation)[:max_mutations]

    # The parent, unchanged, first. Whatever measures these variants has to
    # measure the thing they are variants *of*, or there is no baseline and a
    # predicted stability is a number with nothing to compare against. Putting
    # it in the same list means it goes through the identical downstream call
    # rather than a separate one configured slightly differently.
    variants: list[dict[str, Any]] = [{
        "mutations": "",
        "sequence": parent,
        "num_mutations": 0,
        "mean_rank": None,
        "sources": [],
        "is_parent": True,
    }]

    # Each mutation alone: does it help by itself.
    for entry in chosen:
        parsed = mut.parse(entry["mutation"])
        variants.append({
            "mutations": entry["mutation"],
            "sequence": mut.apply(parent, parsed),
            "num_mutations": 1,
            "mean_rank": entry["mean_rank"],
            "sources": entry["sources"],
            "is_parent": False,
        })

    # Cumulative stacks: do the benefits add up. Built in rank order, so each
    # stack is the previous one plus the next best mutation.
    for size in range(2, len(chosen) + 1):
        group = chosen[:size]
        combined = ":".join(e["mutation"] for e in group)
        parsed = mut.parse(combined)
        variants.append({
            "mutations": combined,
            "sequence": mut.apply(parent, parsed),
            "num_mutations": size,
            "mean_rank": sum(e["mean_rank"] for e in group) / size,
            "sources": sorted({s for e in group for s in e["sources"]}),
            "is_parent": False,
        })

    return {
        "name": name,
        "parent_sequence": parent,
        "length": len(parent),
        "scans": sorted(sources),
        "min_sources": min_sources,
        "num_agreed": len(agreed),
        "agreed": agreed,
        "variants": variants,
    }
