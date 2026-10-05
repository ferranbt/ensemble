"""Agree three scanners on which substitutions are worth making."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from tools.common.mutations import Mutation

# How far apart two mutations must be, in residues, before they are stacked
# into one variant. Neighbouring substitutions interact, so their individual
# effects stop being additive and the stack stops being predictable.
MIN_SEPARATION = 8


@dataclass(frozen=True)
class Scored:
    """One substitution as a single source sees it."""

    mutation: Mutation
    score: float

    @property
    def key(self) -> tuple[int, str]:
        return (self.mutation.position, self.mutation.mt)


@dataclass(frozen=True)
class Reader:
    """Where one kind of scan keeps its residues and its score."""

    wild_type: str
    mutant: str
    score: str
    higher_is_better: bool = True

READERS = {
    "stability": Reader("wild_type", "mutant", "ddG", higher_is_better=False),
    "likelihood": Reader("wild_type", "aa", "log_likelihood_ratio"),
    "structural": Reader("native_aa", "aa", "probability"),
}

def _entries(positions: Iterable[dict[str, Any]], wt_key: str, score_key: str,
             aa_key: str) -> list[tuple[int, str, str, float]]:
    """Flatten one scanner's `positions` into (position, wt, mt, score)."""
    flat = []
    for entry in positions or []:
        position = entry.get("position")
        wild_type = entry.get(wt_key)
        if position is None or not wild_type:
            continue
        for candidate in entry.get("top") or []:
            mutant = candidate.get(aa_key)
            score = candidate.get(score_key)
            if not mutant or score is None or mutant == wild_type:
                continue
            flat.append((int(position), str(wild_type), str(mutant), float(score)))
    return flat


def align_numbering(
    flat: list[tuple[int, str, str, float]], sequence: str, label: str
) -> list[Scored]:
    """Re-number one scanner's substitutions onto the sequence.

    Finds the offset under which every reported wild-type residue matches the
    sequence. Candidate offsets come from the entries themselves rather than a
    blind sweep, so this is cheap, and an offset is only accepted if it
    explains *all* of them.

    Raises:
        ValueError: when no single offset works, which means the scan was run
            against a different sequence than the one being mutated. Guessing
            past that would silently mutate the wrong residues.
    """
    if not flat:
        return []

    positions = {p for p, _, _, _ in flat}
    # An offset maps a reported position to a 1-indexed sequence position.
    # Trying the offsets implied by the first entry is enough: the right one is
    # always among them if the scan describes this sequence at all.
    first = min(positions)
    wanted = {wt for p, wt, _, _ in flat if p == first}
    candidates = sorted(
        {
            index + 1 - first
            for index, residue in enumerate(sequence)
            if residue in wanted
        }
    )

    for offset in candidates:
        if all(
            0 <= p + offset - 1 < len(sequence) and sequence[p + offset - 1] == wt
            for p, wt, _, _ in flat
        ):
            return [
                Scored(Mutation(wt, p + offset, mt), score)
                for p, wt, mt, score in flat
            ]

    sample = ", ".join(f"{wt}{p}" for p, wt, _, _ in flat[:5])
    raise ValueError(
        f"No numbering offset lines {label}'s scan up with this sequence "
        f"({len(sequence)} residues). It reported {sample} and no shift makes "
        f"every wild-type residue match, so the scan describes a different "
        f"sequence. Refusing to guess which residues it meant."
    )


def reader_for(scan: dict[str, Any]) -> Reader:
    """The reader one scan is asking for, with any per-scan overrides applied."""
    kind = scan.get("type")
    if kind not in READERS:
        raise ValueError(
            f"Unknown scan type {kind!r}. Expected one of "
            f"{', '.join(sorted(READERS))}."
        )
    reader = READERS[kind]

    overrides = dict(scan.get("keys") or {})
    unknown = set(overrides) - {"wild_type", "mutant", "score"}
    if unknown:
        raise ValueError(
            f"Unknown key override {', '.join(sorted(unknown))} on a {kind} "
            f"scan. Expected wild_type, mutant or score."
        )
    if "higher_is_better" in scan:
        overrides["higher_is_better"] = bool(scan["higher_is_better"])
    return replace(reader, **overrides) if overrides else reader


def read(scan: dict[str, Any], sequence: str) -> list[Scored]:
    """One scan's substitutions, re-numbered onto the sequence, higher better.

    A scan whose score runs the other way, such as ddG where positive means
    destabilising, is negated here so every source can be ranked the same way.
    """
    reader = reader_for(scan)
    label = scan.get("label") or scan.get("type")
    flat = _entries(scan.get("positions"), reader.wild_type, reader.score,
                    reader.mutant)
    scored = align_numbering(flat, sequence, label)
    if reader.higher_is_better:
        return scored
    return [Scored(s.mutation, -s.score) for s in scored]


def sources(scans: Iterable[dict[str, Any]], sequence: str) -> dict[str, list[Scored]]:
    """Every scan that reported something, keyed by the label it was given.

    The label is the caller's, so nothing here names a tool. A scan that
    reported nothing is left out rather than counted as a source that agreed to
    nothing.
    """
    found: dict[str, list[Scored]] = {}
    seen: set[str] = set()
    for scan in scans or []:
        label = scan.get("label") or scan.get("type")
        if label in seen:
            raise ValueError(
                f"Two scans are both labelled {label!r}. Agreement counts "
                f"sources, so each needs its own label."
            )
        seen.add(label)
        scored = read(scan, sequence)
        if scored:
            found[label] = scored
    return found


def agree(
    sources: dict[str, list[Scored]],
    min_sources: int = 2,
    top_per_source: int = 50,
) -> list[dict[str, Any]]:
    """Substitutions that enough sources independently rank highly, best first.

    **Ranks, not thresholds.** A ddG cutoff, a likelihood cutoff and a
    probability cutoff would each be a number invented here, and published
    pipelines that use such numbers earned them against experimental data we do
    not have. Ranking asks each source only what it is actually good at, which
    is ordering its own candidates, and asks nothing about where to draw a line.

    `top_per_source` is the one number left, and it is a shortlist size rather
    than a claim about what counts as stabilising: a substitution ranked first
    by one source and five hundredth by another has not really been agreed on.

    Ranked by mean rank rather than by a combined score, because the three
    scores are in different units on different scales and adding them would
    silently weight whichever has the widest range.
    """
    ranks: dict[tuple[int, str], dict[str, int]] = {}
    values: dict[tuple[int, str], dict[str, float]] = {}
    mutations: dict[tuple[int, str], Mutation] = {}

    for name, scored in sources.items():
        ordered = sorted(scored, key=lambda s: -s.score)[:top_per_source]
        for position, entry in enumerate(ordered):
            ranks.setdefault(entry.key, {})[name] = position
            values.setdefault(entry.key, {})[name] = entry.score
            mutations[entry.key] = entry.mutation

    agreed = []
    for key, per_source in ranks.items():
        if len(per_source) < min_sources:
            continue
        agreed.append({
            "mutation": str(mutations[key]),
            "position": mutations[key].position,
            "wild_type": mutations[key].wt,
            "mutant": mutations[key].mt,
            "sources": sorted(per_source),
            "mean_rank": sum(per_source.values()) / len(per_source),
            "scores": values[key],
        })
    return sorted(agreed, key=lambda m: (m["mean_rank"], m["position"]))


def spread(agreed: list[dict[str, Any]], separation: int = MIN_SEPARATION) -> list[dict]:
    """The best-ranked mutations, one per position and never too close together."""
    chosen: list[dict[str, Any]] = []
    for candidate in agreed:
        if any(
            abs(candidate["position"] - taken["position"]) < separation
            for taken in chosen
        ):
            continue
        chosen.append(candidate)
    return chosen
