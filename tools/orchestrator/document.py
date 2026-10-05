"""Parsing and substitution for workflow documents.

A workflow is a list of steps. Each names a tool, optionally fans out over a
prior step's results or a workflow input, and supplies parameters. Parameters
may reference three things and nothing else:

    ${inputs.targets}     a value the caller supplied
    ${msa.results}        a prior step's results, by step id
    ${item}               the current element inside a fan-out
    ${item.chains}        a field of it
    ${item.carried.fold.best.structure_uri}
                          a field of an execution this item came from

That last one is not declared anywhere. Every call remembers the item it was
made from, and that item remembers its own, so the chain is available under
`carried` keyed by the step that produced each link. See `lineage`.

Deliberately not a template language. Everything here is pure: no Modal, no
network, no execution. The runner turns what this produces into calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tools.orchestrator import filters

# A whole-value reference, "${x}", keeps the referenced type. Anything else is
# interpolated into a string.
WHOLE = re.compile(r"^\$\{([^}]+)\}$")
EMBEDDED = re.compile(r"\$\{([^}]+)\}")

# Where a result's lineage hangs. One key, so nothing a tool returns is
# shadowed by it.
CARRIED = "carried"


class Unknown:
    """A stand-in for a result that a dry run has not produced.

    It resolves any field access to itself, so `${item.chains}` succeeds
    against a value that does not exist yet. That lets a dry run propagate
    arity through the whole document and report that a fan-out is about to be
    twenty thousand calls, which is the main thing worth knowing before
    spending anything.
    """

    def __contains__(self, _key: object) -> bool:
        return True

    def __getitem__(self, _key: object) -> "Unknown":
        return self

    def __repr__(self) -> str:
        return "<unknown>"


@dataclass
class Step:
    """One step of a workflow."""

    id: str
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    # One reference, or several to fan out over all of them at once. A list is
    # how several branches are collected and given the same treatment: three
    # predictors' results scored by one step rather than three copies of it.
    for_each: str | list[str] | None = None
    needs: list[str] = field(default_factory=list)
    where: str | None = None
    order_by: str | None = None
    limit: int | None = None
    # The field of this step's result that later steps fan out over. Declared
    # by the step that knows its own shape, so a consumer writes `for_each:
    # backbones` rather than repeating a path at every use. A `for_each` path
    # is not checked before the run, so a typo in one costs whatever the
    # previous step already spent.
    emits: str | None = None

    def sources(self) -> list[str]:
        """The references this step fans out over, in order."""
        if self.for_each is None:
            return []
        if isinstance(self.for_each, str):
            return [self.for_each.strip()]
        return [str(reference).strip() for reference in self.for_each]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise ValueError(
                f"Step id {self.id!r} must be a lower-case identifier"
            )
        narrowing = {"where": self.where, "order_by": self.order_by,
                     "limit": self.limit}
        if self.for_each is None and any(v is not None for v in narrowing.values()):
            named = [k for k, v in narrowing.items() if v is not None]
            raise ValueError(
                f"Step {self.id}: {named} narrow the set a step fans out over, "
                f"so they need `for_each`."
            )
        # `limit` without an order takes whatever comes first. That is
        # arbitrary among a step's results, and so almost certainly a mistake,
        # but it is well defined for a list the caller supplied: their order is
        # the order. So the rule applies only to the former.
        over_step_results = any(
            not reference.startswith("inputs.") for reference in self.sources()
        )
        if self.limit is not None and self.order_by is None and over_step_results:
            raise ValueError(
                f"Step {self.id}: `limit` without `order_by` over "
                f"{', '.join(self.sources())} takes an arbitrary subset rather "
                f"than the best ones, because a step's results have no inherent "
                f"order. Add an order, or fan out over a workflow input where "
                f"the order is yours."
            )
        if not re.fullmatch(r"[a-z0-9_]+/[a-z0-9_]+", self.tool):
            raise ValueError(
                f"Step {self.id}: tool must be '<app>/<function>', "
                f"got {self.tool!r}"
            )


@dataclass
class Workflow:
    """A parsed workflow document."""

    name: str
    steps: list[Step]
    inputs: dict[str, dict] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("A workflow needs at least one step")
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"Duplicate step id {step.id!r}")
            for required in step.needs:
                if required not in seen:
                    raise ValueError(
                        f"Step {step.id} needs {required!r}, which does not "
                        f"run before it"
                    )
            # A bare `for_each: <step>` is checked here rather than at run
            # time, because the alternative is discovering a typo after the
            # step it names has already spent its GPU time.
            for source in step.sources():
                if "." not in source and source not in seen:
                    raise ValueError(
                        f"Step {step.id}: for_each names {source!r}, which is "
                        f"not a step that runs before it. Available: "
                        f"{', '.join(sorted(seen)) or 'none'}"
                    )
            seen.add(step.id)


def parse(document: str) -> Workflow:
    """Parse a YAML workflow, raising on anything malformed."""
    import yaml

    raw = yaml.safe_load(document)
    if not isinstance(raw, dict):
        raise ValueError("A workflow document must be a mapping")

    steps = []
    for entry in raw.get("steps") or []:
        known = {"id", "tool", "params", "for_each", "needs",
                 "where", "order_by", "limit", "emits"}
        unknown = sorted(set(entry) - known)
        if unknown:
            raise ValueError(
                f"Step {entry.get('id', '?')}: unknown field(s) {unknown}. "
                f"Known: {', '.join(sorted(known))}"
            )
        steps.append(Step(**entry))

    return Workflow(
        name=raw.get("name", "workflow"),
        description=raw.get("description", ""),
        inputs=raw.get("inputs") or {},
        output=raw.get("output") or {},
        steps=steps,
    )


def bind_inputs(workflow: Workflow, supplied: dict[str, Any]) -> dict[str, Any]:
    """Check supplied inputs against the document's declarations.

    Raises:
        ValueError: for a missing required input or one the workflow does not
            declare, since a silently ignored input is a typo that produces a
            plausible but wrong run.
    """
    declared = workflow.inputs
    unknown = sorted(set(supplied) - set(declared))
    if unknown:
        raise ValueError(
            f"Unknown input(s) {unknown}. This workflow takes: "
            f"{', '.join(sorted(declared)) or 'none'}"
        )

    bound: dict[str, Any] = {}
    for key, spec in declared.items():
        spec = spec or {}
        if key in supplied:
            bound[key] = supplied[key]
        elif "default" in spec:
            bound[key] = spec["default"]
        elif spec.get("required"):
            raise ValueError(f"Input {key!r} is required")
        else:
            bound[key] = None
    return bound


def lookup(path: str, scope: dict[str, Any]) -> Any:
    """Resolve a dotted reference against the current scope.

    Raises:
        KeyError: naming the path and what was available, because an
            unresolvable reference otherwise surfaces as a confusing failure
            inside a tool after the previous step has already run.
    """
    parts = path.strip().split(".")
    value: Any = scope
    walked: list[str] = []
    for part in parts:
        walked.append(part)
        if isinstance(value, Unknown):
            return value
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            value = value[int(part)]
        else:
            available = (
                sorted(value) if isinstance(value, dict) else type(value).__name__
            )
            raise KeyError(
                f"Cannot resolve ${{{path}}}: {'.'.join(walked)} is not "
                f"available. At that point there was: {available}"
            )
    return value


def resolve(value: Any, scope: dict[str, Any]) -> Any:
    """Substitute every reference in a parameter tree.

    A string that is exactly one reference keeps the referenced value's type,
    so `${item.chains}` yields a mapping rather than its printed form. A
    reference embedded in surrounding text is interpolated.
    """
    if isinstance(value, str):
        whole = WHOLE.match(value.strip())
        if whole:
            return lookup(whole.group(1), scope)
        return EMBEDDED.sub(lambda m: str(lookup(m.group(1), scope)), value)
    if isinstance(value, dict):
        return {k: resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, scope) for v in value]
    return value


def emitted(step: Step, outcomes: list[Any]) -> list[Any]:
    """What later steps fan out over, given this step's results.

    Without `emits` that is the results themselves, one per call. With it, the
    named field of every result, concatenated, so a step that produced four
    backbones in one call hands over four items rather than one.

    An emitted item keeps the lineage of the call that produced it, since it
    comes out of a result and a bare design has no memory of what it was
    designed against.

    A result missing the named field contributes nothing rather than failing:
    one call returning an unexpected shape should not end a run whose other
    calls succeeded, and the step report already separates failures from
    filtering.
    """
    if not step.emits:
        return list(outcomes)

    items: list[Any] = []
    for outcome in outcomes:
        if isinstance(outcome, Unknown):
            # A dry run cannot know how many a real call would emit.
            items.append(outcome)
            continue
        value = outcome.get(step.emits) if isinstance(outcome, dict) else None
        inherited = outcome.get(CARRIED) if isinstance(outcome, dict) else None
        produced = value if isinstance(value, list) else [] if value is None else [value]
        for entry in produced:
            items.append(
                {**entry, CARRIED: inherited}
                if inherited and isinstance(entry, dict)
                else entry
            )
    return items


def lineage(item: Any, source: str) -> dict[str, Any]:
    """What a call made from `item` should remember about where it came from.

    One key per execution in the chain, named after the step that produced it,
    so a value stays reachable however many steps run in between:

        ${item.carried.backbones.structure_uri}
        ${item.carried.fold.best.complex_plddt}

    Keying by step rather than flattening is what makes that work. Results
    share field names, `best` and `name` above all, so a flat merge would have
    each hop overwrite the last and a two-hop reach would silently become a
    one-hop one.

    Nothing is declared in the document: a step is handed an item, and what
    that item knows is what its own step and every earlier one produced.
    """
    if not isinstance(item, dict):
        return {source: item}
    inherited = item.get(CARRIED) or {}
    return {
        **inherited,
        source: {k: v for k, v in item.items() if k != CARRIED},
    }


def plan(step: Step, scope: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The (parameters, carried) pairs this step should be invoked with.

    One per element when fanning out, otherwise a single call. This is where a
    step's arity is decided, so it can be reported before anything runs.

    A bare `for_each: <step id>` means that step's emitted items, which the
    runner has already flattened into `<step id>.items`. Anything with a dot is
    an explicit path and resolves unchanged, so `<step id>.results` still means
    one item per call.

    Several references fan out over all of them together, so one step treats
    three branches alike. Each item keeps the source it came from, and the
    narrowing applies across the whole set rather than within each branch,
    which is what makes `limit` mean the best N overall.

    A step that does not fan out carries nothing: there is no item it was made
    from, so there is no lineage to keep.
    """
    if step.for_each is None:
        return [(resolve(step.params, scope), {})]

    # Paired with its source, since the lineage is named after the step that
    # produced the item and a union has more than one.
    paired: list[tuple[Any, str]] = []
    for reference in step.sources():
        source = reference.split(".", 1)[0]
        path = reference if "." in reference else f"{reference}.items"
        found = lookup(path, scope)
        if not isinstance(found, list):
            raise ValueError(
                f"Step {step.id}: for_each must name a list, but "
                f"${{{path}}} is {type(found).__name__}"
            )
        paired.extend((item, source) for item in found)

    # The narrowing may reference workflow inputs, so a threshold can be tuned
    # per run rather than baked into the document. It is resolved against the
    # outer scope only: `item` has no meaning for a rule about the whole set.
    selected = filters.apply(
        paired,
        resolve(step.where, scope),
        resolve(step.order_by, scope),
        resolve(step.limit, scope),
        read=lambda entry: entry[0],
    )
    return [
        (resolve(step.params, {**scope, "item": item}), lineage(item, source))
        for item, source in selected
    ]


def calls_for(step: Step, scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Just the parameter sets, for callers that do not need the lineage."""
    return [params for params, _ in plan(step, scope)]


def reduce_output(workflow: Workflow, results: dict[str, list]) -> Any:
    """Project the final result, so a caller gets a summary not everything."""
    spec = workflow.output
    if not spec:
        return results

    source = spec.get("from")
    if source not in results:
        raise ValueError(
            f"output.from names step {source!r}, which is not in this "
            f"workflow. It has: {', '.join(results) or 'none'}"
        )

    # Narrow first, then project, so an ordering may use a field that `select`
    # does not keep.
    rows = filters.apply(
        results[source], spec.get("where"), spec.get("order_by"), spec.get("limit")
    )
    fields = spec.get("select")
    if fields:
        rows = [
            {f: filters.field(row, f) for f in fields}
            for row in rows
            if isinstance(row, dict)
        ]
    return rows
