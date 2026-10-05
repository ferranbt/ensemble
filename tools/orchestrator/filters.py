"""Narrowing a set of results before a step fans out over it.

Three things a step or an output can declare:

    where:    "best.ptm > 0.9 and alignments.A.num_sequences > 100"
    order_by: "best.ptm desc"
    limit:    10

Applied to the upstream results *before* any call is made, which is what makes
a funnel possible: fold two hundred backbones, keep the confident ones, design
sequences for only those. Work that is filtered out is never started.

`limit` matters as much as `where`. Without it you can filter but not say "take
the best fifty", so a sweep multiplies against an unbounded set and a cross
product quietly becomes twenty thousand calls.

**Expressions are not Python.** A comparison is `<path> <operator> <value>`,
joined by `and`, and nothing else parses. An agent writes these, so the
evaluator must not be able to execute anything, and a malformed one must fail
with a message rather than silently matching nothing.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# One comparison. The path is dotted, the value is a number, a quoted string,
# or a bare word like true.
COMPARISON = re.compile(
    r"^\s*([A-Za-z_][\w.]*)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$"
)
CONJUNCTION = re.compile(r"\s+and\s+", re.IGNORECASE)

OPERATORS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _literal(text: str) -> Any:
    """Read the right-hand side of a comparison."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def field(item: Any, path: str) -> Any:
    """Read a dotted path out of a result, or None if it is not there.

    Missing is None rather than an error, because results from different tools
    have different shapes and a filter that names an absent field should
    exclude the item, not end the run.
    """
    value = item
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def matches(item: Any, expression: str) -> bool:
    """Whether one result satisfies a `where` expression.

    Raises:
        ValueError: for an expression that does not parse, naming the clause.
            A filter that cannot be read must not be treated as matching
            nothing, since that looks identical to a campaign that found no
            candidates.
    """
    if not expression or not expression.strip():
        return True

    for clause in CONJUNCTION.split(expression.strip()):
        parsed = COMPARISON.match(clause)
        if not parsed:
            raise ValueError(
                f"Cannot read filter clause {clause!r}. Expected "
                f"'<field> <operator> <value>', where the operator is one of "
                f"{', '.join(OPERATORS)}, and clauses joined by 'and'."
            )
        path, operator, raw = parsed.groups()
        left, right = field(item, path), _literal(raw)

        if left is None:
            return False
        try:
            if not OPERATORS[operator](left, right):
                return False
        except TypeError:
            # Comparing a string to a number, say. Excluded rather than fatal.
            return False
    return True


def apply(
    items: list[Any],
    where: str | None = None,
    order_by: str | None = None,
    limit: int | None = None,
    read: Callable[[Any], Any] = lambda entry: entry,
) -> list[Any]:
    """Narrow a list of results, in the order the three are declared.

    Filter, then sort, then take. Sorting before limiting is what makes
    `limit` mean "the best N" rather than "the first N".

    Args:
        read: Where to find the result inside each entry, for a caller that
            pairs it with something else. A step fanning out over several
            sources carries each item's source alongside it, and the narrowing
            still has to see the item.
    """
    selected = (
        [i for i in items if matches(read(i), where)] if where else list(items)
    )

    if order_by:
        parts = order_by.strip().split()
        path = parts[0]
        descending = len(parts) > 1 and parts[1].lower() in ("desc", "descending")
        # Items missing the sort field go last either way, rather than
        # comparing None against a number and failing.
        present = [i for i in selected if field(read(i), path) is not None]
        absent = [i for i in selected if field(read(i), path) is None]
        present.sort(key=lambda i: field(read(i), path), reverse=descending)
        selected = present + absent

    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        selected = selected[:limit]

    return selected
