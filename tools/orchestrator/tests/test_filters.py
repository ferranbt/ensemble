"""Tests for narrowing a result set before a step fans out.

Filters decide what runs, so the expensive failures here are silent. A clause
that cannot be read must not behave like one that matched nothing, because
"your filter is malformed" and "your campaign found no candidates" look
identical in a result and mean opposite things. And `limit` without an order
would take an arbitrary subset while looking like it took the best.
"""

from __future__ import annotations

import pytest

from tools.orchestrator import filters

# Shaped like real fold results: nested, and not every item has every field.
RESULTS = [
    {"run_id": "a", "best": {"ptm": 0.95, "iptm": 0.62}, "depth": 6259},
    {"run_id": "b", "best": {"ptm": 0.82, "iptm": 0.20}, "depth": 12},
    {"run_id": "c", "best": {"ptm": 0.88, "iptm": 0.71}, "depth": 340},
    {"run_id": "d", "best": {}, "depth": 5},
]


class TestField:
    def test_reads_a_top_level_field(self):
        assert filters.field(RESULTS[0], "depth") == 6259

    def test_reads_a_nested_path(self):
        assert filters.field(RESULTS[0], "best.ptm") == pytest.approx(0.95)

    def test_indexes_into_a_list(self):
        assert filters.field({"models": [{"ptm": 0.5}]}, "models.0.ptm") == 0.5

    def test_missing_is_none_rather_than_an_error(self):
        """Tools return different shapes; an absent field excludes the item."""
        assert filters.field(RESULTS[3], "best.ptm") is None
        assert filters.field(RESULTS[0], "nope.deeper") is None


class TestMatches:
    def test_an_empty_expression_matches_everything(self):
        assert filters.matches(RESULTS[0], "") is True
        assert filters.matches(RESULTS[0], None) is True

    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("depth > 1000", True),
            ("depth < 1000", False),
            ("depth >= 6259", True),
            ("depth != 0", True),
            ("best.ptm > 0.9", True),
            ("best.ptm > 0.99", False),
            ("run_id == 'a'", True),
            ('run_id == "b"', False),
        ],
    )
    def test_comparisons(self, expression, expected):
        assert filters.matches(RESULTS[0], expression) is expected

    def test_clauses_join_with_and(self):
        assert filters.matches(RESULTS[0], "depth > 100 and best.ptm > 0.9")
        assert not filters.matches(RESULTS[0], "depth > 100 and best.ptm > 0.99")

    def test_a_missing_field_excludes_the_item(self):
        assert filters.matches(RESULTS[3], "best.ptm > 0.5") is False

    def test_comparing_incompatible_types_excludes_rather_than_raises(self):
        assert filters.matches(RESULTS[0], "run_id > 5") is False

    @pytest.mark.parametrize(
        "expression",
        ["depth", "depth >", "> 100", "depth ~ 100", "__import__('os')"],
    )
    def test_an_unreadable_clause_raises(self, expression):
        """Never treat a malformed filter as matching nothing."""
        with pytest.raises(ValueError, match="Cannot read filter clause"):
            filters.matches(RESULTS[0], expression)

    def test_the_error_names_the_valid_operators(self):
        with pytest.raises(ValueError, match=">="):
            filters.matches(RESULTS[0], "depth ~ 5")


class TestApply:
    def test_filters(self):
        kept = filters.apply(RESULTS, where="depth > 100")
        assert [r["run_id"] for r in kept] == ["a", "c"]

    def test_orders_descending(self):
        ordered = filters.apply(RESULTS, order_by="depth desc")
        assert [r["run_id"] for r in ordered] == ["a", "c", "b", "d"]

    def test_orders_ascending_by_default(self):
        ordered = filters.apply(RESULTS, order_by="depth")
        assert [r["run_id"] for r in ordered] == ["d", "b", "c", "a"]

    def test_orders_by_a_nested_field(self):
        ordered = filters.apply(RESULTS, order_by="best.ptm desc")
        assert [r["run_id"] for r in ordered][:3] == ["a", "c", "b"]

    def test_items_missing_the_sort_field_go_last(self):
        """Rather than comparing None against a number and failing."""
        ordered = filters.apply(RESULTS, order_by="best.ptm desc")
        assert ordered[-1]["run_id"] == "d"

    def test_sorts_before_limiting(self):
        """This is what makes `limit` mean the best N, not the first N."""
        best = filters.apply(RESULTS, order_by="depth desc", limit=2)
        assert [r["run_id"] for r in best] == ["a", "c"]

    def test_filters_then_orders_then_limits(self):
        result = filters.apply(
            RESULTS, where="depth > 10", order_by="best.ptm desc", limit=2
        )
        assert [r["run_id"] for r in result] == ["a", "c"]

    def test_limit_larger_than_the_set_is_harmless(self):
        assert len(filters.apply(RESULTS, order_by="depth", limit=99)) == 4

    def test_rejects_a_limit_below_one(self):
        with pytest.raises(ValueError, match="at least 1"):
            filters.apply(RESULTS, order_by="depth", limit=0)

    def test_no_narrowing_returns_everything_unchanged(self):
        assert filters.apply(RESULTS) == RESULTS

    def test_does_not_mutate_its_input(self):
        before = [r["run_id"] for r in RESULTS]
        filters.apply(RESULTS, order_by="depth desc")
        assert [r["run_id"] for r in RESULTS] == before
