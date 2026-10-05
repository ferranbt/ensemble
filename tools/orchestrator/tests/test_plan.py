"""Tests for how a step's calls are planned.

Two features are under test, both there to make one workflow expressible: a
generator produces several designs from one call, and a self-consistency check
needs the backbone a design came from after two steps have run in between.

`emits` is the flattening: the producing step declares its unit once, so a
consumer writes `for_each: backbones` rather than a path repeated at each use.
The lineage is the threading: a call remembers the item it was made from, and
that item remembers its own, so nothing has to be declared to stay reachable.

What matters most here is that neither changes the existing meaning. Documents
already written say `for_each: fold.results` and must keep working.
"""

from __future__ import annotations

import pytest

from tools.orchestrator import document


def step(**kwargs) -> document.Step:
    kwargs.setdefault("id", "s")
    kwargs.setdefault("tool", "app/fn")
    return document.Step(**kwargs)


def scope_with(step_id: str, outcomes: list, emits: str | None = None) -> dict:
    """A scope holding one prior step's results, as the runner builds it."""
    producer = step(id=step_id, emits=emits)
    return {
        step_id: {
            "results": outcomes,
            "items": document.emitted(producer, outcomes),
        }
    }


TWO_CALLS = [
    {"run_id": "r0", "designs": [{"design": 0}, {"design": 1}]},
    {"run_id": "r1", "designs": [{"design": 2}]},
]


class TestEmitted:
    def test_without_emits_one_item_per_call(self):
        assert document.emitted(step(), TWO_CALLS) == TWO_CALLS

    def test_flattens_across_calls(self):
        """The whole point: 2 designs from one call, 1 from another, 3 items."""
        items = document.emitted(step(emits="designs"), TWO_CALLS)
        assert [i["design"] for i in items] == [0, 1, 2]

    def test_a_result_missing_the_field_contributes_nothing(self):
        """One odd result must not end a run whose other calls succeeded."""
        items = document.emitted(step(emits="designs"), [{"run_id": "r"}] + TWO_CALLS)
        assert len(items) == 3

    def test_a_non_list_field_is_one_item(self):
        items = document.emitted(step(emits="best"), [{"best": {"design": 7}}])
        assert items == [{"design": 7}]

    def test_unknown_survives_for_a_dry_run(self):
        placeholders = [document.Unknown()]
        assert len(document.emitted(step(emits="designs"), placeholders)) == 1


class TestForEachResolution:
    def test_bare_step_id_means_its_emitted_items(self):
        scope = scope_with("backbones", TWO_CALLS, emits="designs")
        calls = document.calls_for(
            step(for_each="backbones", params={"n": "${item.design}"}), scope
        )
        assert [c["n"] for c in calls] == [0, 1, 2]

    def test_bare_step_id_without_emits_is_one_per_call(self):
        scope = scope_with("fold", TWO_CALLS)
        calls = document.calls_for(
            step(for_each="fold", params={"r": "${item.run_id}"}), scope
        )
        assert [c["r"] for c in calls] == ["r0", "r1"]

    def test_explicit_results_path_still_means_per_call(self):
        """`for_each: fold.results` is in documents already written."""
        scope = scope_with("fold", TWO_CALLS, emits="designs")
        calls = document.calls_for(
            step(for_each="fold.results", params={"r": "${item.run_id}"}), scope
        )
        assert [c["r"] for c in calls] == ["r0", "r1"]

    def test_deep_path_still_resolves(self):
        scope = scope_with("backbones", TWO_CALLS)
        calls = document.calls_for(
            step(for_each="backbones.results.0.designs",
                 params={"n": "${item.design}"}),
            scope,
        )
        assert [c["n"] for c in calls] == [0, 1]

    def test_a_non_list_is_refused(self):
        scope = {"x": {"items": {"not": "a list"}}}
        with pytest.raises(ValueError, match="must name a list"):
            document.calls_for(step(for_each="x"), scope)


class TestSeveralSources:
    """One step over several branches, so three predictors share one scorer."""

    def scope(self) -> dict:
        return {
            "boltz": {"results": [{"id": "b", "score": 0.9}], "items": []},
            "protenix": {"results": [{"id": "p", "score": 0.5}], "items": []},
        }

    def test_fans_out_over_all_of_them(self):
        calls = document.calls_for(
            step(for_each=["boltz.results", "protenix.results"],
                 params={"i": "${item.id}"}),
            self.scope(),
        )
        assert [c["i"] for c in calls] == ["b", "p"]

    def test_each_item_keeps_the_branch_it_came_from(self):
        planned = document.plan(
            step(for_each=["boltz.results", "protenix.results"]), self.scope()
        )
        assert [sorted(carried) for _, carried in planned] == [
            ["boltz"], ["protenix"]
        ]

    def test_narrowing_applies_across_the_union(self):
        """`limit` means the best N overall, not the best N per branch."""
        planned = document.plan(
            step(for_each=["boltz.results", "protenix.results"],
                 params={"i": "${item.id}"},
                 order_by="score asc", limit=1),
            self.scope(),
        )
        assert [params["i"] for params, _ in planned] == ["p"]

    def test_where_reads_the_item_not_the_pairing(self):
        calls = document.calls_for(
            step(for_each=["boltz.results", "protenix.results"],
                 params={"i": "${item.id}"}, where="score > 0.7"),
            self.scope(),
        )
        assert [c["i"] for c in calls] == ["b"]

    def test_one_reference_still_behaves_as_before(self):
        scope = scope_with("fold", TWO_CALLS)
        assert document.calls_for(
            step(for_each="fold", params={"r": "${item.run_id}"}), scope
        ) == [{"r": "r0"}, {"r": "r1"}]

    def test_an_unknown_branch_is_caught_at_parse_time(self):
        body = """
name: w
steps:
  - id: first
    tool: app/one
  - id: second
    tool: app/two
    for_each: [first, thrid]
"""
        with pytest.raises(ValueError, match="not a step that runs before it"):
            document.parse(body)


class TestUnknownStepIsRejectedAtParseTime:
    """A mistyped step name must not cost the previous step's GPU time."""

    def document_with(self, for_each: str) -> str:
        return f"""
name: w
steps:
  - id: first
    tool: app/one
  - id: second
    tool: app/two
    for_each: {for_each}
"""

    def test_unknown_step_named(self):
        with pytest.raises(ValueError, match="not a step that runs before it"):
            document.parse(self.document_with("frist"))

    def test_error_lists_what_is_available(self):
        with pytest.raises(ValueError, match="first"):
            document.parse(self.document_with("frist"))

    def test_a_later_step_does_not_count(self):
        with pytest.raises(ValueError, match="not a step that runs before it"):
            document.parse(self.document_with("second"))

    def test_a_real_earlier_step_is_accepted(self):
        assert len(document.parse(self.document_with("first")).steps) == 2

    def test_inputs_and_dotted_paths_are_left_alone(self):
        parsed = document.parse(self.document_with("first.results"))
        assert parsed.steps[1].for_each == "first.results"


class TestLineage:
    def test_the_item_is_kept_under_the_step_that_produced_it(self):
        scope = scope_with("fold", TWO_CALLS)
        planned = document.plan(step(for_each="fold"), scope)
        assert planned[0][1] == {"fold": TWO_CALLS[0]}

    def test_nothing_is_carried_without_a_fan_out(self):
        """There is no item the call was made from, so there is no lineage."""
        assert document.plan(step(params={"a": 1}), {}) == [({"a": 1}, {})]

    def test_earlier_hops_are_kept_alongside(self):
        """How the backbone reaches a step two hops later, undeclared."""
        folded = [{
            "run_id": "r",
            "best": {"structure_uri": "s3://b/refold.pdb"},
            "carried": {"backbones": {"structure_uri": "s3://b/backbone.pdb"}},
        }]
        scope = scope_with("fold", folded)
        (_, carried), = document.plan(step(for_each="fold"), scope)
        assert carried["backbones"]["structure_uri"] == "s3://b/backbone.pdb"
        assert carried["fold"]["best"]["structure_uri"] == "s3://b/refold.pdb"

    def test_a_step_is_reachable_however_many_hops_back(self):
        """Keyed by step, so `best` at one hop cannot bury `best` at two.

        Flattening was the alternative and this is what it would break: every
        predictor and every scorer calls its top result `best`.
        """
        scored = [{
            "best": {"tm_score": 0.97},
            "carried": {
                "fold": {"best": {"structure_uri": "s3://b/refold.pdb"}},
                "backbones": {"structure_uri": "s3://b/backbone.pdb"},
            },
        }]
        (_, carried), = document.plan(
            step(for_each="selfconsistency"),
            scope_with("selfconsistency", scored),
        )
        assert carried["selfconsistency"]["best"]["tm_score"] == 0.97
        assert carried["fold"]["best"]["structure_uri"] == "s3://b/refold.pdb"

    def test_named_after_the_step_not_the_path(self):
        scope = scope_with("fold", TWO_CALLS)
        (_, carried), = document.plan(step(for_each="fold.results"), scope)[:1]
        assert set(carried) == {"fold"}

    def test_a_plain_value_is_kept_as_it_is(self):
        scope = {"inputs": {"temperatures": [0.1, 0.3]}}
        planned = document.plan(step(for_each="inputs.temperatures"), scope)
        assert [carried for _, carried in planned] == [
            {"inputs": 0.1}, {"inputs": 0.3}
        ]

    def test_an_emitted_item_inherits_its_calls_lineage(self):
        """A design out of a result has no memory of what it was designed for."""
        produced = [{
            "designs": [{"design": 0}, {"design": 1}],
            "carried": {"backbones": {"structure_uri": "s3://b/backbone.pdb"}},
        }]
        items = document.emitted(step(emits="designs"), produced)
        assert all(
            i["carried"]["backbones"]["structure_uri"] == "s3://b/backbone.pdb"
            for i in items
        )

    def test_lineage_does_not_nest_itself(self):
        """The item's own `carried` is merged, not stored inside the new one."""
        item = {"x": 1, "carried": {"first": {"y": 2}}}
        assert document.lineage(item, "second") == {
            "first": {"y": 2},
            "second": {"x": 1},
        }


class TestCarriedCannotClobber:
    """`carried` is one nested key precisely so a tool's fields are untouchable."""

    def test_merge_keeps_the_tools_fields(self):
        outcome = {"run_id": "r", "rmsd": 1.2}
        carried = {"selfconsistency": {"rmsd": 99.0}}
        merged = {**outcome, document.CARRIED: carried}
        assert merged["rmsd"] == 1.2
        assert merged["carried"]["selfconsistency"]["rmsd"] == 99.0


class TestCallsForStillWorks:
    def test_returns_only_parameters(self):
        scope = scope_with("fold", TWO_CALLS)
        calls = document.calls_for(
            step(for_each="fold", params={"r": "${item.run_id}"}), scope
        )
        assert calls == [{"r": "r0"}, {"r": "r1"}]
