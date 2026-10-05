from __future__ import annotations

import pytest

from tools.variants import consensus

#              1234567890123456789012345
SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEE"


def thermompnn(*entries, offset: int = 0):
    """ThermoMPNN `positions`, numbered as a structure would be."""
    return [
        {
            "position": position + offset,
            "wild_type": SEQUENCE[position - 1],
            "top": [{"mutation": "x", "mutant": mutant, "ddG": ddg}],
        }
        for position, mutant, ddg in entries
    ]


def esm(*entries):
    """ESM `positions`, 1-indexed."""
    return [
        {
            "position": position,
            "wild_type": SEQUENCE[position - 1],
            "top": [{"mutation": "x", "aa": mutant, "log_likelihood_ratio": ratio}],
        }
        for position, mutant, ratio in entries
    ]


def proteinmpnn(*entries):
    """ProteinMPNN `positions`, 0-indexed, wild type under a different key."""
    return [
        {
            "position": position - 1,
            "native_aa": SEQUENCE[position - 1],
            "top": [{"aa": mutant, "probability": probability}],
        }
        for position, mutant, probability in entries
    ]


def read(positions, type_name, **scan):
    return consensus.read(
        {"type": type_name, "positions": positions, **scan}, SEQUENCE
    )


class TestNumbering:
    def test_esm_is_already_aligned(self):
        scored = read(esm((4, "E", 1.0)), "likelihood")
        assert str(scored[0].mutation) == "A4E"

    def test_proteinmpnn_zero_indexing_is_shifted_back(self):
        scored = read(proteinmpnn((4, "E", 0.4)), "structural")
        assert str(scored[0].mutation) == "A4E"

    def test_structure_numbering_is_recovered(self):
        """A PDB numbered from 43 must still land on the right residues."""
        scored = read(thermompnn((4, "E", -1.0), offset=42), "stability")
        assert str(scored[0].mutation) == "A4E"

    def test_all_three_conventions_agree_on_one_substitution(self):
        s = read(thermompnn((4, "E", -1.0), offset=42), "stability")
        l = read(esm((4, "E", 1.0)), "likelihood")
        p = read(proteinmpnn((4, "E", 0.4)), "structural")
        assert {str(x[0].mutation) for x in (s, l, p)} == {"A4E"}

    def test_a_scan_of_another_sequence_is_refused(self):
        wrong = [{
            "position": 4,
            "wild_type": "W",  # not what the sequence has anywhere near here
            "top": [{"aa": "E", "log_likelihood_ratio": 1.0}],
        }]
        with pytest.raises(ValueError, match="different sequence"):
            read(wrong, "likelihood")

    def test_a_partially_matching_scan_is_refused(self):
        """One offset must explain every entry, not most of them."""
        mixed = esm((4, "E", 1.0)) + [{
            "position": 9,
            "wild_type": "C",  # sequence has Q at 9
            "top": [{"aa": "E", "log_likelihood_ratio": 1.0}],
        }]
        with pytest.raises(ValueError, match="No numbering offset"):
            read(mixed, "likelihood")

    def test_the_refusal_names_the_caller_s_label(self):
        """The label is the workflow's, so no tool name is spelled in here."""
        wrong = [{
            "position": 4,
            "wild_type": "W",
            "top": [{"aa": "E", "log_likelihood_ratio": 1.0}],
        }]
        with pytest.raises(ValueError, match="esm2's scan"):
            read(wrong, "likelihood", label="esm2")

    def test_empty_is_empty_not_an_error(self):
        assert read([], "likelihood") == []


class TestReaders:
    def test_an_unknown_type_is_refused(self):
        with pytest.raises(ValueError, match="Unknown scan type"):
            read(esm((4, "E", 1.0)), "vibes")

    def test_the_refusal_lists_what_is_available(self):
        with pytest.raises(ValueError, match="likelihood, stability, structural"):
            read(esm((4, "E", 1.0)), "vibes")

    def test_columns_can_be_renamed_per_scan(self):
        """A second tool answering the same question under other column names."""
        positions = [{
            "position": 4,
            "wt": "A",
            "top": [{"to": "E", "delta_g": -1.0}],
        }]
        scored = read(
            positions,
            "stability",
            keys={"wild_type": "wt", "mutant": "to", "score": "delta_g"},
        )
        assert str(scored[0].mutation) == "A4E"
        assert scored[0].score == 1.0

    def test_direction_can_be_overridden_per_scan(self):
        (scored,) = read(
            thermompnn((4, "E", -1.5)), "stability", higher_is_better=True
        )
        assert scored.score == -1.5

    def test_an_unknown_key_override_is_refused(self):
        with pytest.raises(ValueError, match="Unknown key override"):
            read(esm((4, "E", 1.0)), "likelihood", keys={"position": "idx"})


class TestSources:
    def test_labels_come_from_the_caller(self):
        found = consensus.sources(
            [
                {"label": "esm2", "type": "likelihood",
                 "positions": esm((4, "E", 1.0))},
                {"label": "thermompnn", "type": "stability",
                 "positions": thermompnn((4, "E", -1.0))},
            ],
            SEQUENCE,
        )
        assert sorted(found) == ["esm2", "thermompnn"]

    def test_the_label_defaults_to_the_type(self):
        found = consensus.sources(
            [{"type": "likelihood", "positions": esm((4, "E", 1.0))}], SEQUENCE
        )
        assert list(found) == ["likelihood"]

    def test_two_scans_of_one_type_are_separate_sources(self):
        """Two stability predictors can both count towards agreement."""
        found = consensus.sources(
            [
                {"label": "thermompnn", "type": "stability",
                 "positions": thermompnn((4, "E", -1.0))},
                {"label": "other_ddg", "type": "stability",
                 "positions": thermompnn((4, "E", -2.0))},
            ],
            SEQUENCE,
        )
        assert sorted(found) == ["other_ddg", "thermompnn"]

    def test_a_repeated_label_is_refused(self):
        """Otherwise one source would silently count as two."""
        scans = [
            {"label": "esm2", "type": "likelihood",
             "positions": esm((4, "E", 1.0))},
            {"label": "esm2", "type": "likelihood",
             "positions": esm((9, "A", 1.0))},
        ]
        with pytest.raises(ValueError, match="both labelled"):
            consensus.sources(scans, SEQUENCE)

    def test_a_scan_that_found_nothing_is_not_a_source(self):
        found = consensus.sources(
            [
                {"label": "esm2", "type": "likelihood",
                 "positions": esm((4, "E", 1.0))},
                {"label": "empty", "type": "structural", "positions": []},
            ],
            SEQUENCE,
        )
        assert list(found) == ["esm2"]

    def test_nothing_at_all_is_no_sources(self):
        assert consensus.sources([], SEQUENCE) == {}


class TestOrientation:
    """Every source must end up with higher meaning better, or rank is wrong."""

    def test_stability_score_is_flipped(self):
        """ThermoMPNN's positive ddG destabilises, so the sign has to turn."""
        (scored,) = read(thermompnn((4, "E", -1.5)), "stability")
        assert scored.score == 1.5

    def test_stabilising_outranks_destabilising(self):
        scored = read(thermompnn((4, "E", 2.5), (17, "D", -1.0)), "stability")
        best = max(scored, key=lambda s: s.score)
        assert str(best.mutation) == "S17D"

    def test_nothing_is_cut_on_value(self):
        scored = read(thermompnn((4, "E", -1.0), (17, "D", 2.5)), "stability")
        assert len(scored) == 2

    def test_a_substitution_to_the_wild_type_is_not_a_mutation(self):
        native = SEQUENCE[3]
        assert read(esm((4, native, 2.0)), "likelihood") == []


class TestAgree:
    def scored(self, source_entries):
        return {
            "stability": read(
                thermompnn(*source_entries.get("s", [])), "stability"
            ),
            "likelihood": read(esm(*source_entries.get("l", [])), "likelihood"),
        }

    def test_only_what_both_like_survives(self):
        sources = self.scored({
            "s": [(4, "E", -1.0), (9, "A", -1.0)],
            "l": [(4, "E", 1.0), (17, "D", 1.0)],
        })
        agreed = consensus.agree(sources, min_sources=2)
        assert [m["mutation"] for m in agreed] == ["A4E"]

    def test_records_which_sources_agreed(self):
        sources = self.scored({"s": [(4, "E", -1.0)], "l": [(4, "E", 1.0)]})
        (entry,) = consensus.agree(sources, min_sources=2)
        assert entry["sources"] == ["likelihood", "stability"]
        assert set(entry["scores"]) == {"likelihood", "stability"}

    def test_one_source_is_enough_when_asked(self):
        sources = self.scored({"s": [(4, "E", -1.0)], "l": []})
        assert len(consensus.agree(sources, min_sources=1)) == 1

    def test_ranked_by_mean_rank_not_raw_score(self):
        """Scores are in different units; adding them would weight the widest."""
        sources = self.scored({
            "s": [(4, "E", -0.1), (17, "D", -9.0)],
            "l": [(4, "E", 9.0), (17, "D", 0.1)],
        })
        agreed = consensus.agree(sources, min_sources=2)
        # Each is best in one source and worst in the other, so they tie on
        # mean rank and fall back to position order.
        assert [m["mutation"] for m in agreed] == ["A4E", "S17D"]

    def test_nothing_agreed(self):
        sources = self.scored({"s": [(4, "E", -1.0)], "l": [(17, "D", 1.0)]})
        assert consensus.agree(sources, min_sources=2) == []

    def test_shortlist_excludes_what_a_source_ranks_poorly(self):
        """Agreement means both rank it highly, not merely that both saw it."""
        sources = self.scored({
            "s": [(4, "E", -9.0), (17, "D", -0.1)],
            "l": [(4, "E", 9.0), (17, "D", 0.1)],
        })
        both = consensus.agree(sources, min_sources=2, top_per_source=2)
        assert len(both) == 2
        top = consensus.agree(sources, min_sources=2, top_per_source=1)
        assert [m["mutation"] for m in top] == ["A4E"]


class TestSpread:
    def entry(self, position, rank=0.0):
        return {"mutation": f"X{position}Y", "position": position,
                "mean_rank": rank}

    def test_close_mutations_are_thinned(self):
        """Neighbouring substitutions interact, so stacking them is not additive."""
        chosen = consensus.spread(
            [self.entry(10), self.entry(12), self.entry(30)], separation=8
        )
        assert [e["position"] for e in chosen] == [10, 30]

    def test_the_better_ranked_of_a_close_pair_is_kept(self):
        chosen = consensus.spread([self.entry(10), self.entry(12)], separation=8)
        assert [e["position"] for e in chosen] == [10]

    def test_exactly_at_the_separation_is_allowed(self):
        chosen = consensus.spread([self.entry(10), self.entry(18)], separation=8)
        assert len(chosen) == 2

    def test_one_per_position(self):
        chosen = consensus.spread([self.entry(10), self.entry(10)], separation=8)
        assert len(chosen) == 1
