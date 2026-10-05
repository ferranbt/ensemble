from __future__ import annotations

import pytest

from tools.msa import cache

SEQ = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"


class TestNormalise:
    def test_uppercases_and_strips_whitespace(self):
        assert cache.normalise(" mkt ayi \n") == "MKTAYI"

    def test_strips_internal_newlines(self):
        assert cache.normalise("MKT\nAYI\nGHN") == "MKTAYIGHN"

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError, match="empty"):
            cache.normalise("   ")

    def test_rejects_unsplit_chains(self):
        with pytest.raises(ValueError, match="Split chains"):
            cache.normalise("MKT/GHN")

    def test_accepts_the_ambiguity_codes_a_search_tolerates(self):
        """Unlike design tools, an alignment query may carry B, Z, U, O and X."""
        assert cache.normalise("MKTXBZUO") == "MKTXBZUO"


class TestMode:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "env"),
            ({"use_env": False}, "all"),
            ({"use_filter": False}, "env-nofilter"),
            ({"use_env": False, "use_filter": False}, "nofilter"),
            ({"pair": True}, "pairgreedy-env"),
            ({"pair": True, "use_env": False}, "pairgreedy"),
            ({"pair": True, "pairing_strategy": "complete"}, "paircomplete-env"),
        ],
    )
    def test_builds_the_servers_mode_strings(self, kwargs, expected):
        assert cache.mode(**kwargs) == expected

    def test_rejects_an_unknown_pairing_strategy(self):
        with pytest.raises(ValueError, match="greedy"):
            cache.mode(pair=True, pairing_strategy="partial")


class TestKey:
    def test_is_stable_for_the_same_input(self):
        assert cache.key(SEQ, "env") == cache.key(SEQ, "env")

    def test_ignores_formatting(self):
        wrapped = "\n".join(SEQ[i:i + 20] for i in range(0, len(SEQ), 20))
        assert cache.key(wrapped, "env") == cache.key(SEQ, "env")

    def test_is_case_insensitive(self):
        assert cache.key(SEQ.lower(), "env") == cache.key(SEQ, "env")

    def test_differs_by_mode(self):
        """Two modes give different alignments, so they must not share a key."""
        assert cache.key(SEQ, "env") != cache.key(SEQ, "all")
        assert cache.key(SEQ, "env") != cache.key(SEQ, "env-nofilter")

    def test_differs_by_sequence(self):
        assert cache.key(SEQ, "env") != cache.key(SEQ[:-1], "env")

    def test_carries_the_mode_in_the_name(self):
        """So a cache directory is legible, and one mode can be cleared."""
        assert cache.key(SEQ, "env").startswith("env-")

    def test_is_safe_as_a_filename(self):
        assert "/" not in cache.key(SEQ, "pairgreedy-env")


class TestUnique:
    def test_groups_chains_sharing_a_sequence(self):
        """A homodimer must cost one query, not two."""
        grouped = cache.unique({"A": SEQ, "B": SEQ})
        assert len(grouped) == 1
        assert grouped[SEQ] == ["A", "B"]

    def test_keeps_distinct_sequences_apart(self):
        grouped = cache.unique({"A": SEQ, "B": "MKTAYI"})
        assert len(grouped) == 2

    def test_groups_across_formatting_differences(self):
        grouped = cache.unique({"A": SEQ, "B": SEQ.lower()})
        assert len(grouped) == 1
        assert sorted(grouped[SEQ]) == ["A", "B"]

    def test_preserves_chain_order_within_a_group(self):
        grouped = cache.unique({"C": SEQ, "A": SEQ, "B": SEQ})
        assert grouped[SEQ] == ["C", "A", "B"]

    def test_rejects_a_bad_sequence(self):
        with pytest.raises(ValueError):
            cache.unique({"A": "MKT/GHN"})
