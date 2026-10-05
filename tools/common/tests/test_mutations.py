
from __future__ import annotations

import pytest

from tools.common import mutations as mut

# The 6MRR sequence, reused from the ProteinMPNN fixtures.
SEQUENCE = (
    "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
)


class TestParse:
    def test_reads_a_single_mutation(self):
        assert mut.parse("W2A") == [mut.Mutation("W", 2, "A")]

    def test_reads_several_joined_by_colon(self):
        assert [str(m) for m in mut.parse("W2A:T4K")] == ["W2A", "T4K"]

    @pytest.mark.parametrize("text", ["W2A,T4K", "W2A;T4K", "W2A T4K", "W2A+T4K"])
    def test_accepts_the_common_separators(self, text):
        assert [str(m) for m in mut.parse(text)] == ["W2A", "T4K"]

    def test_is_case_insensitive(self):
        assert mut.parse("w2a") == [mut.Mutation("W", 2, "A")]

    def test_tolerates_surrounding_whitespace(self):
        assert [str(m) for m in mut.parse("  W2A : T4K ")] == ["W2A", "T4K"]

    def test_round_trips_through_str(self):
        assert str(mut.parse("K19R")[0]) == "K19R"

    def test_position_is_one_indexed(self):
        """`index` is the offset into a Python string, one less than written."""
        mutation = mut.parse("W2A")[0]
        assert mutation.position == 2
        assert mutation.index == 1
        assert SEQUENCE[mutation.index] == "W"

    @pytest.mark.parametrize(
        "text, reason",
        [
            ("X2A", "non-standard wild type"),
            ("W2B", "non-standard replacement"),
            ("2A", "missing wild type"),
            ("W2", "missing replacement"),
            ("WA2", "letters and digits transposed"),
            ("W0A", "position below one"),
            ("", "empty"),
            ("   ", "whitespace only"),
        ],
    )
    def test_rejects_malformed_input(self, text, reason):
        with pytest.raises(ValueError):
            mut.parse(text)

    def test_rejects_the_same_position_twice(self):
        """Two changes at one position are contradictory, not additive."""
        with pytest.raises(ValueError, match="more than once"):
            mut.parse("W2A:W2K")

    def test_error_names_the_offending_token(self):
        with pytest.raises(ValueError, match="ZZZ"):
            mut.parse("ZZZ")


class TestValidate:
    def test_accepts_mutations_matching_the_sequence(self):
        mut.validate(mut.parse("W2A:T4K"), SEQUENCE)

    def test_rejects_a_wrong_wild_type_residue(self):
        """The most valuable check: catches offset numbering or a wrong protein."""
        with pytest.raises(ValueError, match="but the sequence has"):
            mut.validate(mut.parse("A2K"), SEQUENCE)

    def test_the_mismatch_error_is_actionable(self):
        with pytest.raises(ValueError) as excinfo:
            mut.validate(mut.parse("A2K"), SEQUENCE)
        message = str(excinfo.value)
        assert "numbering may be offset" in message
        assert "W" in message and "A" in message

    def test_rejects_a_position_past_the_end(self):
        with pytest.raises(ValueError, match="beyond the sequence"):
            mut.validate(mut.parse("G999A"), SEQUENCE)

    def test_accepts_the_final_position(self):
        last = len(SEQUENCE)
        mut.validate(mut.parse(f"{SEQUENCE[-1]}{last}A"), SEQUENCE)


class TestApply:
    def test_applies_one_mutation(self):
        assert mut.apply(SEQUENCE, mut.parse("W2A"))[:3] == "GAS"

    def test_applies_several(self):
        result = mut.apply(SEQUENCE, mut.parse("W2A:T4K"))
        assert result[1] == "A"
        assert result[3] == "K"

    def test_changes_only_the_named_positions(self):
        result = mut.apply(SEQUENCE, mut.parse("W2A"))
        differing = [i for i, (a, b) in enumerate(zip(SEQUENCE, result)) if a != b]
        assert differing == [1]

    def test_preserves_length(self):
        assert len(mut.apply(SEQUENCE, mut.parse("W2A:T4K"))) == len(SEQUENCE)

    def test_leaves_the_original_untouched(self):
        before = SEQUENCE
        mut.apply(SEQUENCE, mut.parse("W2A"))
        assert SEQUENCE == before

    def test_validates_before_applying(self):
        with pytest.raises(ValueError):
            mut.apply(SEQUENCE, mut.parse("A2K"))


class TestCleanSequence:
    def test_uppercases_and_strips_whitespace(self):
        assert mut.clean_sequence(" mkt ayi \n") == "MKTAYI"

    def test_accepts_every_standard_residue(self):
        assert mut.clean_sequence(mut.RESIDUES) == mut.RESIDUES

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError, match="empty"):
            mut.clean_sequence("   ")

    def test_rejects_unsplit_chains(self):
        """ProteinMPNN joins chains with '/', which must be split first."""
        with pytest.raises(ValueError, match="Split chains"):
            mut.clean_sequence("MKT/GHN")

    @pytest.mark.parametrize("sequence", ["MKTX", "MKT-", "MKT*"])
    def test_rejects_non_standard_characters(self, sequence):
        with pytest.raises(ValueError):
            mut.clean_sequence(sequence)

    def test_there_are_twenty_residues(self):
        assert len(mut.RESIDUES) == 20
        assert len(set(mut.RESIDUES)) == 20
