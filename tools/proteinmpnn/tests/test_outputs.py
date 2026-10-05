
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.proteinmpnn import outputs
from tools.proteinmpnn.inputs import ALPHABET

FIXTURES = Path(__file__).parent / "fixtures"

MONOMER_FA = FIXTURES / "seqs" / "6MRR.fa"
COMPLEX_FA = FIXTURES / "seqs" / "3HTN.fa"
SCORES_NPZ = FIXTURES / "score_only" / "3HTN.npz"
PROBS_6MRR = FIXTURES / "unconditional_probs_only" / "6MRR.npz"
PROBS_5L33 = FIXTURES / "unconditional_probs_only" / "5L33.npz"

# Also encoded in the probability archive's `S` array, so the two fixtures
# cross-check each other.
NATIVE_6MRR = (
    "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
)

@pytest.fixture(scope="module")
def monomer():
    return outputs.parse_designs(MONOMER_FA)


@pytest.fixture(scope="module")
def complex_designs():
    return outputs.parse_designs(COMPLEX_FA)


class TestParseDesigns:
    def test_native_comes_first_and_is_flagged(self, monomer):
        assert monomer[0]["is_native"] is True
        assert monomer[0]["name"] == "6MRR"
        assert monomer[0]["sequence"] == NATIVE_6MRR
        assert all(not r["is_native"] for r in monomer[1:])

    def test_native_metrics_are_floats(self, monomer):
        native = monomer[0]
        assert native["score"] == pytest.approx(1.4683)
        assert native["global_score"] == pytest.approx(1.4683)
        assert native["seed"] == 37
        assert isinstance(native["seed"], int)
        assert native["model_name"] == "v_48_020"

    def test_samples_carry_their_own_metrics(self, monomer):
        samples = [r for r in monomer if not r["is_native"]]
        assert len(samples) == 2
        assert [s["sample"] for s in samples] == [1, 2]
        assert samples[0]["score"] == pytest.approx(0.9617)
        assert samples[0]["seq_recovery"] == pytest.approx(0.5000)
        assert samples[0]["T"] == pytest.approx(0.1)

    def test_designs_are_same_length_as_native(self, monomer):
        assert {len(r["sequence"]) for r in monomer} == {len(NATIVE_6MRR)}

    def test_monomer_has_a_single_chain(self, monomer):
        assert all(len(r["chains"]) == 1 for r in monomer)
        assert monomer[0]["designed_chains"] == ["A"]
        assert monomer[0]["fixed_chains"] == []

    def test_chain_lists_survive_their_internal_commas(self, complex_designs):
        """3HTN's header reads `designed_chains=['A', 'B']`; splitting naively on
        "," truncates it to "['A'" and shifts every later field."""
        native = complex_designs[0]
        assert native["designed_chains"] == ["A", "B"]
        assert native["fixed_chains"] == ["C"]

    def test_fields_after_a_chain_list_are_still_read(self, complex_designs):
        native = complex_designs[0]
        assert native["model_name"] == "v_48_020"
        assert native["seed"] == 37
        assert native["git_hash"] == "be1d37b6699dcd2283ab5b6fc8cc88774e2c80e9"

    def test_complex_splits_into_chains_on_slash(self, complex_designs):
        native = complex_designs[0]
        assert len(native["chains"]) == 2
        assert "/" not in native["chains"][0]
        assert native["sequence"] == "/".join(native["chains"])

    def test_unresolved_residues_are_preserved(self, complex_designs):
        # 3HTN has unresolved residues written as X.
        assert "X" in complex_designs[0]["sequence"]

    def test_every_record_has_a_sequence(self, monomer, complex_designs):
        for record in [*monomer, *complex_designs]:
            assert record["sequence"]
            assert set(record["sequence"]) <= set(ALPHABET + "/")


class TestReadScores:
    def test_averages_over_decoding_orders(self):
        result = outputs.read_scores(SCORES_NPZ)
        assert result["num_decoding_orders"] == 10
        assert result["score"] == pytest.approx(1.152583, abs=1e-6)
        assert result["global_score"] == pytest.approx(1.199445, abs=1e-6)


    def test_spread_across_decoding_orders_is_reported(self):
        result = outputs.read_scores(SCORES_NPZ)
        assert result["score_std"] > 0
        assert result["global_score_std"] > 0

    def test_sequence_is_none_when_the_archive_predates_it(self):
        # Older archives carry only the two score arrays.
        assert outputs.read_scores(SCORES_NPZ)["sequence"] is None


class TestReadProbabilities:
    def test_one_entry_per_designable_position(self):
        result = outputs.read_probabilities(PROBS_6MRR)
        assert len(result["positions"]) == len(NATIVE_6MRR)
        assert [p["position"] for p in result["positions"]] == list(
            range(len(NATIVE_6MRR))
        )

    def test_native_residues_decode_to_the_fasta_sequence(self):
        """Confirms ALPHABET's ordering: a wrong one would decode `S` into
        something other than the sequence in the designs FASTA."""
        result = outputs.read_probabilities(PROBS_6MRR)
        decoded = "".join(p["native_aa"] for p in result["positions"])
        assert decoded == NATIVE_6MRR

    def test_alphabet_is_reported(self):
        assert outputs.read_probabilities(PROBS_6MRR)["alphabet"] == ALPHABET
        assert len(ALPHABET) == 21

    def test_top_k_is_sorted_and_bounded(self):
        result = outputs.read_probabilities(PROBS_6MRR, top_k=4)
        for position in result["positions"]:
            probabilities = [t["probability"] for t in position["top"]]
            assert len(probabilities) == 4
            assert probabilities == sorted(probabilities, reverse=True)
            assert all(0.0 <= p <= 1.0 for p in probabilities)

    def test_top_k_is_clamped_to_the_alphabet(self):
        result = outputs.read_probabilities(PROBS_6MRR, top_k=999)
        assert len(result["positions"][0]["top"]) == len(ALPHABET)

    def test_full_matrix_is_a_normalised_distribution(self):
        result = outputs.read_probabilities(PROBS_6MRR, include_matrix=True)
        for position in result["positions"]:
            matrix = position["probabilities"]
            assert set(matrix) == set(ALPHABET)
            assert sum(matrix.values()) == pytest.approx(1.0, abs=1e-6)

    def test_matrix_is_omitted_by_default(self):
        result = outputs.read_probabilities(PROBS_6MRR)
        assert "probabilities" not in result["positions"][0]

    def test_matrix_agrees_with_the_reported_top(self):
        result = outputs.read_probabilities(PROBS_6MRR, top_k=1, include_matrix=True)
        for position in result["positions"]:
            best = max(position["probabilities"].items(), key=lambda kv: kv[1])
            assert position["top"][0]["aa"] == best[0]
            assert position["top"][0]["probability"] == pytest.approx(best[1])

    def test_entropy_is_within_bounds(self):
        result = outputs.read_probabilities(PROBS_6MRR)
        for position in result["positions"]:
            assert 0.0 <= position["entropy_bits"] <= math.log2(len(ALPHABET))

    def test_confident_positions_have_lower_entropy(self):
        result = outputs.read_probabilities(PROBS_6MRR, top_k=1)
        ranked = sorted(result["positions"], key=lambda p: p["entropy_bits"])
        assert ranked[0]["top"][0]["probability"] > ranked[-1]["top"][0]["probability"]

    def test_second_structure_parses_at_its_own_length(self):
        """5L33 is a different length, so nothing may be hard-coded to 6MRR."""
        result = outputs.read_probabilities(PROBS_5L33)
        assert len(result["positions"]) == 106
