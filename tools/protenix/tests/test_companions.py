# What is pinned here is the shape a consumer sees, so that a stage can swap
# Boltz for Protenix unchanged. Fixture keys come from Protenix's
# `sample_confidence.py`; the values are chosen to be checkable by hand.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.protenix import companions, outputs


def full_data(**overrides) -> dict:
    """Protenix's per-atom confidence dump for a two-token, four-atom job."""
    data = {
        # Two tokens, so a 2x2 aligned error matrix.
        "token_pair_pae": [[0.5, 12.0], [12.0, 0.5]],
        # Four atoms: the first two belong to token 0, the last two to token 1.
        "atom_plddt": [0.90, 0.80, 0.60, 0.40],
        "atom_to_token_idx": [0, 0, 1, 1],
    }
    data.update(overrides)
    return data


def summary(**overrides) -> dict:
    data = {
        "plddt": 67.5,
        "ptm": 0.88,
        "iptm": 0.91,
        "ranking_score": 0.89,
        "chain_pair_iptm": [[0.95, 0.91], [0.91, 0.93]],
    }
    data.update(overrides)
    return data


class TestTokenPlddt:
    def test_averages_the_atoms_of_each_token(self):
        result = companions.token_plddt([0.90, 0.80, 0.60, 0.40], [0, 0, 1, 1])
        assert list(result) == pytest.approx([0.85, 0.50])

    def test_uneven_atom_counts_per_token(self):
        """Residues have different atom counts, so the grouping must weight."""
        result = companions.token_plddt([1.0, 0.0, 0.0, 0.6], [0, 0, 0, 1])
        assert list(result) == pytest.approx([1.0 / 3.0, 0.6])

    def test_one_value_per_token_not_per_atom(self):
        result = companions.token_plddt([0.9, 0.8, 0.6, 0.4], [0, 0, 1, 1])
        assert len(result) == 2

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="index the same atoms"):
            companions.token_plddt([0.9, 0.8, 0.6], [0, 0, 1, 1])


class TestPairChainsIptm:
    def test_matrix_becomes_nested_string_indices(self):
        """Boltz's container, which is what the scoring script reads."""
        result = companions.pair_chains_iptm(summary())
        assert result == {
            "0": {"0": 0.95, "1": 0.91},
            "1": {"0": 0.91, "1": 0.93},
        }

    def test_missing_matrix_is_empty_not_an_error(self):
        assert companions.pair_chains_iptm({}) == {}


class TestConfidence:
    def test_translates_to_the_shared_names(self):
        result = companions.confidence(summary())
        assert result["complex_plddt"] == 67.5
        assert result["confidence_score"] == 0.89
        assert result["pair_chains_iptm"]["0"]["1"] == 0.91

    def test_keeps_protenix_own_keys(self):
        result = companions.confidence(summary())
        assert result["ranking_score"] == 0.89
        assert result["iptm"] == 0.91

    def test_does_not_mutate_its_input(self):
        original = summary()
        companions.confidence(original)
        assert "complex_plddt" not in original


class TestWrite:
    @pytest.fixture
    def written(self, tmp_path):
        (tmp_path / "fd.json").write_text(json.dumps(full_data()))
        (tmp_path / "sc.json").write_text(json.dumps(summary()))
        return companions.write(
            structure_name="model_0.cif",
            work_dir=tmp_path,
            full_data=tmp_path / "fd.json",
            summary=tmp_path / "sc.json",
        )

    def test_names_follow_the_convention(self, written, tmp_path):
        assert written["pae"] == tmp_path / "pae_model_0.npz"
        assert written["plddt"] == tmp_path / "plddt_model_0.npz"
        assert written["confidence"] == tmp_path / "confidence_model_0.json"

    def test_pae_is_stored_under_the_expected_key(self, written):
        """`pae` is the key the scoring script loads from the archive."""
        import numpy as np

        with np.load(written["pae"]) as archive:
            assert archive["pae"].tolist() == [[0.5, 12.0], [12.0, 0.5]]

    def test_plddt_is_per_token(self, written):
        import numpy as np

        with np.load(written["plddt"]) as archive:
            assert archive["plddt"].tolist() == pytest.approx([0.85, 0.50])

    def test_plddt_is_as_long_as_the_pae_matrix(self, written):
        """They are indexed together, so a length mismatch is a silent bug."""
        import numpy as np

        with np.load(written["pae"]) as pae, np.load(written["plddt"]) as plddt:
            assert len(plddt["plddt"]) == pae["pae"].shape[0]

    def test_confidence_is_readable_json(self, written):
        data = json.loads(written["confidence"].read_text())
        assert data["pair_chains_iptm"]["0"]["1"] == 0.91

    def test_without_atom_confidence_only_the_summary_is_written(self, tmp_path):
        (tmp_path / "sc.json").write_text(json.dumps(summary()))
        written = companions.write(
            structure_name="model_0.cif",
            work_dir=tmp_path,
            full_data=None,
            summary=tmp_path / "sc.json",
        )
        assert set(written) == {"confidence"}

    def test_nothing_at_all_is_not_an_error(self, tmp_path):
        assert companions.write("model_0.cif", tmp_path, None, None) == {}


class TestPairing:
    """Which json belongs to which structure, when several samples are written."""

    @pytest.fixture
    def two_samples(self, tmp_path):
        for rank in (0, 1):
            (tmp_path / f"job_sample_{rank}.cif").write_text("")
            (tmp_path / f"job_summary_confidence_sample_{rank}.json").write_text("{}")
            (tmp_path / f"job_full_data_sample_{rank}.json").write_text("{}")
        return tmp_path

    def test_matches_on_sample_index(self, two_samples):
        structure = two_samples / "job_sample_1.cif"
        assert outputs._nearest_confidence(structure).name == (
            "job_summary_confidence_sample_1.json"
        )
        assert outputs._nearest_full_data(structure).name == (
            "job_full_data_sample_1.json"
        )

    def test_full_data_is_not_mistaken_for_the_summary(self, two_samples):
        found = outputs._nearest_confidence(two_samples / "job_sample_0.cif")
        assert "full_data" not in found.name

    def test_single_prediction_layout_needs_no_index(self, tmp_path):
        (tmp_path / "model.cif").write_text("")
        (tmp_path / "summary_confidences.json").write_text("{}")
        assert outputs._nearest_confidence(tmp_path / "model.cif") is not None

    def test_ambiguous_match_returns_nothing(self, tmp_path):
        """Guessing here would attach the wrong sample's scores to a structure."""
        (tmp_path / "model.cif").write_text("")
        for tag in ("a", "b"):
            (tmp_path / f"{tag}_summary_confidence.json").write_text("{}")
        assert outputs._nearest_confidence(tmp_path / "model.cif") is None

    def test_absent_full_data_is_none(self, tmp_path):
        (tmp_path / "job_sample_0.cif").write_text("")
        assert outputs._nearest_full_data(tmp_path / "job_sample_0.cif") is None
