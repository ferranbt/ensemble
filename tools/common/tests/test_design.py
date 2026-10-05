"""Tests for the pieces BoltzGen and ProteinMPNN share.

These are the fields a later stage reads, so what the tests pin is that a
candidate from either tool can be handed onward: which chain is the binder, and
each chain's sequence keyed by chain rather than run together in one string.

Structure fixtures are real pipeline outputs, since the point is reading files
other tools wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.common import design
from fixtures import BOLTZ_PDB, BOLTZGEN_CIF


class TestChainLengths:
    def test_reads_pdb(self):
        assert design.chain_lengths(BOLTZ_PDB) == {"A": 68, "B": 68}

    def test_reads_mmcif(self):
        assert design.chain_lengths(BOLTZGEN_CIF) == {"A": 143, "B": 82}


class TestBinderChain:
    def test_a_new_chain_is_the_binder(self):
        assert design.binder_chain({"A": 100, "B": 80}, {"A": 100}) == "B"

    def test_falls_back_to_the_chain_whose_length_changed(self):
        """A generator may reuse the target's letters rather than add one."""
        assert design.binder_chain({"A": 100, "B": 95}, {"A": 100, "B": 80}) == "B"

    def test_unknown_rather_than_a_guess(self):
        """Two candidates is not an answer: scoring the wrong chain gives a
        plausible number for the wrong question."""
        assert design.binder_chain({"C": 80, "D": 90}, {"A": 100}) is None

    def test_nothing_changed(self):
        assert design.binder_chain({"A": 100}, {"A": 100}) is None


class TestReadSequences:
    def test_one_sequence_per_chain(self):
        sequences = design.read_sequences(BOLTZ_PDB)
        assert set(sequences) == {"A", "B"}
        assert len(sequences["A"]) == 68

    def test_residues_are_one_letter(self):
        sequences = design.read_sequences(BOLTZ_PDB)
        assert set(sequences["A"]) <= set("ACDEFGHIKLMNPQRSTVWYX")

    def test_lengths_agree_with_chain_lengths(self):
        """They are read from the same polymer, so a mismatch means one is wrong."""
        sequences = design.read_sequences(BOLTZGEN_CIF)
        lengths = design.chain_lengths(BOLTZGEN_CIF)
        assert {c: len(s) for c, s in sequences.items()} == lengths


class TestSplitSequences:
    def test_maps_onto_the_designed_chains(self):
        assert design.split_sequences("MKT/GHN", ["A", "B"]) == {
            "A": "MKT",
            "B": "GHN",
        }

    def test_single_chain_has_no_separator(self):
        assert design.split_sequences("MKT", ["B"]) == {"B": "MKT"}

    def test_refuses_to_guess_on_a_mismatch(self):
        """Silently zipping would give one chain another's sequence."""
        with pytest.raises(ValueError, match="Refusing to guess"):
            design.split_sequences("MKT/GHN", ["A"])

    def test_refuses_when_a_chain_is_missing_its_sequence(self):
        with pytest.raises(ValueError, match="Refusing to guess"):
            design.split_sequences("MKT", ["A", "B"])


class TestEnvelope:
    def test_best_is_the_first_design(self):
        result = design.envelope(
            name="job", designs=[{"design": 0}, {"design": 1}]
        )
        assert result["best"] == {"design": 0}

    def test_no_designs(self):
        assert design.envelope(name="job", designs=[])["best"] is None

    def test_tool_specific_fields_are_kept(self):
        result = design.envelope(
            name="job", designs=[], protocol="protein-anything"
        )
        assert result["protocol"] == "protein-anything"


class TestSummaryLines:
    def test_renders_either_tools_candidate(self):
        result = design.envelope(
            name="job",
            designs=[
                {
                    "design": 0,
                    "structure_uri": "s3://b/design_0.pdb",
                    "binder_chain": "B",
                    "sequences": {"A": "MKT", "B": "GHNLYQWRS"},
                    "score": 1.2345,
                }
            ],
        )
        (line,) = design.summary_lines(result)
        assert "binder=B" in line
        # The binder's sequence, not the target's.
        assert "seq=GHNLYQWRS" in line
        assert "score=1.2345" in line

    def test_no_designs(self):
        assert design.summary_lines({"designs": []}) == ["no designs produced"]


class TestModifiedResidues:
    """A chemically modified residue is the amino acid it is a form of.

    Selenomethionine is used to phase crystal structures and appears in a large
    share of the PDB. Reading it as unknown turns an ordinary methionine into
    an X, which every downstream sequence tool then rejects.
    """

    def test_selenomethionine_is_methionine(self):
        assert design.THREE_TO_ONE["MSE"] == "M"

    def test_phosphorylated_residues_keep_their_identity(self):
        assert design.THREE_TO_ONE["SEP"] == "S"
        assert design.THREE_TO_ONE["TPO"] == "T"
        assert design.THREE_TO_ONE["PTR"] == "Y"

    def test_standard_residues_are_unchanged(self):
        assert set(design.STANDARD) <= set(design.THREE_TO_ONE)
        assert design.THREE_TO_ONE["MET"] == "M"

    def test_every_mapping_is_a_standard_one_letter_code(self):
        """A modified residue must read as something a sequence can hold."""
        assert set(design.THREE_TO_ONE.values()) <= set(design.STANDARD.values())
