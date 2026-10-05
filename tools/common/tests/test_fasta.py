from __future__ import annotations

from pathlib import Path

from tools.common import fasta

# A real ProteinMPNN output, which is where that fixture lives.
FIXTURES = Path(__file__).resolve().parents[2] / "proteinmpnn" / "tests" / "fixtures"


class TestFasta:
    def test_parses_a_real_proteinmpnn_output(self):
        records = fasta.parse((FIXTURES / "seqs" / "6MRR.fa").read_text())
        assert len(records) == 3
        assert records[0].header.startswith("6MRR, score=")
        assert records[0].sequence.startswith("GWSTELEKHREEL")

    def test_joins_sequences_split_across_lines(self):
        records = fasta.parse(">a\nMKT\nAYI\n\n>b\nGGG\n")
        assert [r.sequence for r in records] == ["MKTAYI", "GGG"]

    def test_ignores_blank_lines_and_leading_whitespace(self):
        assert fasta.parse("\n\n  >x  \n  MKT  \n\n") == [("x", "MKT")]

    def test_returns_nothing_for_empty_input(self):
        assert fasta.parse("") == []
        assert fasta.parse("\n \n") == []

    def test_ignores_sequence_data_before_any_header(self):
        assert fasta.parse("MKT\n>a\nGGG\n") == [("a", "GGG")]

    def test_keeps_an_empty_sequence(self):
        """A header with no residues still counts as a record."""
        assert fasta.parse(">a\n>b\nGGG\n") == [("a", ""), ("b", "GGG")]

    def test_format_round_trips(self):
        pairs = [("query_1", "MKTAYI"), ("query_2", "GGGCCC")]
        assert [tuple(r) for r in fasta.parse(fasta.format(pairs))] == pairs

    def test_format_writes_one_record_per_two_lines(self):
        assert fasta.format([("a", "MKT")]) == ">a\nMKT\n"
