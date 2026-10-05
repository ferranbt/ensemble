from __future__ import annotations

import pytest

from tools.common import a3m

QUERY = "MKTAYIAKQRQISFVKSHFSRQLEE"
VARIANT = "MKTAYIAKQRQISFVKSHFSRQLEK"  # one substitution, at the end


def alignment(tmp_path, query=QUERY, rows=2):
    path = tmp_path / "A.a3m"
    lines = [">101", query]
    for index in range(rows):
        lines += [f">hom{index}", "M" + "A" * (len(query) - 1)]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestReadQuery:
    def test_first_row_after_the_header(self, tmp_path):
        assert a3m.read_query(alignment(tmp_path)) == QUERY

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "A.a3m"
        path.write_text(f"\n\n>101\n{QUERY}\n>h\nAAA\n")
        assert a3m.read_query(path) == QUERY

    def test_no_sequence_is_an_error(self, tmp_path):
        path = tmp_path / "A.a3m"
        path.write_text(">101\n")
        with pytest.raises(ValueError, match="no query sequence"):
            a3m.read_query(path)


class TestRetarget:
    def test_matching_query_is_left_alone(self, tmp_path):
        path = alignment(tmp_path)
        before = path.read_text()
        assert a3m.retarget(path, QUERY) is False
        assert path.read_text() == before

    def test_query_row_is_replaced(self, tmp_path):
        path = alignment(tmp_path)
        assert a3m.retarget(path, VARIANT) is True
        assert a3m.read_query(path) == VARIANT

    def test_the_homologues_are_untouched(self, tmp_path):
        path = alignment(tmp_path, rows=3)
        before = [l for l in path.read_text().splitlines()[2:]]
        a3m.retarget(path, VARIANT)
        assert path.read_text().splitlines()[2:] == before

    def test_a_different_length_is_refused(self, tmp_path):
        """Indels would shift every column; only substitutions are reusable."""
        path = alignment(tmp_path)
        with pytest.raises(ValueError, match="preserve length"):
            a3m.retarget(path, QUERY + "A")

    def test_a_different_protein_is_refused(self, tmp_path):
        path = alignment(tmp_path)
        with pytest.raises(ValueError, match="different protein"):
            a3m.retarget(path, "A" * len(QUERY))

    def test_identity_is_reported_in_the_refusal(self, tmp_path):
        path = alignment(tmp_path)
        with pytest.raises(ValueError, match="%"):
            a3m.retarget(path, "A" * len(QUERY))

    def test_a_realistic_variant_series_is_accepted(self, tmp_path):
        path = alignment(tmp_path)
        many = list(QUERY)
        for position in (0, 5, 10, 15):
            many[position] = "W"
        assert a3m.retarget(path, "".join(many)) is True


class TestIdentity:
    def test_identical(self):
        assert a3m.identity(QUERY, QUERY) == 1.0

    def test_one_difference(self):
        assert a3m.identity(QUERY, VARIANT) == pytest.approx(
            (len(QUERY) - 1) / len(QUERY)
        )

    def test_unequal_lengths_are_not_comparable(self):
        assert a3m.identity(QUERY, QUERY + "A") == 0.0
