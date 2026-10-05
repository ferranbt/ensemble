
from __future__ import annotations

from pathlib import Path

import pytest

from tools.ipsae import outputs

# Verbatim from ipsae.py's OUT.write header, followed by rows in its layout.
HEADER = (
    "Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  "
    "ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   "
    "d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model"
)
ROWS = [
    "A    B      10   15  asym   0.4210   0.3100      0.4010       0.2200   "
    "0.2500         0.3100     0.2800     0.5100    48     206    92      "
    "4.21    7.02    5.44   140     66      31      28     model_0",
    "B    A      10   15  asym   0.6350   0.4400      0.6100       0.2200   "
    "0.2500         0.3100     0.2800     0.5100    52     206    92      "
    "4.40    7.02    5.44   66      140     28      31     model_0",
    "A    B      10   15  max    0.6350   0.4400      0.6100       0.2200   "
    "0.2500         0.3100     0.2800     0.5100    52     206    92      "
    "4.40    7.02    5.44   140     66      31      28     model_0",
]


@pytest.fixture
def summary(tmp_path: Path) -> Path:
    path = tmp_path / "model_0_10_15.txt"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n")
    return path


class TestReportPaths:
    def test_names_outputs_after_the_structure(self, tmp_path):
        paths = outputs.report_paths(tmp_path / "model_0.pdb", 10, 15)
        assert paths["summary"].name == "model_0_10_15.txt"
        assert paths["by_residue"].name == "model_0_10_15_byres.txt"
        assert paths["pymol"].name == "model_0_10_15.pml"

    def test_cutoffs_below_ten_are_zero_padded(self):
        paths = outputs.report_paths(Path("/t/m.pdb"), 8, 9)
        assert paths["summary"].name == "m_08_09.txt"

    def test_writes_beside_the_structure(self, tmp_path):
        paths = outputs.report_paths(tmp_path / "sub" / "m.cif", 10, 15)
        assert paths["summary"].parent == tmp_path / "sub"

    def test_handles_a_cif_structure(self):
        assert outputs.report_paths(Path("/t/m.cif"), 10, 15)["summary"].name == (
            "m_10_15.txt"
        )

    def test_float_cutoffs_are_truncated_to_integers(self):
        assert outputs.report_paths(Path("/t/m.pdb"), 10.0, 15.0)["summary"].name == (
            "m_10_15.txt"
        )


class TestParseSummary:
    def test_reads_every_row(self, summary):
        assert len(outputs.parse_summary(summary)) == 3

    def test_keys_rows_by_the_tables_own_columns(self, summary):
        row = outputs.parse_summary(summary)[0]
        assert row["Chn1"] == "A"
        assert row["Chn2"] == "B"
        assert row["Type"] == "asym"
        assert row["Model"] == "model_0"

    def test_converts_scores_to_floats(self, summary):
        row = outputs.parse_summary(summary)[0]
        assert row["ipSAE"] == pytest.approx(0.4210)
        assert row["pDockQ2"] == pytest.approx(0.2800)
        assert isinstance(row["ipSAE"], float)

    def test_converts_counts_to_integers(self, summary):
        row = outputs.parse_summary(summary)[0]
        assert row["nres1"] == 140
        assert isinstance(row["nres1"], int)
        assert isinstance(row["d0chn"], float)

    def test_ignores_blank_lines(self, tmp_path):
        path = tmp_path / "m_10_15.txt"
        path.write_text(f"\n\n{HEADER}\n\n{ROWS[0]}\n\n")
        assert len(outputs.parse_summary(path)) == 1

    def test_raises_when_there_are_no_rows(self, tmp_path):
        path = tmp_path / "m_10_15.txt"
        path.write_text(HEADER + "\n")
        with pytest.raises(ValueError, match="No score rows"):
            outputs.parse_summary(path)

    def test_keeps_a_model_name_containing_spaces(self, tmp_path):
        """Excess fields fold into the trailing column rather than dropping the row."""
        path = tmp_path / "m_10_15.txt"
        path.write_text(f"{HEADER}\n{ROWS[0]} with spaces\n")
        assert outputs.parse_summary(path)[0]["Model"] == "model_0 with spaces"


class TestBestInterfaces:
    def test_returns_only_the_symmetric_rows(self, summary):
        best = outputs.best_interfaces(outputs.parse_summary(summary))
        assert len(best) == 1
        assert best[0]["Type"] == "max"

    def test_symmetric_score_takes_the_better_direction(self, summary):
        # A/B scored 0.421 and B/A scored 0.635 in the fixture rows.
        best = outputs.best_interfaces(outputs.parse_summary(summary))[0]
        assert best["ipSAE"] == pytest.approx(0.6350)

    def test_sorts_by_ipsae_descending(self, tmp_path):
        weaker = ROWS[2].replace("max    0.6350", "max    0.1000").replace(
            "A    B", "C    D"
        )
        path = tmp_path / "m_10_15.txt"
        path.write_text("\n".join([HEADER, ROWS[2], weaker]) + "\n")
        best = outputs.best_interfaces(outputs.parse_summary(path))
        assert [row["ipSAE"] for row in best] == sorted(
            [row["ipSAE"] for row in best], reverse=True
        )

    def test_falls_back_to_all_rows_when_none_are_symmetric(self, tmp_path):
        path = tmp_path / "m_10_15.txt"
        path.write_text("\n".join([HEADER, ROWS[0], ROWS[1]]) + "\n")
        assert len(outputs.best_interfaces(outputs.parse_summary(path))) == 2
