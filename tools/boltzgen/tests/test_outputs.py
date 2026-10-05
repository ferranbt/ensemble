from __future__ import annotations

from pathlib import Path

import pytest

from tools.boltzgen import outputs

# Exactly the paths the first real run produced.
LAYOUT = [
    "final_ranked_designs/final_1_designs/before_refolding/rank1_binder_1.cif",
    "final_ranked_designs/final_1_designs/rank1_binder_1.cif",
    "final_ranked_designs/intermediate_ranked_10_designs/before_refolding/rank0_binder_1.cif",
    "final_ranked_designs/intermediate_ranked_10_designs/before_refolding/rank1_binder_0.cif",
    "final_ranked_designs/intermediate_ranked_10_designs/rank0_binder_1.cif",
    "final_ranked_designs/intermediate_ranked_10_designs/rank1_binder_0.cif",
    "final_ranked_designs/all_designs_metrics.csv",
    "final_ranked_designs/final_designs_metrics_1.csv",
    "intermediate_designs/backbone_0.cif",
    "intermediate_designs_inverse_folded/aggregate_metrics_analyze.csv",
]


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    for relative in LAYOUT:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            path.write_text("design,rank,iptm\nrank1_binder_1,1,0.72\n")
        else:
            path.write_text("data_x\n")
    return tmp_path


class TestReadDesigns:
    def test_returns_only_the_final_shortlist(self, out_dir):
        # The layout's budget was 1, so exactly one design should come back.
        designs = outputs.read_designs(out_dir)
        assert len(designs) == 1
        assert designs[0]["name"] == "rank1_binder_1"

    def test_excludes_pre_refolding_copies(self, out_dir):
        """Each design exists twice; the refolded one is the result."""
        for design in outputs.read_designs(out_dir):
            assert outputs.BEFORE_REFOLDING not in design["path"]

    def test_ignores_the_wider_intermediate_ranking(self, out_dir):
        paths = [d["path"] for d in outputs.read_designs(out_dir)]
        assert not any("intermediate_ranked" in p for p in paths)

    def test_marks_results_as_final(self, out_dir):
        assert outputs.read_designs(out_dir)[0]["final"] is True

    def test_attaches_metrics_by_design_name(self, out_dir):
        design = outputs.read_designs(out_dir)[0]
        assert design["iptm"] == pytest.approx(0.72)

    def test_falls_back_to_the_intermediate_ranking(self, tmp_path):
        """A run stopped before the final step still returns its designs."""
        for relative in LAYOUT:
            if "final_1_designs" in relative:
                continue
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data_x\n" if path.suffix == ".cif" else "design\n")
        designs = outputs.read_designs(tmp_path)
        assert len(designs) == 2
        assert all("intermediate_ranked" in d["path"] for d in designs)

    def test_falls_back_to_any_structure_produced(self, tmp_path):
        path = tmp_path / "intermediate_designs" / "backbone_0.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_x\n")
        designs = outputs.read_designs(tmp_path)
        assert len(designs) == 1
        assert designs[0]["final"] is False

    def test_reports_the_tree_when_nothing_was_produced(self, tmp_path):
        (tmp_path / "config").mkdir()
        with pytest.raises(FileNotFoundError, match="produced no structures"):
            outputs.read_designs(tmp_path)


class TestReadMetrics:
    def test_prefers_the_final_metrics_table(self, out_dir):
        metrics = outputs.read_metrics(out_dir)
        assert "rank1_binder_1" in metrics
        assert metrics["rank1_binder_1"]["rank"] == pytest.approx(1.0)

    def test_converts_numeric_columns(self, out_dir):
        row = outputs.read_metrics(out_dir)["rank1_binder_1"]
        assert isinstance(row["iptm"], float)
        assert isinstance(row["design"], str)

    def test_returns_nothing_when_there_are_no_tables(self, tmp_path):
        assert outputs.read_metrics(tmp_path) == {}
