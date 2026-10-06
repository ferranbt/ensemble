# A campaign's designs are not its backbones: resequencing appends `_seq<NN>`
# and that longer prefix is what folding and evaluation key off. These assert
# that the prefix change is followed, and that `<prefix>_motif.cif` sitting
# beside `<prefix>.cif` never gets read as a design of its own.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.apnovo import outputs

BACKBONES = ("kemp_0000", "kemp_0001")


def campaign(
    tmp_path: Path,
    backbones=BACKBONES,
    sequences=1,
    states=("monomer", "complex"),
    seeds=(0,),
    evaluate=True,
    **options,
) -> Path:
    """An output directory laid out as run_pipeline.py leaves one."""
    out = tmp_path / "campaign"
    gen = out / outputs.GENERATION
    meta = out / "metadata"
    for directory in (gen, meta):
        directory.mkdir(parents=True, exist_ok=True)

    index = []
    for backbone in backbones:
        (gen / f"{backbone}.cif").write_text(f"data_{backbone}\n")
        if options.get("pdb", True):
            (gen / f"{backbone}.pdb").write_text("ATOM\n")
        if options.get("motif", True):
            (gen / f"{backbone}{outputs.MOTIF_SUFFIX}").write_text("data_motif\n")
        if options.get("fasta", True):
            (gen / f"{backbone}.fa").write_text(f">{backbone}\nMKTAYIAK\nQRQISFVK\n")
        (meta / f"{backbone}{outputs.METADATA_SUFFIX}").write_text(
            json.dumps({"fixed_residues_flag": "A1,A2,A3"})
        )
        index.append({"prefix": backbone, "seed": 0})
    (meta / "designs.json").write_text(json.dumps({"designs": index}))

    prefixes = []
    if sequences:
        reseq = out / outputs.RESEQUENCE
        reseq.mkdir(parents=True, exist_ok=True)
        for backbone in backbones:
            for n in range(sequences):
                prefix = f"{backbone}_seq{n:02d}"
                (reseq / f"{prefix}{outputs.RESEQ_SUFFIX}").write_text("data_reseq\n")
                (reseq / f"{prefix}.fa").write_text(f">{prefix}\nGHNLYQWR\n")
                prefixes.append(prefix)
    else:
        prefixes = list(backbones)

    folded = out / outputs.FOLDED
    folded.mkdir(parents=True, exist_ok=True)
    for prefix in prefixes:
        for state in states:
            for seed in seeds:
                (folded / f"{prefix}_{state}_folded_seed{seed}.cif").write_text("x")
                (folded / f"{prefix}_{state}_confidences_seed{seed}.json").write_text(
                    json.dumps({"ranking_confidence": 0.5 + 0.1 * seed, "ptm": 0.8})
                )

    if evaluate:
        ev = out / outputs.EVAL
        ev.mkdir(parents=True, exist_ok=True)
        for prefix in prefixes:
            (ev / f"{prefix}{outputs.EVALUATION_SUFFIX}").write_text(
                json.dumps({"metrics": {"motif_rmsd": 0.9}})
            )
        (ev / "evaluation_summary.csv").write_text(
            "design,motif_rmsd\n" + "".join(f"{p},0.9\n" for p in prefixes)
        )

    # `stages_run` is the key run_pipeline.py actually writes.
    (out / outputs.PIPELINE_INDEX).write_text(
        json.dumps({"stages_run": ["generation", "resequence",
                                   "folding", "evaluation"]})
    )
    return out


class TestReadIndex:
    def test_prefixes_come_back_in_order(self, tmp_path):
        assert outputs.read_index(campaign(tmp_path)) == list(BACKBONES)

    def test_a_missing_index_says_generation_produced_nothing(self, tmp_path):
        empty = tmp_path / "campaign"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="no design index"):
            outputs.read_index(empty)

    def test_a_bare_list_of_prefixes_is_read_too(self, tmp_path):
        out = tmp_path / "campaign"
        (out / "metadata").mkdir(parents=True)
        (out / "metadata" / "designs.json").write_text(json.dumps(["a_0000"]))
        assert outputs.read_index(out) == ["a_0000"]


class TestReadGenerated:
    def test_one_entry_per_backbone_not_per_cif(self, tmp_path):
        out = campaign(tmp_path)
        assert len(list((out / outputs.GENERATION).glob("*.cif"))) == 4
        assert len(outputs.read_generated(out)) == 2

    def test_the_motif_is_kept_apart_from_the_design(self, tmp_path):
        first = outputs.read_generated(campaign(tmp_path))[0]
        assert first["structure"].name == "kemp_0000.cif"
        assert first["motif"].name == f"kemp_0000{outputs.MOTIF_SUFFIX}"

    def test_sequence_is_joined_without_the_header(self, tmp_path):
        got = outputs.read_generated(campaign(tmp_path))[0]
        assert got["sequence"] == "MKTAYIAKQRQISFVK"

    def test_optional_files_come_back_as_none(self, tmp_path):
        got = outputs.read_generated(campaign(tmp_path, motif=False, pdb=False))[0]
        assert got["motif"] is None
        assert got["pdb"] is None

    def test_an_indexed_design_that_never_wrote_is_an_error(self, tmp_path):
        out = campaign(tmp_path)
        (out / outputs.GENERATION / "kemp_0001.cif").unlink()
        with pytest.raises(FileNotFoundError, match="did not finish"):
            outputs.read_generated(out)


class TestReadResequenced:
    def test_variants_group_under_the_backbone_they_came_from(self, tmp_path):
        grouped = outputs.read_resequenced(campaign(tmp_path, sequences=3))
        assert sorted(grouped) == list(BACKBONES)
        assert [v["prefix"] for v in grouped["kemp_0000"]] == [
            "kemp_0000_seq00", "kemp_0000_seq01", "kemp_0000_seq02"
        ]

    def test_resequencing_off_is_empty_not_an_error(self, tmp_path):
        assert outputs.read_resequenced(campaign(tmp_path, sequences=0)) == {}


class TestReadFolded:
    def test_predictions_are_keyed_by_the_resequenced_prefix(self, tmp_path):
        out = campaign(tmp_path, sequences=1)
        folded = outputs.read_folded(out, ["kemp_0000_seq00", "kemp_0001_seq00"])
        assert sorted(folded) == ["kemp_0000_seq00", "kemp_0001_seq00"]

    def test_one_entry_per_state_and_seed(self, tmp_path):
        out = campaign(tmp_path, states=("monomer", "complex"), seeds=(0, 1))
        entries = outputs.read_folded(out, ["kemp_0000_seq00"])["kemp_0000_seq00"]
        assert len(entries) == 4
        assert {(e["state"], e["seed"]) for e in entries} == {
            ("monomer", 0), ("monomer", 1), ("complex", 0), ("complex", 1)
        }

    def test_state_names_with_underscores_survive_parsing(self, tmp_path):
        out = campaign(tmp_path, states=("ligand_bound",))
        entries = outputs.read_folded(out, ["kemp_0000_seq00"])["kemp_0000_seq00"]
        assert [e["state"] for e in entries] == ["ligand_bound"]

    def test_confidences_travel_with_their_structure(self, tmp_path):
        out = campaign(tmp_path)
        entry = outputs.read_folded(out, ["kemp_0000_seq00"])["kemp_0000_seq00"][0]
        assert entry["confidences"]["ptm"] == 0.8


class TestReadEvaluations:
    def test_metrics_are_unwrapped_from_their_envelope(self, tmp_path):
        got = outputs.read_evaluations(campaign(tmp_path))
        assert got["kemp_0000_seq00"] == {"motif_rmsd": 0.9}

    def test_no_evaluation_stage_is_empty_not_an_error(self, tmp_path):
        assert outputs.read_evaluations(campaign(tmp_path, evaluate=False)) == {}


class TestSummaryAndStages:
    def test_the_summary_is_one_row_per_design(self, tmp_path):
        rows = outputs.read_summary(campaign(tmp_path, sequences=2))
        assert len(rows) == 4
        assert rows[0]["motif_rmsd"] == "0.9"

    def test_stages_completed_is_reported(self, tmp_path):
        assert outputs.stages_completed(campaign(tmp_path)) == [
            "generation", "resequence", "folding", "evaluation"
        ]

    def test_a_missing_record_is_empty_not_an_error(self, tmp_path):
        out = campaign(tmp_path)
        (out / outputs.PIPELINE_INDEX).unlink()
        assert outputs.stages_completed(out) == []

    def test_an_unexpected_key_name_is_tolerated(self, tmp_path):
        out = campaign(tmp_path)
        (out / outputs.PIPELINE_INDEX).write_text(
            json.dumps({"completed_stages": ["generation"]})
        )
        assert outputs.stages_completed(out) == ["generation"]


class TestHelpers:
    def test_fixed_residues_splits_the_flag(self, tmp_path):
        got = outputs.read_generated(campaign(tmp_path))[0]
        assert outputs.fixed_residues(got["metadata"]) == ["A1", "A2", "A3"]

    def test_fixed_residues_absent_is_empty(self):
        assert outputs.fixed_residues({}) == []

    def test_ranking_confidence_takes_the_best_across_predictions(self, tmp_path):
        out = campaign(tmp_path, seeds=(0, 1))
        entries = outputs.read_folded(out, ["kemp_0000_seq00"])["kemp_0000_seq00"]
        assert outputs.ranking_confidence(entries) == pytest.approx(0.6)

    def test_ranking_confidence_is_none_without_predictions(self):
        assert outputs.ranking_confidence([]) is None
