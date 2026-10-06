# These assert on the manifest actually written, because AP Novo takes no
# design arguments on the command line: every way of getting a campaign wrong
# is a key in this JSON, and several are silent rather than loud.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.apnovo import inputs

MOTIF = "A1,A2,A3|10-40,{},2-30,{},2-30,{},10-40/B1"
UNPLACED = "10-250/B1"


def job(**overrides):
    fields = dict(
        name="kemp", input_file="motif.cif", motif_str=MOTIF, num_designs=8
    )
    return inputs.design_job(**{**fields, **overrides})


def manifest(**overrides):
    fields = dict(job=job(), resequence=True)
    return inputs.manifest(**{**fields, **overrides})


class TestDesignJob:
    def test_absent_optionals_are_left_out_not_nulled(self):
        entry = job()
        for key in ("motif_atoms", "seq_length", "unindexed_motif_residues",
                    "partial_diffusion_input_file"):
            assert key not in entry

    def test_supplied_optionals_are_carried(self):
        entry = job(motif_atoms="A1:NE2,ND1 A2:OD1", seq_length="120-160")
        assert entry["motif_atoms"] == "A1:NE2,ND1 A2:OD1"
        assert entry["seq_length"] == "120-160"

    def test_seed_is_always_written_so_a_run_is_reproducible(self):
        assert job()["seed_start"] == 0
        assert job(seed_start=7)["seed_start"] == 7

    def test_unindexed_requires_an_unplaced_motif(self):
        with pytest.raises(ValueError, match="already pins residues"):
            job(unindexed_motif_residues="A1,A2,A3")

    def test_unindexed_is_accepted_when_nothing_is_placed(self):
        entry = job(motif_str=UNPLACED, unindexed_motif_residues="A1,A2,A3")
        assert entry["unindexed_motif_residues"] == "A1,A2,A3"

    def test_ligand_chains_do_not_count_as_placing_the_motif(self):
        # `B1` names an input chain and is always fixed, so reading the whole
        # string rather than the first chain would reject a legitimate job.
        entry = job(motif_str="10-250/B1", unindexed_motif_residues="A1")
        assert entry["motif_str"] == "10-250/B1"

    def test_partial_diffusion_needs_both_halves(self):
        with pytest.raises(ValueError, match="both a starting structure"):
            job(partial_diffusion_input_file="parent.cif")
        with pytest.raises(ValueError, match="both a starting structure"):
            job(partial_diffusion_num_steps=600)

    def test_partial_diffusion_carries_both(self):
        entry = job(
            partial_diffusion_input_file="parent.cif",
            partial_diffusion_num_steps=600,
        )
        assert entry["partial_diffusion_input_file"] == "parent.cif"
        assert entry["partial_diffusion_num_steps"] == 600


class TestManifest:
    def test_every_stage_after_generation_is_configured(self):
        document = manifest(suite="kemp_eliminase")
        assert document["designs"][0]["name"] == "kemp"
        assert document["resequence"]["enabled"] is True
        assert document["folding"]["seeds"] == [0]
        assert document["evaluation"]["suite"] == "kemp_eliminase"

    def test_resequence_off_is_written_not_omitted(self):
        # Upstream gives `enabled` no default, so leaving it out would fail
        # validation rather than default to off.
        assert manifest(resequence=False)["resequence"]["enabled"] is False

    def test_folding_inputs_is_left_for_upstream_to_derive(self):
        # Upstream fills it in from resequence.enabled; writing it here would
        # only create a way for the two to disagree.
        assert "inputs" not in manifest()["folding"]

    def test_folding_states_absent_means_derive_from_the_motif(self):
        assert "states" not in manifest()["folding"]

    def test_folding_states_are_carried_when_given(self):
        states = [{"name": "complex", "ligands": [{"id": "B", "ccd_code": "6NT"}]}]
        assert manifest(folding_states=states)["folding"]["states"] == states

    def test_evaluation_is_omitted_entirely_when_unconfigured(self):
        assert "evaluation" not in manifest()

    def test_an_unknown_suite_is_refused_before_any_gpu_time(self):
        with pytest.raises(ValueError, match="Unknown evaluation suite"):
            manifest(suite="kemp")

    def test_repeated_folding_seeds_are_refused(self):
        with pytest.raises(ValueError, match="more than once"):
            manifest(folding_seeds=[0, 1, 0])

    def test_weights_are_named_in_settings(self):
        assert manifest()["settings"]["model_dir"] == inputs.GENERATOR_DIR


class TestWriteManifest:
    def test_input_file_stays_a_bare_name(self, tmp_path: Path):
        # Paths resolve against the manifest's own directory.
        path = inputs.write_manifest(tmp_path / "manifest.json", manifest())
        written = json.loads(path.read_text())
        assert written["designs"][0]["input_file"] == "motif.cif"


class TestPipelineArgs:
    def base(self, tmp_path: Path, **overrides):
        fields = dict(
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "out",
            generator_dir="/cache/gen",
            af3_dir="/cache/af3",
            resequence=True,
        )
        return inputs.pipeline_args(**{**fields, **overrides})

    def test_resume_is_off_by_default(self, tmp_path):
        assert "--resume=false" in self.base(tmp_path)

    def test_both_model_directories_are_passed(self, tmp_path):
        args = " ".join(self.base(tmp_path))
        assert "--apn_model_dir=/cache/gen" in args
        assert "--af3_model_dir=/cache/af3" in args

    def test_ligandmpnn_is_passed_only_when_resequencing(self, tmp_path):
        with_reseq = " ".join(self.base(tmp_path))
        assert "--ligandmpnn_dir=" in with_reseq
        assert "--ligandmpnn_python=" in with_reseq

        without = " ".join(self.base(tmp_path, resequence=False))
        assert "--ligandmpnn" not in without

    def test_an_unknown_stage_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="is not a stage"):
            self.base(tmp_path, only_stage="folding_but_wrong")

    def test_a_single_stage_can_be_named(self, tmp_path):
        assert "--only_stage=folding" in self.base(tmp_path, only_stage="folding")
