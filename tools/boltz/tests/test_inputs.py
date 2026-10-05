# These assert on the YAML actually written, not on the Chain objects behind it:
# `msa: empty` means single sequence, and omitting the field entirely is how
# Boltz is told to search. Both predict fine, so the mix-up is silent.

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.boltz import inputs
from tools.common import chains as chain_spec

TARGET = "MKTAYIAKQRQISFVK"
BINDER = "GHNLYQWRSTVEDCAP"


def written(tmp_path: Path, chains: dict) -> dict:
    path = inputs.write_job(tmp_path / "job.yaml", chain_spec.parse(chains))
    return yaml.safe_load(path.read_text())


def entity(job: dict, chain_id: str) -> dict:
    for item in job["sequences"]:
        protein = item["protein"]
        if protein["id"] == chain_id:
            return protein
    raise AssertionError(f"no chain {chain_id} in {job}")


class TestWriteJob:
    def test_version_and_one_entity_per_chain(self, tmp_path):
        job = written(tmp_path, {"A": TARGET, "B": BINDER})
        assert job["version"] == 1
        assert len(job["sequences"]) == 2

    def test_no_alignment_means_single_sequence(self, tmp_path):
        job = written(tmp_path, {"A": TARGET})
        assert entity(job, "A")["msa"] == "empty"

    def test_a_server_leaves_the_field_absent(self, tmp_path):
        job = written(tmp_path, {"A": {"sequence": TARGET, "msa": "colabfold"}})
        assert "msa" not in entity(job, "A")

    def test_mixed_per_chain(self, tmp_path):
        """The binder case: target searched, de novo binder single sequence."""
        job = written(
            tmp_path,
            {"A": {"sequence": TARGET, "msa": "colabfold"}, "B": BINDER},
        )
        assert "msa" not in entity(job, "A")
        assert entity(job, "B")["msa"] == "empty"

    def test_a_fetched_alignment_is_referenced_by_path(self, tmp_path):
        alignment = tmp_path / "A.a3m"
        alignment.write_text(">q\nMKT\n")
        parsed = chain_spec.parse({"A": {"sequence": TARGET, "msa": "s3://b/A.a3m"}})
        parsed["A"].alignment = alignment
        path = inputs.write_job(tmp_path / "job.yaml", parsed)
        job = yaml.safe_load(path.read_text())
        assert entity(job, "A")["msa"] == str(alignment)

    def test_sequences_are_carried_verbatim(self, tmp_path):
        job = written(tmp_path, {"A": TARGET, "B": BINDER})
        assert entity(job, "A")["sequence"] == TARGET
        assert entity(job, "B")["sequence"] == BINDER

    def test_refuses_an_empty_complex(self, tmp_path):
        with pytest.raises(ValueError, match="at least one chain"):
            inputs.write_job(tmp_path / "job.yaml", {})
