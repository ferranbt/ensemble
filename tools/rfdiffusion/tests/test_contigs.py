from __future__ import annotations
from pathlib import Path

import pytest

from tools.rfdiffusion import inputs

class TestMonomerContigs:
    def test_a_fixed_length(self):
        assert inputs.monomer_contigs("100") == "[100-100]"

    def test_a_range_varies_per_design(self):
        assert inputs.monomer_contigs("80-120") == "[80-120]"

    def test_nothing_is_held_fixed(self):
        """No chain reference, because there is no input structure."""
        assert "/" not in inputs.monomer_contigs("100")

    @pytest.mark.parametrize("bad", ["", "abc", "120-80", "0"])
    def test_rejects_a_length_that_is_not_one(self, bad):
        with pytest.raises(ValueError):
            inputs.monomer_contigs(bad)

class TestInferenceArgs:
    def common(self, input_pdb):
        return inputs.inference_args(
            input_pdb=input_pdb,
            output_prefix=Path("/tmp/out/run"),
            contigs="[100-100]",
            num_designs=2,
            checkpoint_path="/models/Base_ckpt.pt",
            hotspots=[],
            noise_scale=1.0,
            diffuser_steps=50,
        )

    def test_unconditional_omits_the_input_structure(self):
        """There is nothing to build around, so the override must be absent."""
        args = self.common(None)
        assert not any(a.startswith("inference.input_pdb=") for a in args)

    def test_a_target_is_passed_through(self):
        args = self.common(Path("/tmp/target.pdb"))
        assert "inference.input_pdb=/tmp/target.pdb" in args

    def test_contigs_travel_as_one_argument(self):
        """Brackets and commas must not be shell-split."""
        (contig,) = [a for a in self.common(None) if a.startswith("contigmap.")]
        assert contig == "contigmap.contigs=[100-100]"

    def test_hotspots_are_omitted_when_there_are_none(self):
        assert not any(a.startswith("ppi.") for a in self.common(None))

    def test_num_designs_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            inputs.inference_args(
                input_pdb=None, output_prefix=Path("/tmp/o"),
                contigs="[100-100]", num_designs=0,
                checkpoint_path="/m.pt", hotspots=[],
                noise_scale=1.0, diffuser_steps=50,
            )
