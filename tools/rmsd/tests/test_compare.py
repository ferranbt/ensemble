from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.rmsd import compare as rmsd
from fixtures import BOLTZ_PDB, BOLTZGEN_CIF, PROTENIX_CIF


@pytest.fixture
def moved(tmp_path):
    """The Boltz dimer with chain B pushed 10A along x, chain A untouched."""
    import gemmi

    structure = gemmi.read_structure(str(BOLTZ_PDB))
    structure.setup_entities()
    for chain in structure[0]:
        if chain.name != "B":
            continue
        for residue in chain:
            for atom in residue:
                atom.pos = gemmi.Position(atom.pos.x + 10.0, atom.pos.y, atom.pos.z)
    path = tmp_path / "moved.pdb"
    structure.write_pdb(str(path))
    return path


@pytest.fixture
def rotated(tmp_path):
    """The whole Boltz dimer rigidly moved, both chains together."""
    import gemmi

    structure = gemmi.read_structure(str(BOLTZ_PDB))
    structure.setup_entities()
    # A quarter turn about z, plus a translation.
    transform = gemmi.Transform()
    transform.mat.fromlist([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform.vec.fromlist([12.0, -7.0, 3.0])
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                moved = transform.apply(atom.pos)
                atom.pos = gemmi.Position(moved.x, moved.y, moved.z)
    path = tmp_path / "rotated.pdb"
    structure.write_pdb(str(path))
    return path


class TestParseChains:
    def test_plain_names(self):
        pairs = rmsd.parse_chains("A B")
        assert [(p.reference, p.subject) for p in pairs] == [("A", "A"), ("B", "B")]

    def test_rename(self):
        (pair,) = rmsd.parse_chains("polymer=A")
        assert (pair.reference, pair.subject) == ("polymer", "A")

    def test_str_round_trips(self):
        assert [str(p) for p in rmsd.parse_chains("A B=C")] == ["A", "B=C"]

    def test_empty_is_no_chains(self):
        assert rmsd.parse_chains("   ") == []

    @pytest.mark.parametrize("spec", ["A=", "="])
    def test_rejects_half_a_rename(self, spec):
        with pytest.raises(ValueError, match="Cannot read chain"):
            rmsd.parse_chains(spec)


class TestCaPositions:
    def test_one_per_residue(self):
        model = rmsd.read_model(BOLTZ_PDB)
        assert len(rmsd.ca_positions(model, "A")) == 68

    def test_missing_chain_lists_what_is_there(self):
        model = rmsd.read_model(BOLTZ_PDB)
        with pytest.raises(KeyError, match="A, B"):
            rmsd.ca_positions(model, "Z")


class TestCompare:
    def test_identical_is_zero(self):
        result = rmsd.compare(BOLTZ_PDB, BOLTZ_PDB, align="A", measure="B")
        assert result["rmsd"] == pytest.approx(0.0, abs=1e-6)
        assert result["align_rmsd"] == pytest.approx(0.0, abs=1e-6)
        assert result["measure_rmsd"] == pytest.approx(0.0, abs=1e-6)
        assert result["align_atoms"] == 68
        assert result["measure_atoms"] == 68

    def test_rigid_move_of_everything_changes_nothing(self, rotated):
        result = rmsd.compare(BOLTZ_PDB, rotated, align="A", measure="B")
        assert result["rmsd"] == pytest.approx(0.0, abs=1e-3)
        assert result["align_rmsd"] == pytest.approx(0.0, abs=1e-3)

    def test_displaced_chain_is_measured_not_fitted_away(self, moved):
        result = rmsd.compare(BOLTZ_PDB, moved, align="A", measure="B")
        assert result["align_rmsd"] == pytest.approx(0.0, abs=1e-3)
        # Every atom of B moved by exactly 10A, but its fold is untouched.
        assert result["rmsd"] == pytest.approx(10.0, abs=1e-3)
        assert result["measure_rmsd"] == pytest.approx(0.0, abs=1e-3)

    def test_measuring_the_aligned_chain_hides_the_displacement(self, moved):
        """What `align`/`measure` exists to prevent: superposing on the binder
        makes a badly docked binder look perfect."""
        result = rmsd.compare(BOLTZ_PDB, moved, measure="B")
        assert result["rmsd"] == pytest.approx(0.0, abs=1e-3)

    def test_align_defaults_to_measure(self, moved):
        assert rmsd.compare(BOLTZ_PDB, moved, measure="B") == rmsd.compare(
            BOLTZ_PDB, moved, align="B"
        )

    def test_two_chains_of_a_near_symmetric_dimer(self):
        # Not identical chains, but the same fold: a small non-zero RMSD.
        result = rmsd.compare(BOLTZ_PDB, BOLTZ_PDB, measure="A=B")
        assert 0.0 < result["measure_rmsd"] < 1.0

    def test_reads_mmcif_without_conversion(self):
        result = rmsd.compare(PROTENIX_CIF, PROTENIX_CIF, measure="A")
        assert result["rmsd"] == pytest.approx(0.0, abs=1e-6)
        assert result["measure_atoms"] > 0

    def test_two_predictors_of_the_same_complex(self):
        """Both recover each chain's fold to about 0.5A but place the chains 45A
        apart, which confidence cannot see: neither model is unsure, they simply
        disagree about the docking."""
        result = rmsd.compare(BOLTZ_PDB, PROTENIX_CIF, align="A", measure="B")
        assert result["align_rmsd"] < 1.0
        assert result["measure_rmsd"] < 1.0
        assert result["rmsd"] > 20.0

    def test_requires_a_chain(self):
        with pytest.raises(ValueError, match="at least one chain"):
            rmsd.compare(BOLTZ_PDB, BOLTZ_PDB)

    def test_rejects_unequal_chains(self):
        """Pairing by position across different lengths would be meaningless."""
        with pytest.raises(ValueError, match="residues in the reference"):
            rmsd.compare(BOLTZ_PDB, BOLTZGEN_CIF, measure="A")


class TestRmsd:
    def test_empty_is_zero(self):
        assert rmsd.rmsd([], []) == 0.0

    def test_matches_the_definition(self):
        import gemmi

        fixed = [gemmi.Position(0, 0, 0), gemmi.Position(0, 0, 0)]
        movable = [gemmi.Position(3, 0, 0), gemmi.Position(0, 4, 0)]
        assert rmsd.rmsd(fixed, movable) == pytest.approx(math.sqrt(12.5))
