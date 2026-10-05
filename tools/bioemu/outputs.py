"""Observables for a conformational ensemble, computed by their own libraries.

Everything here delegates. mdtraj computes fluctuation, compactness and
deviation; BioEmu's own benchmark package computes the fraction of native
contacts and the folding free energy. This module converts units and names
things, and that is all it does.

That division is deliberate. An earlier version of this file computed the free
energy here, as a count of samples inside an RMSD cutoff, and reported
-0.97 kcal/mol for a crystallised protein. The method was wrong, not just the
threshold: BioEmu's estimator uses a *soft* foldedness over the fraction of
native contacts, so a sample part-way to unfolded counts part-way, and its
parameters were fitted against experimental free energies. A plausible-looking
number with no calibration behind it is worse than no number.

What each one answers:

`flexibility` is per-residue fluctuation across the ensemble. A design can be
predicted confidently and still be floppy, and this is what separates them.

`compactness` is the radius of gyration per sample. Its spread matters more
than its value: a tight distribution is one state, a broad one is several.

`native_contacts` is how much of a reference structure's contact map each
sample keeps, which is BioEmu's own measure of foldedness and the input their
free energy estimator expects.

`free_energy` is the folding free energy in kcal/mol, negative favouring the
folded state, from BioEmu's fitted estimator rather than from a rule of thumb.

There is deliberately no RMSD against the reference here. `rmsd`, the tool,
compares structures properly, including the chain mapping this would need, and
an ensemble's distance from one snapshot says less than its contact map does
anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# mdtraj works in nanometres; every length in this repo is in Angstroms.
NM_TO_ANGSTROM = 10.0

# The estimator's own defaults, from `compute_dg_ddg_from_fnc`. Named here so a
# caller can see what was assumed rather than having to read their source.
TEMPERATURE_K = 295.0
FOLDED_THRESHOLD = 0.5
STEEPNESS = 10.0


def load(topology: Path, trajectory: Path):
    """A BioEmu ensemble as an mdtraj trajectory.

    BioEmu writes one `topology.pdb` naming the atoms and one `samples.xtc`
    holding every sampled frame.
    """
    import mdtraj

    return mdtraj.load(str(trajectory), top=str(topology))


def load_structure(path: Path):
    """A single reference structure as an mdtraj trajectory of one frame."""
    import mdtraj

    return mdtraj.load(str(path))


def flexibility(ensemble) -> list[float]:
    """Per-residue fluctuation about the ensemble's mean, in Angstroms.

    `mdtraj.rmsf` with no reference uses the average positions and superposes
    the frames itself, so this measures shape change rather than the ensemble
    drifting as a whole.
    """
    import mdtraj

    alpha = ensemble.topology.select("name CA")
    if len(alpha) == 0:
        raise ValueError(
            "The ensemble has no CA atoms, so there is no backbone to measure. "
            "BioEmu writes a backbone representation; this is something else."
        )
    return (
        mdtraj.rmsf(ensemble, None, atom_indices=alpha) * NM_TO_ANGSTROM
    ).tolist()


def compactness(ensemble) -> list[float]:
    """Radius of gyration of each sample, in Angstroms."""
    import mdtraj

    return (mdtraj.compute_rg(ensemble) * NM_TO_ANGSTROM).tolist()


def native_contacts(ensemble, reference):
    """Fraction of the reference's contacts each sample keeps.

    BioEmu's own measure of foldedness, and the input its free energy
    estimator expects. It aligns the sequences itself, so the ensemble and the
    reference need not have identical residue counts.
    """
    from bioemu_benchmarks.eval.folding_free_energies.fraction_native_contacts import (
        get_fnc_from_samples_trajectory,
    )

    return get_fnc_from_samples_trajectory(
        samples=ensemble, reference_conformation=reference
    )


def free_energy(fnc) -> float:
    """Folding free energy in kcal/mol, negative favouring folded.

    BioEmu's estimator, with the defaults their benchmark uses. The public
    wrapper, `compute_dg_ddg_from_fnc`, additionally wants a data frame of
    benchmark metadata in order to report experimental references alongside;
    this calls the function underneath it, which is the calculation itself.
    """
    from bioemu_benchmarks.eval.folding_free_energies.free_energies import (
        _compute_dG,
    )

    return float(
        _compute_dG(
            fnc,
            temperature=TEMPERATURE_K,
            p_fold_thr=FOLDED_THRESHOLD,
            steepness=STEEPNESS,
        )
    )


def describe(ensemble, reference=None) -> dict[str, Any]:
    """Every observable, for one ensemble."""
    import numpy as np

    fluctuation = flexibility(ensemble)
    gyration = compactness(ensemble)
    summary: dict[str, Any] = {
        "num_samples": int(ensemble.n_frames),
        "num_residues": len(fluctuation),
        "rmsf": [round(v, 3) for v in fluctuation],
        "rmsf_mean": round(float(np.mean(fluctuation)), 3),
        "rmsf_max": round(float(np.max(fluctuation)), 3),
        "radius_of_gyration_mean": round(float(np.mean(gyration)), 3),
        "radius_of_gyration_std": round(float(np.std(gyration)), 3),
    }
    if reference is None:
        return summary

    fnc = native_contacts(ensemble, reference)
    summary.update({
        "native_contacts_mean": round(float(np.mean(fnc)), 4),
        "native_contacts_min": round(float(np.min(fnc)), 4),
        "temperature_k": TEMPERATURE_K,
        "free_energy_kcal_mol": round(free_energy(fnc), 3),
    })
    return summary
