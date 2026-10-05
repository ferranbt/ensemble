"""Structure fixtures shared by more than one tool's tests.

Real pipeline outputs, named after the tool that produced each, so a test that
reads one is reading a file another tool actually wrote.
"""

from pathlib import Path

STRUCTURES = Path(__file__).parent / "structures"

BOLTZ_PDB = STRUCTURES / "boltz_model_0.pdb"
BOLTZGEN_CIF = STRUCTURES / "boltzgen_design.cif"
PROTENIX_CIF = STRUCTURES / "protenix_model_0.cif"
