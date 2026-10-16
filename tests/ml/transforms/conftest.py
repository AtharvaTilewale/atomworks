import biotite.structure as struc
import numpy as np
import pytest


@pytest.fixture
def mixed_residues():
    """Return two custom residues and one alanine, each with two unatomized atoms."""
    atoms = struc.AtomArray(6)
    atoms.res_id = np.repeat([1, 2, 3], 2)
    atoms.res_name = np.repeat(["XAA", "XS1", "ALA"], 2)
    atoms.atom_name = np.tile(["C1", "C2"], 3)
    atoms.coord = np.arange(18, dtype=float).reshape(6, 3)
    atoms.set_annotation("atomize", np.zeros(6, dtype=bool))
    return atoms
