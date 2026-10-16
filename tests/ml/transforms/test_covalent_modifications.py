from typing import Any

import biotite.structure as struc
import numpy as np
import pytest

from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.utils.testing import cached_parse

COVALENT_MODIFICATION_TEST_CASES = [
    {
        # 4js1: A_1 61 (protein residue) is covalently bound to B_1 (multi-chain sugar)
        "pdb_id": "4js1",
        "residues_to_be_atomized": [
            {
                "polymer_pn_unit_iid": "A_1",
                "polymer_res_id": 61,
                "non_polymer_pn_unit_iid": "B_1",
                "non_polymer_pn_unit_id": "B",
            }
        ],
    },
]


@pytest.mark.parametrize("test_case", COVALENT_MODIFICATION_TEST_CASES)
def test_covalent_modifications(test_case: dict[str, Any]):
    """Atomize and reassign modified residues without changing modifier atomization."""
    data = cached_parse(test_case["pdb_id"])
    data["atom_array"].set_annotation("atomize", np.zeros(len(data["atom_array"]), dtype=bool))

    # Apply transforms
    unprocessed_atom_array = data["atom_array"].copy()
    covalent_modification_pipeline = Compose(
        [
            AnnotateCovalentModifications(),
        ]
    )
    result_with_covalent_modifications = covalent_modification_pipeline(data)

    processed_atom_array = result_with_covalent_modifications["atom_array"]
    # For all residues bound to non-polymers...
    for residue_dict in test_case["residues_to_be_atomized"]:
        residue_atom_mask = (processed_atom_array.pn_unit_iid == residue_dict["non_polymer_pn_unit_iid"]) & (
            processed_atom_array.res_id == residue_dict["polymer_res_id"]
        )
        residue_atom_array = processed_atom_array[residue_atom_mask]

        assert len(residue_atom_array) > 0

        # ... ensure that we set atomize = True for all atoms in the residue
        assert np.all(residue_atom_array.atomize)

        # Modifier atomization is unchanged.
        non_polymer_residue_atom_mask = unprocessed_atom_array.pn_unit_iid == residue_dict["non_polymer_pn_unit_iid"]
        non_polymer_residue_atom_array = processed_atom_array[non_polymer_residue_atom_mask]
        np.testing.assert_array_equal(
            non_polymer_residue_atom_array.atomize, unprocessed_atom_array.atomize[non_polymer_residue_atom_mask]
        )

        # ... ensure that we set pn_unit_id and pn_unit_iid to that of the non-polymer PN unit
        assert np.all(
            residue_atom_array.pn_unit_iid
            == np.array([residue_dict["non_polymer_pn_unit_iid"]] * len(residue_atom_array))
        )
        assert np.all(
            residue_atom_array.pn_unit_id
            == np.array([residue_dict["non_polymer_pn_unit_id"]] * len(residue_atom_array))
        )

        # ... check that for all other residues, atomize was unchanged
        combined_residue_mask = residue_atom_mask | non_polymer_residue_atom_mask
        other_residues_processed = processed_atom_array[~combined_residue_mask]
        other_residues_unprocessed = unprocessed_atom_array[~combined_residue_mask]
        assert np.all(other_residues_processed.atomize == other_residues_unprocessed.atomize)


@pytest.mark.parametrize("polymer_atom", ["C", "CB"])
@pytest.mark.parametrize("reassign", [False, True])
@pytest.mark.parametrize("atomize_modified_residue", [False, True])
@pytest.mark.parametrize("modifier_atomized", [False, True])
def test_independent_modified_residue_controls(polymer_atom, reassign, atomize_modified_residue, modifier_atomized):
    """Keep reassignment and residue atomization independent while preserving modifier state and bonds."""
    atoms = struc.AtomArray(3)
    atoms.res_name = np.array(["ALA", "ALA", "NAG"])
    atoms.res_id = np.array([1, 1, 2])
    atoms.chain_id = np.array(["A", "A", "B"])
    atoms.atom_name = np.array(["CA", polymer_atom, "C1"])
    atoms.set_annotation("is_polymer", np.array([True, True, False]))
    atoms.set_annotation("pn_unit_id", np.array(["A", "A", "B"]))
    atoms.set_annotation("pn_unit_iid", np.array(["A_1", "A_1", "B_1"]))
    atoms.set_annotation("atomize", np.array([False, False, modifier_atomized]))
    atoms.bonds = struc.BondList(3, np.array([[0, 1, 1], [1, 2, 1]]))
    result = AnnotateCovalentModifications(
        reassign_modified_residue=reassign,
        atomize_modified_residue=atomize_modified_residue,
    )({"atom_array": atoms})["atom_array"]
    np.testing.assert_array_equal(
        result.atomize, [atomize_modified_residue, atomize_modified_residue, modifier_atomized]
    )
    assert result.pn_unit_iid[0] == ("B_1" if reassign else "A_1")
    assert result.is_covalent_modification[2]
    np.testing.assert_array_equal(result.bonds.as_array(), [[0, 1, 1], [1, 2, 1]])


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
