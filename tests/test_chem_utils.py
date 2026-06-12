import pytest
from rdkit import Chem


class TestMolFromSmiles:
    def test_valid_smiles(self):
        from src.chem_utils import mol_from_smiles
        mol = mol_from_smiles("CCO")
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_invalid_smiles(self):
        from src.chem_utils import mol_from_smiles
        assert mol_from_smiles("INVALID") is None

    def test_empty_smiles(self):
        from src.chem_utils import mol_from_smiles
        assert mol_from_smiles("") is None


class TestCanonicalSmiles:
    def test_canonicalization(self):
        from src.chem_utils import mol_from_smiles, canonical_smiles
        mol = mol_from_smiles("C(O)C")
        can = canonical_smiles(mol)
        assert can == "CCO"

    def test_roundtrip(self):
        from src.chem_utils import mol_from_smiles, canonical_smiles
        mol = mol_from_smiles("c1ccccc1")
        can = canonical_smiles(mol)
        mol2 = mol_from_smiles(can)
        can2 = canonical_smiles(mol2)
        assert can == can2


class TestMorganFingerprint:
    def test_fingerprint_shape(self):
        from src.chem_utils import mol_from_smiles, morgan_fingerprint
        mol = mol_from_smiles("CCO")
        fp = morgan_fingerprint(mol)
        assert fp is not None

    def test_similar_molecules_similar_fps(self):
        from src.chem_utils import mol_from_smiles, morgan_fingerprint
        from rdkit import DataStructs
        mol1 = mol_from_smiles("CCO")
        mol2 = mol_from_smiles("CCO")
        fp1 = morgan_fingerprint(mol1)
        fp2 = morgan_fingerprint(mol2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        assert sim == 1.0
