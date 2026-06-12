import pytest
import numpy as np


class TestMorganFingerprintArray:
    def test_output_shape(self):
        from src.features import morgan_fingerprint_array
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        fp = morgan_fingerprint_array(mol)
        assert isinstance(fp, np.ndarray)
        assert fp.ndim == 1

    def test_bit_string_properties(self):
        from src.features import morgan_fingerprint_array
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        fp = morgan_fingerprint_array(mol)
        assert fp.dtype == np.uint8 or fp.dtype == np.int32


class TestPhysicochemicalDescriptors:
    def test_molwt(self):
        from src.features import compute_descriptors
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        desc = compute_descriptors(mol)
        assert "MolWt" in desc
        assert desc["MolWt"] > 0

    def test_lipinski_acceptors(self):
        from src.features import compute_descriptors
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        desc = compute_descriptors(mol)
        assert "NumHAcceptors" in desc
