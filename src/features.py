import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)

def _morgan_bits(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = _MORGAN.GetFingerprint(mol)  # ExplicitBitVect
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    arom = Lipinski.NumAromaticRings(mol)
    tpsa = Descriptors.TPSA(mol)
    return np.array([mw, logp, hbd, hba, rot, arom, tpsa], dtype=np.float32)


def build_feature_matrix(train_df, test_df, smiles_col: str = "canonical_smiles"):
    train_smiles = train_df[smiles_col].tolist()
    test_smiles = test_df[smiles_col].tolist()

    Xfp_train = np.vstack([_morgan_bits(s) for s in train_smiles])
        # cache RDKit bitvectors for fast AD at inference
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in train_smiles]
    with open("data/processed/train_fps.pkl", "wb") as f:
        pickle.dump(train_fps, f)
    Xfp_test = np.vstack([_morgan_bits(s) for s in test_smiles])

    Xdesc_train = np.vstack([_descriptors(s) for s in train_smiles])
    Xdesc_test = np.vstack([_descriptors(s) for s in test_smiles])

    scaler = StandardScaler()
    Xdesc_train_s = scaler.fit_transform(Xdesc_train)
    Xdesc_test_s = scaler.transform(Xdesc_test)

    X_train = np.hstack([Xfp_train, Xdesc_train_s]).astype(np.float32)
    X_test = np.hstack([Xfp_test, Xdesc_test_s]).astype(np.float32)

    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/train_smiles.pkl", "wb") as f:
        pickle.dump(train_smiles, f)
    with open("data/processed/test_smiles.pkl", "wb") as f:
        pickle.dump(test_smiles, f)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/features_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open("data/processed/features_test.pkl", "wb") as f:
        pickle.dump(X_test, f)

    return X_train, X_test, scaler


def build_features(smiles: str, scaler) -> np.ndarray:
    fp = _morgan_bits(smiles)
    desc = _descriptors(smiles).reshape(1, -1)
    desc_s = scaler.transform(desc).ravel()
    return np.hstack([fp.astype(np.float32), desc_s.astype(np.float32)])