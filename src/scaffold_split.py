# src/scaffold_split.py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def murcko_scaffold(smiles: str) -> str | None:
    """Return Bemis Murcko scaffold SMILES (canonical)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None

def scaffold_group_ids(smiles_series: pd.Series) -> pd.Series:
    """Map each molecule to a scaffold id (string). Invalid -> None."""
    scaffolds = smiles_series.apply(murcko_scaffold)
    return scaffolds

def scaffold_split_indices(
    scaffolds: pd.Series,
    test_size: float = 0.2,
    seed: int = 42
):
    """
    Split by scaffolds: whole scaffolds go to train or test.
    Returns (train_idx, test_idx) as numpy arrays of row indices.
    """
    rng = np.random.default_rng(seed)

    valid_mask = scaffolds.notna()
    valid_indices = scaffolds.index[valid_mask].to_numpy()
    valid_scaffolds = scaffolds.loc[valid_mask]

    # group indices by scaffold
    scaffold_to_indices = {}
    for idx, scaf in valid_scaffolds.items():
        scaffold_to_indices.setdefault(scaf, []).append(idx)

    scaffold_keys = np.array(list(scaffold_to_indices.keys()), dtype=object)
    rng.shuffle(scaffold_keys)

    n_test = int(np.ceil(test_size * len(valid_indices)))

    test_idx = []
    train_idx = []

    for scaf in scaffold_keys:
        inds = scaffold_to_indices[scaf]
        # fill test until reaching target
        if len(test_idx) < n_test:
            test_idx.extend(inds)
        else:
            train_idx.extend(inds)

    # If test got too large, move some scaffolds back (rare but can happen)
    test_idx = np.array(test_idx, dtype=int)
    train_idx = np.array(train_idx, dtype=int)

    return train_idx, test_idx

def repeated_scaffold_splits(df: pd.DataFrame, n_repeats: int = 5, test_size: float = 0.2, base_seed: int = 42):
    """
    Generate repeated scaffold splits.
    Requires df['smiles'] column.
    """
    scaffolds = scaffold_group_ids(df["smiles"])
    splits = []
    for i in range(n_repeats):
        seed = base_seed + i
        tr, te = scaffold_split_indices(scaffolds, test_size=test_size, seed=seed)
        splits.append((seed, tr, te))
    return scaffolds, splits