# src/scaffold_split.py
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def _murcko_scaffold_smiles(smiles: str) -> str:
    """Return Bemis Murcko scaffold SMILES (canonical)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "__INVALID__"
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return "__NO_SCAFFOLD__"
        s = Chem.MolToSmiles(scaf, canonical=True)
        return s if s else "__NO_SCAFFOLD__"
    except Exception:
        return "__INVALID__"

def scaffold_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, smiles_col: str = "smiles"):
    """
    Split by scaffolds: whole scaffolds go to train or test.
    Default smiles_col is 'smiles' to match the ml_base.py pipeline.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Dataframe must contain the column: '{smiles_col}'")

    # Generate scaffolds for each molecule
    scaffolds = df[smiles_col].apply(_murcko_scaffold_smiles)
    df_copy = df.copy()
    df_copy["_scaffold"] = scaffolds

    # Group row indices by their scaffold type
    scaffold_to_indices = {}
    for i, scaf in enumerate(df_copy["_scaffold"].tolist()):
        scaffold_to_indices.setdefault(scaf, []).append(i)

    # Shuffle the unique scaffolds
    rng = random.Random(random_state)
    scaffold_keys = list(scaffold_to_indices.keys())
    rng.shuffle(scaffold_keys)

    n_total = len(df_copy)
    n_test_target = int(round(test_size * n_total))

    test_indices = []
    train_indices = []
    test_count = 0

    # Distribute scaffolds until the test set target size is reached
    for scaf in scaffold_keys:
        indices = scaffold_to_indices[scaf]
        if test_count < n_test_target:
            test_indices.extend(indices)
            test_count += len(indices)
        else:
            train_indices.extend(indices)

    # Create final dataframes
    train_df = df_copy.iloc[train_indices].drop(columns=["_scaffold"]).reset_index(drop=True)
    test_df = df_copy.iloc[test_indices].drop(columns=["_scaffold"]).reset_index(drop=True)

    return train_df, test_df