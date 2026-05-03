import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _murcko_scaffold_smiles(canonical_smiles: str) -> str:
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return "__INVALID__"
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return "__NO_SCAFFOLD__"
    s = Chem.MolToSmiles(scaf, canonical=True)
    return s if s else "__NO_SCAFFOLD__"


def scaffold_split(df, test_size: float = 0.2, random_state: int = 42):
    if "canonical_smiles" not in df.columns:
        raise ValueError("df must contain 'canonical_smiles'")

    scaff = df["canonical_smiles"].apply(_murcko_scaffold_smiles)
    df2 = df.copy()
    df2["_scaffold"] = scaff

    scaffold_to_idx = {}
    for i, sc in enumerate(df2["_scaffold"].tolist()):
        scaffold_to_idx.setdefault(sc, []).append(i)

    rng = random.Random(random_state)
    scaffold_keys = list(scaffold_to_idx.keys())
    rng.shuffle(scaffold_keys)

    n_total = len(df2)
    n_test_target = int(round(test_size * n_total))

    test_scaffolds = set()
    test_count = 0
    for sc in scaffold_keys:
        if test_count >= n_test_target:
            break
        test_scaffolds.add(sc)
        test_count += len(scaffold_to_idx[sc])

    test_mask = df2["_scaffold"].isin(test_scaffolds)
    test_df = df2[test_mask].drop(columns=["_scaffold"]).reset_index(drop=True)
    train_df = df2[~test_mask].drop(columns=["_scaffold"]).reset_index(drop=True)

    return train_df, test_df
