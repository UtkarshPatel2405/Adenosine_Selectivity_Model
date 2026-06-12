import pytest


class TestScaffoldSplit:
    def test_split_returns_indices(self):
        from src.scaffold_split import scaffold_train_test_split
        import pandas as pd
        df = pd.DataFrame({
            "smiles": ["CCO", "c1ccccc1", "CCN", "c1ccccc1O", "CCC", "C1CCCCC1"],
            "activity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        train_idx, test_idx = scaffold_train_test_split(df, smiles_col="smiles", test_size=0.3)
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert len(train_idx) + len(test_idx) == len(df)

    def test_no_leakage(self):
        from src.scaffold_split import scaffold_train_test_split
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        df = pd.DataFrame({
            "smiles": ["CCO", "c1ccccc1", "CCN", "c1ccccc1O", "CCC", "C1CCCCC1"],
            "activity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        train_idx, test_idx = scaffold_train_test_split(df, smiles_col="smiles", test_size=0.3)
        train_scaffolds = set()
        for i in train_idx:
            mol = Chem.MolFromSmiles(df.iloc[i]["smiles"])
            train_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        for i in test_idx:
            mol = Chem.MolFromSmiles(df.iloc[i]["smiles"])
            test_scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            assert test_scaff not in train_scaffolds
