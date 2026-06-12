import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class TestPredictor:
    def test_predict_returns_float(self):
        from src.predictor import predict_pic50
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(10, 10)
        y_dummy = np.random.rand(10)
        model.fit(X_dummy, y_dummy)
        result = predict_pic50(model, mol)
        assert isinstance(result, (float, np.floating))

    def test_predict_multiple_molecules(self):
        from src.predictor import batch_predict
        from rdkit import Chem
        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CCN"]]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(10, 10)
        y_dummy = np.random.rand(10)
        model.fit(X_dummy, y_dummy)
        results = batch_predict(model, mols)
        assert len(results) == len(mols)
