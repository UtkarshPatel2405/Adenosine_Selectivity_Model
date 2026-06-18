import pytest
import numpy as np


class TestEnsemblePredict:
    def test_single_model(self):
        from src.predictor import _ensemble_predict
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        model.fit(X, y)
        x = np.random.rand(5)
        mean, std, low, high = _ensemble_predict([model], x)
        assert isinstance(mean, float)
        assert std >= 0
        assert low <= mean <= high


class TestLoadScaler:
    def test_missing_file_raises(self):
        from src.predictor import _load_scaler

        with pytest.raises(FileNotFoundError):
            _load_scaler("dummy_non_existent_mode_123")


class TestDescriptorsDirect:
    def test_descriptors_from_predictor(self):
        from src.features import _descriptors

        desc = _descriptors("CCO")
        assert len(desc) == 7
        assert desc[0] > 0  # MW
