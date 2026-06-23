import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.conformal import train_conformal_model, predict_conformal
from src.config import MAPIE_CONFIDENCE


class TestTrainConformalModel:
    def test_returns_crossconformal_regressor(self):
        X = np.random.rand(50, 10)
        y = np.random.rand(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        wrapped = train_conformal_model(model, X, y, cv=3)
        type_name = type(wrapped).__name__
        assert type_name in ("CrossConformalRegressor",), f"Unexpected type: {type_name}"


class TestPredictConformal:
    def test_returns_prediction_and_intervals(self):
        X_train = np.random.rand(50, 10)
        y_train = np.random.rand(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        wrapped = train_conformal_model(model, X_train, y_train, cv=3)

        X_test = np.random.rand(5, 10)
        y_pred, lower, upper, std = predict_conformal(wrapped, X_test)

        assert len(y_pred) == 5
        assert len(lower) == 5
        assert len(upper) == 5
        assert np.all(lower <= y_pred)
        assert np.all(y_pred <= upper)
        assert np.all(std >= 0)

    def test_single_sample(self):
        X_train = np.random.rand(50, 10)
        y_train = np.random.rand(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        wrapped = train_conformal_model(model, X_train, y_train, cv=3)

        x_single = np.random.rand(10)
        y_pred, lower, upper, std = predict_conformal(wrapped, x_single)

        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == 1

    def test_conformal_coverage(self):
        """
        Statistical test: for a well-calibrated 90% conformal model,
        coverage should be approximately 90% on held-out data.
        """
        np.random.seed(42)
        n = 100
        X = np.random.rand(n, 5)
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1

        model = RandomForestRegressor(n_estimators=20, random_state=42)
        wrapped = train_conformal_model(model, X[:70], y[:70], cv=3)

        X_test = X[70:]
        y_test = y[70:]
        _, lower, upper, _ = predict_conformal(wrapped, X_test)

        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        assert coverage >= 0.70, f"Coverage too low: {coverage:.3f} (expected ~0.90)"
