import logging

import numpy as np
from mapie.regression import CrossConformalRegressor

from src.config import MAPIE_CV_FOLDS, MAPIE_CONFIDENCE

logger = logging.getLogger(__name__)


def train_conformal_model(base_model, X_train, y_train, cv=MAPIE_CV_FOLDS):
    """
    Train a CrossConformalRegressor wrapper around a base estimator.

    Uses Jackknife+ (method="plus") for mathematically guaranteed
    conformal prediction intervals at the specified confidence level.
    """
    mapie = CrossConformalRegressor(
        estimator=base_model,
        cv=cv,
        confidence_level=MAPIE_CONFIDENCE,
        method="plus",
        n_jobs=1,
    )
    mapie.fit_conformalize(X_train, y_train)
    logger.info("Conformal model trained: cv=%d, confidence=%.2f", cv, MAPIE_CONFIDENCE)
    return mapie


def predict_conformal(mapie_model, X: np.ndarray) -> tuple:
    """
    Run inference with prediction intervals.

    NOTE: MAPIE >= 0.8 (confirmed on 1.4.1) no longer accepts alpha or
    confidence_level in predict_interval(). The confidence level is set
    at construction time in CrossConformalRegressor(confidence_level=...).
    predict_interval() without kwargs returns intervals at that fixed level.

    Returns (pred_mean, lower_bound, upper_bound, std_equiv).
    std_equiv = interval_width / 3.29, mapping the 90% CI to
    an equivalent Gaussian standard deviation for downstream use.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    y_pred, y_pis = mapie_model.predict_interval(X)

    # pis shape: (n_samples, 2, n_confidence_levels) when 3D
    # dim-0: samples, dim-1: [lower=0, upper=1], dim-2: confidence levels
    # or (n_samples, 2) when single confidence level
    ci_idx = 0  # index into the confidence level dimension (3rd dim)

    if y_pis.ndim == 3:
        lower = y_pis[:, 0, ci_idx]
        upper = y_pis[:, 1, ci_idx]
    else:
        lower = y_pis[:, 0]
        upper = y_pis[:, 1]

    # 90% CI width / 3.29 ≈ std for Gaussian
    std_equiv = (upper - lower) / 3.29

    return y_pred, lower, upper, std_equiv
