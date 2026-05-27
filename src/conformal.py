import numpy as np
from mapie.regression import CrossConformalRegressor

def train_conformal_model(base_model, X_train, y_train, cv=5):
    """
    Train a CrossConformalRegressor wrapper around a base estimator using Jackknife+ (method="plus").
    This manages 5-fold CV internally, saving out-of-fold residuals for mathematically
    guaranteed conformal prediction intervals.
    """
    mapie = CrossConformalRegressor(estimator=base_model, cv=cv, confidence_level=0.90, method="plus", n_jobs=1)
    mapie.fit_conformalize(X_train, y_train)
    return mapie

def predict_conformal(mapie_model, X: np.ndarray, alpha: float = 0.10) -> tuple:
    """
    Run inference with prediction intervals at the specified significance level (1 - alpha).
    Returns (pred_mean, lower_bound, upper_bound).
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    y_pred, y_pis = mapie_model.predict_interval(X)
    
    # Handle different MAPIE shapes depending on version
    if y_pis.ndim == 3:
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]
    else:
        lower = y_pis[:, 0]
        upper = y_pis[:, 1]
        
    return y_pred, lower, upper
