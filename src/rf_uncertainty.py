# src/rf_uncertainty.py
import numpy as np

def rf_predict_with_uncertainty(model, X):
    """
    Returns:
      y_pred_mean: mean prediction over trees
      y_pred_std: std deviation over trees (uncertainty proxy)
    """
    tree_preds = np.array([t.predict(X) for t in model.estimators_])  # (n_trees, n_samples)
    y_mean = tree_preds.mean(axis=0)
    y_std = tree_preds.std(axis=0)
    return y_mean, y_std

def uncertainty_error_correlation(y_true, y_pred, y_unc):
    """
    Correlation between absolute error and uncertainty.
    """
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    unc = np.asarray(y_unc)
    # handle edge cases
    if np.all(np.isnan(unc)) or np.std(unc) == 0:
        return np.nan
    return float(np.corrcoef(err, unc)[0, 1])