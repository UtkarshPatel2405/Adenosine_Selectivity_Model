import json
import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from src.data_loader import load_and_clean
from src.features import build_feature_matrix, build_features
from src.conformal import train_conformal_model
from src.scaffold_split import split_smiles_globally
from src.smiles_registry import SmilesRegistry
from src.config import (
    SUBTYPES, MODELS_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR,
    SCAFFOLD_SPLIT_SEED, SCAFFOLD_TEST_SIZE, RF_N_ESTIMATORS,
    RF_MAX_DEPTH, RF_MAX_FEATURES, MAPIE_CV_FOLDS, LOG_LEVEL, RUN_ID,
)

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "A1": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
           "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2A": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A2B": {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
    "A3": {"n_estimators": 800, "max_depth": 7, "learning_rate": 0.05,
           "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2},
}


def retrain_production_models(data_path: str = "data/raw"):
    """
    Full-dataset training with MAPIE conformal wrapping.

    CRITICAL FIX:
    - The old code saved raw XGBRegressors without conformal wrapping.
    - Now wraps every XGBoost model with CrossConformalRegressor (Jackknife+)
      to produce real uncertainty estimates.
    - The cost: slightly higher training time (5-fold CV inside MAPIE).
    - The benefit: mathematically valid 90% prediction intervals.
    """
    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING WITH MAPIE CONFORMAL WRAPPING")
    logger.info("=" * 60)

    registry = SmilesRegistry()

    cv_report_path = OUTPUTS_DIR / "nested_cv" / "merged_report.json"
    best_params_per_subtype = {}

    if cv_report_path.exists():
        logger.info("Found Nested CV report at %s. Loading HPO params...", cv_report_path)
        with open(cv_report_path) as f:
            cv_data = json.load(f)
        for st in SUBTYPES:
            if st in cv_data:
                best_params_per_subtype[st] = cv_data[st]["median_params"]
                logger.info("  %s: %s", st, best_params_per_subtype[st])
    else:
        logger.info("No Nested CV HPO params found. Using scientific fallbacks.")
        best_params_per_subtype = DEFAULT_PARAMS

    df, _ = load_and_clean(data_path, mode="precise", include_decoys=True)
    df = df.rename(columns={"canonical_smiles": "smiles"})

    if "barcode" in df.columns:
        logger.info("BARCODE AUDIT: %d rows, %d unique barcodes",
                     len(df), df["barcode"].nunique())

    train_smiles, test_smiles = split_smiles_globally(
        df["smiles"].unique(),
        test_size=SCAFFOLD_TEST_SIZE,
        random_state=SCAFFOLD_SPLIT_SEED,
    )

    split_path = PROCESSED_DATA_DIR / "global_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": list(train_smiles), "test": list(test_smiles)}, f)
    logger.info("Saved global scaffold split to %s", split_path)

    train_df = df[df["smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df["smiles"].isin(test_smiles)].reset_index(drop=True)

    train_lookup = {}
    for smi, subdf in train_df.groupby("smiles"):
        train_lookup[smi] = {
            row["target_subtype"]: float(row["pchembl_value"])
            for _, row in subdf.iterrows()
        }
    train_lookup_path = PROCESSED_DATA_DIR / "db_lookup_train.json"
    with open(train_lookup_path, "w") as f:
        json.dump(train_lookup, f, indent=2, sort_keys=True)
    logger.info("Saved train-only lookup (%d molecules) to %s", len(train_lookup), train_lookup_path)

    X_train_glob, X_test_glob, pipeline = build_feature_matrix(
        train_df, test_df, smiles_col="smiles"
    )

    models_precise_dir = MODELS_DIR / "precise"
    models_precise_dir.mkdir(parents=True, exist_ok=True)

    with open(models_precise_dir / f"scaler_precise.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Saved production scaler pipelines.")

    with open(PROCESSED_DATA_DIR / "features_train.pkl", "wb") as f:
        pickle.dump(X_train_glob, f)
    with open(PROCESSED_DATA_DIR / "features_test.pkl", "wb") as f:
        pickle.dump(X_test_glob, f)
    logger.info("Saved feature matrices.")

    training_summary = {}

    for st in SUBTYPES:
        logger.info("\nTraining %s...", st)

        train_mask = (train_df["target_subtype"] == st).values
        test_mask = (test_df["target_subtype"] == st).values

        X_tr = X_train_glob[train_mask]
        y_tr = train_df.loc[train_mask, "pchembl_value"].values

        X_te = X_test_glob[test_mask]
        y_te = test_df.loc[test_mask, "pchembl_value"].values

        if len(y_tr) < 50:
            logger.warning("SKIP %s: insufficient data (%d samples).", st, len(y_tr))
            continue

        logger.info("  Samples: train=%d, test=%d", len(y_tr), len(y_te))

        params = best_params_per_subtype[st].copy()
        params.update({"tree_method": "hist", "n_jobs": 2, "random_state": 42})
        logger.info("  Params: %s", params)

        base_xgb = xgb.XGBRegressor(**params)
        base_xgb.fit(X_tr, y_tr)

        logger.info("  Wrapping XGBoost with MAPIE CrossConformalRegressor...")
        try:
            conformal_model = train_conformal_model(
                base_model=base_xgb,
                X_train=X_tr,
                y_train=y_tr,
                cv=MAPIE_CV_FOLDS,
            )
            logger.info("  Conformal wrapping succeeded.")
        except Exception as e:
            logger.warning("  Conformal wrapping failed (%s). Saving raw model.", e)
            conformal_model = base_xgb

        rf_model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES,
            random_state=42,
            n_jobs=2,
        )
        rf_model.fit(X_tr, y_tr)

        preds_tr_xgb = conformal_model.predict(X_tr) if hasattr(conformal_model, 'predict') else base_xgb.predict(X_tr)
        preds_te_xgb = conformal_model.predict(X_te) if hasattr(conformal_model, 'predict') else base_xgb.predict(X_te)

        preds_tr_rf = rf_model.predict(X_tr)
        preds_te_rf = rf_model.predict(X_te)

        r2_tr_xgb = r2_score(y_tr, preds_tr_xgb)
        r2_te_xgb = r2_score(y_te, preds_te_xgb)
        r2_tr_rf = r2_score(y_tr, preds_tr_rf)
        r2_te_rf = r2_score(y_te, preds_te_rf)

        logger.info("    XGBoost R2: train=%.3f, test=%.3f", r2_tr_xgb, r2_te_xgb)
        logger.info("    RF R2: train=%.3f, test=%.3f", r2_tr_rf, r2_te_rf)

        xgb_path = models_precise_dir / f"xgboost_{st}_production.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(conformal_model, f)

        rf_path = models_precise_dir / f"rf_{st}_production.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf_model, f)

        training_summary[st] = {
            "train_size": int(len(y_tr)),
            "test_size": int(len(y_te)),
            "xgboost_r2": r2_te_xgb,
            "rf_r2": r2_te_rf,
            "conformal_wrapped": str(type(conformal_model).__name__),
        }
        logger.info("  Saved XGBoost model (type=%s) to %s", type(conformal_model).__name__, xgb_path)

        shutil.copy(xgb_path, MODELS_DIR / f"xgboost_precise_{st.lower()}_model.pkl")
        with open(MODELS_DIR / f"xgboost_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(conformal_model, f)

        with open(models_precise_dir / f"rf_precise_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        with open(MODELS_DIR / f"rf_{st.lower()}_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)

    from src.run_id import save_with_run_id

    training_summary["run_id"] = RUN_ID
    save_with_run_id(training_summary, OUTPUTS_DIR, "training_summary", RUN_ID)
    logger.info("Training summary saved [run_id=%s]", RUN_ID)

    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    retrain_production_models()
