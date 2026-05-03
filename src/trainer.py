import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


DEFAULT_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)


def _fit_xgb(X, y, params):
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def _bootstrap_indices(n, rng):
    return rng.integers(0, n, size=n, endpoint=False)


def train_models(train_df, X_train, out_dir="models", n_bootstrap=5):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train_rows": int(len(train_df)),
        "default_params": DEFAULT_PARAMS,
        "n_bootstrap": int(n_bootstrap),
        "per_subtype_train_size": {},
        "artifacts": {},
    }

    rng = np.random.default_rng(42)

    def train_bundle(name, mask=None):
        if mask is None:
            X = X_train
            y = train_df["pchembl_value"].to_numpy(dtype=np.float32)
        else:
            X = X_train[mask]
            y = train_df.loc[mask, "pchembl_value"].to_numpy(dtype=np.float32)

        n = X.shape[0]
        models = []
        for b in range(n_bootstrap):
            idx = _bootstrap_indices(n, rng)
            params = dict(DEFAULT_PARAMS)
            params["random_state"] = int(1000 + b)
            m = _fit_xgb(X[idx], y[idx], params)
            models.append(m)

        fname = f"xgb_{name}_ens.pkl"
        with open(f"{out_dir}/{fname}", "wb") as f:
            pickle.dump(models, f)

        meta["artifacts"][name] = fname
        return models

    # global ensemble
    train_bundle("global", mask=None)

    # subtype ensembles
    for st in ["A1", "A2A", "A2B", "A3"]:
        mask = (train_df["target_subtype"] == st).to_numpy()
        meta["per_subtype_train_size"][st] = int(mask.sum())
        if mask.sum() < 30:
            continue
        train_bundle(st, mask=mask)

    with open(f"{out_dir}/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return meta