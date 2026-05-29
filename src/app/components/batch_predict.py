import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import pickle
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from src.predictor import _load_scaler, _load_db_lookup, _load_models, SUBTYPES
from src.features import build_features

_SMILES_ALIASES = ["smiles", "SMILES", "Smiles", "canonical_smiles", "smi", "SMI"]

def _infer_smiles_col(df: pd.DataFrame) -> str:
    for alias in _SMILES_ALIASES:
        if alias in df.columns:
            return alias
    return df.columns[0]

def predict_batch(
    df: pd.DataFrame, 
    threshold: float = 6.0, 
    smiles_col: str | None = None,
    show_progress: bool = True,
    mode: str = "standard"
) -> pd.DataFrame:
    col = smiles_col or _infer_smiles_col(df)
    scaler = _load_scaler(mode=mode)
    lookup = _load_db_lookup()
    models = _load_models(mode=mode)
    
    total_mols = len(df)
    
    status_area = st.empty() if show_progress else None
    progress_bar = st.progress(0) if show_progress else None

    # Load train_fps safely for applicability domain
    try:
        with open("data/processed/train_fps.pkl", "rb") as f:
            train_fps = pickle.load(f)
    except FileNotFoundError:
        train_fps = None

    res_df = df.copy()
    res_df['canonical_smiles'] = None
    res_df['error'] = None
    res_df['in_database'] = False
    
    # Initialize receptor columns
    for st_name in SUBTYPES:
        res_df[st_name] = np.nan
        res_df[f"{st_name}_uncertainty"] = 0.0
    res_df['reliability'] = 0.0

    # Phase 1: Canonicalization and DB Check
    for i, (idx, row) in enumerate(res_df.iterrows()):
        if status_area:
            status_area.text(f"Phase 1/3: Validating SMILES ({i+1}/{total_mols})")
            progress_bar.progress((i + 1) / (total_mols * 3))

        raw_smi = str(row[col]).strip()
        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            res_df.at[idx, 'error'] = "Invalid SMILES"
            continue
        
        canon = Chem.MolToSmiles(mol, canonical=True)
        res_df.at[idx, 'canonical_smiles'] = canon
        
        # Check database hit
        if canon in lookup:
            res_df.at[idx, 'in_database'] = True
            exp = lookup[canon]
            for st_name in SUBTYPES:
                val = exp.get(st_name)
                # Professor's Rule: Use DB value if exists, else 0.0
                res_df.at[idx, st_name] = float(val) if pd.notna(val) else 0.0
                res_df.at[idx, f"{st_name}_uncertainty"] = 0.0

    # Phase 2: ML Prediction for Novel Molecules (Not in DB)
    novel_mask = (res_df['in_database'] == False) & (res_df['canonical_smiles'].notna())
    if novel_mask.any():
        to_predict = res_df.loc[novel_mask, 'canonical_smiles'].tolist()
        total_predict = len(to_predict)
        
        x_list = []
        for j, s in enumerate(to_predict):
            if status_area:
                status_area.text(f"Phase 2/3: Featurizing novel molecules ({j+1}/{total_predict})")
                progress_bar.progress(0.33 + ((j + 1) / (total_predict * 3)))
            x_list.append(build_features(s, scaler))
        
        x_batch = np.array(x_list)
        
        for k, st_name in enumerate(SUBTYPES):
            if status_area:
                status_area.text(f"Phase 3/3: Running Inference for {st_name}...")
                progress_bar.progress(0.66 + ((k + 1) / (len(SUBTYPES) * 3)))
            
            ensemble = models[st_name]
            
            # Prediction handling supporting MapieRegressor, CrossConformalRegressor, Legacy Ensembles, and Single Models
            if type(ensemble).__name__ == "CrossConformalRegressor":
                y_pred, y_pis = ensemble.predict_interval(x_batch)
                res_df.loc[novel_mask, st_name] = y_pred
                if y_pis.ndim == 3:
                    lower = y_pis[:, 0, 0]
                    upper = y_pis[:, 1, 0]
                else:
                    lower = y_pis[:, 0]
                    upper = y_pis[:, 1]
                res_df.loc[novel_mask, f"{st_name}_uncertainty"] = (upper - lower) / 3.29
                res_df.loc[novel_mask, f"{st_name}_lower"] = lower
                res_df.loc[novel_mask, f"{st_name}_upper"] = upper
            elif type(ensemble).__name__ == "MapieRegressor":
                y_pred, y_pis = ensemble.predict(x_batch, alpha=0.10)
                res_df.loc[novel_mask, st_name] = y_pred
                if y_pis.ndim == 3:
                    lower = y_pis[:, 0, 0]
                    upper = y_pis[:, 1, 0]
                else:
                    lower = y_pis[:, 0]
                    upper = y_pis[:, 1]
                res_df.loc[novel_mask, f"{st_name}_uncertainty"] = (upper - lower) / 3.29
                res_df.loc[novel_mask, f"{st_name}_lower"] = lower
                res_df.loc[novel_mask, f"{st_name}_upper"] = upper
            elif isinstance(ensemble, (list, tuple)):
                member_preds = np.array([m.predict(x_batch) for m in ensemble])
                res_df.loc[novel_mask, st_name] = member_preds.mean(axis=0)
                res_df.loc[novel_mask, f"{st_name}_uncertainty"] = member_preds.std(axis=0, ddof=0)
            else:
                pred_mean = ensemble.predict(x_batch)
                res_df.loc[novel_mask, st_name] = pred_mean
                res_df.loc[novel_mask, f"{st_name}_uncertainty"] = 0.0

        # Calculate AD / Reliability for novel molecules
        if train_fps:
            query_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in to_predict]
            reliability_scores = []
            for q_fp in query_fps:
                sims = DataStructs.BulkTanimotoSimilarity(q_fp, train_fps)
                reliability_scores.append(max(sims))
            res_df.loc[novel_mask, 'reliability'] = reliability_scores

    valid_mask = res_df['canonical_smiles'].notna()
    if valid_mask.any():
        if status_area: status_area.text("Finalizing result dashboard...")
        res_df.loc[valid_mask, 'best_target'] = res_df.loc[valid_mask, SUBTYPES].idxmax(axis=1)
        res_df.loc[valid_mask, 'target_hits'] = res_df.loc[valid_mask, SUBTYPES].apply(
            lambda r: ", ".join([st for st in SUBTYPES if r[st] >= threshold]), axis=1
        )

    # Clean up the UI
    if status_area: status_area.empty()
    if progress_bar: progress_bar.empty()

    return res_df