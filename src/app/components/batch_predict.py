from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from typing import Any, Dict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from src.predictor import _load_scaler, _load_db_lookup, _load_models, SUBTYPES
from src.features import _morgan_bits, build_features 

_SMILES_ALIASES = ["smiles", "SMILES", "Smiles", "canonical_smiles", "smi", "SMI"]

def _infer_smiles_col(df: pd.DataFrame) -> str:
    for alias in _SMILES_ALIASES:
        if alias in df.columns: return alias
    return df.columns[0]

def predict_batch(
    df: pd.DataFrame, 
    threshold: float = 6.0, 
    smiles_col: str | None = None,
    show_progress: bool = True
) -> pd.DataFrame:
    col = smiles_col or _infer_smiles_col(df)
    scaler = _load_scaler()
    lookup = _load_db_lookup()
    models = _load_models()
    
    total_mols = len(df)
    
    status_area = st.empty() if show_progress else None
    progress_bar = st.progress(0) if show_progress else None

    try:
        with open("data/processed/train_fps.pkl", "rb") as f:
            train_fps = pickle.load(f)
    except FileNotFoundError:
        train_fps = None
        
    res_df = df.copy()
    res_df['canonical_smiles'] = None
    res_df['error'] = None
    res_df['in_database'] = False
    res_df['source'] = "model"
    
    for st_name in SUBTYPES:
        res_df[st_name] = np.nan
        res_df[f"{st_name}_uncertainty"] = 0.0
    res_df['reliability'] = 0.0

    for i, (idx, row) in enumerate(res_df.iterrows()):
        if status_area:
            status_area.text(f" Phase 1/3: Validating SMILES ({i+1}/{total_mols})")
            progress_bar.progress((i + 1) / (total_mols * 3))

        raw_smi = str(row[col]).strip()
        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            res_df.at[idx, 'error'] = "Invalid SMILES"
            continue
        
        canon = Chem.MolToSmiles(mol, canonical=True)
        res_df.at[idx, 'canonical_smiles'] = canon
        
        if canon in lookup:
            res_df.at[idx, 'in_database'] = True
            res_df.at[idx, 'source'] = "database"
            for st_name in SUBTYPES:
                res_df.at[idx, st_name] = lookup[canon].get(st_name, np.nan)

   
    predict_mask = res_df['canonical_smiles'].notna() & (~res_df['in_database'])
    
    if predict_mask.any():
        to_predict = res_df.loc[predict_mask, 'canonical_smiles'].tolist()
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
            member_preds = np.array([m.predict(x_batch) for m in ensemble])
            res_df.loc[predict_mask, st_name] = member_preds.mean(axis=0)
            res_df.loc[predict_mask, f"{st_name}_uncertainty"] = member_preds.std(axis=0)

        if train_fps:
            query_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in to_predict]
            reliability_scores = []
            for q_fp in query_fps:
                sims = DataStructs.BulkTanimotoSimilarity(q_fp, train_fps)
                reliability_scores.append(max(sims))
            res_df.loc[predict_mask, 'reliability'] = reliability_scores

    valid_mask = res_df[SUBTYPES].notna().any(axis=1)
    
    if valid_mask.any():
        if status_area: status_area.text("Finalizing result dashboard...")
        
        res_df.loc[valid_mask, 'best_target'] = res_df.loc[valid_mask, SUBTYPES].idxmax(axis=1)
        sorted_vals = np.sort(res_df.loc[valid_mask, SUBTYPES].values, axis=1)
        res_df.loc[valid_mask, 'selectivity_score'] = sorted_vals[:, -1] - sorted_vals[:, -2]

        res_df.loc[valid_mask, 'target_hits'] = res_df.loc[valid_mask, SUBTYPES].apply(
            lambda r: [st_name for st_name in SUBTYPES if r[st_name] > threshold], axis=1
        )

    if status_area: status_area.empty()
    if progress_bar: progress_bar.empty()

    return res_df
