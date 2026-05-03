import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def build_fps(smiles_list, n_bits=2048, radius=2):
    fps = []
    valid_mask = []
    for s in smiles_list:
        fp = morgan_fp(s, n_bits=n_bits, radius=radius)
        fps.append(fp)
        valid_mask.append(fp is not None)
    return fps, np.array(valid_mask, dtype=bool)

def nearest_tanimoto_similarity(test_fps, train_fps):
   
    sims = np.zeros(len(test_fps), dtype=float)
    for i, fp in enumerate(test_fps):
        if fp is None:
            sims[i] = np.nan
            continue
        
        s = DataStructs.BulkTanimotoSimilarity(fp, [t for t in train_fps if t is not None])
        sims[i] = float(np.max(s)) if len(s) else np.nan
    return sims

def ad_summary(similarities: np.ndarray):
    
    sims = pd.Series(similarities).dropna()
    out = {
        "n": int(sims.shape[0]),
        "mean": float(sims.mean()),
        "median": float(sims.median()),
        "pct_ge_0_4": float((sims >= 0.4).mean()),
        "pct_ge_0_5": float((sims >= 0.5).mean()),
        "pct_ge_0_6": float((sims >= 0.6).mean()),
    }
    return out
