from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    _FILTER_CATALOG_AVAILABLE = True
except ImportError:
    _FILTER_CATALOG_AVAILABLE = False

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)

def canonicalize(smiles: str) -> Optional[str]:
    
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_smiles(smiles: str):
    
    canon = canonicalize(smiles)
    if canon is None:
        return None
    return Chem.MolFromSmiles(canon)

def draw_2d(smiles: str, size: tuple[int, int] = (400, 300)):
    
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None

@lru_cache(maxsize=1)
def _build_pains_catalog():
    if not _FILTER_CATALOG_AVAILABLE:
        return None
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def check_pains(smiles: str) -> list[str]:
    
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    catalog = _build_pains_catalog()
    if catalog is None:
        return []
    matches = []
    for entry in catalog.GetMatches(mol):
        matches.append(entry.GetDescription())
    return matches

def qed_profile(smiles: str) -> Optional[dict]:
   
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return {
        "QED": round(QED.qed(mol), 4),
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotB": int(Lipinski.NumRotatableBonds(mol)),
        "AromaticRings": int(Lipinski.NumAromaticRings(mol)),
        "TPSA": round(Descriptors.TPSA(mol), 2),
    }

@lru_cache(maxsize=1)
def _load_train_fps():
    
    with open("data/processed/train_fps.pkl", "rb") as f:
        return pickle.load(f)


def nearest_tanimoto(smiles: str) -> Optional[float]:
    
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        train_fps = _load_train_fps()
    except FileNotFoundError:
        return None
    qfp = _MORGAN.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
    return float(np.max(sims)) if sims else None

@lru_cache(maxsize=1)
def _load_train_smiles() -> list[str]:
    with open("data/processed/train_smiles.pkl", "rb") as f:
        return pickle.load(f)


def topk_tanimoto(smiles: str, k: int = 5) -> tuple[Optional[str], list[tuple[str, float]]]:
    
    canon = canonicalize(smiles)
    if canon is None:
        return None, []
    mol = Chem.MolFromSmiles(canon)
    qfp = _MORGAN.GetFingerprint(mol)
    try:
        train_fps = _load_train_fps()
        train_smiles = _load_train_smiles()
    except FileNotFoundError:
        return canon, []
    sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
    idx = np.argsort(sims)[::-1][:k]
    top = [(train_smiles[i], float(sims[i])) for i in idx]
    return canon, top
