from __future__ import annotations

import pickle
import os
from functools import lru_cache
from typing import Optional
from pathlib import Path # Ensure Path is imported

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

# Calculate the absolute path to the root of your project
# This assumes chem_utils.py is inside the src/ folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

def draw_2d_svg(smiles: str, size: tuple[int, int] = (400, 300)) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None

def generate_3d_conformer(smiles: str) -> tuple[Optional[str], float, float]:
    """
    Generates a 3D conformer using ETKDGv3, optimizes it using MMFF94 force field,
    and calculates Gasteiger partial charges.
    Returns (mol_block, min_charge, max_charge).
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, 0.0, 0.0
    
    try:
        # Add hydrogens for proper 3D geometry
        mol_3d = Chem.AddHs(mol)
        
        # Generate 3D coordinates using ETKDGv3
        embed_status = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
        if embed_status != 0:
            # Fallback to standard distance geometry if ETKDGv3 fails
            embed_status = AllChem.EmbedMolecule(mol_3d)
            
        if embed_status == 0:
            # Optimize structure using MMFF94 force field
            AllChem.MMFFOptimizeMolecule(mol_3d)
            
        # Compute Gasteiger partial charges
        AllChem.ComputeGasteigerCharges(mol_3d)
        
        # Extract charge bounds
        charges = []
        for atom in mol_3d.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                try:
                    c = float(atom.GetProp("_GasteigerCharge"))
                    if not np.isnan(c) and not np.isinf(c):
                        charges.append(c)
                except ValueError:
                    pass
        
        min_charge = min(charges) if charges else 0.0
        max_charge = max(charges) if charges else 0.0
        
        # Convert to Mol block string
        mol_block = Chem.MolToMolBlock(mol_3d)
        return mol_block, min_charge, max_charge
    except Exception:
        # Fallback: compute Gasteiger charges on 2D mol if 3D conformer fails
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                if atom.HasProp("_GasteigerCharge"):
                    c = float(atom.GetProp("_GasteigerCharge"))
                    if not np.isnan(c) and not np.isinf(c):
                        charges.append(c)
            min_charge = min(charges) if charges else 0.0
            max_charge = max(charges) if charges else 0.0
            return None, min_charge, max_charge
        except Exception:
            return None, 0.0, 0.0

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
    # Use absolute path based on PROJECT_ROOT
    path = PROJECT_ROOT / "data" / "processed" / "train_fps.pkl"
    with open(path, "rb") as f:
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
    # Use absolute path based on PROJECT_ROOT
    path = PROJECT_ROOT / "data" / "processed" / "train_smiles.pkl"
    with open(path, "rb") as f:
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

def generate_pdb_block(smiles: str) -> Optional[str]:
    """Generates a 3D conformer and returns the PDB block string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        mol_3d = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
        if embed_status != 0:
            embed_status = AllChem.EmbedMolecule(mol_3d)
        if embed_status == 0:
            AllChem.MMFFOptimizeMolecule(mol_3d)
        return Chem.MolToPDBBlock(mol_3d)
    except Exception:
        return None