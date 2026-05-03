from __future__ import annotations

from src.chem_utils import check_pains as _check_pains


def check_pains(smiles: str) -> list[str]:
  
    return _check_pains(smiles)
