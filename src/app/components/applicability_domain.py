from __future__ import annotations

from typing import Optional

from src.chem_utils import nearest_tanimoto as _nearest_tanimoto


def nearest_tanimoto(smiles: str) -> Optional[float]:
    return _nearest_tanimoto(smiles)
