from __future__ import annotations

from typing import Optional

from src.chem_utils import qed_profile as _qed_profile


def qed_profile(smiles: str) -> Optional[dict]:

    return _qed_profile(smiles)
