from __future__ import annotations

from typing import Optional

from src.chem_utils import draw_2d as _draw_2d


def draw_2d(smiles: str, size: tuple[int, int] = (400, 300)):
    
    return _draw_2d(smiles, size=size)
