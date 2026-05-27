from src.chem_utils import draw_2d as _draw_2d, draw_2d_svg as _draw_2d_svg, generate_3d_conformer as _generate_3d_conformer


def draw_2d(smiles: str, size: tuple[int, int] = (400, 300)):
    return _draw_2d(smiles, size=size)

def draw_2d_svg(smiles: str, size: tuple[int, int] = (400, 300)):
    return _draw_2d_svg(smiles, size=size)

def generate_3d_conformer(smiles: str):
    return _generate_3d_conformer(smiles)
