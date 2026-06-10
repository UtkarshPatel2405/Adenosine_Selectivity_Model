"""
SMILES Barcode Registry — Deterministic molecular identity tracking and deduplication.

Every unique canonical SMILES gets a fixed 12-char barcode via SHA-256 hashing.
Duplicate SMILES (even in different representations) map to the same barcode,
preventing duplicate training and ensuring consistent molecule identity across pipelines.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from rdkit import Chem


_DEFAULT_REGISTRY_PATH = Path("data/processed/smiles_registry.json")


class SmilesRegistry:
    """Thread-safe, persistent SMILES-to-barcode mapping."""

    def __init__(self, registry_path: Optional[Path] = None):
        self.path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY_PATH
        # barcode -> canonical SMILES
        self._barcode_to_smiles: dict[str, str] = {}
        # canonical SMILES -> barcode (reverse index for fast lookup)
        self._smiles_to_barcode: dict[str, str] = {}
        self._load()

    # ── public API ──────────────────────────────────────────────

    @staticmethod
    def canonicalize(smiles: str) -> Optional[str]:
        """Canonicalize SMILES via RDKit. Returns None if invalid."""
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    @staticmethod
    def generate_barcode(canonical_smiles: str) -> str:
        """Deterministic 12-char hex barcode from canonical SMILES via SHA-256."""
        return hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:12]

    def register(self, smiles: str) -> tuple[Optional[str], bool]:
        """
        Register a SMILES string.
        Returns (barcode, is_new).
        - is_new=True  → first time this molecule is seen
        - is_new=False → duplicate detected (same barcode already exists)
        - barcode=None → invalid SMILES
        """
        canon = self.canonicalize(smiles)
        if canon is None:
            return None, False

        if canon in self._smiles_to_barcode:
            return self._smiles_to_barcode[canon], False

        barcode = self.generate_barcode(canon)
        self._barcode_to_smiles[barcode] = canon
        self._smiles_to_barcode[canon] = barcode
        return barcode, True

    def lookup_barcode(self, smiles: str) -> Optional[str]:
        """Get barcode for a SMILES without registering it."""
        canon = self.canonicalize(smiles)
        if canon is None:
            return None
        return self._smiles_to_barcode.get(canon)

    def lookup_smiles(self, barcode: str) -> Optional[str]:
        """Get canonical SMILES from a barcode."""
        return self._barcode_to_smiles.get(barcode)

    def check_duplicates(self, smiles_list: list[str]) -> dict:
        """
        Analyze a list of SMILES for duplicates using barcodes.
        Returns:
          - unique_barcodes: set of unique barcodes
          - duplicate_indices: list of (index, barcode) for duplicates
          - dedup_map: {barcode: [list of original indices]}
          - stats: summary dict
        """
        dedup_map: dict[str, list[int]] = {}
        invalid_indices: list[int] = []

        for idx, smi in enumerate(smiles_list):
            canon = self.canonicalize(smi)
            if canon is None:
                invalid_indices.append(idx)
                continue
            barcode = self.generate_barcode(canon)
            dedup_map.setdefault(barcode, []).append(idx)

        duplicate_indices = []
        for barcode, indices in dedup_map.items():
            if len(indices) > 1:
                for i in indices[1:]:
                    duplicate_indices.append((i, barcode))

        unique_barcodes = set(dedup_map.keys())

        return {
            "unique_barcodes": unique_barcodes,
            "duplicate_indices": duplicate_indices,
            "dedup_map": dedup_map,
            "invalid_indices": invalid_indices,
            "stats": {
                "total_input": len(smiles_list),
                "unique_molecules": len(unique_barcodes),
                "duplicate_rows": len(duplicate_indices),
                "invalid_smiles": len(invalid_indices),
            },
        }

    def get_registry_stats(self) -> dict:
        """Summary statistics of the current registry."""
        return {
            "total_registered": len(self._barcode_to_smiles),
            "registry_path": str(self.path),
        }

    def save(self) -> None:
        """Persist registry to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "barcode_to_smiles": self._barcode_to_smiles,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    # ── internal ────────────────────────────────────────────────

    def _load(self) -> None:
        """Load registry from disk if it exists."""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            b2s = data.get("barcode_to_smiles", {})
            self._barcode_to_smiles = b2s
            self._smiles_to_barcode = {v: k for k, v in b2s.items()}
        except (json.JSONDecodeError, KeyError):
            # Corrupted file — start fresh
            self._barcode_to_smiles = {}
            self._smiles_to_barcode = {}

    def __len__(self) -> int:
        return len(self._barcode_to_smiles)

    def __contains__(self, smiles: str) -> bool:
        canon = self.canonicalize(smiles)
        if canon is None:
            return False
        return canon in self._smiles_to_barcode


if __name__ == "__main__":
    # Quick self-test
    reg = SmilesRegistry()
    b1, new1 = reg.register("CCO")
    b2, new2 = reg.register("OCC")  # Same molecule, different representation
    assert b1 == b2, f"Barcode mismatch: {b1} != {b2}"
    assert new1 is True
    assert new2 is False
    print(f"[PASS] Barcode dedup works. Barcode for ethanol: {b1}")

    # Test batch duplicate detection
    dupes = reg.check_duplicates(["CCO", "OCC", "c1ccccc1", "CCO", "INVALID"])
    print(f"[PASS] Duplicate check: {dupes['stats']}")

    reg.save()
    print(f"[PASS] Registry saved to {reg.path}")
