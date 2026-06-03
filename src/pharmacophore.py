# src/pharmacophore.py
from typing import Dict, Any, List
from rdkit import Chem

class AdenosinePharmacophoreAnalyzer:
    """
    Chemically rigorous 2D pharmacophore matcher for Adenosine Receptors (A1, A2A, A2B, A3).
    Evaluates key features required for orthosteric GPCR pocket binding and subtype selectivity.
    """
    
    # Core binding features
    CORE_FEATURES = {
        "Purine/Adenine Core": "c1nc2c(n1)ncnc2",
        "Xanthine Core": "O=C1NC(=O)c2[nH]cnc2N1",
        "General Fused Bicyclic Aromatic (6-5)": "a12aaaaa1aaa2",
        "General Fused Bicyclic Aromatic (6-6)": "a12aaaaa1aaaa2",
        "Conserved H-Bond Donor (Asn253/Glu169 interaction)": "[!#6;!H0]",
        "Conserved H-Bond Acceptor": "[#7,#8,#9]"
    }
    
    # Subtype-specific selectivity features
    SUBTYPE_SELECTIVITY_FEATURES = {
        "A1": {
            "Exocyclic Bulky N6 Substituent (A1 Selectivity)": "[NH]-C1CCCC1", # e.g., cyclopentyl-adenosine derivatives
            "Aromatic Secondary Amine": "c-[NH]-[#6]"
        },
        "A2A": {
            "Conserved Furan/Triazole Ring (A2A Selectivity)": "o1cccc1", # Furan substituent found in SCH58261 / ZM241385
            "Triazole/Pyrazole Linker": "n1cnn1"
        },
        "A2B": {
            "Xanthine derivative N-alkyl chains (A2B Selectivity)": "O=C1NC(=O)c2[nH]cnc2N1CCN", 
            "Amide Linker Motif": "C(=O)NC"
        },
        "A3": {
            "N6-Benzyl / Exocyclic Halobenzyl (A3 Selectivity)": "[NH]-Cc1ccccc1", # Exocyclic benzylamine motif
            "Ribose-like Carboxamide Modification": "C(O)C(=O)NC"
        }
    }
    
    @classmethod
    def analyze_molecule(cls, smiles: str) -> Dict[str, Any]:
        """
        Analyzes a compound SMILES for pharmacophore matching across all AR subtypes.
        Returns a dictionary containing matching scores, detected features, and explanations.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
            
        results = {
            "core_features": {},
            "subtype_features": {st: {} for st in ["A1", "A2A", "A2B", "A3"]},
            "scores": {st: 0.0 for st in ["A1", "A2A", "A2B", "A3"]},
            "summary": {}
        }
        
        # Check core binding features (2 points max)
        core_matched = 0
        
        # Check Pi-Stacking core
        pi_stacking = False
        for name, smarts in cls.CORE_FEATURES.items():
            if "Core" in name or "Bicyclic" in name:
                pat = Chem.MolFromSmarts(smarts)
                if pat and mol.HasSubstructMatch(pat):
                    pi_stacking = True
                    results["core_features"]["Pi-Stacking Core"] = {
                        "matched": True,
                        "description": f"Contains core scaffold: {name}"
                    }
                    break
        
        if not pi_stacking:
            # Fallback check for any aromatic ring
            any_aromatic = Chem.MolFromSmarts("a1aaaaa1")
            if any_aromatic and mol.HasSubstructMatch(any_aromatic):
                pi_stacking = True
                results["core_features"]["Pi-Stacking Core"] = {
                    "matched": True,
                    "description": "Contains generic monocyclic aromatic ring"
                }
            else:
                results["core_features"]["Pi-Stacking Core"] = {
                    "matched": False,
                    "description": "Missing required aromatic ring for Pi-Pi stacking with conserved Phe168"
                }
                
        if pi_stacking:
            core_matched += 1.0
            
        # Check H-Bond Donor/Acceptor network
        hbd_pat = Chem.MolFromSmarts(cls.CORE_FEATURES["Conserved H-Bond Donor (Asn253/Glu169 interaction)"])
        hba_pat = Chem.MolFromSmarts(cls.CORE_FEATURES["Conserved H-Bond Acceptor"])
        
        hbd_match = hbd_pat and mol.HasSubstructMatch(hbd_pat)
        hba_match = hba_pat and mol.HasSubstructMatch(hba_pat)
        
        if hbd_match and hba_match:
            core_matched += 1.0
            results["core_features"]["H-Bond Network"] = {
                "matched": True,
                "description": "Valid H-bond donor & acceptor network present"
            }
        else:
            results["core_features"]["H-Bond Network"] = {
                "matched": False,
                "description": "Incomplete H-bond network (requires at least one donor and one acceptor)"
            }
            
        # Check subtype selectivity features
        for st, features in cls.SUBTYPE_SELECTIVITY_FEATURES.items():
            st_matched = 0
            total_st_features = len(features)
            
            for name, smarts in features.items():
                pat = Chem.MolFromSmarts(smarts)
                matched = bool(pat and mol.HasSubstructMatch(pat))
                results["subtype_features"][st][name] = matched
                if matched:
                    st_matched += 1
            
            # Subtype score = Core score (out of 2) + Subtype specific score
            # Normalize to 0-100%
            subtype_specific_score = (st_matched / total_st_features) if total_st_features > 0 else 0
            total_score = (core_matched + subtype_specific_score) / 3.0
            results["scores"][st] = round(total_score * 100, 1)
            
        return results
