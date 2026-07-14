import json
from pathlib import Path

def patch_a3_manually():
    json_path = Path("data/processed/adenosine_pdb_ligands.json")
    if json_path.exists():
        with open(json_path) as f:
            registry = json.load(f)
    else:
        registry = {}
        
    registry["A3"] = [
        {
            "pdb_id": "7LD3",
            "ligands": [
                {
                    "ccd": "XTD",
                    "name": "{2-amino-4-[3,5-bis(trifluoromethyl)phenyl]thiophen-3-yl}(4-chlorophenyl)methanone",
                    "smiles": "c1cc(ccc1C(=O)c2c(csc2N)c3cc(cc(c3)C(F)(F)F)C(F)(F)F)Cl",
                    "formula": "C19 H10 Cl F6 N O S",
                    "mw": 449.797
                }
            ]
        },
        {
            "pdb_id": "8J78",
            "ligands": [
                {
                    "ccd": "BTI",
                    "name": "5-(HEXAHYDRO-2-OXO-1H-THIENO[3,4-D]IMIDAZOL-6-YL)PENTANAL",
                    "smiles": "C1C2C(C(S1)CCCCC=O)NC(=O)N2",
                    "formula": "C10 H16 N2 O2 S",
                    "mw": 228.311
                }
            ]
        }
    ]
    
    if "A2A" not in registry:
        registry["A2A"] = []
        
    if not any(x.get("pdb_id") == "8RLN" for x in registry["A2A"]):
        registry["A2A"].append({
            "pdb_id": "8RLN",
            "ligands": [
                {
                    "ccd": "A1H1S",
                    "name": "2-amino-4-(4-hydroxyphenyl)-6-[(1H-imidazol-2-ylmethyl)thio]-3,5-pyridinedicarbonitrile",
                    "smiles": "NC1=NC(SCC3=NC=CN3)=C(C#N)C(C2=CC=C(O)C=C2)=C1C#N",
                    "formula": "C17 H12 N6 O S",
                    "mw": 348.38
                }
            ]
        })
    
    with open(json_path, "w") as f:
        json.dump(registry, f, indent=2)
    print("Successfully patched A3 and A2A manually in adenosine_pdb_ligands.json!")

if __name__ == "__main__":
    patch_a3_manually()
