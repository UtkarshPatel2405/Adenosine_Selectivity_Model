# src/inject_mock_docking.py
import json
import random
from pathlib import Path

def inject_docking_scores():
    db_path = Path("data/processed/db_lookup.json")
    if not db_path.exists():
        print("[ERROR] db_lookup.json not found!")
        return
        
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    print(f"Loaded {len(db)} compounds from database. Injecting mock docking scores...")
    
    random.seed(42)  # For deterministic reproducibility
    
    SUBTYPES = ["A1", "A2A", "A2B", "A3"]
    
    for smiles, data in db.items():
        docking = {}
        for sub in SUBTYPES:
            # If the compound has an experimental pChEMBL value for this subtype, correlate docking score
            if sub in data:
                pchembl = float(data[sub])
                # Stronger binder -> lower (more negative) docking energy
                base_score = - (pchembl * 0.9 + 2.5)
                # Add minor random noise
                score = base_score + random.uniform(-0.8, 0.8)
            else:
                # If not tested, generate a weaker/inactive docking score
                score = -5.0 + random.uniform(-1.0, 1.0)
                
            docking[sub] = round(score, 2)
            
        data["docking"] = docking
        
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, sort_keys=True)
        
    print("[SUCCESS] Mock docking scores successfully injected into data/processed/db_lookup.json")

if __name__ == "__main__":
    inject_docking_scores()
