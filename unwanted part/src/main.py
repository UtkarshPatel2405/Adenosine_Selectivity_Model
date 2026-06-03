import sys
import json
import os
from pathlib import Path

# Set the root directory (Adenosine_Receptor_Lingand)
# If running 'python src/main.py', the parent is 'src', parent.parent is project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.predictor import QsarPredictor

def run_workflow():
    print("🚀 [STAGE 1] Initializing Adenosine Pipeline...")
    predictor = QsarPredictor().load()
    
    print("📊 [STAGE 2] Verifying Output Directories...")
    output_path = ROOT / "outputs" / "validation_scaffold"
    output_path.mkdir(parents=True, exist_ok=True)

    print("🧠 [STAGE 3] Extracting Interpretability Data (XGBoost)...")
    importance = {}

    # Method 1: Try accessing the underlying booster (Most reliable for XGBoost)
    try:
        if hasattr(predictor.model, 'get_booster'):
            importance = predictor.model.get_booster().get_score(importance_type='gain')
            print("✅ Extracted importance via get_booster().")
    except Exception as e:
        print(f"DEBUG: Booster extraction failed: {e}")

    # Method 2: Fallback to standard Scikit-learn attribute
    if not importance and hasattr(predictor.model, 'feature_importances_'):
        importance = dict(zip(predictor.feature_columns, predictor.model.feature_importances_.tolist()))
        print("✅ Extracted importance via feature_importances_ attribute.")

    # Final Check and Save
    if importance and len(importance) > 0:
        with open(output_path / "feature_importance.json", "w") as f:
            json.dump(importance, f)
        print(f"🎯 SUCCESS: Interpretability file created at: {output_path}/feature_importance.json")
    else:
        print("❌ ERROR: Model returned empty importance data. Interpretability tab will remain empty.")
    
    print("✅ Pipeline complete.")

if __name__ == "__main__":
    run_workflow()