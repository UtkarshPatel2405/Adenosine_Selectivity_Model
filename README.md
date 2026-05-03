# Adenosine Receptor Ligand QSAR (pChEMBL Regression)


This project trains and validates machine‑learning models to predict **pChEMBL_value** from **SMILES** strings.  
It is designed for early‑stage drug discovery on Adenosine receptors.

## 🔬 Key Features
- Single‑molecule prediction with full diagnostic report (bioactivity, safety, interpretability)
- Batch CSV screening with downloadable results
- Side‑by‑side candidate comparison (up to 5 molecules)
- Applicability Domain (AD) via Tanimoto similarity to training set
- PAINS structural alerts & drug‑likeness (QED, Lipinski)
- Mechanistic interpretability (top Morgan fingerprint bits)
- Model benchmarking across representations (Morgan, RDKit descriptors, MACCS)
- Interactive Results Dashboard (scaffold‑split validation, AD coverage, uncertainty)

## 📊 Model Performance (XGBoost – 5‑repeat Scaffold Split)

| Metric | Value |
|--------|-------|
| **Mean $R^2$**   | *{INSERT XGB_R2_MEAN HERE}* |
| **Mean $MAE$**   | *{INSERT XGB_MAE_MEAN HERE}* |
| **Validation**   | 5‑repeat scaffold‑aware cross‑validation |
| **Featurization**| Morgan fingerprints (ECFP4‑like, 2048 bits) + RDKit descriptors |
| **Training data**| ~9 600 curated ChEMBL compounds |

> **How to get the exact numbers:**  
> Run the validation pipeline: `python ml_validate_scaffold_ad.py`  
> Then open `outputs/validation_scaffold/validation_report.json` and copy the `r2_mean` and `mae_mean` for the XGBoost model.

## 🚀 Quick Start

### 1. Clone & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
