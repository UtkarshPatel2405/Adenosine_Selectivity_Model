# Adenosine Receptor Selectivity & Affinity Predictor (CADD / QSAR)

This project trains and validates publication-grade machine-learning models to predict **pChEMBL_value** (affinity) and **selectivity** across four human adenosine receptor subtypes (A1, A2A, A2B, A3) from **SMILES** strings. 

It is specifically designed for rigorous early-stage drug discovery, incorporating conformal prediction, zero-leakage data pipelines, and strict chemical sanity diagnostics.

## 🔬 Key Features

### 1. Zero-Leakage Data Pipeline (`precise` mode)
- **Global Scaffold Splitting**: Prevents structural data leakage between training and testing folds.
- **Mutual Decoy Ingestion**: Integrates GPCRdb non-binders (decoys) across subtypes to provide rigorous negative controls.
- **SMILES Hash Deduplication**: Identical molecules across active and decoy sets are collapsed to prevent conflicting ground truths.

### 2. Advanced Predictive Modeling
- **Conformal XGBoost**: Provides point predictions alongside mathematically guaranteed prediction intervals (Confidence Intervals).
- **Random Forest Baseline**: Evaluates performance robustness against a secondary tree-based method.
- **GNN Model Pipeline**: Includes a Graph Neural Network implementation as a deep learning comparison point.
- **Subtype Selectivity Prediction**: Dedicated gradient boosting models to classify multi-subtype selectivity preferences (e.g., A1 vs A2A).

### 3. Rigorous Chemical Diagnostics
- **Y-Randomization**: Ensures the model actually learns chemical patterns (R² drops to <0.0 on shuffled targets).
- **SHAP Analysis**: Deconstructs model decisions to verify reliance on plausible molecular features rather than structural noise or artifacts.
- **Publication Benchmark Checking**: Automatically compares local metrics against literature thresholds for state-of-the-art CADD models.

### 4. Interactive Streamlit Dashboard
- **Single Molecule & Batch Prediction**: Screening tools supporting CSV inputs.
- **Diagnostic Viewing**: Visualizes Y-Randomization distributions, SHAP beeswarm/bar plots, and benchmarking criteria directly in the UI.

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the Full Production Suite
To trigger the end-to-end retraining pipeline (data cleaning -> modeling -> diagnostics):
```bash
python -m src.retrain_production
python -m src.y_randomization
python -m src.shap_analysis
```

### 3. Launch the Dashboard
```bash
streamlit run streamlit_app.py
```

## 📊 Evaluation & Artifacts
The pipeline caches models and reports directly into the repository structure:
- **Models**: Saved as standard `.pkl` (and `.pt` for GNNs) within `models/`.
- **Validation Results**: Stored in `outputs/validoutput/` and `outputs/training_summary.json`.
- **Diagnostics**: `outputs/shap/`, `outputs/y_randomization/`, and `outputs/benchmark/`.
