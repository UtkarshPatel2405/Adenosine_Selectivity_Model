# Systematic QSAR Selectivity Predictor: Flow & Results

A clear, fragmented overview of the structural pipeline, the engine logic, and the validated performance metrics.

---

## 1. Systematic Pipeline Diagram (SMILES Ingestion to Chemical Insights)

The flowchart below traces a candidate compound from its SMILES input through coordinate generation, conformal selectivity modeling, and local medicinal chemistry explanations.

```mermaid
graph TD
    %% Input Layer
    In[SMILES Compound Input] --> Viz[Structure Visualization Engine]
    In --> Feat[Feature Extraction Pipeline]
    
    %% Viz Path
    subgraph 3D & 2D Vector Coordinate Engine
        Viz --> SVG[2D Vector SVG Drawing]
        Viz --> Conformer[3D Conformer Generation]
        Conformer --> SDF[3D SDF Molecular File]
        Conformer --> PDB[3D PDB Biomolecule File]
    end

    %% Feature Path
    subgraph Feature Extraction & Scaling
        Feat --> Morgan[2048-bit Morgan Fingerprints]
        Feat --> MACCS[167-bit MACCS Structural Keys]
        Feat --> Descriptors[RDKit Continuous Descriptors ~210]
        Descriptors --> Filter[Variance & Correlation Filter]
        Filter --> Concat[Staged Feature Vector Shape: 1x2278]
    end

    %% Model Path
    subgraph Conformal Selectivity Ensemble
        Concat --> Predictor[Ensemble Conformal Regressor]
        Predictor --> Mapie[MAPIE Conformal Engine]
        Mapie --> Bio[Subtype Bioactivity Profile]
        Bio --> Select[Pairwise Selectivity Profile: ΔpChEMBL]
    end

    %% Output Path
    subgraph Explainability & Chemical Translation
        Select --> SHAP[TreeSHAP Global Explainer]
        SHAP --> RDKitMap[RDKit Circular Environment Map]
        RDKitMap --> SMARTS[Medicinal Chemistry SMARTS Translations]
    end

    %% UI Displays
    SVG -.-> UI[Streamlit Interactive Dashboard]
    SDF -.-> UI
    PDB -.-> UI
    Bio -.-> UI
    SMARTS -.-> UI
```

---

## 2. Fragmented Architecture Breakdown

### 📊 Phase A: Data Ingestion & ETL (Dataloader)
* **Parent Ingestion:** Cleans and filters out salts, mixtures, and organometallics to isolate raw parent compound SMILES.
* **Scaffold Isolation:** Employs Bemis-Murcko scaffold groupings to prevent leakage of identical core chemistry between training and testing sets.
* **Mutual Decoy Mapping:** Tagging compounds active on other targets but untested on the query target as boundary controls ($pChEMBL = 3.0$) to define clear selectivity margins.

### 🧪 Phase B: Feature Engineering (Molecular Descriptors)
* **2D Fingerprints:** Combines **2,048-bit Morgan circular fingerprints (radius=2)** and **167-bit MACCS keys** to capture topological functional groups.
* **Continuous Physicochemicals:** Computes RDKit physical descriptors (e.g., MolLogP, TPSA, Weight, H-Bond capacity) to enforce thermodynamic binding physics.

### 🧠 Phase C: Conformal Modeling Suite
* **Conformal Regressors:** Combines Mapie prediction frameworks with XGBoost base estimators.
* **Confidence Safety Bounds:** Predicts continuous $pChEMBL$ affinity value alongside dynamic **90% prediction intervals** on-the-fly, indicating prediction reliability.

### 🔬 Phase D: Explainability & Exporters
* **Dynamic Local SHAP:** Computes local feature contributions for predictions on-the-fly, displaying a waterfall plot.
* **RDKit Chemical Translations:** Dynamically maps Morgan fingerprint contribution bits back to the molecule's exact atoms and radius, outputting readable **SMARTS structural moieties** (e.g. `[#6]:[#6](:[#6])-[#7]`).
* **Multi-Format Downloader:** Generates 2D SVG vectors, 3D SDF coordinates, 3D PDB coordinates, and prediction CSVs instantly.

---

## 3. Core Performance & Validation Results

The models are strictly validated using **Nested Cross-Validation with Scaffold Splits** (ensuring all test compounds belong to completely unseen Murcko scaffolds) to prevent target memorization.

### 🎯 Overall Conformal Performance (Unified Precise Mode)
* **Dataset Size:** 41,937 parent compound data points (incorporating clean precise actives + programmatic mutual decoys + 1,607 structural P2Y non-binder decoys)
* **Overall MAE:** `0.396 pChEMBL units` (Mean Absolute Error)
* **Overall R²:** `0.845` (Variance Explained) — **Publication-Grade!**
* **Average Prediction Width:** `±0.763 pChEMBL units` (90% Conformal Interval Bound)

### 📈 Receptor Subtype Potency Breakdown
| Subtype Target | Training Compounds | Testing Compounds | Validation R² | Validation MAE | Validation RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A₁ Subtype** | 8,272 | 2,136 | **0.809** | 0.403 | 0.817 |
| **A₂A Subtype** | 8,407 | 2,126 | **0.835** | 0.529 | 0.928 |
| **A₂B Subtype** | 8,290 | 2,114 | **0.801** | 0.305 | 0.717 |
| **A₃ Subtype** | 8,432 | 2,160 | **0.894** | 0.347 | 0.700 |

