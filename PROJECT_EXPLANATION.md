# Comprehensive Project Guide: Adenosine Receptor Selectivity Model

This document explains the "why" and "how" behind every major component of your project. It is written so you can fully understand the technical logic and confidently explain the syntax and architecture to your professor.

---

## 1. The Core Objective
We are performing **QSAR (Quantitative Structure-Activity Relationship)** modeling. Specifically, it is a **regression** task (predicting a continuous number). 
The goal is to input a molecule's structure (via a SMILES string) and predict its **pChEMBL binding affinity** across 4 different Adenosine Receptor Subtypes: **A1, A2A, A2B, and A3**.

By predicting all four at once, we can measure **Selectivity** (e.g., a drug might bind very strongly to A2A but ignore A1, making it highly selective).

---

## 2. Data Processing & Cleaning (`src/data_loader.py`)
Before training, we must ensure the data is clean and scientifically sound.

**The Logic:**
- **Canonicalization:** SMILES strings can be written in multiple ways for the exact same molecule (e.g., `CCO` vs `OCC` for ethanol). RDKit's `Chem.MolToSmiles(mol, canonical=True)` forces every molecule into a single, standardized string format.
- **Deduplication:** We might have multiple experimental values for the same molecule. We sort by `pchembl_value` descending and drop duplicates, keeping the highest binding affinity.
- **Standard vs. Strict Modes:** 
  - *Standard Mode:* Drops `IC50` values (which can be heavily dependent on assay conditions) but keeps other binding metrics.
  - *Strict Mode:* Highly rigorous. Only accepts `KI` or `KD` standard types, exact relations (`=`), units in `nM`, binding assays (`B`), and confidence scores `>= 7`.

**The Code Syntax in Python (Pandas):**
```python
# Filtering Pandas DataFrame based on specific criteria
df = df[df["standard_relation"] == "="]
df = df[df["confidence_score"] >= 7]
# Applying a function to a column to create a new one
df["canonical_smiles"] = df["smiles"].apply(_canonicalize_smiles)
```

---

## 3. Feature Engineering (`src/features.py`, `src/molecular_features.py`)
Machine learning models only understand numbers, not letters. We must convert SMILES strings into numeric features.

**The Logic:**
We represent each molecule using **2,055 features**:
1. **Morgan Fingerprints (2048 bits):** We use a radius of 2 (equivalent to ECFP4). This analyzes the molecule's substructures (rings, bonds) and creates an array of 2048 1s and 0s indicating if a specific substructure is present.
2. **Physicochemical Descriptors (7 features):** We calculate 7 key RDKit properties: Molecular Weight (MW), LogP (lipophilicity), Hydrogen Bond Donors (HBD), H-Bond Acceptors (HBA), Rotatable Bonds, Aromatic Rings, and TPSA (Topological Polar Surface Area).

**Selective Feature Scaling:**
We apply a `StandardScaler` (subtracting the mean and dividing by standard deviation) **only to the 7 continuous descriptors**. We intentionally do *not* scale the 2048 fingerprint bits, because scaling binary (0/1) arrays destroys their sparse meaning.

**The Code Syntax (RDKit):**
```python
# Generate Morgan Fingerprint as a Bit Vector
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
# Calculate a physical descriptor
logp = Descriptors.MolLogP(mol)
```

---

## 4. Scaffold Splitting (`src/data_splitter.py`)
Normally in ML, you use a Random Split (e.g., 80% train, 20% test). In Cheminformatics, random splits are considered **"cheating"**.

**The Logic:**
If two highly similar molecules end up in train and test, the model isn't learning chemistry; it's just memorizing patterns. To fix this, we use a **Bemis-Murcko Scaffold Split**. 
This strips away the side-chains of a molecule to find its "core ring structure" (scaffold). We force all molecules sharing the same scaffold into either the Train set OR the Test set, never both. This proves that our model can generalize to *novel* chemical classes it hasn't seen before.

**The Code Syntax:**
```python
from rdkit.Chem.Scaffolds import MurckoScaffold
# Extract the core scaffold of a molecule
scaf = MurckoScaffold.GetScaffoldForMol(mol)
```

---

## 5. Machine Learning Models (`src/ml_xgboost.py`, `src/train_subtypes.py`)
We train **four independent XGBoost models**, one for each subtype (A1, A2A, etc.).

**The Logic:**
- **Why XGBoost?** Extreme Gradient Boosting builds sequential decision trees. It is highly robust to sparse data (like our 2048-bit fingerprints), captures non-linear chemical relationships beautifully, and provides "Feature Importance" (telling us which fingerprint bits mattered most).
- **Hyperparameter Tuning:** You tuned parameters like `max_depth`, `n_estimators`, and `learning_rate` to find the "sweet spot" that prevents the model from memorizing (overfitting) the training data.

**The Code Syntax:**
```python
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=800, max_depth=7, learning_rate=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 6. Evaluation Metrics & Baseline (`src/evaluator.py`)
When you present to your professor, you need to prove the model is actually "learning."

**The Logic:**
- **R² (R-squared):** Tells you what percentage of the variance in binding affinity is explained by your model (1.0 is perfect, 0 is guessing).
- **MAE (Mean Absolute Error):** The average error in pChEMBL units. If MAE is 0.35, our prediction is usually off by ±0.35 units.
- **The Baseline Test (DummyRegressor):** We compare our model to a "dumb" model that simply predicts the average `pchembl_value` of the training set for every molecule. If our model has a significantly lower MAE than the Dummy, it is truly learning chemical patterns. (Your model strongly beats the baseline).

---

## 7. The Applicability Domain & Uncertainty (`src/applicability_domain.py`)
No model is perfect on every molecule in the universe. We must define where our model is "safe" to use.

**The Logic:**
When a user inputs a novel SMILES string, we calculate its **Tanimoto Similarity** to the molecules in our training data.
- Tanimoto > 0.6: High Reliability (Very similar to training data).
- Tanimoto 0.4 - 0.6: Medium Reliability.
- Tanimoto < 0.4: Low Reliability (Out of distribution, model is guessing blindly).

---

## 8. The Web Application (`src/streamlit_app.py`, `src/predictor.py`)
We built a beautiful Streamlit UI to interact with the models.

**The Logic:**
- **The Database Lookup Rule:** If a user searches for a molecule that we already have exact experimental ChEMBL data for, the app returns the *actual experimental value* instead of a prediction. This is best scientific practice.
- **Safety Checks:** We integrated `pains_checker.py` (Pan Assay Interference Compounds) to warn the user if a molecule is structurally known to trigger false positives in biological assays.
- **Batch Processing:** Allows the user to upload a CSV of thousands of SMILES and get predictions for A1, A2A, A2B, and A3, allowing them to rapidly sort and filter for Selective compounds.
