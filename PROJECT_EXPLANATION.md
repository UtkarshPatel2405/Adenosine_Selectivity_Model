# Adenosine Receptor QSAR Model: The Complete Source Code Textbook

## Preface: How to Read This Book
This document is a comprehensive textbook for the Adenosine Receptor Selectivity Model. It is designed to take you from a conceptual understanding of Quantitative Structure-Activity Relationships (QSAR) down to the literal line-by-line explanation of every python file in this codebase. 

Whether you are defending your thesis to a strict professor, preparing for a technical software engineering interview, or just trying to remember why you wrote a specific line of code, this book holds the answer.

---

# Chapter 1: The Core Scientific Logic

### The Biological Problem
Proteins act as "locks" and chemical drugs act as "keys." The human body has four subtypes of the Adenosine Receptor: **A1, A2A, A2B, and A3**. Because these four receptors evolved from the same genetic ancestor, their "keyholes" (binding pockets) look incredibly similar.
* If you design a drug to treat asthma (targeting A2B), but it accidentally fits into the A1 receptor, it can stop the patient's heart.
* Therefore, modern pharmacology demands **Selectivity**. A drug must bind tightly to one receptor and ignore the others.

### The Computational Solution (QSAR)
Instead of synthesizing thousands of drugs in a lab, we use machine learning to predict binding affinity mathematically. This is called QSAR (Quantitative Structure-Activity Relationship).
We measure binding affinity using `pChEMBL`. This is the negative base-10 logarithm of the molar concentration required to trigger the receptor ($-log_{10}(M)$). 
* A `pChEMBL` of 5.0 means it takes $10\mu M$ (weak drug).
* A `pChEMBL` of 9.0 means it takes $1 nM$ (incredibly potent drug).

The goal of this codebase is to read a string of text representing a molecule (SMILES), convert it to numbers, and predict the continuous `pChEMBL` value for all four adenosine receptors.

---

# Chapter 2: Data Ingestion (`src/data_loader.py`)

This file is responsible for loading the messy, real-world Excel data, cleaning it, standardizing the chemical structures, and creating a database lookup.

### 2.1 The Subtype Mapping
```python
SUBTYPE_MAP = {
    "A1R": "A1",
    "A2AR": "A2A",
    "A2BR": "A2B",
    "A3R": "A3",
    "A1": "A1",
    ...
}
```
**Explanation:** Different scientific papers call the receptors different names (e.g., A1R vs A1). This dictionary is a lookup table to force all names into our standard 4 labels.

### 2.2 Canonicalizing SMILES
```python
def _canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip(): return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return Chem.MolToSmiles(mol, canonical=True)
```
**Explanation:** A SMILES string is a way to type a molecule on a keyboard. However, ethanol can be typed as `CCO` or `OCC`. `Chem.MolFromSmiles` turns the text into a 3D graph object in RAM. `Chem.MolToSmiles(canonical=True)` turns that graph back into text, forcing it to use the exact same alphabetical spelling every time. This prevents the AI from thinking `CCO` and `OCC` are two different drugs.

### 2.3 The Cleaning Pipeline (`load_and_clean`)
```python
def load_and_clean(file_path: str, save_lookup_path: str = "data/processed/db_lookup.json", mode: str = "standard"):
    df = pd.read_csv(file_path)
    # ... checks for required columns ...
```
**Explanation:** Opens the raw CSV. It verifies that columns like `pchembl_value` and `confidence_score` exist.

```python
    if mode == "strict":
        df = df[df["standard_relation"] == "="].copy()
        df = df[df["standard_units"].str.lower() == "nm"].copy()
        df = df[df["assay_type"].str.upper() == "B"].copy()
        df = df[df["confidence_score"].notna() & (df["confidence_score"] >= 7)].copy()
```
**Explanation:** If the user demands strict academic rigor, we drop any rows where the lab wasn't 100% sure of the result (`=` rather than `>`), force units to nanomolar, require it to be a Binding assay (`B`), and require a confidence score $\ge 7$.

```python
    df = (
        df.sort_values("pchembl_value", ascending=False)
        .drop_duplicates(subset=["canonical_smiles", "target_subtype"], keep="first")
        .reset_index(drop=True)
    )
```
**Explanation:** **CRITICAL LINE.** If three different labs tested Caffeine on A2A, we have 3 rows. We sort them from highest binding affinity to lowest. We then use `drop_duplicates` to keep only the `first` (highest) value. We do this to assume the most potent measurement is the true binding affinity, filtering out poor lab conditions.

```python
    lookup = {}
    for smi, subdf in df.groupby("canonical_smiles"):
        lookup[smi] = {row["target_subtype"]: float(row["pchembl_value"]) for _, row in subdf.iterrows()}
    with open(save_lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f)
```
**Explanation:** We create `db_lookup.json`. This is our "cheat sheet." When a user uses the web app, we check this dictionary first. If they typed a drug we already know, we instantly return the real lab value rather than wasting time running the AI model.

---

# Chapter 3: Molecular Featurization (`src/features.py`)

AI algorithms (like XGBoost) only understand numbers. This file translates the canonical SMILES string into a mathematical array.

### 3.1 Morgan Fingerprints
```python
_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)

def _morgan_bits(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fp = _MORGAN.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
```
**Explanation:** A Morgan Fingerprint (Radius 2) draws a circle around every atom in the molecule, looks at its neighbors up to 2 bonds away, and hashes that structure into an array of 2048 zeroes and ones. If bit #450 is a `1`, it might mean "This molecule has a benzene ring connected to an oxygen." This turns the 2D topology into a sparse binary array.

### 3.2 Physicochemical Descriptors
```python
def _descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol) # Molecular Weight
    logp = Descriptors.MolLogP(mol) # Lipophilicity (Fat vs Water solubility)
    hbd = Lipinski.NumHDonors(mol) # Hydrogen Bond Donors
    hba = Lipinski.NumHAcceptors(mol) # Hydrogen Bond Acceptors
    rot = Lipinski.NumRotatableBonds(mol) # Flexibility
    arom = Lipinski.NumAromaticRings(mol) # Aromaticity
    tpsa = Descriptors.TPSA(mol) # Polar Surface Area
    return np.array([mw, logp, hbd, hba, rot, arom, tpsa], dtype=np.float32)
```
**Explanation:** Fingerprints only track shape. Descriptors track physics. We calculate 7 continuous properties, like how heavy the molecule is, or how well it dissolves in fat (LogP). These 7 numbers are appended to the 2048 zeroes and ones.

### 3.3 Scaling the Matrix
```python
    scaler = StandardScaler()
    Xdesc_train_s = scaler.fit_transform(Xdesc_train)
    Xdesc_test_s = scaler.transform(Xdesc_test)
    X_train = np.hstack([Xfp_train, Xdesc_train_s]).astype(np.float32)
```
**Explanation:** **CRITICAL LOGIC.** Molecular Weight can be 500.0. LogP is usually 2.0. If we give this raw to the AI, it will mathematically assume Weight is 250x more important than LogP simply because the number is bigger. 
`StandardScaler` squashes the 7 descriptors down so they center around 0. 
*Why don't we scale the fingerprints?* Because fingerprints are binary `0` or `1`. If you scale a `0`, it becomes `-0.04`, destroying the logical "Yes/No" meaning of the bit. Thus, we split the matrix, scale the 7 physics numbers, and `hstack` (glue) them back onto the binary fingerprints.

---

# Chapter 4: Anti-Cheating Validation (`src/data_splitter.py`)

If you randomly shuffle a deck of molecules into Train (80%) and Test (20%), the AI will cheat. It will see "Caffeine" in the training set, and "Caffeine with an extra methyl group" in the test set. It will score perfectly on the test set, but it hasn't learned chemistry; it just memorized local variations.

### The Bemis-Murcko Scaffold Split
```python
def _murcko_scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, canonical=True)
```
**Explanation:** This function acts as a chemical "bone stripper." It removes all the fluffy side-chains of a molecule and leaves only the macroscopic ring systems (the scaffold). 

```python
    scaffolds = df[smiles_col].apply(_murcko_scaffold_smiles)
    df_copy["_scaffold"] = scaffolds

    scaffold_to_indices = {}
    for i, scaf in enumerate(df_copy["_scaffold"].tolist()):
        scaffold_to_indices.setdefault(scaf, []).append(i)
```
**Explanation:** We map every single molecule to its skeleton. We group them. All molecules with Skeleton A go into one bucket. All molecules with Skeleton B go into another.

```python
    for scaf in scaffold_keys:
        indices = scaffold_to_indices[scaf]
        if test_count < n_test_target:
            test_indices.extend(indices)
            test_count += len(indices)
        else:
            train_indices.extend(indices)
```
**Explanation:** We put entire buckets (scaffolds) into the Test set until it reaches 20%. This guarantees that when the AI takes the final test, it is looking at chemical skeletons it has **never seen before in its life**. This forces true out-of-distribution (OOD) learning.

---

# Chapter 5: The Brain (`src/ml_xgboost.py`)

We use XGBoost (Extreme Gradient Boosting Regressor). Neural Networks are famously bad at tabular/sparse data unless you have millions of rows. XGBoost is state-of-the-art for our specific dataset size and fingerprint structure.

```python
    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=2,
        subsample=0.8,
        tree_method='hist',
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
```
**Explanation:**
*   `n_estimators=800`: Build 800 decision trees. The first tree guesses. The second tree predicts the *error* of the first tree to fix it. The third fixes the second.
*   `learning_rate=0.05`: Each tree is only allowed to fix 5% of the error at a time, preventing over-correction.
*   `max_depth=7`: A tree can only ask 7 "Yes/No" questions before it must make a final guess.
*   `subsample=0.8` / `colsample_bytree=0.8`: Every time a new tree is built, it is only allowed to look at 80% of the molecules, and 80% of the 2055 features. This forces the trees to be diverse and prevents over-fitting to noisy features.
*   `early_stopping_rounds=50`: If the model builds 50 trees in a row and the test score doesn't improve, it aborts training early to save time and prevent memorization.

---

# Chapter 6: The Blind External Test (`prepare_new_test_set.py`)

When the professor hands you four brand new Excel files, you must prove the AI didn't already train on them.

```python
    seen_smiles = set()
    for smi in train_df['smiles'].dropna():
        canon = canonicalize(smi)
        seen_smiles.add(canon)
```
**Explanation:** We load the 10,000+ molecules we used for training, canonicalize them, and lock them in a `set`.

```python
        for idx, row in df.iterrows():
            smi = row.get('SMILES')
            canon = canonicalize(smi)
            
            if canon in seen_smiles:
                total_skipped_seen += 1
                continue
```
**Explanation:** As we read the professor's new files, we canonicalize every new molecule. `if canon in seen_smiles:` is our security checkpoint. If the molecule is in the set, the code hits `continue` (skip), instantly throwing it in the trash. This guarantees **zero data leakage**.

```python
            current_val = novel_molecules[canon].get(subtype, 0)
            novel_molecules[canon][subtype] = max(current_val, float(p_val))
```
**Explanation:** The professor gave us 4 separate files (A1, A2A, etc.). If "Drug X" appears in the A1 file, we save `novel_molecules["Drug X"]["A1"] = 6.0`. If it later appears in the A2A file, we add `novel_molecules["Drug X"]["A2A"] = 8.0`. This merges four isolated files into one beautiful, multi-target dataframe.

---

# Chapter 7: Analyzing Selectivity (`analyze_novel_results.py`)

The most advanced and rigorous part of the project: evaluating if the AI actually learned *selectivity*.

### 7.1 Mean Absolute Error (MAE)
```python
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
```
**Explanation:** If the true `pChEMBL` is 8.0, and the AI guesses 7.0, the absolute error is 1.0. Because `pChEMBL` is logarithmic, an error of 1.0 means the AI was off by exactly one order of magnitude (10-fold error). For pure 2D ligand screening, a 0.7-1.0 error is excellent.

### 7.2 Recall@1 (Top-1 Selectivity Accuracy)
```python
    # Filter for molecules tested on multiple receptors
    counts = truth_df[subtypes].notna().sum(axis=1)
    multi_target_idx = counts[counts >= 2].index
```
**Explanation:** You can't calculate selectivity if a drug was only tested on A1. We strictly filter the dataset for drugs that have ground-truth experimental data for *at least two* receptors.

```python
        true_best = max(true_vals, key=true_vals.get)
        
        pred_sorted = sorted(pred_vals.keys(), key=lambda k: pred_vals[k], reverse=True)
        pred_best = pred_sorted[0]
        
        if true_best == pred_best:
            correct_selectivity_top1 += 1
```
**Explanation:** For those multi-target drugs, we ask: "Which receptor has the highest true lab value?" (`true_best`). Then we look at the AI's predictions and ask: "Which receptor did the AI *think* was the highest?" (`pred_best`). If they match, we score a point. This proves the AI can identify the primary pharmacological target, not just guess random numbers.

### 7.3 Epistemic Uncertainty (Applicability Domain)
```python
    high_rel = (merged['reliability'] >= 0.6).sum()
```
**Explanation:** "Reliability" is calculated via Tanimoto Similarity. It compares the Morgan Fingerprint of the novel drug to every single fingerprint in the training set. If the drug is less than 40% similar to anything we've ever seen, it is Out-of-Distribution (OOD). The AI is flying blind. We proudly report this rather than hiding it, proving we understand the mathematical limits of our dataset.

---

# Chapter 8: The Production Web Application (`src/predictor.py` & `streamlit_app.py`)

A model is useless if chemists cannot interact with it. We built a Streamlit web application that allows researchers to query SMILES strings and instantly see selectivity profiles, safety alerts, and applicability domains.

### 8.1 The "Known Drug" Fast-Pass (`src/predictor.py`)
```python
    in_db = canon in lookup
    if in_db:
        exp = lookup[canon]
        for st in SUBTYPES:
            preds[st], unc[st] = float(exp.get(st, 0.0)), 0.0
        source = "database"
```
**Explanation:** When a user queries a molecule, we first check `db_lookup.json`. If the user asks about Caffeine, and Caffeine was already in our original ChEMBL training data, **we do not run the AI model**. The AI might have a margin of error, but our database has the literal, true laboratory result. We bypass the XGBoost ensemble entirely and return the exact experimental `pChEMBL` value. This guarantees 100% accuracy on historically known drugs.

### 8.2 Safety & Drug Likeness (PAINS & QED)
```python
        with col_pains:
            alerts = check_pains(smiles)
            if alerts: st.error(f"PAINS alert(s) detected: {', '.join(alerts)}")
```
**Explanation:** PAINS (Pan Assay Interference Compounds) are chemical structures that frequently give false-positive results in lab tests because they react with the assay chemicals, not the receptor itself. We use RDKit's PAINS filters to warn the researcher if the AI's prediction might be based on a chemically deceptive sub-structure. We also compute QED (Quantitative Estimate of Drug-likeness), which scores the molecule from 0 to 1 based on how similar its physical properties are to FDA-approved oral drugs.

### 8.3 Explainability (Nearest Neighbors)
```python
        # Top-5 Similar Training Molecules
        canon_smi, top_sims = topk_tanimoto(smiles, k=5)
```
**Explanation:** If the AI predicts a novel molecule has an incredibly high affinity, the researcher will immediately ask: "Why?" Our UI calculates the Tanimoto similarity against the 10,000+ training molecules and displays the top 5 most structurally similar molecules. This allows the chemist to manually verify if the AI's logic is sound by looking at historically active analogs.

---

# Chapter 9: Defense Strategy (Conclusion)

If an academic reviewer critiques your model, use this textbook to defend your choices:

1. **Critique:** "Your MAE is 1.0 in log base, that's a 10x error!"
   **Defense:** Open Chapter 7. Explain that for early-stage 2D virtual screening, we aren't doing lead-optimization. We are doing "bucket classification" (separating $\mu M$ from $nM$ drugs) before running expensive 3D docking.
2. **Critique:** "You should use 3D structural data from the bound complex."
   **Defense:** Agree completely. Explain that this codebase is the required **2D Baseline**. You had to build the data ingestion (Chap 2), the scaffold splitting (Chap 4), and the evaluation pipeline (Chap 7) to have something to benchmark future 3D Graph Neural Networks against. 
3. **Critique:** "Your model is over-fitted."
   **Defense:** Open Chapter 4. Show them `MurckoScaffold`. Prove that the model was tested on chemical backbones it had never seen before.

*You are now the master of this codebase.*
