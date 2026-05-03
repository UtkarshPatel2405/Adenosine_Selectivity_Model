import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FUNCTION 1: VALIDATE SMILES
# ============================================================================

def validate_smiles(smiles):
    """
    Check if SMILES string is valid.
    
    LOGIC:
    - RDKit parses SMILES
    - If parsing fails, SMILES is invalid
    - Return None for invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        return None
    except:
        return None


# ============================================================================
# FUNCTION 2: GENERATE MORGAN FINGERPRINTS
# ============================================================================

def get_morgan_fingerprints(smiles, n_bits=2048, radius=2):
    """
    Generate Morgan Fingerprints from SMILES.
    
    MORGAN FINGERPRINTS:
    - 2048-bit vector representing molecular structure
    - Each bit = presence/absence of molecular substructure
    - Similar molecules → similar fingerprints
    - Used in drug discovery for similarity searching
    
    PARAMETERS:
    - n_bits: Length of fingerprint (2048 is standard)
    - radius: ECFP radius (2 = ECFP4)
    
    RETURNS:
    - numpy array of 2048 binary values (0 or 1)
    
    EXAMPLE:
    Aspirin SMILES: CC(=O)Oc1ccccc1C(=O)O
    → Generates 2048-bit fingerprint
    → Each bit represents a structural feature
    """
    
    mol = validate_smiles(smiles)
    
    if mol is None:
        return None
    
    try:
        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None


# ============================================================================
# FUNCTION 3: CALCULATE PHYSICOCHEMICAL PROPERTIES
# ============================================================================

def get_physicochemical_properties(smiles):
    """
    Calculate molecular descriptors from SMILES.
    
    PROPERTIES:
    - Molecular Weight (MW): Size of molecule
    - LogP: Lipophilicity (oil/water solubility)
    - H-Bond Donors: Can accept hydrogen bonds
    - H-Bond Acceptors: Can donate hydrogen bonds
    - Rotatable Bonds: Molecular flexibility
    - Aromatic Rings: Aromatic systems
    - Molecular Refractivity: Polarizability
    
    LIPINSKI'S RULE OF 5 (for drug-likeness):
    - MW ≤ 500
    - LogP ≤ 5
    - H-Bond Donors ≤ 5
    - H-Bond Acceptors ≤ 10
    
    RETURNS:
    - Dictionary with property values
    """
    
    mol = validate_smiles(smiles)
    
    if mol is None:
        return None
    
    try:
        properties = {
            'MW': Descriptors.MolWt(mol),              # Molecular Weight
            'LogP': Crippen.MolLogP(mol),              # Lipophilicity
            'HBD': Descriptors.NumHDonors(mol),        # H-Bond Donors
            'HBA': Descriptors.NumHAcceptors(mol),     # H-Bond Acceptors
            'RotBonds': Descriptors.NumRotatableBonds(mol),  # Rotatable Bonds
            'AromRings': Descriptors.NumAromaticRings(mol),  # Aromatic Rings
            'TPSA': Descriptors.TPSA(mol),             # Topological Polar Surface Area
        }
        return properties
    except:
        return None


# ============================================================================
# FUNCTION 4: EXTRACT ALL FEATURES FROM SMILES
# ============================================================================

def extract_features_from_smiles(df, use_fingerprints=True, use_properties=True, n_bits=2048):
    """
    Extract all molecular features from SMILES column.
    
    PARAMETERS:
    - df: DataFrame with 'smiles' column
    - use_fingerprints: Include Morgan Fingerprints
    - use_properties: Include physicochemical properties
    - n_bits: Fingerprint length (2048 standard)
    
    RETURNS:
    - X: Feature matrix (fingerprints + properties)
    - valid_indices: Indices with valid SMILES
    
    LOGIC:
    1. For each SMILES string:
       - Extract Morgan Fingerprint (2048 bits)
       - Extract physicochemical properties (7 properties)
    2. Combine into single feature matrix
    3. Track which molecules are valid
    """
    
    print("\n" + "="*80)
    print("EXTRACTING MOLECULAR FEATURES FROM SMILES")
    print("="*80)
    
    n_molecules = len(df)
    print(f"\nProcessing {n_molecules} molecules...")
    
    fingerprints = []
    properties_list = []
    valid_indices = []
    
    # Progress tracking
    for idx, smiles in enumerate(df['smiles'].values):
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1}/{n_molecules} molecules...")
        
        if pd.isna(smiles):
            continue
        
        # Extract fingerprint
        fp = None
        if use_fingerprints:
            fp = get_morgan_fingerprints(smiles, n_bits=n_bits)
        
        # Extract properties
        props = None
        if use_properties:
            props = get_physicochemical_properties(smiles)
        
        # Only keep if both are valid
        if fp is not None and props is not None:
            fingerprints.append(fp)
            properties_list.append(props)
            valid_indices.append(idx)
    
    print(f"\n✅ Successfully processed {len(valid_indices)}/{n_molecules} molecules")
    print(f"   Valid molecules: {len(valid_indices)}")
    print(f"   Invalid molecules: {n_molecules - len(valid_indices)}")
    
    if len(valid_indices) == 0:
        print("❌ No valid molecules found!")
        return None, None
    
    # Convert fingerprints to DataFrame
    if use_fingerprints:
        fingerprint_names = [f'FP_{i}' for i in range(n_bits)]
        X_fingerprints = pd.DataFrame(fingerprints, columns=fingerprint_names, index=valid_indices)
    else:
        X_fingerprints = None
    
    # Convert properties to DataFrame
    if use_properties:
        X_properties = pd.DataFrame(properties_list, index=valid_indices)
        print(f"\n📊 Physicochemical Properties:")
        print(X_properties.describe())
    else:
        X_properties = None
    
    # Combine features
    if X_fingerprints is not None and X_properties is not None:
        X = pd.concat([X_fingerprints, X_properties], axis=1)
    elif X_fingerprints is not None:
        X = X_fingerprints
    else:
        X = X_properties
    
    print(f"\n🔬 FINAL FEATURE MATRIX:")
    print(f"   Shape: {X.shape}")
    print(f"   Rows (molecules): {X.shape[0]}")
    print(f"   Columns (features): {X.shape[1]}")
    print(f"      - Morgan Fingerprints: {n_bits if use_fingerprints else 0} bits")
    print(f"      - Properties: {len(X_properties.columns) if use_properties else 0} properties")
    
    return X, valid_indices


# ============================================================================
# FUNCTION 5: FEATURE STATISTICS
# ============================================================================

def analyze_features(X, y, valid_indices):
    """
    Analyze extracted features.
    
    ANALYSIS:
    - Feature correlations
    - Property statistics
    - Fingerprint sparsity
    """
    
    print("\n" + "="*80)
    print("FEATURE ANALYSIS")
    print("="*80)
    
    # Get properties (last 7 columns)
    properties_cols = X.columns[-7:]
    X_properties = X[properties_cols]
    
    print("\n📊 PHYSICOCHEMICAL PROPERTIES STATISTICS:")
    print(X_properties.describe())
    
    # Correlation with target
    print("\n\n🔗 PROPERTY CORRELATIONS WITH TARGET (pchembl_value):")
    y_aligned = y.loc[valid_indices]
    
    for col in properties_cols:
        corr = X_properties[col].corr(y_aligned)
        print(f"   {col}: {corr:.6f}")
    
    # Fingerprint statistics
    X_fingerprints = X.iloc[:, :-7]  # All columns except last 7
    n_bits = X_fingerprints.shape[1]
    
    print(f"\n🔬 FINGERPRINT STATISTICS:")
    print(f"   Total bits: {n_bits}")
    print(f"   Average bits set per molecule: {X_fingerprints.sum(axis=1).mean():.2f}")
    print(f"   Min bits set: {X_fingerprints.sum(axis=1).min()}")
    print(f"   Max bits set: {X_fingerprints.sum(axis=1).max()}")
    print(f"   Sparsity: {(1 - X_fingerprints.sum().sum() / (X_fingerprints.shape[0] * X_fingerprints.shape[1])) * 100:.2f}%")


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    # Test with example SMILES
    test_smiles = [
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'O=C(O)Cc1ccccc1NC(=O)c2c(Cl)cccc2Cl'  # Diclofenac
    ]
    
    print("Testing SMILES Extraction:")
    for smiles in test_smiles:
        print(f"\nSMILES: {smiles}")
        
        # Test fingerprint
        fp = get_morgan_fingerprints(smiles)
        if fp is not None:
            print(f"  Fingerprint: {fp.shape} bits, {fp.sum()} bits set")
        
        # Test properties
        props = get_physicochemical_properties(smiles)
        if props is not None:
            print(f"  MW: {props['MW']:.2f}, LogP: {props['LogP']:.2f}")