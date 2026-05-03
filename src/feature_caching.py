import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureCache:
   
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.fingerprints_file = f'{cache_dir}/morgan_fingerprints.pkl'
        self.properties_file = f'{cache_dir}/physicochemical_properties.pkl'
        self.metadata_file = f'{cache_dir}/feature_metadata.json'
  
    def validate_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None, mol
        except:
            return False, None
  
    def get_morgan_fingerprints_new_api(self, smiles, n_bits=2048, radius=2):
        
        is_valid, mol = self.validate_smiles(smiles)
        
        if not is_valid:
            return None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp, dtype=np.uint8)
        except:
            return None
  
    def get_physicochemical_properties(self, smiles):
        is_valid, mol = self.validate_smiles(smiles)
        
        if not is_valid:
            return None
        
        try:
            properties = {
                'MW': float(Descriptors.MolWt(mol)),
                'LogP': float(Crippen.MolLogP(mol)),
                'HBD': int(Descriptors.NumHDonors(mol)),
                'HBA': int(Descriptors.NumHAcceptors(mol)),
                'RotBonds': int(Descriptors.NumRotatableBonds(mol)),
                'AromRings': int(Descriptors.NumAromaticRings(mol)),
                'TPSA': float(Descriptors.TPSA(mol))
            }
            return properties
        except:
 
    def extract_and_cache_features(self, df, n_bits=2048, radius=2, verbose=True):
        
        if verbose:
            print("\n" + "="*80)
            print("PHASE 1: EXTRACT & CACHE FEATURES")
            print("="*80)
            print("\nExtracting Morgan Fingerprints and Properties...")
            print(f"   Total molecules: {len(df)}")
            print(f"   Fingerprint size: {n_bits} bits")
            print(f"   Using NEW RDKit API (no deprecation warnings)")
        
        smiles_list = df['smiles'].values
        
        fingerprints = []
        properties_list = []
        valid_indices = []
        failed_count = 0
        
        for idx, smiles in enumerate(smiles_list):
            if verbose and (idx + 1) % 1000 == 0:
                print(f"    Processed {idx + 1}/{len(smiles_list)} molecules...")
            
            if pd.isna(smiles):
                failed_count += 1
                continue
            
            fp = self.get_morgan_fingerprints_new_api(smiles, n_bits=n_bits, radius=radius)
            
            props = self.get_physicochemical_properties(smiles)
            
            if fp is not None and props is not None:
                fingerprints.append(fp)
                properties_list.append(props)
                valid_indices.append(idx)
            else:
                failed_count += 1
        
        if verbose:
            print(f"\nFeature extraction complete!")
            print(f"   Successfully extracted: {len(valid_indices)} molecules")
            print(f"   Failed: {failed_count} molecules")
        
        fp_names = [f'FP_{i}' for i in range(n_bits)]
        X_fingerprints = pd.DataFrame(fingerprints, columns=fp_names, index=valid_indices)
        X_properties = pd.DataFrame(properties_list, index=valid_indices)
    
        X_combined = pd.concat([X_fingerprints, X_properties], axis=1)
        
        if verbose:
            print(f"\nCACHING TO DISK:")
            print(f"   Fingerprints shape: {X_fingerprints.shape}")
            print(f"   Properties shape: {X_properties.shape}")
            print(f"   Combined shape: {X_combined.shape}")
        
        with open(self.fingerprints_file, 'wb') as f:
            pickle.dump({
                'fingerprints': X_fingerprints,
                'valid_indices': valid_indices,
                'n_bits': n_bits,
                'radius': radius,
                'timestamp': datetime.now().isoformat(),
                'total_molecules': len(df),
                'valid_molecules': len(valid_indices)
            }, f)

        with open(self.properties_file, 'wb') as f:
            pickle.dump({
                'properties': X_properties,
                'valid_indices': valid_indices,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        if verbose:
            print(f"\n Cached files created:")
            print(f"   {self.fingerprints_file}")
            print(f"   {self.properties_file}")
            print(f"   Size: ~{(os.path.getsize(self.fingerprints_file) + os.path.getsize(self.properties_file)) / 1024 / 1024:.1f} MB")
        
        return X_combined, valid_indices

    def load_cached_features(self, verbose=True):
        
        if verbose:
            print("\n" + "="*80)
            print("LOADING CACHED FEATURES")
            print("="*80)
        
        if not os.path.exists(self.fingerprints_file):
            print(" Cache not found! Run Phase 1 first to extract features.")
            return None, None

        with open(self.fingerprints_file, 'rb') as f:
            fp_data = pickle.load(f)
  
        with open(self.properties_file, 'rb') as f:
            props_data = pickle.load(f)
        
        X_fingerprints = fp_data['fingerprints']
        X_properties = props_data['properties']
        valid_indices = fp_data['valid_indices']
        
        X_combined = pd.concat([X_fingerprints, X_properties], axis=1)
        
        if verbose:
            print(f"\n Cache loaded successfully!")
            print(f"   Fingerprints: {X_fingerprints.shape}")
            print(f"   Properties: {X_properties.shape}")
            print(f"   Combined: {X_combined.shape}")
            print(f"   Valid molecules: {len(valid_indices)}")
            print(f"   Cache created: {fp_data['timestamp']}")
        
        return X_combined, valid_indices
 
    def get_cache_statistics(self):
        
        if not os.path.exists(self.fingerprints_file):
            return None
        
        with open(self.fingerprints_file, 'rb') as f:
            fp_data = pickle.load(f)
        
        with open(self.properties_file, 'rb') as f:
            props_data = pickle.load(f)
        
        X_fp = fp_data['fingerprints']
        X_props = props_data['properties']
        
        stats = {
            'total_molecules': fp_data['total_molecules'],
            'valid_molecules': fp_data['valid_molecules'],
            'failed_molecules': fp_data['total_molecules'] - fp_data['valid_molecules'],
            'fingerprint_bits': fp_data['n_bits'],
            'fingerprint_radius': fp_data['radius'],
            'property_count': X_props.shape[1],
            'total_features': X_fp.shape[1] + X_props.shape[1],
            'fingerprint_sparsity': 1 - (X_fp.sum().sum() / (X_fp.shape[0] * X_fp.shape[1])),
            'cache_size_mb': (os.path.getsize(self.fingerprints_file) + os.path.getsize(self.properties_file)) / 1024 / 1024,
            'cache_created': fp_data['timestamp']
        }
        
        return stats
