"""
COMPREHENSIVE RESULTS LOGGER
=============================
Captures EVERY metric, number, and detail from model training
Saves to structured JSON and detailed report
NO manipulation - just raw facts
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

class ResultsLogger:
    """
    Logs all results to JSON format for complete transparency
    """
    
    def __init__(self, project_name="QSAR_Analysis"):
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results = {
            'metadata': {},
            'dataset': {},
            'preprocessing': {},
            'feature_extraction': {},
            'models': {},
            'model_details': {},
            'comparisons': {}
        }
        
    def log_metadata(self, info_dict):
        """Log project metadata"""
        self.results['metadata'] = {
            'project_name': self.project_name,
            'timestamp': self.timestamp,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.now().strftime("%H:%M:%S"),
            **info_dict
        }
        return self.results['metadata']
    
    def log_dataset_info(self, df, target_col='pchembl_value'):
        """Log ALL dataset statistics"""
        self.results['dataset'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'column_dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'target_column': target_col,
            'target_statistics': {
                'min': float(df[target_col].min()),
                'max': float(df[target_col].max()),
                'mean': float(df[target_col].mean()),
                'median': float(df[target_col].median()),
                'std': float(df[target_col].std()),
                'variance': float(df[target_col].var()),
                'skewness': float(df[target_col].skew()),
                'kurtosis': float(df[target_col].kurtosis()),
                'q1': float(df[target_col].quantile(0.25)),
                'q3': float(df[target_col].quantile(0.75)),
                'iqr': float(df[target_col].quantile(0.75) - df[target_col].quantile(0.25)),
                'non_null_count': int(df[target_col].count()),
                'null_count': int(df[target_col].isnull().sum()),
                'null_percentage': float((df[target_col].isnull().sum() / len(df)) * 100)
            },
            'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns},
            'missing_percentages': {col: float((df[col].isnull().sum() / len(df)) * 100) for col in df.columns},
            'duplicate_rows': int(df.duplicated().sum()),
            'shape': list(df.shape)
        }
        return self.results['dataset']
    
    def log_smiles_processing(self, total_smiles, valid_smiles, invalid_count):
        """Log SMILES processing details"""
        self.results['preprocessing']['smiles_processing'] = {
            'total_smiles_input': int(total_smiles),
            'valid_smiles': int(valid_smiles),
            'invalid_smiles': int(invalid_count),
            'valid_percentage': float((valid_smiles / total_smiles) * 100),
            'invalid_percentage': float((invalid_count / total_smiles) * 100)
        }
        return self.results['preprocessing']['smiles_processing']
    
    def log_feature_extraction(self, X_df, fingerprint_bits=2048, n_properties=7):
        """Log detailed feature extraction info"""
        self.results['feature_extraction'] = {
            'total_samples': X_df.shape[0],
            'total_features': X_df.shape[1],
            'fingerprint_bits': fingerprint_bits,
            'physicochemical_properties': n_properties,
            'feature_names': list(X_df.columns),
            'feature_statistics': {
                'min_values': {col: float(X_df[col].min()) for col in X_df.columns},
                'max_values': {col: float(X_df[col].max()) for col in X_df.columns},
                'mean_values': {col: float(X_df[col].mean()) for col in X_df.columns},
                'std_values': {col: float(X_df[col].std()) for col in X_df.columns},
                'median_values': {col: float(X_df[col].median()) for col in X_df.columns}
            },
            'fingerprint_sparsity': float(1 - (X_df.iloc[:, :fingerprint_bits].sum().sum() / (X_df.shape[0] * fingerprint_bits))),
            'fingerprint_bits_set_per_molecule': {
                'min': int(X_df.iloc[:, :fingerprint_bits].sum(axis=1).min()),
                'max': int(X_df.iloc[:, :fingerprint_bits].sum(axis=1).max()),
                'mean': float(X_df.iloc[:, :fingerprint_bits].sum(axis=1).mean()),
                'median': float(X_df.iloc[:, :fingerprint_bits].sum(axis=1).median()),
                'std': float(X_df.iloc[:, :fingerprint_bits].sum(axis=1).std())
            }
        }
        return self.results['feature_extraction']
    
    def log_train_test_split(self, X_train, X_test, y_train, y_test, test_size=0.2):
        """Log train/test split details"""
        self.results['preprocessing']['train_test_split'] = {
            'test_size_ratio': float(test_size),
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'total_samples': X_train.shape[0] + X_test.shape[0],
            'train_percentage': float((X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])) * 100),
            'test_percentage': float((X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])) * 100),
            'y_train_statistics': {
                'min': float(y_train.min()),
                'max': float(y_train.max()),
                'mean': float(y_train.mean()),
                'median': float(y_train.median()),
                'std': float(y_train.std()),
                'count': int(len(y_train))
            },
            'y_test_statistics': {
                'min': float(y_test.min()),
                'max': float(y_test.max()),
                'mean': float(y_test.mean()),
                'median': float(y_test.median()),
                'std': float(y_test.std()),
                'count': int(len(y_test))
            }
        }
        return self.results['preprocessing']['train_test_split']
    
    def log_model_training(self, model_name, hyperparameters, training_time_seconds):
        """Log model training details"""
        if 'training' not in self.results['model_details']:
            self.results['model_details']['training'] = {}
        
        self.results['model_details']['training'][model_name] = {
            'model_name': model_name,
            'hyperparameters': hyperparameters,
            'training_time_seconds': float(training_time_seconds),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return self.results['model_details']['training'][model_name]
    
    def log_model_evaluation(self, model_name, y_true, y_pred, dataset_type='test'):
        """Log ALL evaluation metrics"""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
        
        # Calculate all metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Residuals
        residuals = y_true.values - y_pred
        
        # Additional metrics
        median_ae = np.median(np.abs(residuals))
        mean_bias = np.mean(residuals)
        
        metrics = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'predictions': {
                'total_predictions': int(len(y_pred)),
                'min_prediction': float(np.min(y_pred)),
                'max_prediction': float(np.max(y_pred)),
                'mean_prediction': float(np.mean(y_pred)),
                'median_prediction': float(np.median(y_pred)),
                'std_prediction': float(np.std(y_pred))
            },
            'actual_values': {
                'total_values': int(len(y_true)),
                'min_value': float(y_true.min()),
                'max_value': float(y_true.max()),
                'mean_value': float(y_true.mean()),
                'median_value': float(y_true.median()),
                'std_value': float(y_true.std())
            },
            'error_metrics': {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape),
                'Median_Absolute_Error': float(median_ae),
                'Mean_Bias': float(mean_bias),
                'R2_Score': float(r2)
            },
            'residuals': {
                'min': float(residuals.min()),
                'max': float(residuals.max()),
                'mean': float(residuals.mean()),
                'median': float(np.median(residuals)),
                'std': float(residuals.std()),
                'q1': float(np.percentile(residuals, 25)),
                'q3': float(np.percentile(residuals, 75)),
                'iqr': float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
            },
            'prediction_error_distribution': {
                'within_0.5': int((np.abs(residuals) <= 0.5).sum()),
                'within_0.5_percentage': float((np.abs(residuals) <= 0.5).sum() / len(residuals) * 100),
                'within_1.0': int((np.abs(residuals) <= 1.0).sum()),
                'within_1.0_percentage': float((np.abs(residuals) <= 1.0).sum() / len(residuals) * 100),
                'within_1.5': int((np.abs(residuals) <= 1.5).sum()),
                'within_1.5_percentage': float((np.abs(residuals) <= 1.5).sum() / len(residuals) * 100),
                'over_1.5': int((np.abs(residuals) > 1.5).sum()),
                'over_1.5_percentage': float((np.abs(residuals) > 1.5).sum() / len(residuals) * 100)
            }
        }
        
        if 'evaluations' not in self.results['model_details']:
            self.results['model_details']['evaluations'] = {}
        
        key = f"{model_name}_{dataset_type}"
        self.results['model_details']['evaluations'][key] = metrics
        
        return metrics
    
    def log_feature_importance(self, model_name, feature_names, importances, top_n=20):
        """Log feature importance details"""
        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'percentage': (importances / importances.sum()) * 100
        }).sort_values('importance', ascending=False)
        
        self.results['model_details'][f'{model_name}_feature_importance'] = {
            'total_features': len(feature_names),
            'top_features': importance_df.head(top_n).to_dict('records'),
            'top_20_sum_percentage': float(importance_df.head(20)['percentage'].sum()),
            'importance_summary': {
                'min': float(importances.min()),
                'max': float(importances.max()),
                'mean': float(importances.mean()),
                'median': float(np.median(importances)),
                'std': float(importances.std())
            }
        }
        return self.results['model_details'][f'{model_name}_feature_importance']
    
    def log_model_comparison(self, comparison_df):
        """Log model comparison"""
        self.results['comparisons']['model_comparison'] = comparison_df.to_dict('records')
        return self.results['comparisons']['model_comparison']
    
    def save_json_report(self, filename='outputs/COMPLETE_RESULTS.json'):
        """Save all results to JSON"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\n✅ JSON Report saved: {filename}")
        return filename
    
    def save_text_report(self, filename='outputs/COMPLETE_RESULTS_DETAILED.txt'):
        """Save all results as formatted text"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE QSAR MODEL RESULTS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Metadata
            f.write("METADATA\n")
            f.write("-"*80 + "\n")
            for key, value in self.results['metadata'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n\n")
            
            # Dataset
            f.write("DATASET INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Rows: {self.results['dataset']['total_rows']}\n")
            f.write(f"Total Columns: {self.results['dataset']['total_columns']}\n")
            f.write(f"Memory Usage: {self.results['dataset']['memory_usage_mb']:.2f} MB\n")
            f.write(f"\nTarget Variable ({self.results['dataset']['target_column']}) Statistics:\n")
            for stat, value in self.results['dataset']['target_statistics'].items():
                f.write(f"  {stat}: {value}\n")
            
            f.write(f"\nMissing Values:\n")
            for col, count in self.results['dataset']['missing_values'].items():
                pct = self.results['dataset']['missing_percentages'][col]
                f.write(f"  {col}: {count} ({pct:.2f}%)\n")
            
            f.write(f"\nDuplicate Rows: {self.results['dataset']['duplicate_rows']}\n\n")
            
            # Preprocessing
            f.write("PREPROCESSING DETAILS\n")
            f.write("-"*80 + "\n")
            
            if 'smiles_processing' in self.results['preprocessing']:
                f.write("SMILES Processing:\n")
                for key, value in self.results['preprocessing']['smiles_processing'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            if 'train_test_split' in self.results['preprocessing']:
                f.write("Train/Test Split:\n")
                split = self.results['preprocessing']['train_test_split']
                f.write(f"  Train Samples: {split['train_samples']} ({split['train_percentage']:.2f}%)\n")
                f.write(f"  Test Samples: {split['test_samples']} ({split['test_percentage']:.2f}%)\n")
                f.write(f"\n  Training Target Statistics:\n")
                for stat, value in split['y_train_statistics'].items():
                    f.write(f"    {stat}: {value}\n")
                f.write(f"\n  Test Target Statistics:\n")
                for stat, value in split['y_test_statistics'].items():
                    f.write(f"    {stat}: {value}\n")
            
            f.write("\n")
            
            # Feature Extraction
            f.write("FEATURE EXTRACTION\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Samples: {self.results['feature_extraction']['total_samples']}\n")
            f.write(f"Total Features: {self.results['feature_extraction']['total_features']}\n")
            f.write(f"Fingerprint Bits: {self.results['feature_extraction']['fingerprint_bits']}\n")
            f.write(f"Physicochemical Properties: {self.results['feature_extraction']['physicochemical_properties']}\n")
            f.write(f"Fingerprint Sparsity: {self.results['feature_extraction']['fingerprint_sparsity']:.4f}\n")
            f.write(f"\nFingerprint Bits Set Per Molecule:\n")
            for stat, value in self.results['feature_extraction']['fingerprint_bits_set_per_molecule'].items():
                f.write(f"  {stat}: {value}\n")
            f.write("\n\n")
            
            # Model Details
            f.write("MODEL TRAINING & EVALUATION\n")
            f.write("-"*80 + "\n")
            
            if 'evaluations' in self.results['model_details']:
                for eval_key, metrics in self.results['model_details']['evaluations'].items():
                    f.write(f"\n{eval_key}:\n")
                    f.write(f"  Error Metrics:\n")
                    for metric, value in metrics['error_metrics'].items():
                        f.write(f"    {metric}: {value:.6f}\n")
                    f.write(f"  Residuals:\n")
                    for stat, value in metrics['residuals'].items():
                        f.write(f"    {stat}: {value:.6f}\n")
                    f.write(f"  Prediction Error Distribution:\n")
                    for key, value in metrics['prediction_error_distribution'].items():
                        f.write(f"    {key}: {value}\n")
                    f.write("\n")
            
            # Feature Importance
            f.write("\nFEATURE IMPORTANCE\n")
            f.write("-"*80 + "\n")
            for key in self.results['model_details']:
                if 'feature_importance' in key:
                    f.write(f"\n{key}:\n")
                    importance_data = self.results['model_details'][key]
                    f.write(f"Total Features: {importance_data['total_features']}\n")
                    f.write(f"Top 20 Sum Percentage: {importance_data['top_20_sum_percentage']:.2f}%\n")
                    f.write(f"\nTop 20 Features:\n")
                    for feature in importance_data['top_features'][:20]:
                        f.write(f"  {feature['feature']}: {feature['importance']:.6f} ({feature['percentage']:.2f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        print(f"✅ Text Report saved: {filename}")
        return filename
    
    def generate_summary_dataframe(self):
        """Generate summary as DataFrame"""
        summaries = []
        
        if 'evaluations' in self.results['model_details']:
            for eval_key, metrics in self.results['model_details']['evaluations'].items():
                summaries.append({
                    'Model': eval_key,
                    'R² Score': metrics['error_metrics']['R2_Score'],
                    'RMSE': metrics['error_metrics']['RMSE'],
                    'MAE': metrics['error_metrics']['MAE'],
                    'MAPE': metrics['error_metrics']['MAPE'],
                    'Mean_Bias': metrics['error_metrics']['Mean_Bias'],
                    'Predictions_Within_±0.5': f"{metrics['prediction_error_distribution']['within_0.5_percentage']:.2f}%",
                    'Predictions_Within_±1.0': f"{metrics['prediction_error_distribution']['within_1.0_percentage']:.2f}%"
                })
        
        return pd.DataFrame(summaries)
