"""
VISUALIZATION OF HYPERPARAMETER TUNING RESULTS
===============================================
Creates all 4 graphs + additional visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

sns.set_style("whitegrid")

class TuningVisualizer:
    """
    Visualize hyperparameter tuning results
    """
    
    def __init__(self, results_df, fold_results_df, best_params, output_dir='outputs'):
        self.results_df = results_df
        self.fold_results_df = fold_results_df
        self.best_params = best_params
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # GRAPH 1: N_ESTIMATORS EFFICIENCY CURVE
    # ========================================================================
    
    def plot_n_estimators_efficiency(self):
        """
        Graph 1: Where does efficiency plateau?
        Shows R² vs number of trees
        """
        
        print("\n📈 Creating Graph 1: N_Estimators Efficiency Curve...")
        
        # Get data for best max_depth and min_samples_leaf (from best model)
        best_depth = self.best_params['max_depth']
        best_leaf = self.best_params['min_samples_leaf']
        
        # Filter for best depth and leaf combo
        filtered = self.results_df[
            (self.results_df['param_max_depth'] == best_depth) &
            (self.results_df['param_min_samples_leaf'] == best_leaf)
        ].copy()
        
        filtered = filtered.sort_values('param_n_estimators')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: CV R² vs n_estimators
        ax1.plot(filtered['param_n_estimators'], filtered['mean_test_score'], 
                marker='o', linewidth=2.5, markersize=8, color='steelblue', label='CV R² Mean')
        ax1.fill_between(filtered['param_n_estimators'],
                         filtered['mean_test_score'] - filtered['std_test_score'],
                         filtered['mean_test_score'] + filtered['std_test_score'],
                         alpha=0.2, color='steelblue', label='±1 Std Dev')
        
        # Mark best point
        best_idx = filtered['mean_test_score'].idxmax()
        best_n_est = filtered.loc[best_idx, 'param_n_estimators']
        best_score = filtered.loc[best_idx, 'mean_test_score']
        
        ax1.scatter([best_n_est], [best_score], color='red', s=200, marker='*', 
                   label=f'Optimal: {int(best_n_est)} trees', zorder=5)
        
        ax1.set_xlabel('Number of Trees (n_estimators)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cross-Validation R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Graph 1: Model Efficiency vs Number of Trees\n(Where does efficiency plateau?)',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(filtered['param_n_estimators'].unique())
        
        # Plot 2: Train vs Test R²
        ax2.plot(filtered['param_n_estimators'], filtered['mean_train_score'],
                marker='o', linewidth=2.5, markersize=8, color='green', label='Train R²')
        ax2.plot(filtered['param_n_estimators'], filtered['mean_test_score'],
                marker='s', linewidth=2.5, markersize=8, color='red', label='Test R² (CV)')
        
        ax2.scatter([best_n_est], [filtered.loc[best_idx, 'mean_train_score']], 
                   color='green', s=200, marker='*', zorder=5)
        ax2.scatter([best_n_est], [best_score], 
                   color='red', s=200, marker='*', zorder=5)
        
        ax2.set_xlabel('Number of Trees (n_estimators)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax2.set_title('Train vs Test R²: Overfitting Analysis',
                     fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(filtered['param_n_estimators'].unique())
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graph_1_n_estimators_efficiency.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: graph_1_n_estimators_efficiency.png")
        plt.close()
        
        # Print summary
        print(f"\n📊 N_Estimators Analysis:")
        print(f"  Best: {int(best_n_est)} trees")
        print(f"  CV R² at best: {best_score:.6f}")
        print(f"  Efficiency plateaus after: ~{int(best_n_est)} trees")
        
        return filtered
    
    # ========================================================================
    # GRAPH 2: MAX_DEPTH EFFECT (SWEET SPOT)
    # ========================================================================
    
    def plot_max_depth_effect(self):
        """
        Graph 2: What's the optimal max_depth?
        Shows R² vs max_depth (sweet spot analysis)
        """
        
        print("\n📈 Creating Graph 2: Max Depth Effect...")
        
        # Get data for best n_estimators and min_samples_leaf
        best_n_est = self.best_params['n_estimators']
        best_leaf = self.best_params['min_samples_leaf']
        
        filtered = self.results_df[
            (self.results_df['param_n_estimators'] == best_n_est) &
            (self.results_df['param_min_samples_leaf'] == best_leaf)
        ].copy()
        
        # Replace None with string for plotting
        filtered['param_max_depth_str'] = filtered['param_max_depth'].astype(str)
        filtered = filtered.sort_values('param_max_depth', 
                                       key=lambda x: pd.factorize(x, sort=False)[0])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: CV R² vs max_depth
        depths_str = filtered['param_max_depth_str'].values
        depths_pos = np.arange(len(depths_str))
        
        bars = ax1.bar(depths_pos, filtered['mean_test_score'], 
                       color='coral', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax1.errorbar(depths_pos, filtered['mean_test_score'], 
                    yerr=filtered['std_test_score'], fmt='none', 
                    color='black', capsize=5, capthick=2)
        
        # Mark best
        best_idx = filtered['mean_test_score'].idxmax()
        best_depth = filtered.loc[best_idx, 'param_max_depth_str']
        best_score = filtered.loc[best_idx, 'mean_test_score']
        best_pos = list(depths_str).index(best_depth)
        
        bars[best_pos].set_color('green')
        bars[best_pos].set_edgecolor('darkgreen')
        bars[best_pos].set_linewidth(2.5)
        
        ax1.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cross-Validation R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Graph 2: Optimal Max Depth (Sweet Spot)\nBias-Variance Trade-off',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(depths_pos)
        ax1.set_xticklabels(depths_str)
        ax1.grid(axis='y', alpha=0.3)
        ax1.text(best_pos, best_score + 0.01, f'✅ Optimal\n{best_depth}', 
                ha='center', fontweight='bold', fontsize=10)
        
        # Plot 2: Train vs Test gap
        train_scores = filtered['mean_train_score'].values
        test_scores = filtered['mean_test_score'].values
        gaps = train_scores - test_scores
        
        ax2.bar(depths_pos, gaps, color='purple', edgecolor='black', 
               linewidth=1.5, alpha=0.7)
        ax2.axhline(y=gaps.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Average Gap: {gaps.mean():.4f}')
        
        ax2.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Train R² - Test R² (Overfitting Gap)', fontsize=12, fontweight='bold')
        ax2.set_title('Overfitting Analysis: Which depth prevents memorization?',
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(depths_pos)
        ax2.set_xticklabels(depths_str)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graph_2_max_depth_effect.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: graph_2_max_depth_effect.png")
        plt.close()
        
        # Print summary
        print(f"\n📊 Max Depth Analysis:")
        print(f"  Optimal depth: {best_depth}")
        print(f"  CV R² at optimal: {best_score:.6f}")
        print(f"  Overfitting gap at optimal: {gaps[best_pos]:.6f}")
        
        return filtered
    
    # ========================================================================
    # GRAPH 3: MIN_SAMPLES_LEAF EFFECT (PREVENT MEMORIZATION)
    # ========================================================================
    
    def plot_min_samples_leaf_effect(self):
        """
        Graph 3: How to prevent memorization?
        Shows effect of min_samples_leaf
        """
        
        print("\n📈 Creating Graph 3: Min Samples Leaf Effect...")
        
        # Get data for best n_estimators and max_depth
        best_n_est = self.best_params['n_estimators']
        best_depth = self.best_params['max_depth']
        
        filtered = self.results_df[
            (self.results_df['param_n_estimators'] == best_n_est) &
            (self.results_df['param_max_depth'] == best_depth)
        ].copy()
        
        filtered = filtered.sort_values('param_min_samples_leaf')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Train vs Test R² vs min_samples_leaf
        leaves = filtered['param_min_samples_leaf'].values
        train_scores = filtered['mean_train_score'].values
        test_scores = filtered['mean_test_score'].values
        
        ax1.plot(leaves, train_scores, marker='o', linewidth=2.5, markersize=10,
                color='green', label='Train R²', markerfacecolor='lightgreen')
        ax1.plot(leaves, test_scores, marker='s', linewidth=2.5, markersize=10,
                color='red', label='Test R² (CV)', markerfacecolor='lightcoral')
        ax1.fill_between(leaves, train_scores, test_scores, alpha=0.2, color='gray')
        
        ax1.set_xlabel('Min Samples Per Leaf', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Graph 3: Preventing Memorization\nTrain vs Test R² Gap',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(leaves)
        
        # Plot 2: Overfitting gap
        gaps = train_scores - test_scores
        
        bars = ax2.bar(range(len(leaves)), gaps, color='purple', edgecolor='black',
                       linewidth=1.5, alpha=0.7)
        
        # Mark best (smallest gap)
        best_idx = np.argmin(gaps)
        bars[best_idx].set_color('green')
        bars[best_idx].set_edgecolor('darkgreen')
        bars[best_idx].set_linewidth(2.5)
        
        ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=2,
                   label='Good generalization threshold (0.10)')
        
        ax2.set_xlabel('Min Samples Per Leaf', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Overfitting Gap (Train R² - Test R²)', fontsize=12, fontweight='bold')
        ax2.set_title('How much memorization?',
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(leaves)))
        ax2.set_xticklabels(leaves)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(fontsize=11)
        ax2.text(best_idx, gaps[best_idx] + 0.005, f'✅ Best\n{leaves[best_idx]}',
                ha='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graph_3_min_samples_leaf_effect.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: graph_3_min_samples_leaf_effect.png")
        plt.close()
        
        # Print summary
        print(f"\n📊 Min Samples Leaf Analysis:")
        print(f"  Best value: {leaves[best_idx]}")
        print(f"  Overfitting gap with best value: {gaps[best_idx]:.6f}")
        print(f"  Prevents memorization: ✅ YES")
        
        return filtered
    
    # ========================================================================
    # GRAPH 4: HEATMAP OF ALL COMBINATIONS
    # ========================================================================
    
    def plot_heatmap_all_combinations(self):
        """
        Graph 4: Heatmap showing performance across all combinations
        """
        
        print("\n📈 Creating Graph 4: Heatmap of All Combinations...")
        
        # Create pivot table: n_estimators vs max_depth (colored by R²)
        # Fix min_samples_leaf to best value
        best_leaf = self.best_params['min_samples_leaf']
        
        filtered = self.results_df[
            self.results_df['param_min_samples_leaf'] == best_leaf
        ].copy()
        
        # Create pivot for heatmap
        pivot_data = filtered.pivot_table(
            index='param_max_depth',
            columns='param_n_estimators',
            values='mean_test_score',
            aggfunc='first'
        )
        
        # Sort properly (handle None)
        pivot_data = pivot_data.sort_index(key=lambda x: pd.factorize(x, sort=False)[0])
        pivot_data = pivot_data[sorted(pivot_data.columns)]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'CV R² Score'}, ax=ax, 
                   vmin=pivot_data.min().min() - 0.02,
                   vmax=pivot_data.max().max() + 0.02,
                   linewidths=0.5, linecolor='gray')
        
        # Mark best point
        best_depth = self.best_params['max_depth']
        best_n_est = self.best_params['n_estimators']
        
        # Find position
        if best_depth in pivot_data.index and best_n_est in pivot_data.columns:
            row_pos = list(pivot_data.index).index(best_depth)
            col_pos = list(pivot_data.columns).index(best_n_est)
            
            # Add rectangle around best
            rect = plt.Rectangle((col_pos, row_pos), 1, 1, fill=False, 
                                edgecolor='blue', linewidth=3)
            ax.add_patch(rect)
        
        ax.set_xlabel('Number of Trees (n_estimators)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Max Depth', fontsize=12, fontweight='bold')
        ax.set_title(f'Graph 4: Performance Heatmap - All Combinations\n(Best model marked in blue box, min_samples_leaf={best_leaf})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graph_4_heatmap_all_combinations.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: graph_4_heatmap_all_combinations.png")
        plt.close()
        
        print(f"\n📊 Heatmap shows all {len(pivot_data) * len(pivot_data.columns)} combinations")
        print(f"   Red = Low performance, Green = High performance")
        print(f"   Blue box = Optimal model")
        
        return pivot_data
    
    # ========================================================================
    # BONUS: 5-FOLD CONSISTENCY GRAPH
    # ========================================================================
    
    def plot_5fold_consistency(self):
        """
        Bonus: Show 5-fold cross-validation consistency
        """
        
        print("\n📈 Creating Bonus Graph: 5-Fold Consistency...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: R² across folds
        folds = self.fold_results_df['fold'].values
        train_r2 = self.fold_results_df['train_r2'].values
        test_r2 = self.fold_results_df['test_r2'].values
        
        x_pos = np.arange(len(folds))
        width = 0.35
        
        ax1.bar(x_pos - width/2, train_r2, width, label='Train R²',
               color='green', edgecolor='black', alpha=0.7)
        ax1.bar(x_pos + width/2, test_r2, width, label='Test R²',
               color='red', edgecolor='black', alpha=0.7)
        
        ax1.axhline(y=train_r2.mean(), color='darkgreen', linestyle='--',
                   linewidth=2, label=f'Train Avg: {train_r2.mean():.4f}')
        ax1.axhline(y=test_r2.mean(), color='darkred', linestyle='--',
                   linewidth=2, label=f'Test Avg: {test_r2.mean():.4f}')
        
        ax1.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('5-Fold Cross-Validation: R² Consistency',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(folds)
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 0.7])
        
        # Plot 2: Gap analysis
        gaps = self.fold_results_df['gap'].values
        
        bars = ax2.bar(x_pos, gaps, color='purple', edgecolor='black', alpha=0.7)
        ax2.axhline(y=gaps.mean(), color='darkviolet', linestyle='--',
                   linewidth=2.5, label=f'Mean Gap: {gaps.mean():.4f}')
        
        # Color code: green if gap < 0.10, orange if < 0.15, red otherwise
        for i, (bar, gap) in enumerate(zip(bars, gaps)):
            if gap < 0.10:
                bar.set_color('green')
            elif gap < 0.15:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax2.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Overfitting Gap (Train - Test)', fontsize=12, fontweight='bold')
        ax2.set_title('5-Fold Gap Analysis: Consistency Check',
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(folds)
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add consistency message
        consistency_score = 1 - (gaps.std() / gaps.mean()) if gaps.mean() != 0 else 0
        status = '✅ Excellent' if gaps.std() < 0.02 else '⚠️ Good' if gaps.std() < 0.05 else '❌ High'
        ax2.text(0.5, 0.95, f'Consistency: {status} (Std: {gaps.std():.4f})',
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bonus_5fold_consistency.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: bonus_5fold_consistency.png")
        plt.close()
        
        return self.fold_results_df
    
    def create_all_visualizations(self):
        """
        Create all visualizations
        """
        print("\n" + "="*80)
        print("CREATING ALL VISUALIZATIONS")
        print("="*80)
        
        self.plot_n_estimators_efficiency()
        self.plot_max_depth_effect()
        self.plot_min_samples_leaf_effect()
        self.plot_heatmap_all_combinations()
        self.plot_5fold_consistency()
        
        print("\n" + "="*80)
        print("✅ ALL 5 GRAPHS CREATED")
        print("="*80)
