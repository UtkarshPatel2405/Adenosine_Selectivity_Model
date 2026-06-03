import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_style("whitegrid")

# Create output folder if not exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Histogram of the data
def plot_histogram(df, column, bins=50):
    plt.figure(figsize=(12, 6))
    plt.hist(df[column].dropna(), bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of {column}", fontsize=14, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filename = f"outputs/histogram_{column}.png"
    plt.savefig(filename, dpi=300)
    print(f" Saved: {filename}")
    plt.close()

# Bar Chart
def plot_bar_chart(df, column, top_n=10):
    counts = df[column].value_counts().head(top_n)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(counts)), counts.values, color='lightcoral', edgecolor='black')
    plt.xticks(range(len(counts)), counts.index, rotation=45, ha='right')
    plt.title(f"Top {top_n} {column}", fontsize=14, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filename = f"outputs/bar_chart_{column}.png"
    plt.savefig(filename, dpi=300)
    print(f" Saved: {filename}")
    plt.close()

# Box Plot
def plot_box_plot(df, column, group_by=None):
    plt.figure(figsize=(12, 6))
    if group_by:
        sns.boxplot(x=group_by, y=column, data=df, palette='Set2')
        plt.title(f"Box Plot of {column} by {group_by}", fontsize=14, fontweight='bold')
    else:
        sns.boxplot(y=df[column], color='lightseagreen')
        plt.title(f"Box Plot of {column}", fontsize=14, fontweight='bold')
    plt.xlabel(group_by if group_by else '')
    plt.ylabel(column)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filename = f"outputs/box_plot_{column}.png"
    plt.savefig(filename, dpi=300)
    print(f" Saved: {filename}")
    plt.close()


# Scatter Plot
def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(12, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50, edgecolors='black', color='steelblue')
    plt.title(f"{x_col} vs {y_col}", fontsize=14, fontweight='bold')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filename = f"outputs/scatter_{x_col}_vs_{y_col}.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved: {filename}")
    plt.close()

# CORRELATION HEATMAP
def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        print("No numeric columns found")
        return
    
    corr = numeric_df.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = "outputs/correlation_heatmap.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved: {filename}")
    plt.close()