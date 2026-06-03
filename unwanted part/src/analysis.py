import pandas as pd
import numpy as np


def analyze_column(df, column_name):
    print("\n" + "="*80)
    print(f"Analysis: {column_name}")
    print("="*80)

    col = df[column_name]
    print(f"Data Type: {col.dtype}")
    print(f"Non-null Count: {col.count()}")
    print(f"Null Count: {col.isnull().sum()}")
    

    if col.dtype in ['int64', 'float64']:
        print(f"\nMean: {col.mean():.4f}")
        print(f"Median: {col.median():.4f}")
        print(f"Standard Deviation: {col.std():.4f}")
        print(f"Min: {col.min():.4f}")
        print(f"Max: {col.max():.4f}")
        print(f"Q1 (25%): {col.quantile(0.25):.4f}")
        print(f"Q3 (75%): {col.quantile(0.75):.4f}")
    else:
        print(f"\nUnique Values: {col.nunique()}")
        print(f"\nTop 10 Most Frequent Values:\n{col.value_counts().head(10)}")

def group_analysis(df, group_col, value_col, operation= 'mean'):
    print("\n" + "="*80)
    print(f"Group ANalysis")
    print("="*80)
    print(f"\nGrouping by : {group_col}")
    print(f"Analyzing: {value_col}")
    print(f"Operation: {operation}")

    result = df.groupby(group_col)[value_col].agg(operation)
    print(f"\nResult:\n{result}")
    print(result)
    return result

def value_counts_analysis(df, column):
    print("\n" + "="*80)
    print(f"Value frequency: {column}")
    print("="*80)

    col = df[column].value_counts()
    total = len(df)

    for value, count in col.items():
        percentage = (count / total) * 100
        print(f"{value}: {count} ({percentage:.2f}%)")
    

def find_outliers(df, column):
    print("\n" + "="*80)
    print(f"Outliers detection: {column}")
    print("="*80)


    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"Q1: {Q1:.4f}")
    print(f"Q3: {Q3:.4f}")
    print(f"IQR: {IQR:.4f}")
    print(f"Lower bound: {lower:.4f}")
    print(f"Upper bound: {upper:.4f}")
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"\nOutliers found: {len(outliers)}")
    return outliers

# Correlation analysis
def correlation_analysis(df):
    print("\n" + "="*80)
    print(f"Correlation Analysis")
    print("="*80)
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    print("\nHighly correlated pairs (.0.7: ")
    found = False
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                print(f"{corr.columns[i]} and {corr.columns[j]}: {corr.iloc[i, j]:.4f}")
                found = True
    if not found:
        print("No pairs with correlation > 0.7")
