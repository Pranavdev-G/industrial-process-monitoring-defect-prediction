import pandas as pd
import numpy as np

def get_descriptive_stats(df):
    """
    Compute descriptive statistics for numeric columns.
    Returns formatted DataFrame.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()

    stats = numeric.describe().T
    stats = stats.round(4)
    return stats

def get_correlation_matrix(df):
    """Return correlation matrix for numeric columns"""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr()

def get_covariance_matrix(df):
    """Return covariance matrix for numeric columns"""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.cov()
