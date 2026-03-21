import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """
    Load CSV data and perform basic cleaning.
    - Handle missing values
    - Detect data types
    - Return cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")

    # Basic cleaning
    # Drop columns with all NaN
    df = df.dropna(axis=1, how='all')

    # For numeric columns, fill NaN with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # For categorical, fill with mode or 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')

    return df

def get_numeric_columns(df):
    """Return list of numeric column names"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def validate_dataset(df):
    """Basic validation"""
    if df.empty:
        return False, "Dataset is empty"
    if len(df.columns) == 0:
        return False, "No columns found"
    return True, "Valid"
