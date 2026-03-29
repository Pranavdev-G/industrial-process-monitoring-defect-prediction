import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df):
    """
    Clean and prepare dataset for ML and analysis
    """

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # 3. Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 4. Separate features & target
    target_col = None
    for col in df.columns:
        if "defect" in col.lower():
            target_col = col
            break

    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df