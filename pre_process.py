import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # ================= CLEANING =================
    df = df.reset_index(drop=True)  # Create numeric index as time proxy
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # ================= TARGET COLUMN =================
    if "DefectStatus" in df.columns:
        df.rename(columns={"DefectStatus": "Defect"}, inplace=True)

    # Ensure binary (0/1)
    if "Defect" in df.columns and df["Defect"].dtype != int:
        df["Defect"] = df["Defect"].apply(lambda x: 1 if str(x).lower() in ["1", "yes", "defect", "true"] else 0)

    # ================= TIME SORTING & LAG FEATURES =================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != "defect"]
    
    # Sort by index (time proxy) - NO SHUFFLE
    df = df.sort_index()
    
    # Add lag features, rolling stats (window=3)
    new_features = []
    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=3, min_periods=1).std()
        new_features.extend([f'{col}_lag1', f'{col}_lag2', f'{col}_rolling_mean', f'{col}_rolling_std'])
    
    # ================= FEATURES (original + new) =================
    all_feature_cols = feature_cols + new_features
    df = df.dropna()  # Drop rows with NaN from lags
    
    X = df[all_feature_cols]
    y = df["Defect"] if "Defect" in df.columns else None

    # ================= SCALING =================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df
