import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # ================= CLEANING =================
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
    if df["Defect"].dtype != int:
        df["Defect"] = df["Defect"].apply(lambda x: 1 if x in [1, "Yes", "Defect"] else 0)

    # ================= FEATURES =================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != "Defect"]

    X = df[feature_cols]
    y = df["Defect"]

    # ================= SCALING =================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df
