import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

INDUSTRY_REQUIRED_COLUMNS = [
    "ProductionVolume",
    "ProductionCost",
    "SupplierQuality",
    "DeliveryDelay",
    "DefectRate",
    "QualityScore",
    "MaintenanceHours",
    "DowntimePercentage",
    "InventoryTurnover",
    "StockoutRate",
    "WorkerProductivity",
    "SafetyIncidents",
    "EnergyConsumption",
    "EnergyEfficiency",
    "AdditiveProcessTime",
    "AdditiveMaterialCost",
]

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

    # Keep only required industry columns (+ target if available)
    required_cols = [col for col in INDUSTRY_REQUIRED_COLUMNS if col in df.columns]
    if "Defect" in df.columns:
        required_cols.append("Defect")
    if required_cols:
        df = df[required_cols].copy()

    # Ensure binary (0/1)
    if "Defect" in df.columns and df["Defect"].dtype != int:
        df["Defect"] = df["Defect"].apply(lambda x: 1 if str(x).lower() in ["1", "yes", "defect", "true"] else 0)

    # ================= FEATURE PREPARATION =================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != "defect"]
    
    # Sort by index (time proxy) - NO SHUFFLE
    df = df.sort_index()

    X = df[feature_cols] if feature_cols else pd.DataFrame(index=df.index)
    y = df["Defect"] if "Defect" in df.columns else None

    # ================= SCALING =================
    if not X.empty:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = np.empty((len(df), 0))

    return X_scaled, y, df
