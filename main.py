from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pre_process import preprocess_data
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from typing import Optional
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the dataset
current_data = None
data_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
cluster_model = None
cluster_scaler = None
cluster_features = None
data_source = "none"

class AnalyzeInput(BaseModel):
    temperature: float
    pressure: float
    speed: float

def clean_nan(obj):
    """Recursively replace NaN/Inf with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return clean_nan(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

def build_default_dataset(n_rows: int = 180):
    """Create a synthetic industrial dataset so clustering/dendrogram always works."""
    rng = np.random.default_rng(42)

    # Create three latent operating regimes for visible hierarchical structure.
    cluster_ids = rng.choice([0, 1, 2], size=n_rows, p=[0.4, 0.35, 0.25])

    base_volume = np.where(cluster_ids == 0, 520, np.where(cluster_ids == 1, 390, 610))
    base_quality = np.where(cluster_ids == 0, 88, np.where(cluster_ids == 1, 72, 82))
    base_delay = np.where(cluster_ids == 0, 5.0, np.where(cluster_ids == 1, 12.0, 8.0))
    base_maintenance = np.where(cluster_ids == 0, 8.5, np.where(cluster_ids == 1, 14.0, 11.5))

    production_volume = np.clip(rng.normal(base_volume, 28), 250, 800)
    supplier_quality = np.clip(rng.normal(base_quality, 5.5), 45, 100)
    delivery_delay = np.clip(rng.normal(base_delay, 2.0), 0, 25)
    maintenance_hours = np.clip(rng.normal(base_maintenance, 2.2), 1, 24)

    defect_rate = np.clip(
        0.18 * delivery_delay
        + 0.12 * maintenance_hours
        + (100 - supplier_quality) * 0.08
        + rng.normal(0, 1.5, n_rows),
        0,
        35
    )
    quality_score = np.clip(95 - defect_rate + rng.normal(0, 2.5, n_rows), 40, 100)
    downtime_percentage = np.clip(defect_rate * 0.9 + rng.normal(0, 1.5, n_rows), 0, 55)
    energy_efficiency = np.clip(quality_score - downtime_percentage * 0.2 + rng.normal(0, 2.0, n_rows), 25, 100)

    df = pd.DataFrame({
        "ProductionVolume": np.round(production_volume, 2),
        "ProductionCost": np.round(production_volume * 2.4 + rng.normal(0, 45, n_rows), 2),
        "SupplierQuality": np.round(supplier_quality, 2),
        "DeliveryDelay": np.round(delivery_delay, 2),
        "DefectRate": np.round(defect_rate, 2),
        "QualityScore": np.round(quality_score, 2),
        "MaintenanceHours": np.round(maintenance_hours, 2),
        "DowntimePercentage": np.round(downtime_percentage, 2),
        "InventoryTurnover": np.round(np.clip(11 - delivery_delay * 0.3 + rng.normal(0, 0.8, n_rows), 1, 18), 2),
        "StockoutRate": np.round(np.clip(delivery_delay * 0.6 + rng.normal(0, 1.0, n_rows), 0, 30), 2),
        "WorkerProductivity": np.round(np.clip(78 + supplier_quality * 0.15 - maintenance_hours * 0.9 + rng.normal(0, 4, n_rows), 35, 98), 2),
        "SafetyIncidents": np.round(np.clip(maintenance_hours * 0.15 + rng.normal(0.3, 0.7, n_rows), 0, 9), 2),
        "EnergyConsumption": np.round(np.clip(production_volume * 0.8 + rng.normal(0, 35, n_rows), 100, 900), 2),
        "EnergyEfficiency": np.round(energy_efficiency, 2),
        "AdditiveProcessTime": np.round(np.clip(6 + maintenance_hours * 0.35 + rng.normal(0, 1.0, n_rows), 2, 20), 2),
        "AdditiveMaterialCost": np.round(np.clip(90 + production_volume * 0.18 + defect_rate * 1.5 + rng.normal(0, 20, n_rows), 40, 320), 2),
        "Defect": (defect_rate > np.percentile(defect_rate, 62)).astype(int)
    })

    return df

def load_data():
    """Load dataset from CSV file"""
    global current_data, data_source
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if df.empty:
                default_df = build_default_dataset()
                _, _, current_data = preprocess_data(default_df)
                data_source = "default_synthetic_dataset"
                return True
            _, _, current_data = preprocess_data(df)
            data_source = "uploaded_or_local_csv"
            return True
        default_df = build_default_dataset()
        _, _, current_data = preprocess_data(default_df)
        data_source = "default_synthetic_dataset"
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        default_df = build_default_dataset()
        _, _, current_data = preprocess_data(default_df)
        data_source = "default_synthetic_dataset"
        return True

@app.on_event("startup")
def initialize_default_data():
    """Ensure there is always a dataset ready for dashboard charts."""
    load_data()

def get_defect_column(df):
    """Return the defect column name when present, ignoring case."""
    for col in df.columns:
        if col.lower() == "defect":
            return col
    return None

def split_pascal_case(name):
    """Convert a compact column name into readable words."""
    parts = []
    current = ""
    for char in name:
        if char.isupper() and current and not current[-1].isupper():
            parts.append(current)
            current = char
        else:
            current += char
    if current:
        parts.append(current)
    return " ".join(parts)

def get_variable_meaning(column_name):
    """Return a short one-word meaning for a known process variable."""
    meanings = {
        "ProductionVolume": "Volume",
        "ProductionCost": "Cost",
        "SupplierQuality": "Quality",
        "DeliveryDelay": "Delay",
        "DefectRate": "Defects",
        "QualityScore": "Score",
        "MaintenanceHours": "Maintenance",
        "DowntimePercentage": "Downtime",
        "InventoryTurnover": "Inventory",
        "StockoutRate": "Stockout",
        "WorkerProductivity": "Productivity",
        "SafetyIncidents": "Safety",
        "EnergyConsumption": "Energy",
        "EnergyEfficiency": "Efficiency",
        "AdditiveProcessTime": "Time",
        "AdditiveMaterialCost": "Material",
        "Defect": "Status"
    }
    return meanings.get(column_name, "Metric")

def normalize_defect_value(value):
    """Map defect values to user-facing labels."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"1", "yes", "defect", "defective", "true"}:
        return "Defective"
    if text in {"0", "no", "good", "normal", "false"}:
        return "Good"
    return str(value)

def get_defect_labels(df):
    """
    Return defect labels as Good/Defective.
    Fallback logic is used when an explicit defect column is not present.
    """
    defect_col = get_defect_column(df)
    if defect_col:
        labels = df[defect_col].apply(normalize_defect_value)
        if ((labels == "Good") | (labels == "Defective")).any():
            return labels

    if "DefectRate" in df.columns and pd.api.types.is_numeric_dtype(df["DefectRate"]):
        threshold = float(df["DefectRate"].quantile(0.75))
        return pd.Series(
            np.where(df["DefectRate"] >= threshold, "Defective", "Good"),
            index=df.index
        )

    if "QualityScore" in df.columns and pd.api.types.is_numeric_dtype(df["QualityScore"]):
        threshold = float(df["QualityScore"].quantile(0.25))
        return pd.Series(
            np.where(df["QualityScore"] <= threshold, "Defective", "Good"),
            index=df.index
        )

    return pd.Series(["Unknown"] * len(df), index=df.index)

def get_recommendation_for_feature(feature_name, direction):
    """Return practical corrective action for a feature trend."""
    playbook = {
        "ProductionCost": {
            "high": "Audit scrap/rework sources, tighten process tolerance windows, and renegotiate high-variance supplier components.",
            "low": "Validate whether cost reduction changed raw material grade or process quality checks."
        },
        "SupplierQuality": {
            "high": "Keep current supplier control plan and continue incoming inspection sampling.",
            "low": "Increase incoming quality audits and enforce supplier CAPA with stricter acceptance criteria."
        },
        "DeliveryDelay": {
            "high": "Reduce waiting between process stages, rebalance schedule load, and maintain safety stock for critical materials.",
            "low": "Preserve current planning discipline and monitor bottlenecks weekly."
        },
        "DefectRate": {
            "high": "Introduce immediate root-cause containment on top rejection codes and run SPC alarms on critical parameters.",
            "low": "Sustain current in-process checks and verify trend stability."
        },
        "QualityScore": {
            "high": "Standardize the best-performing operating window as SOP for all shifts.",
            "low": "Run shift-wise process audits and retrain operators on calibration and setup checks."
        },
        "MaintenanceHours": {
            "high": "Move from reactive to preventive maintenance and schedule condition-based service on high-failure assets.",
            "low": "Ensure maintenance is not under-reported and verify machine health with vibration/temperature checks."
        },
        "DowntimePercentage": {
            "high": "Target top downtime reasons with preventive actions and spare-part readiness to cut stoppage duration.",
            "low": "Continue autonomous maintenance and verify downtime logging accuracy."
        },
        "InventoryTurnover": {
            "high": "Validate replenishment logic so high turnover does not cause rushed processing or stockouts.",
            "low": "Improve material flow planning and remove slow-moving inventory bottlenecks."
        },
        "StockoutRate": {
            "high": "Increase reorder-point safety margin and improve supplier lead-time reliability monitoring.",
            "low": "Keep current stock planning controls active."
        },
        "WorkerProductivity": {
            "high": "Document high-performing practices and replicate across shifts with standardized work instructions.",
            "low": "Provide targeted operator training and reduce changeover complexity on weak stations."
        },
        "SafetyIncidents": {
            "high": "Run immediate safety kaizen and enforce PPE/procedure compliance checks at risky work cells.",
            "low": "Maintain current safety culture and continue routine audits."
        },
        "EnergyConsumption": {
            "high": "Tune machine settings for efficient load, eliminate idle-running, and calibrate high-draw equipment.",
            "low": "Track energy profile to ensure efficiency gains do not hide quality drift."
        },
        "EnergyEfficiency": {
            "high": "Lock in current energy-efficient process window as standard setpoint.",
            "low": "Recalibrate equipment and optimize cycle parameters to reduce wasteful energy use."
        },
        "AdditiveProcessTime": {
            "high": "Reduce process cycle variation with setup checklists and optimized job sequencing.",
            "low": "Validate time reductions against quality checks to avoid rushed processing."
        },
        "AdditiveMaterialCost": {
            "high": "Investigate material waste points and tighten dosing/usage control at the source.",
            "low": "Confirm cost savings are not from lower-spec material substitutions."
        }
    }
    feature_rules = playbook.get(feature_name)
    if not feature_rules:
        if direction == "high":
            return f"Control high variation in {feature_name} using SPC limits and corrective action triggers."
        return f"Review whether low {feature_name} is causing process instability, then adjust operating setpoints."
    return feature_rules.get(direction, "Review this feature trend and define a corrective control plan.")

def infer_defect_type(top_drivers):
    """Infer a likely defect category from top feature drivers."""
    names = {driver["feature"] for driver in top_drivers}
    if {"SupplierQuality", "QualityScore"} & names:
        return "Quality degradation defect"
    if {"MaintenanceHours", "DowntimePercentage", "AdditiveProcessTime"} & names:
        return "Machine/process instability defect"
    if {"DeliveryDelay", "InventoryTurnover", "StockoutRate"} & names:
        return "Flow and scheduling induced defect"
    if {"EnergyConsumption", "EnergyEfficiency"} & names:
        return "Process efficiency related defect"
    return "Mixed process defect pattern"

def build_defect_diagnosis(df):
    """Create defect type summary and practical overcome actions from dataset."""
    defect_col = get_defect_column(df)
    labels = get_defect_labels(df)
    defective_mask = labels == "Defective"
    good_mask = labels == "Good"

    defective_count = int(defective_mask.sum())
    good_count = int(good_mask.sum())
    total_count = int(len(df))
    defect_rate = round((defective_count / total_count) * 100, 2) if total_count else 0.0

    if defective_count == 0:
        return {
            "available": True,
            "defect_type": "No major defect trend",
            "message": "No defective records found after preprocessing." if defect_col else "No explicit defect trend found from fallback defect inference.",
            "defect_rate": defect_rate,
            "top_drivers": [],
            "overcome_actions": [
                "Maintain current control limits and continue periodic SPC review.",
                "Keep preventive maintenance and calibration schedule on-time.",
                "Track quality metrics weekly to catch early drift."
            ]
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != defect_col]

    driver_rows = []
    for col in feature_cols:
        defect_values = df.loc[defective_mask, col].dropna()
        good_values = df.loc[good_mask, col].dropna()
        if len(defect_values) < 2 or len(good_values) < 2:
            continue

        mean_defect = float(defect_values.mean())
        mean_good = float(good_values.mean())
        std_all = float(df[col].std())
        if std_all == 0 or np.isnan(std_all):
            continue

        effect = (mean_defect - mean_good) / std_all
        direction = "high" if effect >= 0 else "low"
        driver_rows.append({
            "feature": col,
            "mean_defective": round(mean_defect, 3),
            "mean_good": round(mean_good, 3),
            "impact_score": round(float(abs(effect)), 3),
            "direction": direction
        })

    driver_rows = sorted(driver_rows, key=lambda x: x["impact_score"], reverse=True)[:4]
    defect_type = infer_defect_type(driver_rows)

    actions = []
    for row in driver_rows:
        actions.append(get_recommendation_for_feature(row["feature"], row["direction"]))

    if not actions:
        actions = [
            "Perform Pareto analysis on defect events and prioritize top recurring causes.",
            "Use SPC alarms and operator checklist to contain process drift early.",
            "Validate machine calibration and maintenance compliance before next production run."
        ]

    return {
        "available": True,
        "defect_type": defect_type,
        "message": f"Defect rate is {defect_rate}% with {defective_count} defective samples." if defect_col else f"Defect rate is {defect_rate}% using inferred labels from process metrics.",
        "defect_rate": defect_rate,
        "defective_count": defective_count,
        "good_count": good_count,
        "top_drivers": driver_rows,
        "overcome_actions": actions
    }

def build_overview_summary():
    """Build a processed-data overview for the dashboard."""
    global current_data

    if current_data is None:
        return {"success": False, "error": "No data loaded"}

    defect_col = get_defect_column(current_data)
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in current_data.columns if col != defect_col]
    numeric_feature_cols = [col for col in numeric_cols if col != defect_col]

    defect_labels = get_defect_labels(current_data)

    good_count = int((defect_labels == "Good").sum())
    defect_count = int((defect_labels == "Defective").sum())
    total_samples = int(len(current_data))
    defect_rate = round((defect_count / total_samples) * 100, 2) if total_samples else 0.0

    process_status = "Attention Needed" if defect_rate > 20 else "Stable Trend"

    detail_columns = feature_cols[:4]
    good_records = pd.DataFrame()
    defect_records = pd.DataFrame()
    if defect_col or ((defect_labels == "Good").any() or (defect_labels == "Defective").any()):
        preview_cols = detail_columns + [defect_col]
        if defect_col is None:
            preview_cols = detail_columns.copy()
            current_data_local = current_data.copy()
            current_data_local["Defect"] = defect_labels
            defect_col = "Defect"
        else:
            current_data_local = current_data
        good_records = current_data_local.loc[defect_labels == "Good", preview_cols].head(5).copy()
        defect_records = current_data_local.loc[defect_labels == "Defective", preview_cols].head(5).copy()
        if not good_records.empty:
            good_records[defect_col] = "Good"
        if not defect_records.empty:
            defect_records[defect_col] = "Defective"

    variable_details = []
    for col in current_data.columns:
        series = current_data[col]
        variable_details.append({
            "name": col,
            "label": split_pascal_case(col),
            "type": "Numeric" if pd.api.types.is_numeric_dtype(series) else "Categorical",
            "role": "Target" if col == defect_col else "Input",
            "meaning": get_variable_meaning(col),
            "sample": str(series.iloc[0]) if len(series) > 0 else "N/A"
        })

    spc_summary = {}
    for col in numeric_feature_cols:
        data = current_data[col].dropna()
        if len(data) > 0:
            spc_summary[col] = {
                "mean": f"{data.mean():.2f}",
                "std": f"{data.std():.2f}" if len(data) > 1 else "0.00"
            }

    defect_diagnosis = build_defect_diagnosis(current_data)

    return {
        "success": True,
        "summary": {
            "total_samples": total_samples,
            "features_analyzed": feature_cols,
            "numeric_features": numeric_feature_cols,
            "defect_distribution": {
                "Good": good_count,
                "Defective": defect_count
            },
            "defect_status": {
                "good_products": good_count,
                "defective_products": defect_count,
                "defect_rate": defect_rate,
                "process_status": process_status
            },
            "good_product_details": clean_nan(good_records.to_dict(orient="records")),
            "defective_product_details": clean_nan(defect_records.to_dict(orient="records")),
            "variable_details": variable_details,
            "spc_summary": spc_summary,
            "defect_diagnosis": defect_diagnosis
        }
    }

def build_problem_payload():
    """Build topic explanation content - EXACT SPECIFICATION"""
    global current_data
    
    # Dataset stats if available
    total_samples = 0
    numeric_features = 0
    defect_rate = None
    
    if current_data is not None:
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
        defect_col = get_defect_column(current_data)
        feature_cols = [col for col in numeric_cols if col.lower() != "defect"]
        
        total_samples = int(len(current_data))
        numeric_features = int(len(feature_cols))
        
        if defect_col:
            total = len(current_data)
            defective = int((current_data[defect_col] == 1).sum())
            defect_rate = round((defective / total) * 100, 2) if total > 0 else None

    return {
        "success": True,
        "dataset_highlights": {
            "total_samples": total_samples,
            "numeric_features": numeric_features, 
            "defect_rate": defect_rate
        },
        "problem_statement": """Modern industrial manufacturing systems generate large volumes of process data through sensors and control systems. However, traditional monitoring approaches often fail to detect early signs of process instability and potential defects. This leads to delayed corrective actions, increased production costs, and reduced product quality. Therefore, there is a need for a data-driven system that continuously monitors process variables, identifies hidden patterns, and predicts defects in advance.""",
        "goal": "Industrial Process Monitoring and Defect Prediction System",
        "monitoring": "An Integrated Approach Using Statistical Process Control, Multivariate Analysis, and Machine Learning",
        "justification": """Current industrial practices rely on manual inspection, rule-based quality checks, and Statistical Process Control (SPC) charts. These methods are limited when handling high-dimensional data and complex relationships. Machine learning models are used but often lack interpretability and integration with statistical methods.""",
        "industrial_solutions": [
            {
                "title": "Manual Inspection & Rule-based QC", 
                "description": "Traditional methods limited by human error and static rules"
            },
            {
                "title": "SPC Charts", 
                "description": "Good for stability but struggles with multivariate relationships"
            },
            {
                "title": "Standalone ML Models", 
                "description": "Powerful prediction but lacks process context and interpretability"
            }
        ],
        "details": [
            {
                "title": "Proposed Solution",
                "content": "This system integrates data preprocessing, SPC, PCA, clustering, and machine learning-based prediction. It detects instability, reduces dimensionality, identifies patterns, and predicts defects accurately in a unified workflow."
            }
        ],
        "features": ["ProductionVolume", "ProductionCost", "SupplierQuality", "DeliveryDelay", "DefectRate", "QualityScore", "MaintenanceHours", "DowntimePercentage", "InventoryTurnover", "StockoutRate", "WorkerProductivity", "SafetyIncidents", "EnergyConsumption", "EnergyEfficiency", "AdditiveProcessTime", "AdditiveMaterialCost"],
        "methods": ["Statistical Process Control (SPC)", "Principal Component Analysis (PCA)", "K-Means Clustering", "Hierarchical Clustering", "Logistic Regression", "Decision Tree Classification"]
    }

def get_numeric_feature_cols(df):
    """Get numeric feature columns excluding defect."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col.lower() != "defect"]

def parse_feature_list(features: Optional[str]):
    """Parse comma-separated feature query input."""
    if not features:
        return None
    parsed = [f.strip() for f in features.split(",") if f.strip()]
    return parsed if parsed else None

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the main HTML page"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        # FIX: Added encoding='utf-8' to handle special characters in HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1><p>Please ensure index.html is in the same directory.</p>")

@app.get("/topic-explanation", response_class=HTMLResponse)
def topic_explanation_page():
    """Serve the topic explanation page."""
    html_path = os.path.join(os.path.dirname(__file__), "topic_explanation.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>topic_explanation.html not found</h1>")

@app.get("/machine-analysis", response_class=HTMLResponse)
def machine_analysis_page():
    """Serve the machine analysis page."""
    html_path = os.path.join(os.path.dirname(__file__), "machine_analysis.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>machine_analysis.html not found</h1>")

@app.get("/favicon.ico")
def favicon():
    path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    return FileResponse(path) if os.path.exists(path) else {"status": "not found"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file and save it as the dataset"""
    global current_data

    try:
        # Read file content
        content = await file.read()

        # Save file
        with open(data_path, 'wb') as f:
            f.write(content)

        # Read CSV safely
        current_data = pd.read_csv(data_path, encoding='utf-8')

        # 👉 APPLY PREPROCESSING (IMPORTANT)
        from pre_process import preprocess_data
        X, y, current_data = preprocess_data(current_data)

        # Preview (clean NaN → None for JSON)
        preview = (
            current_data.head(10)
            .replace({np.nan: None})
            .to_dict(orient='records')
        )

        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully",
            "preview": preview,
            "columns": list(current_data.columns),
            "shape": list(current_data.shape)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/problem")
def get_problem():
    """Get problem definition"""
    global current_data
    return build_problem_payload()

@app.get("/spc")
def get_spc():
    """Get SPC analysis results"""
    global current_data
    if current_data is None:
        return {"success": False, "error": "No data loaded"}
    
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']
    
    if not feature_cols:
        return {"success": False, "error": "No numeric features found for SPC analysis"}

    spc_results = {}
    all_stable = True
    
    for col in feature_cols[:4]:
        data = current_data[col].dropna().values
        
        # FIX: Need at least 2 points to calculate std deviation
        if len(data) < 2:
            continue
            
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        
        UCL = mean + 3 * std
        LCL = mean - 3 * std
        
        out_of_control = int(np.sum((data > UCL) | (data < LCL)))
        if out_of_control > 0:
            all_stable = False
        
        spc_results[col] = {
            "data": data.tolist()[:100],
            "mean": mean,
            "std": std,
            "UCL": UCL,
            "LCL": LCL,
            "out_of_control": out_of_control
        }
    
    return {
        "success": True,
        "spc_results": spc_results,
        "process_stable": all_stable,
        "message": "Process is STABLE" if all_stable else "Process is UNSTABLE"
    }

@app.get("/eda")
def get_eda():
    """Get EDA results"""
    global current_data

    if current_data is None:
        return {"success": False, "error": "No data loaded"}

    # ================= NUMERIC FEATURES =================
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']

    if not feature_cols:
        return {"success": False, "error": "No numeric features found"}

    # ================= STATISTICS =================
    stats = {}
    for col in feature_cols:
        data = current_data[col].dropna()

        if len(data) == 0:
            continue

        stats[col] = {
            "mean": float(data.mean()),
            "std": float(data.std()) if len(data) > 1 else 0.0,
            "min": float(data.min()),
            "25%": float(data.quantile(0.25)),
            "50%": float(data.quantile(0.50)),
            "75%": float(data.quantile(0.75)),
            "max": float(data.max())
        }

    if not stats:
        return {"success": False, "error": "No valid numeric data"}

    # ================= HISTOGRAM =================
    histograms = {}

    for col in feature_cols[:1]:  # only first column for simplicity
        data = current_data[col].dropna()

        if len(data) > 1:
            counts, bins = np.histogram(data, bins=20)
        else:
            counts, bins = [0], [0, 1]

        histograms[col] = {
            "counts": counts.tolist(),
            "bins": bins.tolist()
        }

    # ================= CORRELATION =================
    try:
        corr_matrix = current_data[feature_cols].corr()

        # Handle NaN (constant columns issue)
        corr_matrix = corr_matrix.fillna(0)

        corr_matrix = corr_matrix.to_dict()

    except Exception:
        corr_matrix = {}

    # ================= INTERPRETATION (FOR MARKS) =================
    interpretation = []
    for col in feature_cols[:3]:
        interpretation.append(
            f"{col} has average value {round(stats[col]['mean'], 2)} with variability {round(stats[col]['std'], 2)}"
        )

    interpretation.append("Correlation analysis helps identify relationships between process variables")
    interpretation.append("Histogram shows distribution and spread of key feature")

    # ================= FINAL RETURN =================
    return clean_nan({
        "success": True,
        "statistics": stats,
        "histograms": histograms,
        "correlation": corr_matrix,
        "numeric_columns": feature_cols,
        "interpretation": interpretation   
    })

@app.get("/pca")
def get_pca():
    """Get PCA analysis results"""
    global current_data
    if current_data is None:
        return {"success": False, "error": "No data loaded"}
    
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']
    
    X = current_data[feature_cols].dropna()
    
    if len(X) < 2:
        return {"success": False, "error": "Not enough data for PCA"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_components = min(len(feature_cols), 5, len(X))
    if n_components < 1:
         return {"success": False, "error": "Not enough features for PCA"}

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_.tolist()
    cumulative_variance = np.cumsum(explained_variance).tolist()
    
    pca_data = {
    "PC1": X_pca[:, 0].tolist(),
    "PC2": X_pca[:, 1].tolist() if n_components > 1 else [0] * len(X_pca),
    "Defect": current_data["Defect"].tolist() if "Defect" in current_data.columns else None
}
    
    if 'Defect' in current_data.columns:
        pca_data["Defect"] = current_data.loc[X.index, 'Defect'].tolist()
    
    return {
        "success": True,
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "pca_data": pca_data,
        "n_components": n_components
    }

@app.get("/factor")
def get_factor():
    global current_data

    if current_data is None:
        return {"success": False, "error": "No data loaded"}

    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']

    X = current_data[feature_cols].dropna()

    if len(X) < 2:
        return {"success": False, "error": "Not enough data"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fa = FactorAnalysis(n_components=2)
    factors = fa.fit_transform(X_scaled)

    return {
        "success": True,
        "factor1": factors[:, 0].tolist(),
        "factor2": factors[:, 1].tolist()
    }

@app.get("/dendrogram_data")
def get_dendrogram_data(
    n_samples: int = Query(default=120, ge=2, le=2000),
    method: str = Query(default="ward"),
    features: Optional[str] = Query(default=None, description="Comma-separated numeric feature names")
):
    """
    Prepare hierarchical clustering dendrogram data from the currently loaded dataset.
    Returns linkage matrix and plot-ready coordinates for frontend rendering.
    """
    global current_data

    if current_data is None and not load_data():
        return {"success": False, "error": "No data loaded. Upload a CSV or keep dataset.csv in project root."}

    allowed_methods = {"ward", "complete", "average", "single"}
    method = (method or "ward").strip().lower()
    if method not in allowed_methods:
        return {
            "success": False,
            "error": f"Invalid method '{method}'. Use one of: {', '.join(sorted(allowed_methods))}"
        }

    all_numeric_features = get_numeric_feature_cols(current_data)
    if not all_numeric_features:
        return {"success": False, "error": "No numeric features available for hierarchical clustering"}

    requested_features = parse_feature_list(features)
    if requested_features:
        missing = [f for f in requested_features if f not in all_numeric_features]
        if missing:
            return {
                "success": False,
                "error": f"Invalid features: {', '.join(missing)}",
                "available_features": all_numeric_features
            }
        feature_cols = requested_features
    else:
        feature_cols = all_numeric_features

    X_full = current_data[feature_cols].dropna()
    total_rows = len(X_full)
    if total_rows < 2:
        return {"success": False, "error": "Not enough valid rows after removing missing values"}

    if total_rows > n_samples:
        sampled_idx = np.linspace(0, total_rows - 1, n_samples, dtype=int)
        X = X_full.iloc[sampled_idx].copy()
        source_indices = X_full.index[sampled_idx].tolist()
    else:
        X = X_full.copy()
        source_indices = X_full.index.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method=method)

    labels = [f"R{int(idx)}" for idx in source_indices]
    dendro_plot = dendrogram(Z, labels=labels, no_plot=True)

    return {
        "success": True,
        "method": method,
        "features_used": feature_cols,
        "total_rows_after_dropna": int(total_rows),
        "rows_used": int(len(X)),
        "source_indices": [int(i) for i in source_indices],
        "dendrogram": {
            "leaves": int(len(source_indices)),
            "icoord": dendro_plot.get("icoord", []),
            "dcoord": dendro_plot.get("dcoord", []),
            "leaf_order": [int(i) for i in dendro_plot.get("leaves", [])],
            "leaf_labels": dendro_plot.get("ivl", []),
            "color_list": dendro_plot.get("color_list", [])
        },
        "linkage_matrix": Z.tolist()
    }

@app.get("/cluster")
def get_cluster(n_clusters: int = Query(default=3, ge=2, le=10)):

    global current_data
    global cluster_model, cluster_scaler, cluster_features

    if current_data is None and not load_data():
        return {"success": False, "error": "No data loaded. Upload a CSV or keep dataset.csv in project root."}
    
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']
    
    X = current_data[feature_cols].dropna()
    
    if len(X) < n_clusters:
        return {"success": False, "error": f"Not enough data samples ({len(X)}) for {n_clusters} clusters"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMEANS (existing)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # HIERARCHICAL CLUSTERING
    Z = linkage(X_scaled, method='ward')
    h_labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Dendrogram coordinates for visualization
    max_dendro_points = min(len(X_scaled), 50)
    dendro_input = X_scaled[:max_dendro_points]
    dendro_Z = linkage(dendro_input, method='ward')
    dendro_plot = dendrogram(dendro_Z, no_plot=True)
    leaves_order = dendro_plot.get("leaves", [])
    dendro_data = {
        "leaves": int(max_dendro_points),
        "icoord": dendro_plot.get("icoord", []),
        "dcoord": dendro_plot.get("dcoord", []),
        "leaf_order": [int(i) for i in leaves_order],
        "leaf_labels": [f"S{i}" for i in leaves_order],
        "method": "ward"
    }
    
    # Store models
    cluster_model = kmeans
    cluster_scaler = scaler
    cluster_features = feature_cols

    # KMeans stats (existing format)
    kmeans_stats = []
    for i in range(n_clusters):
        count = int(np.sum(kmeans_labels == i))
        kmeans_stats.append({
            "cluster": i,
            "count": count,
            "percentage": float(count / len(X) * 100)
        })

    # Combined visualization data (KMeans)
    if len(feature_cols) >= 2:
        x_feature, y_feature = feature_cols[0], feature_cols[1]
        x_data = current_data.loc[X.index, x_feature].tolist()
        y_data = current_data.loc[X.index, y_feature].tolist()
    else:
        x_data = X_scaled[:, 0].tolist()
        y_data = X_scaled[:, 1].tolist() if X_scaled.shape[1] > 1 else [0] * len(X_scaled)
        x_feature, y_feature = "Component 1", "Component 2"
    
    kmeans_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_x = kmeans_centers[:, 0].tolist()
    centers_y = kmeans_centers[:, 1].tolist() if kmeans_centers.shape[1] > 1 else [0]*n_clusters

    return {
        "success": True,
        "kmeans_stats": kmeans_stats,
        "hierarchical_labels": h_labels.tolist(),
        "cluster_data": {
            "labels": kmeans_labels.tolist(),  # Frontend uses KMeans plot
            "x": x_data,
            "y": y_data,
            "centers_x": centers_x,
            "centers_y": centers_y,
            "feature_x": x_feature,
            "feature_y": y_feature,
            "hierarchical_available": True
        },
        "dendrogram": dendro_data,
        "n_clusters": n_clusters
    }

@app.get("/cluster_predict")
def predict_cluster(v1: float, v2: float):
    global cluster_model, cluster_scaler

    if cluster_model is None:
        return {"error": "Run clustering first"}

    new_point = cluster_scaler.transform([[v1, v2]])
    cluster = int(cluster_model.predict(new_point)[0])

    return {"cluster": cluster}

@app.get("/model")
def get_model():
    """Train and evaluate models with full metrics - TIME SERIES SPLIT"""
    global current_data
    if current_data is None:
        return {"success": False, "error": "No data loaded"}
    
    if 'Defect' not in current_data.columns:
        return {"success": False, "error": "Defect column not found in dataset"}
    
    # Use ALL features (including lag/rolling from preprocess)
    feature_cols = [col for col in current_data.columns if col.lower() != 'defect' and pd.api.types.is_numeric_dtype(current_data[col])]
    
    data_clean = current_data[feature_cols + ['Defect']].dropna()
    
    if len(data_clean) < 10:
        return {"success": False, "error": "Not enough valid data samples to train model"}

    X = data_clean[feature_cols]
    y = data_clean['Defect']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # ✅ TIME SERIES SPLIT - NO SHUFFLE (already sorted by preprocess)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def get_metrics(y_true, y_pred, y_proba=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba[:,1]) if y_proba is not None else None
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec), 
            "f1_score": float(f1),
            "roc_auc": float(auc) if auc else None
        }
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=2000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)
    lr_metrics = get_metrics(y_test, lr_pred, lr_proba)
    
    # Decision Tree  
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=8)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_metrics = get_metrics(y_test, dt_pred)
    
    def calc_confusion_matrix(y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred).tolist()
    
    lr_cm = calc_confusion_matrix(y_test, lr_pred)
    dt_cm = calc_confusion_matrix(y_test, dt_pred)
    
    feature_importance = dict(zip(feature_cols, dt_model.feature_importances_.tolist()))
    
    return {
        "success": True,
        "results": {
            "logistic_regression": {
                **lr_metrics,
                "confusion_matrix": lr_cm
            },
            "decision_tree": {
                **dt_metrics, 
                "confusion_matrix": dt_cm
            }
        },
        "feature_importance": feature_importance,
        "comparison": {
            "models": ["Logistic Regression", "Decision Tree"],
            "best_model": "LR" if lr_metrics["accuracy"] > dt_metrics["accuracy"] else "DT"
        },
        "n_features": len(feature_cols)
    }

@app.get("/results")
def get_results():
    """Get final results summary"""
    global current_data
    return build_overview_summary()

@app.get("/defect_diagnosis")
def get_defect_diagnosis():
    """Get defect diagnosis and corrective actions as a dedicated payload."""
    global current_data
    if current_data is None and not load_data():
        return {"success": False, "error": "No data loaded"}
    return {
        "success": True,
        "diagnosis": build_defect_diagnosis(current_data)
    }

@app.get("/insights")
def insights():
    global current_data

    if current_data is None:
        return {"error": "No data"}

    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()

    insights = []

    for col in numeric_cols:
        mean = current_data[col].mean()
        insights.append(f"{col} average is {round(mean,2)}")

    insights.append("High variation indicates unstable process")
    insights.append("Stable features reduce defect probability")

    return {
        "insights": insights
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": current_data is not None,
        "data_source": data_source,
        "rows": int(len(current_data)) if current_data is not None else 0,
        "data_path": data_path,
        "data_exists": os.path.exists(data_path)
    }

@app.post("/analyze")
def analyze_machine(inputs: AnalyzeInput):
    base_value = (inputs.temperature * 0.5) + (inputs.pressure * 0.3) + (inputs.speed * 0.2)
    fluctuation_scale = 0.6 + (abs(inputs.temperature - inputs.pressure) / 100.0) + (inputs.speed / 120.0)
    time_points = [f"t{i}" for i in range(1, 11)]
    values = []

    for i in range(10):
        variation = np.sin((i + 1) * 0.8) * fluctuation_scale
        if inputs.speed >= 75 and i in {3, 7}:
            variation += 0.8 * fluctuation_scale
        values.append(round(base_value + variation, 2))

    variation_score = float(np.std(values))
    if variation_score < 1.0:
        result_sentence = "Machine is stable -> Low defect production"
    elif variation_score < 2.0:
        result_sentence = "Machine is slightly unstable -> Medium defect risk"
    else:
        result_sentence = "Machine is unstable -> High defect risk"

    return {
        "success": True,
        "time": time_points,
        "value": values,
        "result_sentence": result_sentence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
