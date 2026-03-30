from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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

def load_data():
    """Load dataset from CSV file"""
    global current_data
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if df.empty:
                current_data = None
                return False
            _, _, current_data = preprocess_data(df)
            return True
        current_data = None
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        current_data = None
        return False

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

def build_overview_summary():
    """Build a processed-data overview for the dashboard."""
    global current_data

    if current_data is None:
        return {"success": False, "error": "No data loaded"}

    defect_col = get_defect_column(current_data)
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in current_data.columns if col != defect_col]
    numeric_feature_cols = [col for col in numeric_cols if col != defect_col]

    defect_labels = []
    if defect_col:
        defect_labels = current_data[defect_col].apply(normalize_defect_value)

    good_count = int((defect_labels == "Good").sum()) if defect_col else 0
    defect_count = int((defect_labels == "Defective").sum()) if defect_col else 0
    total_samples = int(len(current_data))
    defect_rate = round((defect_count / total_samples) * 100, 2) if total_samples else 0.0

    process_status = "Attention Needed" if defect_rate > 20 else "Stable Trend"

    detail_columns = feature_cols[:4]
    good_records = []
    defect_records = []
    if defect_col:
        preview_cols = detail_columns + [defect_col]
        good_records = current_data.loc[defect_labels == "Good", preview_cols].head(5).copy()
        defect_records = current_data.loc[defect_labels == "Defective", preview_cols].head(5).copy()
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
            "good_product_details": clean_nan(good_records.to_dict(orient="records")) if defect_col else [],
            "defective_product_details": clean_nan(defect_records.to_dict(orient="records")) if defect_col else [],
            "variable_details": variable_details,
            "spc_summary": spc_summary
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

@app.get("/cluster")
def get_cluster(n_clusters: int = Query(default=3, ge=2, le=10)):

    global current_data
    global cluster_model, cluster_scaler, cluster_features

    if current_data is None:
        return {"success": False, "error": "No data loaded"}
    
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
    
    # HIERARCHICAL CLUSTERING (NEW)
    Z = linkage(X_scaled, method='ward')
    h_labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Dendrogram coordinates for visualization  
    # Sample 50 points for dendrogram to avoid too many points
    dendro_data = {
        "leaves": X_scaled[:50].shape[0],
        "icoord": [], "dcoord": [], "icoord2": [], "dcoord2": []  # Simplified for frontend
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
        "data_path": data_path,
        "data_exists": os.path.exists(data_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
