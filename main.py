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
            current_data = df
            return True
        current_data = None
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        current_data = None
        return False

# Load data on startup
load_data()

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the main HTML page"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        # FIX: Added encoding='utf-8' to handle special characters in HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1><p>Please ensure index.html is in the same directory.</p>")

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
    if current_data is None:
        load_data()
    
    if current_data is None:
        return {
            "goal": "No data loaded. Please upload a dataset.",
            "monitoring": "N/A",
            "features": [],
            "methods": []
        }
    
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    
    return {
    "goal": "Develop a predictive system to classify defective products using machine learning",
    "justification": "Helps industries reduce defects and improve quality",
    "objective": "Analyze process parameters and predict defects",
    "methods": [
        "EDA",
        "SPC",
        "PCA",
        "Factor Analysis",
        "Clustering",
        "Machine Learning Models"
    ]
}

@app.get("/spc")
def get_spc():
    """Get SPC analysis results"""
    global current_data
    if current_data is None:
        load_data()
    
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
        load_data()
    
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
        load_data()
    
    if current_data is None:
        return {"success": False, "error": "No data loaded"}
    
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']
    
    X = current_data[feature_cols].dropna()
    
    if len(X) < n_clusters:
        return {"success": False, "error": f"Not enough data samples ({len(X)}) for {n_clusters} clusters"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # ✅ FIX → STORE MODEL HERE
    cluster_model = kmeans
    cluster_scaler = scaler
    cluster_features = feature_cols

    cluster_stats = []
    for i in range(n_clusters):
        count = int(np.sum(labels == i))
        cluster_stats.append({
            "cluster": i,
            "count": count,
            "percentage": float(count / len(labels) * 100)
        })
    
    # Visualization
    if len(feature_cols) >= 2:
        x_feature, y_feature = feature_cols[0], feature_cols[1]
        x_data = current_data.loc[X.index, x_feature].tolist()
        y_data = current_data.loc[X.index, y_feature].tolist()
    else:
        x_data = X_scaled[:, 0].tolist()
        y_data = X_scaled[:, 1].tolist() if X_scaled.shape[1] > 1 else [0] * len(X_scaled)
        x_feature, y_feature = "Component 1", "Component 2"
    
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_x = centers[:, 0].tolist()
    centers_y = centers[:, 1].tolist() if centers.shape[1] > 1 else [0]*n_clusters

    return {
        "success": True,
        "cluster_stats": cluster_stats,
        "cluster_data": {
            "labels": labels.tolist(),
            "x": x_data,
            "y": y_data,
            "centers_x": centers_x,
            "centers_y": centers_y,
            "feature_x": x_feature,
            "feature_y": y_feature
        },
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
    """Train and evaluate models"""
    global current_data
    if current_data is None:
        load_data()
    
    if current_data is None:
        return {"success": False, "error": "No data loaded"}
    
    if 'Defect' not in current_data.columns:
        return {"success": False, "error": "Defect column not found in dataset"}
    
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']
    
    data_clean = current_data[feature_cols + ['Defect']].dropna()
    
    if len(data_clean) < 5:
        return {"success": False, "error": "Not enough valid data samples to train model"}

    X = data_clean[feature_cols]
    y = data_clean['Defect']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_accuracy = float(np.mean(lr_predictions == y_test))
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_accuracy = float(np.mean(dt_predictions == y_test))
    
    def calc_confusion_matrix(y_true, y_pred):
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        return [[tn, fp], [fn, tp]]
    
    lr_cm = calc_confusion_matrix(y_test, lr_predictions)
    dt_cm = calc_confusion_matrix(y_test, dt_predictions)
    
    feature_importance = {feat: float(imp) for feat, imp in zip(feature_cols, dt_model.feature_importances_)}
    
    return {
    "success": True,
    "models": {
        "logistic_regression": {
            "accuracy": lr_accuracy,
            "confusion_matrix": lr_cm
        },
        "decision_tree": {
            "accuracy": dt_accuracy,
            "confusion_matrix": dt_cm
        }
    },
    "comparison": {
        "best_model": "Logistic Regression" if lr_accuracy > dt_accuracy else "Decision Tree",
        "difference": abs(lr_accuracy - dt_accuracy)
    }
}

@app.get("/results")
def get_results():
    """Get final results summary"""
    global current_data
    if current_data is None:
        load_data()
    
    if current_data is None:
        return {"success": False, "error": "No data loaded"}
    
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col.lower() != 'defect']
    
    total_samples = len(current_data)
    
    if 'Defect' in current_data.columns:
        defect_dist = current_data['Defect'].value_counts().to_dict()
        defect_distribution = {str(k): int(v) for k, v in defect_dist.items()}
    else:
        defect_distribution = {"No": total_samples, "Yes": 0}
    
    spc_summary = {}
    for col in feature_cols:
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
            "defect_distribution": defect_distribution,
            "spc_summary": spc_summary
        }
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
        "data_path": data_path,
        "data_exists": os.path.exists(data_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)