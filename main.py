import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

app = FastAPI()
templates = Jinja2Templates(directory="templates")

dataset = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    global dataset

    context = {
        "request": request,
        "has_data": False,
        "message": "Upload CSV",
        "rows": 0,
        "columns": 0,
        "missing": 0,
        "duplicate_rows": 0,
        "numeric_count": 0,
        "categorical_count": 0,
        "completeness": 0,
        "top_missing_column": "None",
        "target_column": "Not available",
        "preview": [],
        "headers": [],
        "stats": [],
        "numeric_col": None,
        "line_data": [],
        "hist_data": [],
        "hist_labels": [],
        "spc_mean": 0,
        "spc_std": 0,
        "spc_ucl": 0,
        "spc_lcl": 0,
        "spc_data": [],
        "ml_accuracy": None,
        "ml_message": "No ML performed",
        "cluster_labels": [],
        "cluster_points": [],
        "cluster_centers": [],
        "cluster_columns": [],
        "cluster_counts": [],
        "cluster_message": "No clustering",
        "corr_matrix": [],
        "corr_headers": [],
        "linreg_score": None,
        "linreg_message": "No regression",
        "linreg_columns": [],
        "pca_data": [],
        "pca_variance": [],
        "pca_message": "No PCA",
    }

    if dataset is not None:
        try:
            context["has_data"] = True
            context["rows"] = len(dataset)
            context["columns"] = len(dataset.columns)
            context["missing"] = int(dataset.isnull().sum().sum())
            context["duplicate_rows"] = int(dataset.duplicated().sum())
            context["target_column"] = str(dataset.columns[-1]) if len(dataset.columns) else "Not available"

            context["headers"] = list(dataset.columns)
            context["preview"] = dataset.head().to_dict("records")

            numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = [col for col in dataset.columns if col not in numeric_cols]
            context["numeric_count"] = len(numeric_cols)
            context["categorical_count"] = len(categorical_cols)

            total_cells = max(len(dataset) * max(len(dataset.columns), 1), 1)
            context["completeness"] = round(((total_cells - context["missing"]) / total_cells) * 100, 2)

            missing_by_col = dataset.isnull().sum().sort_values(ascending=False)
            if len(missing_by_col) and int(missing_by_col.iloc[0]) > 0:
                context["top_missing_column"] = str(missing_by_col.index[0])

            if numeric_cols:
                col = numeric_cols[0]
                context["numeric_col"] = col

                context["stats"] = [
                    {
                        "column": c,
                        "mean": round(dataset[c].mean(), 2),
                        "std": round(dataset[c].std(), 2),
                    }
                    for c in numeric_cols
                ]

                data = dataset[col].dropna().tolist()
                context["line_data"] = data[:100]
                hist_counts, bin_edges = np.histogram(dataset[col].dropna(), bins=min(10, max(5, len(set(data[:50])) if data else 5)))
                context["hist_data"] = hist_counts.tolist()
                context["hist_labels"] = [
                    f"{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}"
                    for i in range(len(bin_edges) - 1)
                ]

                mean = dataset[col].mean()
                std = dataset[col].std()

                context["spc_mean"] = round(mean, 2)
                context["spc_std"] = round(std, 2)
                context["spc_ucl"] = round(mean + 3 * std, 2)
                context["spc_lcl"] = round(mean - 3 * std, 2)
                context["spc_data"] = data[:100]

                try:
                    corr = dataset[numeric_cols].corr().round(2)
                    context["corr_matrix"] = corr.values.tolist()
                    context["corr_headers"] = corr.columns.tolist()
                except Exception:
                    context["corr_matrix"] = []
                    context["corr_headers"] = []

                try:
                    target_col = dataset.columns[-1]

                    if target_col not in numeric_cols:
                        X = dataset[numeric_cols].fillna(0)
                        y = dataset[target_col]

                        if y.dtype == "object":
                            y = pd.factorize(y)[0]

                        if len(np.unique(y)) >= 2:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42
                            )

                            model = LogisticRegression(max_iter=1000)
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)

                            context["ml_accuracy"] = round(
                                accuracy_score(y_test, pred) * 100, 2
                            )
                            context["ml_message"] = "Logistic Regression completed"
                        else:
                            context["ml_message"] = "Need at least 2 classes"
                    else:
                        context["ml_message"] = "Target column should not be numeric"

                except Exception as e:
                    context["ml_message"] = f"ML Error: {str(e)[:50]}"

                if len(numeric_cols) >= 2:
                    try:
                        X = dataset[[numeric_cols[0]]].fillna(0)
                        y = dataset[numeric_cols[1]].fillna(0)

                        model = LinearRegression()
                        model.fit(X, y)
                        pred = model.predict(X)

                        context["linreg_score"] = round(r2_score(y, pred), 2)
                        context["linreg_message"] = "Linear Regression done"
                        context["linreg_columns"] = [numeric_cols[0], numeric_cols[1]]

                    except Exception as e:
                        context["linreg_message"] = f"Regression Error: {str(e)[:50]}"

                if len(numeric_cols) >= 2:
                    try:
                        X = dataset[numeric_cols].fillna(0)

                        pca = PCA(n_components=2)
                        comp = pca.fit_transform(X)

                        context["pca_data"] = comp.tolist()[:100]
                        context["pca_variance"] = [
                            round(value * 100, 2)
                            for value in pca.explained_variance_ratio_.tolist()
                        ]
                        context["pca_message"] = "PCA applied"

                    except Exception as e:
                        context["pca_message"] = f"PCA Error: {str(e)[:50]}"

                if len(numeric_cols) >= 2:
                    try:
                        X = dataset[numeric_cols[:2]].fillna(0)

                        kmeans = KMeans(n_clusters=3, random_state=42)
                        labels = kmeans.fit_predict(X)
                        counts = np.bincount(labels, minlength=3)

                        context["cluster_labels"] = labels.tolist()[:20]
                        context["cluster_points"] = X.iloc[:100].values.tolist()
                        context["cluster_centers"] = kmeans.cluster_centers_.tolist()
                        context["cluster_columns"] = numeric_cols[:2]
                        context["cluster_counts"] = counts.tolist()
                        context["cluster_message"] = "Clustering done"

                    except Exception as e:
                        context["cluster_message"] = f"Clustering Error: {str(e)[:50]}"

            else:
                context["message"] = "No numeric columns found"

        except Exception as e:
            context["message"] = f"Error processing dataset: {str(e)[:50]}"

    return templates.TemplateResponse("index.html", context)


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    global dataset

    try:
        content = await file.read()
        dataset = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception:
        dataset = None

    return await home(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
