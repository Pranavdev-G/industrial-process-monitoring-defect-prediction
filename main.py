"""
Industrial Process Monitoring Web App (Upgraded)
"""

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA
import io

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
        "preview": [],
        "headers": [],
        "stats": [],
        "numeric_col": None,
        "line_data": [],
        "hist_data": [],
        "spc_mean": 0,
        "spc_std": 0,
        "spc_ucl": 0,
        "spc_lcl": 0,
        "spc_data": [],
        "ml_accuracy": None,
        "ml_message": "",
        "cluster_labels": [],
        "cluster_message": "",
        "corr_matrix": [],
        "linreg_score": None,
        "linreg_message": "",
        "pca_data": [],
        "pca_message": ""
    }

    if dataset is not None:
        context["has_data"] = True
        context["rows"] = len(dataset)
        context["columns"] = len(dataset.columns)
        context["missing"] = int(dataset.isnull().sum().sum())

        context["headers"] = list(dataset.columns)
        context["preview"] = dataset.head().to_dict('records')

        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            col = numeric_cols[0]
            context["numeric_col"] = col

            # Stats
            context["stats"] = [
                {"column": c, "mean": round(dataset[c].mean(), 2), "std": round(dataset[c].std(), 2)}
                for c in numeric_cols
            ]

            data = dataset[col].dropna().tolist()
            context["line_data"] = data[:100]
            context["hist_data"] = data

            # SPC
            mean = dataset[col].mean()
            std = dataset[col].std()
            context["spc_mean"] = round(mean, 2)
            context["spc_std"] = round(std, 2)
            context["spc_ucl"] = round(mean + 3 * std, 2)
            context["spc_lcl"] = round(mean - 3 * std, 2)
            context["spc_data"] = data[:100]

            # Correlation
            try:
                corr = dataset[numeric_cols].corr().round(2)
                context["corr_matrix"] = corr.values.tolist()
            except:
                pass

            # Logistic Regression
            try:
                target_col = dataset.columns[-1]
                X = dataset[numeric_cols].fillna(0)
                y = dataset[target_col]

                if y.dtype == 'object':
                    y = pd.factorize(y)[0]

                if len(np.unique(y)) >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    context["ml_accuracy"] = round(accuracy_score(y_test, pred) * 100, 2)
                    context["ml_message"] = "Logistic Regression completed"
            except:
                context["ml_message"] = "ML failed"

            # Linear Regression
            if len(numeric_cols) >= 2:
                try:
                    X = dataset[[numeric_cols[0]]].fillna(0)
                    y = dataset[numeric_cols[1]].fillna(0)

                    model = LinearRegression()
                    model.fit(X, y)
                    pred = model.predict(X)

                    context["linreg_score"] = round(r2_score(y, pred), 2)
                    context["linreg_message"] = "Linear Regression done"
                except:
                    context["linreg_message"] = "Regression failed"

            # PCA
            if len(numeric_cols) >= 2:
                try:
                    X = dataset[numeric_cols].fillna(0)
                    pca = PCA(n_components=2)
                    comp = pca.fit_transform(X)
                    context["pca_data"] = comp.tolist()[:100]
                    context["pca_message"] = "PCA applied"
                except:
                    context["pca_message"] = "PCA failed"

            # Clustering
            if len(numeric_cols) >= 2:
                try:
                    X = dataset[numeric_cols[:2]].fillna(0)
                    kmeans = KMeans(n_clusters=3)
                    labels = kmeans.fit_predict(X)
                    context["cluster_labels"] = labels.tolist()[:20]
                    context["cluster_message"] = "Clustering done"
                except:
                    context["cluster_message"] = "Clustering failed"

    return templates.TemplateResponse("index.html", context)


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    global dataset
    try:
        content = await file.read()
        dataset = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except:
        dataset = None

    return await home(request)