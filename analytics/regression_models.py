import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict_defects(df, target_col=None):
    """
    Perform logistic regression for defect prediction.
    If target_col is None, try to find a suitable target.
    Returns accuracy, confusion matrix fig, and model info.
    """
    if df.empty:
        return None, None, "No data available"

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None, None, "Need at least 2 numeric columns"

    # If no target specified, assume last column or look for 'defect' or 'target'
    if target_col is None:
        possible_targets = [col for col in df.columns if 'defect' in col.lower() or 'target' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            target_col = df.columns[-1]

    if target_col not in df.columns:
        return None, None, f"Target column '{target_col}' not found"

    # Prepare data
    X = numeric.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    # Ensure y is binary
    unique_y = y.unique()
    if len(unique_y) != 2:
        return None, None, "Target must be binary for logistic regression"

    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mode()[0] if not y.mode().empty else 0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Target values:", y.unique())
    print("Target distribution:\n", y.value_counts())

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    return accuracy, fig, f"Accuracy: {accuracy:.2%}"
