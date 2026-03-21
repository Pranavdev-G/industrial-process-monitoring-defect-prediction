import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def perform_clustering(df, n_clusters=3):
    """
    Perform K-Means clustering on numeric data.
    Returns cluster labels, centroids, and plot figure.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None, None, None, "Need at least 2 numeric columns for clustering"

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # Plot (using first 2 components if more than 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    if numeric.shape[1] >= 2:
        ax.scatter(numeric.iloc[:, 0], numeric.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
        ax.set_xlabel(numeric.columns[0])
        ax.set_ylabel(numeric.columns[1])
    else:
        ax.scatter(range(len(clusters)), numeric.iloc[:, 0], c=clusters, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Index')
        ax.set_ylabel(numeric.columns[0])

    ax.set_title(f'K-Means Clustering (k={n_clusters})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return clusters, centroids, fig, f"Clustered into {n_clusters} groups"