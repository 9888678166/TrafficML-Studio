"""
K-Means Clustering and Dimensionality Reduction Module
Evaluates k in [2..10] under Euclidean and Manhattan evaluation metrics.
Computes PCA projections and feature loadings.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import joblib

RANDOM_STATE = 42

def run_kmeans_experiments(X: np.ndarray, k_range=range(2, 11), n_init: int = 25, random_state: int = RANDOM_STATE):
    """
    Executes K-Means across k_range.
    Computes WCSS, Euclidean Silhouette, Manhattan Silhouette,
    Davies-Bouldin index, and cluster size distributions.
    """
    results = []
    models = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X)
        
        wcss = kmeans.inertia_
        sil_euclidean = silhouette_score(X, cluster_labels, metric="euclidean")
        sil_manhattan = silhouette_score(X, cluster_labels, metric="manhattan")
        db_score = davies_bouldin_score(X, cluster_labels)
        ch_score = calinski_harabasz_score(X, cluster_labels)
        
        counts = pd.Series(cluster_labels).value_counts()
        min_cluster_size = counts.min()
        max_cluster_size = counts.max()
        min_cluster_pct = (min_cluster_size / len(X)) * 100.0
        
        results.append({
            "k": k,
            "wcss": round(wcss, 2),
            "silhouette_euclidean": round(sil_euclidean, 4),
            "silhouette_manhattan": round(sil_manhattan, 4),
            "davies_bouldin": round(db_score, 4),
            "calinski_harabasz": round(ch_score, 2),
            "min_cluster_size": int(min_cluster_size),
            "min_cluster_pct": round(min_cluster_pct, 2),
            "max_cluster_size": int(max_cluster_size)
        })
        models[k] = kmeans
        
    results_df = pd.DataFrame(results)
    return results_df, models

def compute_pca_projection(X: np.ndarray, feature_names: list, n_components: int = 2, random_state: int = RANDOM_STATE):
    """
    Computes PCA 2D/3D projection, explained variance, and component loadings.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    return X_pca, explained_var, loadings, pca

def compute_silhouette_profiles(X: np.ndarray, labels: np.ndarray):
    """
    Computes per-sample silhouette scores for detailed cluster diagnostic plots.
    """
    sample_sil_values = silhouette_samples(X, labels, metric="euclidean")
    return sample_sil_values

if __name__ == "__main__":
    from preprocessing import load_and_inspect_raw_data, engineer_features, prepare_clustering_matrix
    df, _ = load_and_inspect_raw_data()
    df_eng = engineer_features(df)
    X_scaled_df, feats, _, _ = prepare_clustering_matrix(df_eng)
    
    results_df, models = run_kmeans_experiments(X_scaled_df.values)
    print("K-Means Grid Results:")
    print(results_df.to_string(index=False))

