"""
Evaluation, Stability, and Profiling Module
Analyzes cluster profiles, cross-tabs with held-out labels,
and runs multi-seed and bootstrap ARI stability tests.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from itertools import combinations

RANDOM_STATE = 42

def generate_cluster_profiles(df_original: pd.DataFrame, cluster_labels: np.ndarray):
    """
    Computes mean and median of original-unit features per cluster,
    plus cluster volume and percentage of dataset.
    """
    df = df_original.copy()
    df["cluster"] = cluster_labels
    
    analysis_cols = [
        "CarCount", "BikeCount", "BusCount", "TruckCount",
        "total_volume", "car_ratio", "bike_ratio", "bus_ratio", "truck_ratio",
        "hour_float", "is_weekend"
    ]
    
    # Filter only available columns
    available_cols = [c for c in analysis_cols if c in df.columns]
    
    means = df.groupby("cluster")[available_cols].mean().round(2)
    medians = df.groupby("cluster")[available_cols].median().round(2)
    counts = df["cluster"].value_counts().sort_index()
    percentages = (counts / len(df) * 100.0).round(2)
    
    summary = pd.DataFrame({
        "cluster": counts.index,
        "sample_count": counts.values,
        "percentage": percentages.values
    }).set_index("cluster")
    
    profile_mean = summary.join(means)
    profile_median = summary.join(medians)
    
    return profile_mean, profile_median

def cross_tabulate_traffic_situation(held_out_labels: pd.Series, cluster_labels: np.ndarray):
    """
    Cross-tabulates unsupervised clusters against held-out ground truth Traffic Situation.
    Computes both count and row-wise percentages.
    """
    df_ct = pd.DataFrame({
        "Cluster": [f"Cluster {c}" for c in cluster_labels],
        "Traffic Situation": held_out_labels.values
    })
    
    counts_table = pd.crosstab(df_ct["Cluster"], df_ct["Traffic Situation"])
    norm_table = (pd.crosstab(df_ct["Cluster"], df_ct["Traffic Situation"], normalize="index") * 100.0).round(2)
    
    return counts_table, norm_table

def test_random_seed_stability(X: np.ndarray, k: int, num_seeds: int = 15):
    """
    Runs K-Means with different random seeds and computes pairwise Adjusted Rand Index (ARI).
    Returns list of ARI values and summary statistics.
    """
    seeds = [42 + i * 17 for i in range(num_seeds)]
    clusterings = []
    
    for seed in seeds:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        clusterings.append(km.fit_predict(X))
        
    ari_scores = []
    for c1, c2 in combinations(clusterings, 2):
        ari_scores.append(adjusted_rand_score(c1, c2))
        
    stats = {
        "k": k,
        "num_seeds": num_seeds,
        "mean_ari": round(np.mean(ari_scores), 4),
        "std_ari": round(np.std(ari_scores), 4),
        "min_ari": round(np.min(ari_scores), 4),
        "max_ari": round(np.max(ari_scores), 4)
    }
    return stats, ari_scores

def test_subsampling_stability(X: np.ndarray, k: int, sample_ratio: float = 0.8, n_iterations: int = 20):
    """
    Subsamples data (e.g. 80%) repeatedly, fits K-Means, and compares predictions on overlapping samples.
    """
    np.random.seed(RANDOM_STATE)
    n_samples = len(X)
    subsample_size = int(n_samples * sample_ratio)
    
    reference_km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    reference_labels = reference_km.fit_predict(X)
    
    ari_scores = []
    for i in range(n_iterations):
        idx = np.random.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X[idx]
        
        km_sub = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE + i)
        sub_labels = km_sub.fit_predict(X_sub)
        
        # Compare reference labels on the subset with subsample labels
        ari = adjusted_rand_score(reference_labels[idx], sub_labels)
        ari_scores.append(ari)
        
    stats = {
        "k": k,
        "sample_ratio": sample_ratio,
        "n_iterations": n_iterations,
        "mean_subsample_ari": round(np.mean(ari_scores), 4),
        "std_subsample_ari": round(np.std(ari_scores), 4),
        "min_subsample_ari": round(np.min(ari_scores), 4)
    }
    return stats, ari_scores

