"""
Preprocessing and Feature Engineering Module
For Traffic Congestion Pattern Discovery
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

RANDOM_STATE = 42

def load_and_inspect_raw_data(csv_path: str = "data/raw/Traffic.csv"):
    """Loads raw traffic dataset and returns initial inspection metadata."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw data file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    inspection = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }
    return df, inspection

def engineer_features(df_raw: pd.DataFrame):
    """
    Cleans, parses time, and engineers features per time-slot.
    Avoids multi-collinearity pitfalls (e.g. Total = Car + Bike + Bus + Truck).
    """
    df = df_raw.copy()
    
    # 1. Parse time
    # Convert 'Time' column to 24-hour decimal float (0.0 to 23.75)
    parsed_time = pd.to_datetime(df["Time"], format="%I:%M:%S %p", errors="coerce").dt.time
    # Fallback to general parsing if format differs
    if parsed_time.isnull().all():
        parsed_time = pd.to_datetime(df["Time"], errors="coerce").dt.time
        
    df["hour_float"] = parsed_time.apply(lambda t: t.hour + t.minute / 60.0 if pd.notnull(t) else np.nan)
    
    # Missing value handling (if any)
    if df["hour_float"].isnull().any():
        df["hour_float"] = df["hour_float"].fillna(df["hour_float"].median())
        
    for col in ["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # 2. Cyclical time encoding (preserves 23:45 -> 00:00 continuity)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_float"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_float"] / 24.0)
    
    # 3. Calendar features
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    df["day_num"] = df["Day of the week"].map(day_map).fillna(0).astype(int)
    df["is_weekend"] = (df["day_num"] >= 5).astype(int)
    
    # 4. Total and Vehicle Mix Ratios
    # Recalculate true total to verify consistency
    raw_total = df["CarCount"] + df["BikeCount"] + df["BusCount"] + df["TruckCount"]
    safe_total = np.maximum(raw_total, 1) # Avoid division by zero
    
    df["car_ratio"] = df["CarCount"] / safe_total
    df["bike_ratio"] = df["BikeCount"] / safe_total
    df["bus_ratio"] = df["BusCount"] / safe_total
    df["truck_ratio"] = df["TruckCount"] / safe_total
    df["total_volume"] = raw_total
    
    # 5. Outlier capping (IQR winsorization on total_volume to safeguard K-Means)
    q1 = df["total_volume"].quantile(0.01)
    q3 = df["total_volume"].quantile(0.99)
    # Traffic spikes can be valid patterns, but extreme sensor glitches > 3*IQR can be capped
    iqr = df["total_volume"].quantile(0.75) - df["total_volume"].quantile(0.25)
    upper_bound = df["total_volume"].quantile(0.75) + 3.0 * iqr
    df["total_volume_capped"] = np.clip(df["total_volume"], 0, upper_bound)
    
    return df

def prepare_clustering_matrix(df_engineered: pd.DataFrame, scaler_path: str = "data/processed/scaler.joblib"):
    """
    Selects non-redundant clustering features, separates held-out labels,
    applies StandardScaler, and persists transformed matrices.
    """
    # Held-out ground-truth label for post-hoc validation (NEVER clustered on)
    held_out_label = df_engineered["Traffic Situation"].copy() if "Traffic Situation" in df_engineered.columns else None
    
    # Selected feature subset:
    # Captures:
    # - Temporal cyclicality: sin_hour, cos_hour
    # - Day profile: is_weekend
    # - Traffic magnitude: total_volume_capped
    # - Traffic composition: car_ratio, bike_ratio, bus_ratio, truck_ratio
    feature_cols = [
        "sin_hour",
        "cos_hour",
        "is_weekend",
        "total_volume_capped",
        "car_ratio",
        "bike_ratio",
        "bus_ratio",
        "truck_ratio"
    ]
    
    X = df_engineered[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_engineered.index)
    
    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    return X_scaled_df, feature_cols, held_out_label, scaler

if __name__ == "__main__":
    df, inspect_dict = load_and_inspect_raw_data()
    print("Raw Data Shape:", inspect_dict["shape"])
    df_eng = engineer_features(df)
    X_scaled_df, feats, labels, _ = prepare_clustering_matrix(df_eng)
    print("Final Scaled Matrix Shape:", X_scaled_df.shape)
    print("Features selected:", feats)

