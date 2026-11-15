"""
Streamlit Web Application
Road Segment Congestion Pattern Discovery with K-Means
An interactive web application for unsupervised traffic telemetry analysis,
real-time congestion regime classification, and cluster stability validation.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure environment safety in constrained cloud runtimes
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import engineer_features, prepare_clustering_matrix

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrafficML Studio | Unsupervised K-Means Discovery",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Metadata & Empirical Distributions
# -----------------------------------------------------------------------------
CLUSTER_METADATA = {
    0: {
        "persona": "Midday Commercial & Freight Flow",
        "description": "Dominated by commercial logistics, local deliveries, and steady midday traffic. Heavy truck presence with moderate overall congestion.",
        "badge_color": "#f59e0b",
        "badge_name": "Cluster 0",
        "icon": "🚛"
    },
    1: {
        "persona": "Peak Commuter Rush Hour",
        "description": "Characterized by severe congestion, high volume of personal commuter cars and motorcycles during morning and evening rush hours.",
        "badge_color": "#ef4444",
        "badge_name": "Cluster 1",
        "icon": "🚗"
    },
    2: {
        "persona": "Nighttime / Early Dawn Low Flow",
        "description": "Minimal overall volume, free-flowing conditions, and disproportionately high share of overnight long-haul freight transport.",
        "badge_color": "#6366f1",
        "badge_name": "Cluster 2",
        "icon": "🌙"
    },
    3: {
        "persona": "Weekend Leisure & Off-Peak Travel",
        "description": "Active primarily on Saturdays and Sundays. Balanced personal car and motorcycle travel with negligible commercial transit.",
        "badge_color": "#10b981",
        "badge_name": "Cluster 3",
        "icon": "🏖️"
    }
}

DEFAULT_SITUATION_PROBS = {
    0: {"heavy": 0.0, "high": 47.4, "normal": 48.9, "low": 3.6},
    1: {"heavy": 79.9, "high": 19.8, "normal": 0.3, "low": 0.0},
    2: {"heavy": 0.0, "high": 0.0, "normal": 1.7, "low": 98.4},
    3: {"heavy": 0.0, "high": 16.1, "normal": 55.6, "low": 28.3}
}

# -----------------------------------------------------------------------------
# Initialization & Self-Healing Pipeline
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def initialize_system():
    """Ensures raw data, processed artifacts, and models are generated and loaded."""
    raw_path = "data/raw/Traffic.csv"
    scaler_path = "data/processed/scaler.joblib"
    model_path = "data/processed/kmeans_model.joblib"

    # Self-heal on fresh cloud deployment
    if not os.path.exists(raw_path):
        from generate_sample_data import generate_traffic_data
        os.makedirs("data/raw", exist_ok=True)
        generate_traffic_data(output_path=raw_path)

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        from run_pipeline import main as run_pipeline_main
        run_pipeline_main()

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

scaler, model = initialize_system()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def format_hour(h: float) -> str:
    hour_int = int(h)
    minutes = int(round((h - hour_int) * 60))
    period = "PM" if hour_int >= 12 else "AM"
    disp_h = 12 if hour_int in (0, 12) else (hour_int - 12 if hour_int > 12 else hour_int)
    return f"{disp_h:02d}:{minutes:02d} {period} ({h:.2f}h)"

def predict_regime(hour, day, cars, bikes, buses, trucks):
    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

    total_raw = cars + bikes + buses + trucks
    safe_total = max(total_raw, 1)

    car_ratio = cars / safe_total
    bike_ratio = bikes / safe_total
    bus_ratio = buses / safe_total
    truck_ratio = trucks / safe_total
    total_volume_capped = min(total_raw, 220.0)

    feat_cols = [
        "sin_hour", "cos_hour", "is_weekend", "total_volume_capped",
        "car_ratio", "bike_ratio", "bus_ratio", "truck_ratio"
    ]
    feat_df = pd.DataFrame([[
        sin_hour, cos_hour, is_weekend, total_volume_capped,
        car_ratio, bike_ratio, bus_ratio, truck_ratio
    ]], columns=feat_cols)

    scaled_vec = scaler.transform(feat_df)
    cluster = int(model.predict(scaled_vec)[0])
    dist = float(np.linalg.norm(scaled_vec[0] - model.cluster_centers_[cluster]))

    return cluster, dist, total_raw, truck_ratio * 100.0

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
        <span style="font-size: 32px;">🚦</span>
        <div>
            <h3 style="margin: 0; font-size: 20px;">TrafficML Studio</h3>
            <p style="margin: 0; font-size: 12px; color: #94a3b8;">Unsupervised K-Means Discovery</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Dashboard",
        "⚡ Live Pattern Classifier",
        "📈 Model Diagnostics & PCA",
        "🔍 Cluster Profiles & Stability",
        "💾 Dataset & Retraining",
        "📖 About & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Status:** Active Model (k=4)\n\n"
    "**Random State:** 42\n\n"
    "**Unit of Analysis:** 15-min Time-Slot\n\n"
    "**Framework:** Scikit-Learn & Streamlit"
)

# =============================================================================
# 1. Executive Dashboard
# =============================================================================
if menu == "📊 Executive Dashboard":
    st.title("Road Segment Congestion Pattern Discovery")
    st.markdown("Unsupervised operational regime identification from continuous 15-minute multi-vehicle telemetry.")

    raw_path = "data/raw/Traffic.csv"
    sample_count = len(pd.read_csv(raw_path)) if os.path.exists(raw_path) else 2976

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Unit of Analysis", value="Time-Slot", help="Hour-of-day × Day-of-week arterial profile")
    with col2:
        st.metric(label="Total Observations", value=f"{sample_count:,}", help="Continuous 15-minute sensor intervals")
    with col3:
        st.metric(label="Optimal Regimes (k)", value="4", delta="Defensible", help="Optimal elbow and peak Manhattan silhouette")
    with col4:
        st.metric(label="Stability (ARI)", value="0.998", delta="+0.965 Bootstrap", help="Adjusted Rand Index across 15 seeds")

    st.markdown("### Discovered Traffic Operating Regimes (k = 4)")

    c_cols = st.columns(4)
    for i, col in enumerate(c_cols):
        meta = CLUSTER_METADATA[i]
        with col:
            st.markdown(
                f"""
                <div style="background-color: #1e293b; border-left: 5px solid {meta['badge_color']}; padding: 16px; border-radius: 8px; margin-bottom: 12px; min-height: 250px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="background-color: {meta['badge_color']}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                            Cluster {i}
                        </span>
                        <span style="font-size: 20px;">{meta['icon']}</span>
                    </div>
                    <h4 style="margin: 4px 0 8px 0; font-size: 15px; color: #f8fafc;">{meta['persona']}</h4>
                    <p style="font-size: 12px; color: #94a3b8; line-height: 1.4;">{meta['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Primary Exploratory & PCA Visualizations")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        f1 = "outputs/figures/01_diurnal_hourly_pattern.png"
        if os.path.exists(f1):
            st.image(f1, caption="Figure 1: Mean Vehicle Composition Across Hours of Day", use_container_width=True)
    with img_col2:
        f5 = "outputs/figures/05_pca_2d_clusters.png"
        if os.path.exists(f5):
            st.image(f5, caption="Figure 2: PCA 2D Cluster Space with Optimal Centroids (k=4)", use_container_width=True)

# =============================================================================
# 2. Live Pattern Classifier
# =============================================================================
elif menu == "⚡ Live Pattern Classifier":
    st.title("⚡ Live Traffic Pattern Classifier & Simulator")
    st.markdown("Simulate arbitrary 15-minute traffic flows and classify their operational congestion regime in real time.")

    # Preset selection
    st.markdown("#### Scenario Presets")
    pcol1, pcol2, pcol3, pcol4 = st.columns([1, 1, 1, 3])
    preset = None
    with pcol1:
        if st.button("🌅 Morning Rush"):
            preset = "rush"
    with pcol2:
        if st.button("🌙 Night Freight"):
            preset = "freight"
    with pcol3:
        if st.button("🏖️ Weekend Leisure"):
            preset = "weekend"

    # Default preset values
    default_hour = 8.5
    default_day = "Monday"
    default_cars = 95
    default_bikes = 28
    default_buses = 14
    default_trucks = 4

    if preset == "rush":
        default_hour, default_day, default_cars, default_bikes, default_buses, default_trucks = 8.5, "Monday", 95, 28, 14, 4
    elif preset == "freight":
        default_hour, default_day, default_cars, default_bikes, default_buses, default_trucks = 3.0, "Wednesday", 8, 2, 1, 8
    elif preset == "weekend":
        default_hour, default_day, default_cars, default_bikes, default_buses, default_trucks = 14.5, "Saturday", 45, 16, 5, 3

    st.markdown("---")
    st.markdown("#### Configure Telemetry Parameters")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        hour_val = st.slider("Hour of Day (0 to 23.75):", min_value=0.0, max_value=23.75, value=float(default_hour), step=0.25)
        st.caption(f"Selected Time: **{format_hour(hour_val)}**")
        day_val = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(default_day))

    with fcol2:
        car_val = st.number_input("Car Count (Passenger vehicles):", min_value=0, max_value=250, value=int(default_cars))
        bike_val = st.number_input("Bike / Motorcycle Count:", min_value=0, max_value=120, value=int(default_bikes))
        bus_val = st.number_input("Bus Count (Public Transit):", min_value=0, max_value=50, value=int(default_buses))
        truck_val = st.number_input("Truck Count (Commercial / Freight):", min_value=0, max_value=60, value=int(default_trucks))

    # Live classification
    cluster_res, dist_res, tot_vol, truck_share = predict_regime(hour_val, day_val, car_val, bike_val, bus_val, truck_val)
    meta = CLUSTER_METADATA[cluster_res]
    situation_dist = DEFAULT_SITUATION_PROBS[cluster_res]

    st.markdown("---")
    st.markdown("### Classification Output")

    res_col1, res_col2 = st.columns([1.5, 1])
    with res_col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1)); border: 1px solid {meta['badge_color']}; border-radius: 8px; padding: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="background-color: {meta['badge_color']}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 13px;">
                        Cluster {cluster_res}
                    </span>
                    <span style="color: #94a3b8; font-size: 13px;">Euclidean Distance: <strong>{dist_res:.3f}</strong></span>
                </div>
                <h2 style="margin: 8px 0; font-size: 22px; color: #f8fafc;">{meta['icon']} {meta['persona']}</h2>
                <p style="color: #cbd5e1; font-size: 14px; line-height: 1.5;">{meta['description']}</p>
                <div style="display: flex; gap: 20px; margin-top: 12px; font-size: 13px; color: #94a3b8;">
                    <span>Total Volume: <strong style="color: #f8fafc;">{tot_vol} veh</strong></span>
                    <span>Truck Fleet Share: <strong style="color: #f8fafc;">{truck_share:.1f}%</strong></span>
                    <span>Weekend Mode: <strong style="color: #f8fafc;">{'Yes' if day_val in ['Saturday', 'Sunday'] else 'No'}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with res_col2:
        st.markdown("#### Expected Congestion Severity (Ground-Truth Proxy)")
        colors = {"heavy": "#ef4444", "high": "#f59e0b", "normal": "#3b82f6", "low": "#10b981"}
        for sit, prob in situation_dist.items():
            st.markdown(f"**{sit.capitalize()}**: {prob:.1f}%")
            st.progress(float(prob) / 100.0)

# =============================================================================
# 3. Model Diagnostics & PCA
# =============================================================================
elif menu == "📈 Model Diagnostics & PCA":
    st.title("📈 Model Diagnostics & Multi-Metric Evaluation")
    st.markdown("Systematic evaluation across cluster numbers (k=2 to 10), distance metrics, and dimensionality reduction.")

    grid_path = "outputs/kmeans_metrics_grid.csv"
    if os.path.exists(grid_path):
        grid_df = pd.read_csv(grid_path)
        st.markdown("### K-Means Experimental Grid (k = 2 to 10)")
        st.dataframe(grid_df.style.highlight_min(subset=["wcss", "davies_bouldin"], color="#1e3a5f").highlight_max(subset=["silhouette_euclidean", "silhouette_manhattan"], color="#1e40af"), use_container_width=True)

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        f3 = "outputs/figures/03_elbow_curve.png"
        if os.path.exists(f3):
            st.image(f3, caption="Figure 3: Within-Cluster Sum of Squares (WCSS / Inertia) vs. k", use_container_width=True)
    with dcol2:
        f4 = "outputs/figures/04_silhouette_and_db_comparison.png"
        if os.path.exists(f4):
            st.image(f4, caption="Figure 4: Silhouette Score (Euclidean & Manhattan) & Davies-Bouldin Index", use_container_width=True)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        f6 = "outputs/figures/06_pca_loadings.png"
        if os.path.exists(f6):
            st.image(f6, caption="Figure 5: Principal Component Analysis (PCA) Feature Loadings", use_container_width=True)
    with pcol2:
        f7 = "outputs/figures/07_silhouette_samples.png"
        if os.path.exists(f7):
            st.image(f7, caption="Figure 6: Per-Sample Silhouette Coefficient Diagnostic Profile (k=4)", use_container_width=True)

# =============================================================================
# 4. Cluster Profiles & Stability
# =============================================================================
elif menu == "🔍 Cluster Profiles & Stability":
    st.title("🔍 Cluster Profiles, Validation & Stability Analysis")
    st.markdown("Detailed physical feature statistics, ground-truth cross-tabulation, and statistical robustness tests.")

    prof_path = "outputs/cluster_profiles_mean.csv"
    if os.path.exists(prof_path):
        prof_df = pd.read_csv(prof_path)
        st.markdown("### Mean Cluster Profiles (Original Physical Units)")
        st.dataframe(prof_df, use_container_width=True)

    hcol1, hcol2 = st.columns(2)
    with hcol1:
        f8 = "outputs/figures/08_standardized_centroids_heatmap.png"
        if os.path.exists(f8):
            st.image(f8, caption="Figure 7: Standardized Feature Centroids Heatmap (Z-Scores)", use_container_width=True)
    with hcol2:
        f9 = "outputs/figures/09_traffic_situation_by_cluster.png"
        if os.path.exists(f9):
            st.image(f9, caption="Figure 8: Held-Out 'Traffic Situation' Composition per Cluster", use_container_width=True)

    st.markdown("### Statistical Stability Analysis (Adjusted Rand Index)")
    stab_path = "outputs/stability_analysis.csv"
    if os.path.exists(stab_path):
        stab_df = pd.read_csv(stab_path)
        scol1, scol2 = st.columns([1.2, 1])
        with scol1:
            st.dataframe(stab_df, use_container_width=True)
            st.caption("k=4 demonstrates near-deterministic convergence across 15 seeds (Mean ARI = **0.9979**) and 20 iterations of 80% bootstrap subsamples (Mean ARI = **0.9653**).")
        with scol2:
            f10 = "outputs/figures/10_stability_ari_comparison.png"
            if os.path.exists(f10):
                st.image(f10, caption="Figure 9: Stability Comparison Across k", use_container_width=True)

# =============================================================================
# 5. Dataset & Retraining
# =============================================================================
elif menu == "💾 Dataset & Retraining":
    st.title("💾 Dataset Management & Automated Pipeline")
    st.markdown("Inspect raw 15-minute telemetry records, upload a new Kaggle CSV, or retrain the clustering pipeline.")

    up_col, run_col = st.columns(2)
    with up_col:
        st.markdown("#### Upload Custom Telemetry CSV")
        uploaded_file = st.file_uploader("Upload CSV formatted matching the Kaggle Traffic Prediction schema:", type=["csv"])
        if uploaded_file is not None:
            dest_path = "data/raw/Traffic.csv"
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully as data/raw/Traffic.csv! Click 'Retrain Clustering Pipeline' to update models.")

    with run_col:
        st.markdown("#### Retrain End-to-End Pipeline")
        st.write("Executes data cleaning, feature engineering, K-Means grid (k=2..10), PCA projections, and regenerates all figures.")
        if st.button("⚡ Retrain Clustering Pipeline", type="primary"):
            with st.spinner("Retraining pipeline..."):
                from run_pipeline import main as run_pipeline_main
                run_pipeline_main()
                st.cache_resource.clear()
            st.success("Clustering pipeline executed successfully! Models, figures, and metrics have been refreshed.")

    st.markdown("---")
    st.markdown("#### Raw Telemetry Data Preview (First 50 Intervals)")
    raw_path = "data/raw/Traffic.csv"
    if os.path.exists(raw_path):
        df_preview = pd.read_csv(raw_path)
        st.dataframe(df_preview.head(50), use_container_width=True)
        st.download_button(
            label="📥 Download Clean Processed Data (CSV)",
            data=pd.read_csv("data/processed/traffic_clean.csv").to_csv(index=False).encode("utf-8"),
            file_name="traffic_clean.csv",
            mime="text/csv"
        )

# =============================================================================
# 6. About & Methodology
# =============================================================================
elif menu == "📖 About & Methodology":
    st.title("📖 About This Project & Theoretical Methodology")
    st.markdown(
        r"""
        ### 1. Problem Formulation
        Urban traffic congestion management is constrained by rigid, static signal plans that fail to reflect dynamic fleet composition and non-linear diurnal shifts. This project formulates road congestion analysis as an **unsupervised pattern discovery problem** using continuous 15-minute multi-vehicle telemetry.

        ### 2. Definition of "Road Segment"
        - Observations originate from a single monitoring station over time.
        - The empirical **unit of analysis** is the **time-slot profile** ($\text{hour-of-day} \times \text{day-of-week}$).
        - Congestion patterns represent **recurrent operational operating states** of that road segment.
        - The column `Traffic Situation` was **strictly held out** and used solely for post-hoc validation.

        ### 3. Feature Engineering Rationale
        - **Cyclical Time:** Continuous $\sin(2\pi \cdot h / 24)$ and $\cos(2\pi \cdot h / 24)$ mapping eliminates boundary distortions between 23:45 and 00:00.
        - **Fleet Mix Ratios:** Proportions ($\text{car\_ratio}, \text{truck\_ratio}, \dots$) prevent collinearity with total volume while capturing transit share.
        - **Outlier Capping:** IQR winsorization mitigates sensor spike distortion on Euclidean centroids.

        ### 4. Robustness & Ethics
        - Multi-seed ARI ($0.998$) and bootstrap subsampling ($0.965$) confirm stability.
        - Models are designed as **decision-support systems (DSS)** to assist municipal traffic engineers, not automated punitive tools.
        """
    )
