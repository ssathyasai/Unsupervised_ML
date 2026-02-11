# ==========================================================
# 🚕 ClusterCab Pro - Multi-Experiment DBSCAN Analysis
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="ClusterCab Pro",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 ClusterCab Pro - Taxi Pickup Hotspot Detection")
st.markdown("Density-Based Spatial Clustering using **DBSCAN**")
st.markdown("---")

# ==========================================================
# 1️⃣ Load Dataset
# ==========================================================

uploaded_file = st.file_uploader("📂 Upload NYC Taxi Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📊 Dataset Preview (First 5 Rows)")
st.dataframe(df.head())

# ==========================================================
# 2️⃣ Feature Selection + Data Cleaning
# ==========================================================

required_cols = ['pickup_latitude', 'pickup_longitude']

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain 'pickup_latitude' and 'pickup_longitude' columns.")
    st.stop()

X = df[['pickup_latitude', 'pickup_longitude']].copy()

# Convert to numeric
X['pickup_latitude'] = pd.to_numeric(X['pickup_latitude'], errors='coerce')
X['pickup_longitude'] = pd.to_numeric(X['pickup_longitude'], errors='coerce')

# Remove infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Drop missing values
before_rows = len(X)
X = X.dropna()
after_rows = len(X)

st.write(f"🧹 Removed {before_rows - after_rows} invalid rows during cleaning.")

if len(X) < 10:
    st.error("Not enough valid data points after cleaning.")
    st.stop()

# ==========================================================
# 3️⃣ Data Preprocessing (StandardScaler)
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# 4️⃣ 5️⃣ 6️⃣ DBSCAN Experiments
# ==========================================================

eps_values = [0.2, 0.3, 0.5]
results = {}

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)
    results[eps] = labels

# ==========================================================
# 7️⃣ Cluster Evaluation
# ==========================================================

st.markdown("---")
st.subheader("📈 Cluster Evaluation")

evaluation_data = []

for eps, labels in results.items():

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)

    evaluation_data.append({
        "eps": eps,
        "Clusters": n_clusters,
        "Noise Points": n_noise,
        "Noise Ratio": round(noise_ratio, 4)
    })

eval_df = pd.DataFrame(evaluation_data)
st.dataframe(eval_df)

# ==========================================================
# 8️⃣ Silhouette Score Calculation
# ==========================================================

st.markdown("---")
st.subheader("📊 Silhouette Scores")

silhouette_results = {}

for eps, labels in results.items():
    mask = labels != -1

    if len(set(labels[mask])) > 1:
        score = silhouette_score(X_scaled[mask], labels[mask])
        silhouette_results[eps] = round(score, 4)
        st.write(f"eps = {eps} → Silhouette Score: {round(score, 4)}")
    else:
        silhouette_results[eps] = None
        st.write(f"eps = {eps} → Silhouette Score: Not Applicable")

# ==========================================================
# 9️⃣ Visualization
# ==========================================================

st.markdown("---")
st.subheader("🗺 Cluster Visualizations")

for eps, labels in results.items():

    st.markdown(f"### 📍 Experiment with eps = {eps}")

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            color = 'black'
            marker = 'x'
            label_name = "Noise"
        else:
            color = None
            marker = 'o'
            label_name = f"Cluster {label}"

        ax.scatter(
            X_scaled[labels == label, 0],
            X_scaled[labels == label, 1],
            c=color,
            marker=marker,
            s=10,
            label=label_name
        )

    ax.set_xlabel("Latitude (scaled)")
    ax.set_ylabel("Longitude (scaled)")
    ax.legend()
    st.pyplot(fig)

# ==========================================================
# 🔟 Best Model Selection
# ==========================================================

st.markdown("---")
st.subheader("🏆 Best Model Selection")

best_eps = None
best_score = -1

for eps in eps_values:
    score = silhouette_results[eps]
    noise_ratio = eval_df[eval_df["eps"] == eps]["Noise Ratio"].values[0]

    if score is not None:
        if score > best_score and noise_ratio < 0.5:
            best_score = score
            best_eps = eps

if best_eps is not None:
    st.success(f"✅ Best eps value = {best_eps}")
else:
    st.warning("Best eps value = Not Applicable")

st.markdown("---")
st.info("ClusterCab Pro | Multi-Experiment DBSCAN Dashboard 🚕")
