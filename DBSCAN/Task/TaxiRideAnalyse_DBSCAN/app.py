# ==========================================================
# 🚕 TaxiRideAnalyse - Stable DBSCAN Multi Experiment App
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Taxi Ride Analyse", layout="wide")
st.title("🚕 Taxi Pickup Hotspot Detection using DBSCAN")
st.markdown("---")

# ==========================================================
# 1️⃣ Load Dataset
# ==========================================================

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================================================
# 2️⃣ Feature Selection + STRONG Cleaning
# ==========================================================

required_cols = ['pickup_latitude', 'pickup_longitude']

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain pickup_latitude and pickup_longitude columns.")
    st.stop()

# Select only required columns
X = df[['pickup_latitude', 'pickup_longitude']].copy()

# Convert EVERYTHING to numeric safely
X = X.apply(lambda col: pd.to_numeric(col, errors='coerce'))

# Replace infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values
X.dropna(inplace=True)

# Reset index
X.reset_index(drop=True, inplace=True)

st.write("Valid rows after cleaning:", len(X))

if len(X) < 5:
    st.error("Not enough valid data points after cleaning.")
    st.stop()

# ==========================================================
# 3️⃣ Safe Scaling
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# FINAL SAFETY CHECK
if np.isnan(X_scaled).any():
    st.error("NaN detected after scaling. Please check dataset.")
    st.stop()

# ==========================================================
# 4️⃣ 5️⃣ 6️⃣ DBSCAN Experiments
# ==========================================================

eps_values = [0.2, 0.3, 0.5]
results = {}

for eps in eps_values:
    model = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(X_scaled)
    results[eps] = labels

# ==========================================================
# 7️⃣ Cluster Evaluation
# ==========================================================

st.markdown("---")
st.subheader("Cluster Evaluation")

evaluation_rows = []

for eps, labels in results.items():
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    noise_ratio = n_noise / len(labels)

    evaluation_rows.append({
        "eps": eps,
        "Clusters": n_clusters,
        "Noise Points": n_noise,
        "Noise Ratio": round(noise_ratio, 4)
    })

eval_df = pd.DataFrame(evaluation_rows)
st.dataframe(eval_df)

# ==========================================================
# 8️⃣ Silhouette Score
# ==========================================================

st.markdown("---")
st.subheader("Silhouette Scores")

silhouette_scores = {}

for eps, labels in results.items():
    mask = labels != -1

    if len(set(labels[mask])) > 1:
        score = silhouette_score(X_scaled[mask], labels[mask])
        silhouette_scores[eps] = round(score, 4)
        st.write(f"eps={eps} → Silhouette Score: {round(score,4)}")
    else:
        silhouette_scores[eps] = None
        st.write(f"eps={eps} → Silhouette Score: Not Applicable")

# ==========================================================
# 9️⃣ Visualization
# ==========================================================

st.markdown("---")
st.subheader("Cluster Visualizations")

for eps, labels in results.items():
    st.markdown(f"### Experiment with eps = {eps}")

    fig, ax = plt.subplots(figsize=(6,5))
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            ax.scatter(
                X_scaled[labels == label, 0],
                X_scaled[labels == label, 1],
                marker='x',
                s=10,
                label="Noise"
            )
        else:
            ax.scatter(
                X_scaled[labels == label, 0],
                X_scaled[labels == label, 1],
                s=10,
                label=f"Cluster {label}"
            )

    ax.set_xlabel("Latitude (scaled)")
    ax.set_ylabel("Longitude (scaled)")
    ax.legend()
    st.pyplot(fig)

# ==========================================================
# 🔟 Best Model Selection
# ==========================================================

st.markdown("---")
st.subheader("Best Model Selection")

best_eps = None
best_score = -1

for eps in eps_values:
    score = silhouette_scores[eps]
    noise_ratio = eval_df[eval_df["eps"] == eps]["Noise Ratio"].values[0]

    if score is not None:
        if score > best_score and noise_ratio < 0.5:
            best_score = score
            best_eps = eps

if best_eps is not None:
    st.success(f"Best eps value = {best_eps}")
else:
    st.warning("Best eps value = Not Applicable")

st.markdown("---")
st.info("App Running Successfully 🚀")
