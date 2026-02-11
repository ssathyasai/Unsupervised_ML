import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="NYC Taxi DBSCAN Clustering", layout="wide")

st.title("🚖 NYC Taxi Pickup Clustering using DBSCAN")
st.markdown("Upload your dataset and explore clustering interactively.")

# Drag & Drop Upload
uploaded_file = st.file_uploader(
    "📂 Drag and Drop your CSV file here",
    type=["csv"]
)

if uploaded_file is not None:

    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Check required columns
    if not {'pickup_latitude', 'pickup_longitude'}.issubset(df.columns):
        st.error("Required columns not found! Dataset must contain pickup_latitude and pickup_longitude.")
        st.stop()

    # Sampling for performance
    if len(df) > 100000:
        st.warning("Large dataset detected. Sampling 50,000 rows for performance.")
        df = df.sample(50000, random_state=42)

    # Feature Selection
    X = df[['pickup_latitude', 'pickup_longitude']]

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("Data Preprocessing Completed")

    st.sidebar.header("⚙️ DBSCAN Parameters")

    eps_values = [0.2, 0.3, 0.5]
    min_samples = st.sidebar.slider("Min Samples", 3, 10, 5)

    results = []

    for eps in eps_values:

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = np.sum(labels == -1)
        noise_ratio = noise_points / len(labels)

        mask = labels != -1
        if len(set(labels[mask])) > 1:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
        else:
            sil_score = None

        results.append({
            "eps": eps,
            "clusters": n_clusters,
            "noise_points": noise_points,
            "noise_ratio": round(noise_ratio, 3),
            "silhouette_score": sil_score,
            "labels": labels
        })

    # Evaluation Table
    st.subheader("📊 Cluster Evaluation")

    eval_df = pd.DataFrame([{
        "eps": r["eps"],
        "Clusters": r["clusters"],
        "Noise Ratio": r["noise_ratio"],
        "Silhouette Score": r["silhouette_score"]
    } for r in results])

    st.dataframe(eval_df)

    # Visualization
    st.subheader("📍 Cluster Visualizations")

    for r in results:

        st.markdown(f"### eps = {r['eps']}")

        fig, ax = plt.subplots()

        labels = r["labels"]
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                subset = X[labels == -1]
                ax.scatter(
                    subset['pickup_latitude'],
                    subset['pickup_longitude'],
                    marker='x',
                    label='Noise'
                )
            else:
                subset = X[labels == label]
                ax.scatter(
                    subset['pickup_latitude'],
                    subset['pickup_longitude'],
                    label=f'Cluster {label}'
                )

        ax.set_xlabel("Latitude")
        ax.set_ylabel("Longitude")
        ax.legend()

        st.pyplot(fig)

    # Best Model Selection
    st.subheader("🏆 Best Model Selection")

    best_eps = None
    best_score = -1

    for r in results:
        if r["silhouette_score"] is not None:
            score = r["silhouette_score"] * (1 - r["noise_ratio"])
            if score > best_score:
                best_score = score
                best_eps = r["eps"]

    if best_eps:
        st.success(f"Best eps value = {best_eps}")
    else:
        st.warning("No suitable model found.")
