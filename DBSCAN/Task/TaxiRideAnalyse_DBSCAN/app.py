import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Taxi Ride DBSCAN Analysis", layout="wide")

st.title("🚖 Taxi Ride Clustering using DBSCAN")

uploaded_file = st.file_uploader(
    "📂 Drag and Drop your CSV file here",
    type=["csv"]
)

if uploaded_file is not None:

    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    st.subheader("Preview")
    st.dataframe(df.head())

    # Check required columns
    required_cols = ['pickup_latitude', 'pickup_longitude']

    if not set(required_cols).issubset(df.columns):
        st.error("Dataset must contain pickup_latitude and pickup_longitude columns.")
        st.stop()

    # ======================================================
    # CLEANING DATA (IMPORTANT FIX)
    # ======================================================

    df = df[required_cols]

    # Convert to numeric (handle string issues)
    df['pickup_latitude'] = pd.to_numeric(df['pickup_latitude'], errors='coerce')
    df['pickup_longitude'] = pd.to_numeric(df['pickup_longitude'], errors='coerce')

    # Remove NaN values
    df = df.dropna()

    # Remove impossible coordinates
    df = df[
        (df['pickup_latitude'].between(-90, 90)) &
        (df['pickup_longitude'].between(-180, 180))
    ]

    if len(df) == 0:
        st.error("No valid data left after cleaning!")
        st.stop()

    # Sampling large dataset
    if len(df) > 100000:
        st.warning("Large dataset detected. Sampling 50,000 rows.")
        df = df.sample(50000, random_state=42)

    st.success(f"Cleaned Dataset Size: {len(df)} rows")

    # ======================================================
    # FEATURE SELECTION
    # ======================================================
    X = df[['pickup_latitude', 'pickup_longitude']]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================================================
    # DBSCAN Experiments
    # ======================================================

    eps_values = [0.2, 0.3, 0.5]
    results = []

    for eps in eps_values:

        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = np.sum(labels == -1)
        noise_ratio = noise_points / len(labels)

        # Silhouette score
        mask = labels != -1
        if len(set(labels[mask])) > 1:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
        else:
            sil_score = None

        results.append({
            "eps": eps,
            "clusters": n_clusters,
            "noise_ratio": round(noise_ratio, 3),
            "silhouette": sil_score,
            "labels": labels
        })

    # ======================================================
    # EVALUATION TABLE
    # ======================================================
    st.subheader("📊 Cluster Evaluation")

    eval_df = pd.DataFrame([{
        "eps": r["eps"],
        "Clusters": r["clusters"],
        "Noise Ratio": r["noise_ratio"],
        "Silhouette Score": r["silhouette"]
    } for r in results])

    st.dataframe(eval_df)

    # ======================================================
    # VISUALIZATION
    # ======================================================
    st.subheader("📍 Cluster Visualizations")

    for r in results:

        st.markdown(f"### eps = {r['eps']}")

        fig, ax = plt.subplots()
        labels = r["labels"]

        for label in set(labels):

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

    # ======================================================
    # BEST MODEL SELECTION
    # ======================================================
    st.subheader("🏆 Best Model Selection")

    best_eps = None
    best_score = -1

    for r in results:
        if r["silhouette"] is not None:
            score = r["silhouette"] * (1 - r["noise_ratio"])
            if score > best_score:
                best_score = score
                best_eps = r["eps"]

    if best_eps:
        st.success(f"Best eps value = {best_eps}")
    else:
        st.warning("Silhouette score not applicable.")
