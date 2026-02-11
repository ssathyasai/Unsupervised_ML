import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="NYC Taxi DBSCAN Clustering",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 NYC Taxi Pickup Clustering using DBSCAN")

# --------------------------------------------------
# Safe Dataset Loader
# --------------------------------------------------
@st.cache_data
def load_data():
    try:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "NewYorkCityTaxiTripDuration.csv")

        if not os.path.exists(file_path):
            return None, "Dataset file not found in project folder."

        df = pd.read_csv(file_path)

        required_columns = ["pickup_latitude", "pickup_longitude"]

        for col in required_columns:
            if col not in df.columns:
                return None, f"Column '{col}' not found in dataset."

        df = df[required_columns].dropna()

        if df.empty:
            return None, "Dataset is empty after removing null values."

        return df, None

    except Exception as e:
        return None, str(e)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "NewYorkCityTaxiTripDuration.csv")

        if not os.path.exists(file_path):
            return None, "Dataset file not found."

        df = pd.read_csv(
            file_path,
            usecols=["pickup_latitude", "pickup_longitude"]
        ).dropna()

        return df, None

    except Exception as e:
        return None, str(e)


st.success(f"Dataset Loaded Successfully ✅ | Total Records: {len(df)}")

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙️ DBSCAN Parameters")

eps = st.sidebar.slider("eps value", 0.1, 1.0, 0.3, 0.1)
min_samples = st.sidebar.slider("min_samples", 3, 20, 5)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
except Exception as e:
    st.error(f"Scaling Error: {e}")
    st.stop()

# --------------------------------------------------
# Model Training
# --------------------------------------------------
try:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

# --------------------------------------------------
# Metrics
# --------------------------------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
noise_ratio = n_noise / len(labels)

mask = labels != -1

if len(set(labels[mask])) > 1:
    silhouette = silhouette_score(X_scaled[mask], labels[mask])
else:
    silhouette = None

col1, col2, col3 = st.columns(3)

col1.metric("Clusters", n_clusters)
col2.metric("Noise Points", n_noise)
col3.metric("Noise Ratio", round(noise_ratio, 4))

if silhouette is not None:
    st.info(f"Silhouette Score: {round(silhouette, 4)}")
else:
    st.warning("Silhouette Score Not Applicable")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("Cluster Visualization")

try:
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            ax.scatter(
                X_scaled[labels == label, 0],
                X_scaled[labels == label, 1],
                c='black',
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

    ax.set_xlabel("Latitude (Scaled)")
    ax.set_ylabel("Longitude (Scaled)")
    ax.legend()

    st.pyplot(fig)

except Exception as e:
    st.error(f"Plotting Error: {e}")

st.markdown("---")
st.markdown("Built with Streamlit 🚀 | Fully Error-Handled Version")
