import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="NYC Taxi DBSCAN Clustering",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 NYC Taxi Pickup Clustering using DBSCAN")
st.markdown("Density-Based Clustering Analysis of Pickup Locations")

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
import os

@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "NewYorkCityTaxiTripDuration.csv")
    
    df = pd.read_csv(file_path)
    df = df[['pickup_latitude', 'pickup_longitude']].dropna()
    return df


st.success(f"Dataset Loaded Successfully ✅ | Total Records: {len(df)}")

# ------------------------------------------------
# Sidebar Controls
# ------------------------------------------------
st.sidebar.header("⚙️ DBSCAN Parameters")

eps = st.sidebar.slider("Select eps value", 0.1, 1.0, 0.3, 0.1)
min_samples = st.sidebar.slider("Select min_samples", 3, 20, 5)

# ------------------------------------------------
# Data Scaling
# ------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ------------------------------------------------
# Model Training
# ------------------------------------------------
db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(X_scaled)

# ------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
noise_ratio = n_noise / len(labels)

# Remove noise for silhouette
mask = labels != -1

if len(set(labels[mask])) > 1:
    silhouette = silhouette_score(X_scaled[mask], labels[mask])
else:
    silhouette = None

# ------------------------------------------------
# Display Metrics
# ------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📌 Clusters", n_clusters)
col2.metric("⚠️ Noise Points", n_noise)
col3.metric("📊 Noise Ratio", round(noise_ratio, 4))

if silhouette:
    st.info(f"Silhouette Score: {round(silhouette, 4)}")
else:
    st.warning("Silhouette Score Not Applicable")

# ------------------------------------------------
# Visualization
# ------------------------------------------------
st.subheader("📍 Cluster Visualization")

fig, ax = plt.subplots(figsize=(8,6))

unique_labels = set(labels)

for label in unique_labels:
    if label == -1:
        color = 'black'
        marker = 'x'
        label_name = 'Noise'
    else:
        color = None
        marker = 'o'
        label_name = f'Cluster {label}'
    
    ax.scatter(
        X_scaled[labels == label, 0],
        X_scaled[labels == label, 1],
        c=color,
        marker=marker,
        label=label_name,
        s=10
    )

ax.set_xlabel("Latitude (Scaled)")
ax.set_ylabel("Longitude (Scaled)")
ax.legend()

st.pyplot(fig)

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.markdown("👨‍💻 Built with Streamlit | DBSCAN Unsupervised Learning Project")
