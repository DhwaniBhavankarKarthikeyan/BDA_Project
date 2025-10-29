import streamlit as st
import pandas as pd
from pymongo import MongoClient
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1️⃣ MongoDB Connection
# ----------------------------
@st.cache_resource
def get_mongo_collection():
    uri = "mongodb+srv://dhwanibhavankarbtech2022_db_user:ETocwBaxBKa9Py8O@bigdataproject.h8jfu8w.mongodb.net/?retryWrites=true&w=majority&appName=BigDataProject"
    client = MongoClient(uri)
    db = client["big_data_clustered_db"]
    return db["customers_clustered"]

# ----------------------------
# 2️⃣ Load Data
# ----------------------------
def load_data(collection):
    df = pd.DataFrame(list(collection.find()))
    if df.empty:
        st.error("No data found in the MongoDB collection.")
        return None

    if "Cluster" not in df.columns:
        st.error("'Cluster' column not found in the data.")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["Cluster", "_id"]]

    if len(feature_cols) < 2:
        st.error("Not enough numeric columns for PCA.")
        return None

    return df, feature_cols

# ----------------------------
# 3️⃣ PCA + Visualization
# ----------------------------
def plot_clusters(df, feature_cols):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df[feature_cols])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                         c=df["Cluster"], cmap="viridis", alpha=0.7)
    plt.title("Customer Clusters (PCA Visualization)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")
    st.pyplot(fig)

# ----------------------------
# 4️⃣ Streamlit UI
# ----------------------------
st.title("🧩 Customer Clusters Dashboard")
st.write("Visualizing pre-computed customer clusters from MongoDB.")

collection = get_mongo_collection()

if st.button("Load and Visualize Clusters"):
    df, feature_cols = load_data(collection)
    if df is not None:
        st.success(f"✅ Loaded {len(df)} customer records from MongoDB.")
        plot_clusters(df, feature_cols)
