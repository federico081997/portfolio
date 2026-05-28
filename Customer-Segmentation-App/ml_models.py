import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st


@st.cache_data
def prepare_clustering_data(df: pd.DataFrame):
    """
    Isolates behavioral and demographic features and scales them for clustering.

    Because the dataset is flat (one row per interaction), no aggregation is needed.
    Scaling is applied using StandardScaler to ensure the distance-based
    K-Means algorithm does not bias towards features with larger magnitudes
    (like Purchase Amount over Review Rating).

    Args:
        df (pd.DataFrame): The cleaned B2C dataframe.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A copy of the input dataframe.
            - np.ndarray: The mathematically scaled 2D numerical array ready for modeling.

    Raises:
        KeyError: If the required columns are missing from the input dataframe.
        ValueError: If the input dataframe is empty.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty. Cannot prepare clustering data.")

    features_to_scale = [
        "Age",
        "Purchase Amount (USD)",
        "Previous Purchases",
        "Review Rating",
    ]

    if not set(features_to_scale).issubset(df.columns):
        raise KeyError(
            f"Missing one or more required columns for clustering: {features_to_scale}"
        )

    # Standardize the data (mean=0, variance=1)
    scaler = StandardScaler()

    # Fit and transform the behavioral features
    scaled_features = scaler.fit_transform(df[features_to_scale])

    # Return a copy to prevent SettingWithCopy warnings when appending labels later
    return df.copy(), scaled_features


@st.cache_data
def calculate_wcss(scaled_features, max_k: int = 10):
    """
    Computes the Within-Cluster Sum of Squares (WCSS) to determine the optimal 'k'.

    Iterates through a range of cluster counts (k), fitting a model for each,
    and extracting the inertia (SSE) to generate the coordinates for an Elbow plot.

    Args:
        scaled_features (np.ndarray): The standardized numerical feature matrix.
        max_k (int, optional): The maximum number of clusters to evaluate. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - list: The integer values of k tested (e.g., [1, 2, ..., max_k]).
            - list: The corresponding WCSS (inertia) float values for each k.

    Raises:
        ValueError: If max_k is less than 1 or if scaled_features is empty.
    """
    if max_k < 1:
        raise ValueError("max_k must be an integer greater than or equal to 1.")
    if len(scaled_features) == 0:
        raise ValueError("Scaled features array is empty. Cannot calculate WCSS.")

    wcss = []
    k_values = list(range(1, max_k + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    return k_values, wcss


def train_kmeans_model(df: pd.DataFrame, scaled_features, n_clusters: int = 4):
    """
    Executes final K-Means clustering and appends labels directly to the dataset.

    Args:
        df (pd.DataFrame): The dataframe prepared for clustering.
        scaled_features (np.ndarray): The standardized feature matrix.
        n_clusters (int, optional): The chosen number of clusters. Defaults to 4.

    Returns:
        pd.DataFrame: The original dataframe with an appended 'Cluster' column.

    Raises:
        ValueError: If n_clusters is less than 1.
    """
    if n_clusters < 1:
        raise ValueError("n_clusters must be an integer greater than or equal to 1.")

    # Fit the final model and assign labels directly to the dataset
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_features)

    # Cast Cluster to string and add a prefix so Plotly treats it as a
    # discrete category in legends, rather than a continuous color scale.
    df["Cluster"] = "Cluster " + df["Cluster"].astype(str)

    return df
