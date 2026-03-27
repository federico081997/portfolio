from pathlib import Path

import numpy as np
import umap
from sklearn.cluster import KMeans

from labeling import label_clusters
from recommender import load_artifacts


def run_kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """
    Cluster embedding vectors using K-means.

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features).
        n_clusters: Number of clusters to create.
        random_state: Random seed for reproducibility.
        n_init: Number of K-means initializations.

    Returns:
        np.ndarray: Cluster label for each input sample.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    # Fit the model and assign a cluster label to each embedding.
    cluster_labels = kmeans.fit_predict(embeddings)

    return cluster_labels


def run_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 40,
    n_components: int = 2,
    random_state: int = 42,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Project high-dimensional embeddings into a lower-dimensional space using UMAP.

    This is mainly used for visualization.

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features).
        n_neighbors: Number of neighboring points used by UMAP.
        n_components: Number of output dimensions.
        random_state: Random seed for reproducibility.
        metric: Distance metric used by UMAP.

    Returns:
        np.ndarray: Projected coordinates of shape (n_samples, n_components).
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        metric=metric,
    )

    # Compute the low-dimensional embedding coordinates.
    coords = reducer.fit_transform(embeddings)

    return coords


def main() -> None:
    """
    Cluster papers, compute 2D visualization coordinates, generate cluster labels,
    and save the resulting outputs.
    """
    # Define the project root directory.
    project_root = Path(__file__).parent

    # Define output file paths.
    clustered_output_path = project_root / "data" / "processed" / "papers_clustered.csv"
    cluster_summary_path = project_root / "data" / "processed" / "cluster_summary.csv"

    # Load the dataset, embeddings, and faiss index.
    df, embeddings, _ = load_artifacts()

    print(f"Dataset shape: {df.shape}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Use the number of unique known categories as the number of K-means clusters.
    n_clusters = df["category"].nunique()
    print(f"Running K-means clustering with {n_clusters} clusters...")

    cluster_labels = run_kmeans_clustering(
        embeddings=embeddings,
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    df["cluster_id"] = cluster_labels

    # Generate 2D UMAP coordinates for visualization.
    print("Computing 2D UMAP projection...")
    coords_2d = run_umap_projection(
        embeddings=embeddings,
        n_neighbors=40,
        n_components=2,
        random_state=42,
        metric="cosine",
    )
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

    # Generate TF-IDF-based cluster keywords and optional Ollama-based cluster labels.
    print("Generating cluster keywords and labels...")
    df, cluster_summary = label_clusters(
        df=df,
        cluster_col="cluster_id",
        text_col="combined_text",
        top_n_words=10,
        max_docs_per_cluster=None,
        use_ollama=True,
        ollama_batch_size=1,
        ollama_model="llama3.2",
        host="http://localhost:11434",
    )

    # Save the enriched paper dataset and cluster summary.
    df.to_csv(clustered_output_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)

    print(f"Clustered dataset saved to: {clustered_output_path}")
    print(f"Cluster summary saved to: {cluster_summary_path}")
    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
