from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import umap

from recommender import load_artifacts
from labeling import label_clusters


def run_kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """
    Run K-means clustering on the embedding matrix.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    # Predict cluster labels
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
    Reduce embeddings with UMAP.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        metric=metric,
    )

    # Perform the UMAP projection on the embeddings
    coords = reducer.fit_transform(embeddings)

    return coords


def main() -> None:
    # Define project root
    project_root = Path(__file__).parent

    # Define paths where to save paper clustering outputs
    clustered_output_path = project_root / "data" / "processed" / "papers_clustered.csv"
    cluster_summary_path = project_root / "data" / "processed" / "cluster_summary.csv"

    # Load dataset and embeddings
    df, embeddings, _ = load_artifacts()

    print(f"Dataset shape: {df.shape}")
    print(f"Embeddings shape: {embeddings.shape}")

    # K-means clustering
    # Use number of known categories as the number of clusters
    n_clusters = df["category"].nunique()
    print(f"Running K-means with {n_clusters} clusters...")
    cluster_labels = run_kmeans_clustering(
        embeddings=embeddings,
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    df["cluster_id"] = cluster_labels

    # UMAP for visualization
    print("Calculating 2D UMAP for visualization...")
    coords_2d = run_umap_projection(
        embeddings=embeddings,
        n_neighbors=40,
        n_components=2,
        random_state=42,
        metric="cosine",
    )
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

    # TF-IDF cluster keywords
    print("Creating TF-IDF cluster keywords and generating labels with Ollama...")
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

    # Save the outputs
    df.to_csv(clustered_output_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)

    print("Clustered dataset saved to:", clustered_output_path)
    print("Cluster summary saved to:", cluster_summary_path)
    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
