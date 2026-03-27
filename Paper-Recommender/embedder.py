from pathlib import Path

import numpy as np

from data_loader import load_processed_data
from recommender import get_model


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generate dense vector embeddings for a list of input texts.

    This function:
    - loads the embedding model
    - encodes the input texts into dense vectors
    - normalizes embeddings for cosine similarity usage

    Args:
        texts: List of text strings to encode.

    Returns:
        np.ndarray: Array of shape (n_samples, embedding_dim).
    """
    # Load the model
    model = get_model()

    # Encode texts into embeddings.
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def main() -> None:
    """
    Generate and save embeddings for the processed dataset.

    Workflow:
    - load cleaned dataset
    - extract combined text field
    - generate embeddings
    - save embeddings to disk
    """
    # Define project root directory.
    project_root = Path(__file__).parent

    # Load processed dataset.
    df = load_processed_data()

    # Extract text data for embedding generation.
    texts = df["combined_text"].tolist()

    # Generate embeddings.
    embeddings = generate_embeddings(texts)

    # Define output path and save embeddings.
    embeddings_path = project_root / "data" / "processed" / "paper_embeddings.npy"
    np.save(embeddings_path, embeddings)

    # Minimal informative output.
    print(f"Embeddings saved: {embeddings.shape} -> {embeddings_path}")


if __name__ == "__main__":
    main()
