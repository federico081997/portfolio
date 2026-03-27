from pathlib import Path

import faiss
import numpy as np


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """
    Load precomputed embeddings from disk.

    The embeddings are converted to float32 because this is the expected
    numeric format for FAISS indexing.

    Args:
        embeddings_path: Path to the saved NumPy embeddings file.

    Returns:
        np.ndarray: Embedding matrix as float32.
    """
    return np.load(embeddings_path).astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for similarity search.

    Since the embeddings are already normalized, inner product similarity
    behaves like cosine similarity.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).

    Returns:
        faiss.Index: Populated FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, index_path: Path) -> None:
    """
    Save a FAISS index to disk.

    Args:
        index: FAISS index to save.
        index_path: Output path for the saved index file.
    """
    faiss.write_index(index, str(index_path))


def main() -> None:
    """
    Load embeddings, build a FAISS index, and save it to disk.
    """
    # Define the project root directory.
    project_root = Path(__file__).parent

    # Define input and output paths.
    embeddings_path = project_root / "data" / "processed" / "paper_embeddings.npy"
    index_path = project_root / "data" / "processed" / "faiss_paper_index.bin"

    # Load embeddings, build the index, and save it.
    embeddings = load_embeddings(embeddings_path)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, index_path)

    print(f"FAISS index saved to: {index_path}")
    print(f"Total vectors indexed: {index.ntotal}")


if __name__ == "__main__":
    main()
