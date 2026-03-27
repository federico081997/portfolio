from pathlib import Path
import numpy as np
import faiss


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load generated embeddings"""
    return np.load(embeddings_path).astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a Faiss index for cosine-style similarity search
    Since embeddings are normalised, inner product works like cosine similarity
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, index_path: Path) -> None:
    """Save FAISS index to local path"""
    faiss.write_index(index, str(index_path))


def main() -> None:
    # Project directory
    project_root = Path(__file__).parent

    # Embeddings and Faiss index paths
    embeddings_path = project_root / "data" / "processed" / "paper_embeddings.npy"
    index_path = project_root / "data" / "processed" / "faiss_paper_index.bin"

    # Load generated embeddings and create FAISS index
    embeddings = load_embeddings(embeddings_path)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, index_path)

    print(f"FAISS index saved to {index_path}")
    print(f"Total vectors indexed: {index.ntotal}")


if __name__ == "__main__":
    main()
