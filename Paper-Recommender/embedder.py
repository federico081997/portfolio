from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import load_processed_data
from recommender import get_model


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generate dense embeddings for a list of texts
    """
    print(f"Downloading model...")
    model = get_model()

    print("Generating embeddings...")
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )

    return embeddings


def main() -> None:
    # Define project root
    project_root = Path(__file__).parent

    # Load processed DataFrame
    df = load_processed_data()

    # Generate embeddings
    texts = df["combined_text"].tolist()
    embeddings = generate_embeddings(texts)

    # Save embeddings
    embeddings_path = project_root / "data" / "processed" / "paper_embeddings.npy"
    np.save(embeddings_path, embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to: {embeddings_path}")


if __name__ == "__main__":
    main()
