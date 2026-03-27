from pathlib import Path
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from data_loader import load_processed_data


# Default embedding model used across the project
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Instantiate and return the sentence transformer model.

    Args:
        model_name: Name of the pretrained model.

    Returns:
        SentenceTransformer: Loaded embedding model.
    """
    return SentenceTransformer(model_name)


def load_artifacts():
    """
    Load all core artifacts required for the recommender system.

    This includes:
    - cleaned dataset
    - precomputed embeddings
    - FAISS index for fast similarity search

    Returns:
        tuple: (df, embeddings, faiss_index)
    """
    # Define project root
    project_root = Path(__file__).parent

    # Define artifact paths
    embeddings_path = project_root / "data" / "processed" / "paper_embeddings.npy"
    faiss_index_path = project_root / "data" / "processed" / "faiss_paper_index.bin"

    # Load dataset
    df = load_processed_data()

    # Ensure required files exist
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at: {embeddings_path}")

    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {faiss_index_path}")

    # Load embeddings and FAISS index
    embeddings = np.load(embeddings_path).astype("float32")
    faiss_index = faiss.read_index(str(faiss_index_path))

    return df, embeddings, faiss_index


def tokenize_technical(text: str) -> list[str]:
    """
    Tokenizer tailored for scientific and technical text.

    Keeps:
    - alphabetic words
    - acronyms (e.g. PDE, CFD, ML)
    - alphanumeric tokens (e.g. 3D, L2)
    - hyphenated terms (e.g. finite-volume, GPU-based)

    Removes:
    - stopwords
    - very short tokens
    - pure numeric tokens

    Args:
        text: Input text string.

    Returns:
        list: Cleaned list of tokens.
    """
    STOPWORDS = set(ENGLISH_STOP_WORDS)

    # Normalize text
    text = str(text).lower()

    # Regex pattern for technical tokens
    pattern = r"(?u)\b[a-z0-9]+(?:[-–][a-z0-9]+)*\b"

    # Extract tokens
    raw_tokens = re.findall(pattern, text)

    final_tokens = []

    for token in raw_tokens:
        # Skip pure numbers
        if token.isdigit():
            continue

        # Keep full token if meaningful
        if token not in STOPWORDS and len(token) >= 3:
            final_tokens.append(token)

        # If token contains hyphen, also split into parts
        if "-" in token or "–" in token:
            parts = re.split(r"[-–]", token)
            for part in parts:
                if part not in STOPWORDS and len(part) >= 3 and not part.isdigit():
                    final_tokens.append(part)

    return final_tokens


def keyword_overlap_score(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard-style keyword overlap between two texts.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        float: Overlap score in [0, 1].
    """
    tokens_a = set(tokenize_technical(text_a))
    tokens_b = set(tokenize_technical(text_b))

    # Avoid division by zero
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))

    return intersection / union


def recency_score(year: float, min_year: float, max_year: float) -> float:
    """
    Normalize publication year into a [0, 1] score.

    More recent papers receive higher scores.

    Args:
        year: Paper year.
        min_year: Minimum year in dataset.
        max_year: Maximum year in dataset.

    Returns:
        float: Normalized recency score.
    """
    if pd.isna(year):
        return 0.0

    # Prevent division by zero
    if min_year == max_year:
        return 1.0

    return (year - min_year) / (max_year - min_year)


def hybrid_score(
    semantic_similarity: float,
    category_bonus: float,
    keyword_overlap: float,
    recency_score: float,
    w_semantic: int = 0.75,
    w_category: int = 0.10,
    w_keyword: int = 0.10,
    w_recency: int = 0.05,
):
    """
    Compute final hybrid ranking score.

    Combines:
    - semantic similarity
    - category match bonus
    - keyword overlap
    - recency

    Args:
        semantic_similarity: Embedding similarity score.
        category_bonus: Bonus if categories match.
        keyword_overlap: Jaccard keyword score.
        recency_score: Recency normalization score.
        w_*: Weights for each component.

    Returns:
        float: Final weighted score.
    """
    return (
        w_semantic * semantic_similarity
        + w_category * category_bonus
        + w_keyword * keyword_overlap
        + w_recency * recency_score
    )


def build_explanation(row: dict) -> str:
    """
    Generate a human-readable explanation for a recommendation.

    Args:
        row: Dictionary or row containing scoring components.

    Returns:
        str: Explanation string.
    """
    reasons = []

    semantic_similarity = row.get("semantic_similarity", 0.0)
    same_category = row.get("same_category", False)
    keyword_overlap = row.get("keyword_overlap", 0.0)
    recency_score = row.get("recency_score", 0.0)

    # Add a reason based on semantic similarity strength.
    if semantic_similarity >= 0.80:
        reasons.append("very strong semantic similarity")
    elif semantic_similarity >= 0.65:
        reasons.append("strong topical similarity")
    elif semantic_similarity >= 0.50:
        reasons.append("moderate semantic similarity")

    # Add a reason if the candidate paper belongs to the same category.
    if same_category:
        reasons.append("matching research category")

    # Add a reason based on technical keyword overlap.
    if keyword_overlap >= 0.12:
        reasons.append("clear overlap in technical keywords")
    elif keyword_overlap >= 0.06:
        reasons.append("some overlap in technical terminology")

    # Add a reason if the paper is relatively recent.
    if recency_score >= 0.75:
        reasons.append("relatively recent publication")

    # Fallback explanation if no specific reason crosses the thresholds.
    if not reasons:
        return "Recommended based on the overall embedding similarity pattern."

    if len(reasons) == 1:
        return f"Recommended because of {reasons[0]}."

    return "Recommended because of " + ", ".join(reasons[:-1]) + f", and {reasons[-1]}."


def get_similar_by_paper(
    paper_idx: int,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    faiss_index: faiss.Index,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Recommend papers similar to a selected paper using embedding similarity
    and rerank them using hybrid scoring.

    Args:
        paper_idx: Integer row index of the selected paper.
        df: DataFrame containing paper metadata.
        embeddings: Precomputed embedding matrix.
        faiss_index: FAISS index used for nearest-neighbor search.
        top_k: Number of recommendations to return.

    Returns:
        pd.DataFrame: Ranked recommendation results.
    """
    # Ensure the provided paper index is treated as an integer row position.
    paper_idx = int(paper_idx)

    # Work on a copy to avoid modifying the original dataframe.
    working_df = df.copy()
    working_df["published_date"] = pd.to_datetime(
        working_df["published_date"],
        errors="coerce",
    )

    # Retrieve the embedding vector for the selected paper.
    query_vector = embeddings[paper_idx].reshape(1, -1)

    # Search for nearest neighbors in the FAISS index.
    # top_k + 1 is used because the paper itself will usually be returned too.
    scores, indices = faiss_index.search(query_vector, top_k + 1)

    candidate_scores = scores[0]
    candidate_indices = indices[0]

    results = []

    source_row = working_df.iloc[paper_idx]

    # Compute dataset year range for recency normalization.
    min_year = working_df["published_date"].dt.year.min()
    max_year = working_df["published_date"].dt.year.max()

    for idx, score in zip(candidate_indices, candidate_scores):
        idx = int(idx)

        # Skip the selected source paper.
        if idx == paper_idx:
            continue

        candidate_row = working_df.iloc[idx]

        # Compute keyword overlap between the source paper and candidate paper.
        overlap = keyword_overlap_score(
            source_row["combined_text"],
            candidate_row["combined_text"],
        )

        # Check whether both papers belong to the same category.
        same_category = source_row["category"] == candidate_row["category"]

        # Compute the candidate paper recency score.
        candidate_year = candidate_row["published_date"].year
        recent = recency_score(candidate_year, min_year, max_year)

        # Compute the final hybrid score used for reranking.
        final_score = hybrid_score(
            semantic_similarity=float(score),
            category_bonus=same_category,
            keyword_overlap=overlap,
            recency_score=recent,
        )

        # Build a readable explanation for the recommendation.
        explanation = build_explanation(
            {
                "semantic_similarity": float(score),
                "same_category": same_category,
                "keyword_overlap": overlap,
                "recency_score": recent,
            }
        )

        results.append(
            {
                "paper_index": idx,
                "title": candidate_row["title"],
                "category": candidate_row["category"],
                "authors": candidate_row["authors"],
                "year": candidate_year,
                "semantic_similarity": float(score),
                "keyword_overlap": overlap,
                "same_category": same_category,
                "recency_score": recent,
                "final_score": final_score,
                "explanation": explanation,
            }
        )

    # Convert results into a dataframe, sort by final score, and keep top_k rows.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("final_score", ascending=False).head(top_k)

    return results_df.reset_index(drop=True)


def get_similar_by_query(
    query: str,
    df: pd.DataFrame,
    faiss_index: faiss.Index,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Recommend papers for a free-text query using hybrid ranking.

    Args:
        query: Free-text search query.
        df: DataFrame containing paper metadata.
        faiss_index: FAISS index used for nearest-neighbor search.
        top_k: Number of recommendations to return.

    Returns:
        pd.DataFrame: Ranked recommendation results.
    """
    # Generate an embedding for the input query.
    model = get_model()
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Search for the most similar papers in the FAISS index.
    scores, indices = faiss_index.search(query_vector, top_k)

    candidate_scores = scores[0]
    candidate_indices = indices[0]

    results = []

    # Ensure publication dates are in datetime format.
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    # Compute dataset year range for recency normalization.
    min_year = df["published_date"].dt.year.min()
    max_year = df["published_date"].dt.year.max()

    for idx, score in zip(candidate_indices, candidate_scores):
        candidate_row = df.iloc[idx]

        # Compute keyword overlap between the query and candidate paper.
        overlap = keyword_overlap_score(query, candidate_row["combined_text"])

        # Compute recency score for the candidate paper.
        recent = recency_score(candidate_row["published_date"].year, min_year, max_year)

        # Compute final hybrid score.
        # Category bonus is disabled here because free-text queries do not
        # inherently belong to a known paper category.
        final_score = hybrid_score(
            semantic_similarity=float(score),
            category_bonus=False,
            keyword_overlap=overlap,
            recency_score=recent,
            w_category=0.0,
        )

        # Build a human-readable explanation.
        explanation = build_explanation(
            {
                "semantic_similarity": float(score),
                "same_category": False,
                "keyword_overlap": overlap,
                "recency_score": recent,
            }
        )

        # Store result for later sorting and display.
        results.append(
            {
                "paper_index": idx,
                "title": candidate_row["title"],
                "category": candidate_row["category"],
                "authors": candidate_row["authors"],
                "year": candidate_row["published_date"].year,
                "semantic_similarity": float(score),
                "keyword_overlap": overlap,
                "recency_score": recent,
                "final_score": final_score,
                "explanation": explanation,
            }
        )

        # Convert results into a dataframe, sort by final score, and keep top_k rows.
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("final_score", ascending=False).head(top_k)

    return results_df.reset_index(drop=True)
