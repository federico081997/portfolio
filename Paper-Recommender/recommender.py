from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import faiss
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from data_loader import load_processed_data

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Create an instance of the chosen model"""
    return SentenceTransformer(model_name)


def load_artifacts():
    """Load dataset, embeddings and FAISS index"""
    # Define project root
    project_root = Path(__file__).parent

    # Define embeddings and FAISS index paths
    embeddings_path = project_root / "data" / "processed" / "paper_embeddings.npy"
    faiss_index_path = project_root / "data" / "processed" / "faiss_paper_index.bin"

    # Load dataset
    df = load_processed_data()

    # Check if embedding and FAISS index paths exist
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
    Tokenizer designed for scientific / technical text.

    Keeps:
    - alphabetic words
    - acronyms like PDE, CFD, ML
    - alphanumeric tokens like 3D, L2
    - hyphenated technical phrases like finite-volume, GPU-based
    """
    # Define stopwords
    STOPWORDS = set(ENGLISH_STOP_WORDS)

    # Convert text to lowercase
    text = str(text).lower()

    # Define regex patten of tokenizer
    pattern = r"(?u)\b[a-z0-9]+(?:[-–][a-z0-9]+)*\b"

    # Extract raw tokens
    raw_tokens = re.findall(pattern, text)

    final_tokens = []

    # Add token to the final list
    for token in raw_tokens:
        # Skip pure numbers
        if token.isdigit():
            continue

        # Keep full token only if meaningful
        if token not in STOPWORDS and len(token) >= 3:
            final_tokens.append(token)

        # Only split if hyphen exists
        if "-" in token or "–" in token:
            parts = re.split(r"[-–]", token)
            for part in parts:
                if part not in STOPWORDS and len(part) >= 3 and not part.isdigit():
                    final_tokens.append(part)

    return final_tokens


def keyword_overlap_score(text_a: str, text_b: str) -> float:
    """Jaccard-style overlap score between"""
    # Tokenize inputs
    tokens_a = set(tokenize_technical(text_a))
    tokens_b = set(tokenize_technical(text_b))

    # Prevent division by 0
    if not tokens_a or not tokens_b:
        return 0.0

    # Jaccard similarity formula is calculated as:
    # intersection(A, B)/union(A,B)
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))

    return intersection / union


def recency_score(year: float, min_year: float, max_year: float) -> float:
    """
    Normalize year to [0, 1]
    More recent papers get a higher score
    """
    # Check the year is not null
    if pd.isna(year):
        return 0.0

    # Prevents division by 0
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
    """Weightd hybrid score"""
    return (
        w_semantic * semantic_similarity
        + w_category * category_bonus
        + w_keyword * keyword_overlap
        + w_recency * recency_score
    )


def build_explanation(row: dict) -> str:
    """Build a short explanation for why a paper was recommended"""
    reasons = []

    semantic_similarity = row.get("semantic_similarity", 0.0)
    same_category = row.get("same_category", False)
    keyword_overlap = row.get("keyword_overlap", 0.0)
    recency_score = row.get("recency_score", 0.0)

    if semantic_similarity >= 0.80:
        reasons.append("very strong semantic similarity")
    elif semantic_similarity >= 0.65:
        reasons.append("strong topical similarity")
    elif semantic_similarity >= 0.50:
        reasons.append("moderate semantic similarity")

    if same_category:
        reasons.append("matching research category")

    if keyword_overlap >= 0.12:
        reasons.append("clear overlap in technical keywords")
    elif keyword_overlap >= 0.06:
        reasons.append("some overlap in technical terminology")

    if recency_score >= 0.75:
        reasons.append("relatively recent publication")

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
    """

    # Ensure integer row position
    paper_idx = int(paper_idx)

    working_df = df.copy()
    working_df["published_date"] = pd.to_datetime(
        working_df["published_date"], errors="coerce"
    )

    # Retrieve embedding for selected paper
    query_vector = embeddings[paper_idx].reshape(1, -1)

    # Search for similar papers using FAISS index
    scores, indices = faiss_index.search(query_vector, top_k + 1)

    candidate_scores = scores[0]
    candidate_indices = indices[0]

    results = []

    source_row = working_df.iloc[paper_idx]

    min_year = working_df["published_date"].dt.year.min()
    max_year = working_df["published_date"].dt.year.max()

    for idx, score in zip(candidate_indices, candidate_scores):
        idx = int(idx)

        # Skip the chosen paper
        if idx == paper_idx:
            continue

        candidate_row = working_df.iloc[idx]

        # Compute keyword overlap score
        overlap = keyword_overlap_score(
            source_row["combined_text"],
            candidate_row["combined_text"],
        )

        # Check whether papers belong to same category
        same_category = source_row["category"] == candidate_row["category"]

        # Compute recency score safely
        candidate_year = candidate_row["published_date"].year
        recent = recency_score(candidate_year, min_year, max_year)

        # Compute hybrid score
        final_score = hybrid_score(
            semantic_similarity=float(score),
            category_bonus=same_category,
            keyword_overlap=overlap,
            recency_score=recent,
        )

        # Build explanation
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
    """
    # Generate embedding for the query
    model = get_model()
    query_vector = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    # Search for similar papers using FAISS index
    scores, indices = faiss_index.search(query_vector, top_k)

    # Candidate scores and indices
    candidate_scores = scores[0]
    candidate_indices = indices[0]

    results = []
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    min_year = df["published_date"].dt.year.min()
    max_year = df["published_date"].dt.year.max()

    for idx, score in zip(candidate_indices, candidate_scores):

        candidate_row = df.iloc[idx]

        # Compute keyword overlap score (Jaccard similarity score)
        overlap = keyword_overlap_score(query, candidate_row["combined_text"])

        # Compute recency score
        recent = recency_score(candidate_row["published_date"].year, min_year, max_year)

        # Compute hybrid score
        final_score = hybrid_score(
            semantic_similarity=float(score),
            category_bonus=False,
            keyword_overlap=overlap,
            recency_score=recent,
            w_category=0.0,
        )

        # Build a short explanation for why a paper was recommended
        explanation = build_explanation(
            {
                "semantic_similarity": float(score),
                "same_category": False,
                "keyword_overlap": overlap,
                "recency_score": recent,
            }
        )

        # Assemble final dataFrame
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

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("final_score", ascending=False).head(top_k)

    return results_df.reset_index(drop=True)
