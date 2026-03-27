import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

from ollama_labeling import generate_labels_with_ollama


def tokenize_for_labels(text: str) -> list[str]:
    """
    Cleaner tokenizer for cluster labeling.

    Keeps:
    - alphabetic words
    - hyphenated alphabetic terms like finite-volume, navier-stokes

    Removes:
    - pure numbers
    - years
    - alphanumeric tokens like 16gb, 13b, 128k
    - short acronyms like ml, ai, cnn, gpu
    """
    # Define stopwords
    STOPWORDS = set(ENGLISH_STOP_WORDS)

    # Convert text to lowercase
    text = str(text).lower()

    # Define regex patten of tokenizer
    pattern = r"(?u)\b[a-z]+(?:[-–][a-z]+)*\b"

    # Extract raw tokens
    raw_tokens = re.findall(pattern, text)

    final_tokens = []

    # Add token to the final list
    for token in raw_tokens:
        # Keep full tokens meaningful
        if token not in STOPWORDS and len(token) >= 3:
            final_tokens.append(token)

        # Split hyphenated terms only if needed
        if "-" in token or "–" in token:
            parts = re.split(r"[-–]", token)
            for part in parts:
                if part not in STOPWORDS and len(part) >= 3:
                    final_tokens.append(part)

    return final_tokens


def build_cluster_documents(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    text_col: str = "combined_text",
    max_docs_per_cluster: int | None = 300,
) -> tuple[list[int], list[str]]:
    # Sort the papers in by cluster
    cluster_ids = sorted(df[cluster_col].unique())

    cluster_docs = []
    for cluster_id in cluster_ids:
        cluster_series = df.loc[df[cluster_col] == cluster_id, text_col]
        if (
            max_docs_per_cluster is not None
            and len(cluster_series) > max_docs_per_cluster
        ):
            cluster_series = cluster_series.sample(
                n=max_docs_per_cluster, random_state=42
            )

        tokens = []
        for text in cluster_series.astype(str):
            tokens.extend(tokenize_for_labels(text))
        cluster_docs.append(" ".join(tokens))

    return cluster_ids, cluster_docs


def has_repeated_words(
    term: str,
) -> bool:
    """Return True if a term contains the same word twice in a row.
    Example: learning learning
    """
    words = term.split()
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            return True
    return False


def is_too_similar_to_selected(term: str, selected_terms: list[str]) -> bool:
    """
    Return True if this term is too similar to one of the selected terms
    Example:
     - term = 'graph neural'
     - selected = 'graph neural network'
    """
    term_words = set(term.split())
    for selected in selected_terms:
        selected_words = set(selected.split())

        # Rule 1: Exact containment
        if term_words.issubset(selected_words):
            return True

        # Rule 2: Strong overlap
        overlap = len(term_words & selected_words)
        similarity_ratio = overlap / max(len(term_words), len(selected_words))
        if similarity_ratio >= 0.5:
            return True

    return False


def get_top_terms_per_cluster(
    cluster_docs: list[str],
    top_n_words: int = 5,
) -> list[list[str]]:
    """
    Compute TF-IDF across cluster documents and return top terms for each cluster.
    """
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-z]+\b",
        lowercase=False,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(cluster_docs)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Get top terms per cluster
    top_terms_by_cluster = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().ravel()

        # If all values in on row are zeros then define the cluster as "Unknown"
        if np.all(row == 0):
            top_terms_by_cluster.append(["Unknown"])
            continue

        sorted_indices = np.argsort(row)[::-1]

        # Append top words to the list
        selected_terms = []
        for idx in sorted_indices:
            term = feature_names[idx]
            # Skip bad terms like "learning learning"
            if has_repeated_words(term):
                continue

            # Skip if already contained inside a selected longer term
            if is_too_similar_to_selected(term, selected_terms):
                continue

            selected_terms.append(term)

            if len(selected_terms) == top_n_words:
                break

        if not selected_terms:
            selected_terms = ["Unknown"]

        top_terms_by_cluster.append(selected_terms)

    return top_terms_by_cluster


def label_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    text_col: str = "combined_text",
    top_n_words: int = 5,
    max_docs_per_cluster: int | None = 300,
    use_ollama: bool = True,
    ollama_batch_size: int = 10,
    ollama_model: str = "llama3.2",
    host: str = "http://localhost:11434",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create automatic labels for each cluster using TF-IDF keywords
    and optionally refine them into human-readable labels using Ollama.
    """
    # Create a copy of original dataset
    labeled_df = df.copy()

    # Build large cluster documents
    cluster_ids, cluster_docs = build_cluster_documents(
        df=labeled_df,
        cluster_col=cluster_col,
        text_col=text_col,
        max_docs_per_cluster=max_docs_per_cluster,
    )

    # Obtain top terms by cluster
    top_terms_by_cluster = get_top_terms_per_cluster(
        cluster_docs=cluster_docs, top_n_words=top_n_words
    )

    # Map cluster_id -> list of top keyword terms
    cluster_keyword_list_map = {
        cluster_id: top_terms
        for cluster_id, top_terms in zip(cluster_ids, top_terms_by_cluster)
    }

    # Map cluster_id -> keyword string
    cluster_keywords_map = {
        cluster_id: ", ".join(top_terms)
        for cluster_id, top_terms in cluster_keyword_list_map.items()
    }

    # Generate human-readable labels with Ollama
    if use_ollama:
        cluster_label_map = generate_labels_with_ollama(
            clusters_keywords=cluster_keyword_list_map,
            batch_size=ollama_batch_size,
            model=ollama_model,
            host=host,
            verbose=True,
        )
    else:
        # Fallback: use first few keywords as label
        cluster_label_map = {
            cluster_id: ", ".join(top_terms[:3])
            for cluster_id, top_terms in cluster_keyword_list_map.items()
        }

    # Add cluster keyword and label columns to original dataframe
    labeled_df["cluster_keywords"] = labeled_df[cluster_col].map(cluster_keywords_map)
    labeled_df["cluster_label"] = labeled_df[cluster_col].map(cluster_label_map)

    # Add cluster label columns to dataframe
    cluster_rows = []
    for cluster_id in cluster_ids:
        cluster_size = int((labeled_df[cluster_col] == cluster_id).sum())

        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": cluster_label_map.get(cluster_id, "Unknown Topic"),
                "cluster_keywords": cluster_keywords_map.get(cluster_id, "Unknown"),
                "cluster_size": cluster_size,
            }
        )

    # Create a cluster summary dataframe
    cluster_summary = (
        pd.DataFrame(cluster_rows)
        .sort_values(
            by="cluster_size",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return labeled_df, cluster_summary
