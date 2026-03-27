from pathlib import Path
import ast

import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean a text value with minimal preprocessing.

    This function:
    - converts the input to a string
    - replaces line breaks with spaces
    - removes extra whitespace

    Args:
        text: Input text value.

    Returns:
        A cleaned string. If the input is missing, an empty string is returned.
    """
    if pd.isna(text):
        return ""

    # Convert the value to string in case it is not already text.
    text = str(text)

    # Replace line breaks with spaces to keep the text in one line.
    text = text.replace("\n", " ")

    # Collapse repeated whitespace into single spaces.
    text = " ".join(text.split())

    return text


def parse_authors(authors):
    """
    Convert the authors field into a human-readable string.

    If the value is a string representation of a Python list, it is parsed
    and joined into a comma-separated string. If parsing fails, the original
    string is returned unchanged.

    Args:
        authors: Authors value from the dataset.

    Returns:
        A readable string representation of the authors field.
    """
    if isinstance(authors, str):
        try:
            authors = ast.literal_eval(authors)
        except (ValueError, SyntaxError):
            return authors

    if isinstance(authors, list):
        return ", ".join(authors)

    return str(authors)


def main():
    """
    Load the raw arXiv dataset, clean selected fields, and save the processed file.

    Processing steps:
    - load the raw CSV file
    - rename the summary column to abstract
    - keep only the relevant columns
    - remove rows with missing title or abstract
    - remove rows with very short abstracts
    - clean title and abstract text
    - remove duplicate title/abstract pairs
    - convert published_date to datetime
    - format authors as a readable string
    - create a combined_text column for semantic search
    - save the cleaned dataset to the processed folder
    """
    # Define the project root directory.
    project_root = Path(__file__).parent

    # Define input and output file paths.
    raw_path = project_root / "data" / "raw" / "arxiv_papers.csv"
    processed_path = project_root / "data" / "processed" / "arxiv_cleaned.csv"

    # Ensure the raw dataset exists before attempting to load it.
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    # Load the raw dataset.
    df = pd.read_csv(raw_path)

    print("\nDataset loaded")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    # Rename the summary column to abstract for consistency.
    df = df.rename(columns={"summary": "abstract"})

    # Keep only the columns needed for the project.
    df = df[["id", "title", "abstract", "category", "authors", "published_date"]]

    # Remove rows missing title or abstract, since they are not useful for search.
    df = df.dropna(subset=["title", "abstract"])

    # Remove rows with very short abstracts, which are likely poor-quality samples.
    df = df[df["abstract"].str.len() > 100]

    # Clean text fields.
    df["title"] = df["title"].apply(clean_text)
    df["abstract"] = df["abstract"].apply(clean_text)

    # Remove duplicate records based on title and abstract.
    df = df.drop_duplicates(subset=["title", "abstract"])

    # Convert published_date to datetime format.
    df["published_date"] = pd.to_datetime(df["published_date"], format="mixed")

    # Convert authors into a readable string format.
    df["authors"] = df["authors"].apply(parse_authors)

    # Create a combined text field for downstream semantic search tasks.
    df["combined_text"] = df["title"] + " " + df["abstract"]

    # Reset the dataframe index after filtering and deduplication.
    df = df.reset_index(drop=True)

    print("\nCleaned dataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    # Save the cleaned dataset.
    df.to_csv(processed_path, index=False)

    print("\nCleaned dataset saved to:")
    print(processed_path)


if __name__ == "__main__":
    main()
