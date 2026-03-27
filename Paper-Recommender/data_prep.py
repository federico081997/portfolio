from pathlib import Path
import pandas as pd
import ast


def clean_text(text: str) -> str:
    """
    Simple text cleaning

    - converts to string
    - removes line breaks
    - removes extra spaces
    """
    if pd.isna(text):
        return ""

    # 1. Converts text to string
    text = str(text)

    # 2. Removs line breaks
    text = text.replace("\n", " ")

    # 3. Removes extra spaces
    text = " ".join(text.split())

    return text


def parse_authors(authors):
    """Convert authors into a human-readable string"""
    if isinstance(authors, str):
        try:
            authors = ast.literal_eval(authors)
        except (ValueError, SyntaxError):
            return authors

    if isinstance(authors, list):
        return ", ".join(authors)

    return str(authors)


def main():
    # Project root folder
    project_root = Path(__file__).parent

    # File paths
    raw_path = project_root / "data" / "raw" / "arxiv_papers.csv"
    processed_path = project_root / "data" / "processed" / "arxiv_cleaned.csv"

    # Check if the raw file exists
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    # Load the raw dayaset
    df = pd.read_csv(raw_path)

    print("\nDataset loaded")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    # Rename summary column to abstract
    df = df.rename(columns={"summary": "abstract"})

    # Select useful columns
    df = df[["id", "title", "abstract", "category", "authors", "published_date"]]

    # Remove missing title or abstract as it would not be useful for semantic research
    df = df.dropna(subset=["title", "abstract"])

    # Remove rows with very small abstracts (bad samples)
    df = df[df["abstract"].str.len() > 100]

    # Clean text
    df["title"] = df["title"].apply(clean_text)
    df["abstract"] = df["abstract"].apply(clean_text)

    # Remove duplicates
    df = df.drop_duplicates(subset=["title", "abstract"])

    # Convert published_date to datetime format
    df["published_date"] = pd.to_datetime(df["published_date"], format="mixed")

    # Convert authors into a properly formatted string
    df["authors"] = df["authors"].apply(parse_authors)

    # Create a combined title + abstract column, useful for semantic search
    df["combined_text"] = df["title"] + " " + df["abstract"]

    # Reset index
    df = df.reset_index(drop=True)

    print("\nCleaned dataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    # Save cleaned dataset
    df.to_csv(processed_path, index=False)

    print("\nCleaned dataset saved to:")
    print(processed_path)


if __name__ == "__main__":
    main()
