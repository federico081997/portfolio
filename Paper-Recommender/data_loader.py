from pathlib import Path
import pandas as pd


def load_processed_data() -> pd.DataFrame:
    """
    Load the cleaned dataset from the processed folder
    """
    # Define cleaned data directory
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "processed" / "arxiv_cleaned.csv"

    # Check if the cleaned file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {data_path}")

    # Load the cleaned dataset
    df = pd.read_csv(data_path)

    # Double-check the required columns are present in the dataframe
    required_columns = [
        "title",
        "abstract",
        "category",
        "published_date",
        "combined_text",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert date column
    df["published_date"] = pd.to_datetime(df["published_date"])

    return df
