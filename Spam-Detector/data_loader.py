from pathlib import Path

import pandas as pd


def load_processed_data() -> pd.DataFrame:
    """
    Load the cleaned SMS dataset from the processed folder.

    This function:
    - locates the processed dataset file
    - verifies that it exists
    - loads it into a pandas DataFrame
    - checks for required columns

    Returns:
        pd.DataFrame: The cleaned dataset ready for downstream use.

    Raises:
        FileNotFoundError: If the processed dataset file is not found.
        ValueError: If required columns are missing from the dataset.
    """
    # Define the project root and path to the processed dataset.
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "processed" / "sms_cleaned.csv"

    # Ensure the processed dataset exists before loading.
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {data_path}")

    # Load the dataset into a DataFrame.
    df = pd.read_csv(data_path)

    # Define the columns required for the application to function correctly.
    required_columns = ["message", "label"]

    # Identify any missing required columns.
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    return df
