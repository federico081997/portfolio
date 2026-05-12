from pathlib import Path

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


def main():
    """
    Load the raw SMS dataset, clean selected fields, and save the processed file.

    Processing steps:
    - load the raw CSV file
    - rename Message and spamORham columns to "message" and "label"
    - keep only the relevant columns
    - clean message text by removing repeated whitespaces and replacing \n with " "
    - one-hot encode the "label" column: ham --> 0, spam --> 1
    - save the cleaned dataset to the processed folder
    """
    # Define the project root directory.
    base_dir = Path(__file__).parent

    # Define input and output file paths.
    raw_path = base_dir / "data" / "raw" / "sms_dataset.csv"
    processed_path = base_dir / "data" / "processed" / "sms_cleaned.csv"

    # Ensure the raw dataset exists before attempting to load it.
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    # Load the raw dataset.
    df = pd.read_csv(raw_path)

    print("\nDataset loaded")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    # Rename the the Message and spamORham columns for consistency.
    df = df.rename(columns={"Message": "message", "spamORham": "label"})

    # Keep only the columns needed for the project.
    df = df[["message", "label"]]

    # Clean message text by removing repeated whitespaces and replacing \n with " "
    df["label"] = df["label"].apply(clean_text)

    # One-hot encode the "label" column: ham --> 0, spam --> 1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Reset the dataframe index.
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
