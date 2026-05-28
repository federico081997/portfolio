import pandas as pd


def load_and_clean_data(uploaded_file):
    """
    Safely loads, validates, and cleans the B2C shopping behavior dataset
    using the native column names.

    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The CSV or Excel file uploaded via the Streamlit interface.

    Returns:
        tuple: (bool, pd.DataFrame or str)
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            return (
                False,
                "Unsupported file format. Please upload a CSV or Excel (.xlsx) file.",
            )

        df.columns = df.columns.str.strip()

        # Gatekeeper using the exact column names
        required_columns = [
            "Age",
            "Gender",
            "Item Purchased",
            "Category",
            "Purchase Amount (USD)",
            "Review Rating",
            "Previous Purchases",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"

        # Cleaning
        df = df.dropna(subset=["Item Purchased"])
        df = df.drop_duplicates()

        # Ensure purchase amounts are strictly positive
        df = df[df["Purchase Amount (USD)"] > 0]

        # Text Standardization for cleaner visuals
        df["Item Purchased"] = df["Item Purchased"].astype(str).str.strip().str.title()
        df["Category"] = df["Category"].astype(str).str.strip().str.title()

        return True, df

    except Exception as e:
        return False, f"An unexpected error occurred during data processing: {str(e)}"
