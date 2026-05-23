import pandas as pd
import streamlit as st

from config import DATA_PATH, REQUIRED_COLUMNS


@st.cache_data
def load_data():
    """
    Loads the cleaned phone specification dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned phone specification dataframe.
    """

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]

    if missing_columns:
        st.error(f"Missing required dataset columns: {missing_columns}")
        st.stop()

    return data
