"""Dataset loading utilities for the Streamlit application.

This module provides cached helper functions for loading CSV datasets into
pandas DataFrames from either a user-uploaded file or a remote URL.

The functions are designed for interactive Streamlit workflows: successful
loads return populated DataFrames, while invalid inputs or parsing failures are
reported through the UI and return empty DataFrames to keep the application
running safely.
"""

from typing import Any

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Loading dataset into memory...")
def load_data_from_file(uploaded_file: Any) -> pd.DataFrame:
    """Load a CSV dataset from a Streamlit-uploaded file.

    Uses Streamlit caching to avoid repeated file parsing across interface
    reruns when the uploaded file has not changed.

    Args:
        uploaded_file: File-like object uploaded through the Streamlit interface.

    Returns:
        A DataFrame containing the loaded CSV data, or an empty DataFrame if
        the file is empty, malformed, or cannot be loaded.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is empty. Please upload a valid CSV.")
        return pd.DataFrame()

    except pd.errors.ParserError:
        st.error("❌ Failed to parse the file. Ensure it is a properly formatted CSV.")
        return pd.DataFrame()

    except Exception as error:
        st.error(f"❌ An unexpected error occurred while loading the file: {error}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Fetching dataset from URL...")
def load_data_from_url(url: str) -> pd.DataFrame:
    """Load a remote CSV dataset from a URL.

    Uses Streamlit caching to avoid repeated network requests and CSV parsing
    when the same URL is reused across interface reruns.

    Args:
        url: Direct URL pointing to a CSV file.

    Returns:
        A DataFrame containing the loaded remote CSV data, or an empty DataFrame
        if the URL is invalid, unreachable, or cannot be parsed as CSV.
    """
    try:
        df = pd.read_csv(url)
        return df

    except ValueError:
        st.error(
            "❌ Invalid URL format. Please ensure the link includes http:// or https://"
        )
        return pd.DataFrame()

    except Exception as error:
        st.error(f"❌ Failed to fetch or parse data from the provided URL: {error}")
        return pd.DataFrame()
