import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Loading dataset into memory...")
def load_data_from_file(uploaded_file) -> pd.DataFrame:
    """
    Reads a CSV dataset into a Pandas DataFrame from a Streamlit file buffer.

    Utilizes Streamlit's caching mechanism to keep the dataset in memory
    across UI reruns, preventing redundant I/O operations.

    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The in-memory file buffer uploaded by the user via the Streamlit interface.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
                      Returns an empty DataFrame if the file cannot be parsed.

    Raises:
        pd.errors.EmptyDataError: If the provided CSV file is entirely empty.
        pd.errors.ParserError: If the file contains invalid CSV formatting.
        Exception: Catches generic unexpected errors during the file reading process.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid CSV.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        st.error("Failed to parse the file. Ensure it is a properly formatted CSV.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Fetching dataset from URL...")
def load_data_from_url(url: str) -> pd.DataFrame:
    """
    Fetches and loads a remote CSV dataset into a Pandas DataFrame via a URL.

    Results are cached to optimize performance and prevent repeated network
    requests when the user interacts with dashboard widgets.

    Args:
        url (str): The direct web link to the target CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the remote data.
                      Returns an empty DataFrame if the network request or parsing fails.

    Raises:
        ValueError: If the provided URL is malformed or lacks a valid schema.
        urllib.error.URLError: If the network request fails (e.g., unreachable host).
        pd.errors.ParserError: If the remote file is not a valid CSV format.
    """
    try:
        df = pd.read_csv(url)
        return df
    except ValueError:
        st.error(
            "Invalid URL format. Please ensure the link includes http:// or https://"
        )
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch or parse data from the provided URL: {e}")
        return pd.DataFrame()
