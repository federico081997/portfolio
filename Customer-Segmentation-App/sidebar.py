import streamlit as st
import pandas as pd


def render_file_uploader():
    """
    Renders the file upload widget for data ingestion.

    Returns:
        streamlit.runtime.uploaded_file_manager.UploadedFile or None:
        The uploaded CSV or Excel file object, or None if no file is uploaded.
    """
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx"],
        help="Limit 200MB. Ensure the file contains B2C demographic and behavioral data.",
    )
    return uploaded_file


def render_analysis_controls() -> dict:
    """
    Renders the core action buttons that trigger different analysis phases.

    Utilizes 'width="stretch"' to ensure buttons scale elegantly
    regardless of the user's screen size or sidebar width.

    Returns:
        dict: A dictionary mapping control names to their boolean click states.
    """
    st.sidebar.markdown("### Analysis Options")

    controls = {
        "demographics": st.sidebar.button(
            "Age & Persona Demographics", width="stretch"
        ),
        "revenue": st.sidebar.button(
            "Revenue Drivers (Gender/Category)", width="stretch"
        ),
        "build_model": st.sidebar.button(
            "Build K-Means Model", type="primary", width="stretch"
        ),
    }

    return controls


def render_cluster_filters(df: pd.DataFrame) -> list:
    """
    Renders interactive multi-select filters for exploring specific clusters.

    This function dynamically extracts the unique cluster labels generated
    by the machine learning model and allows the user to filter the downstream
    visualizations. It defaults to selecting all available clusters.

    Args:
        df (pd.DataFrame): The dataset containing the newly generated 'Cluster' column.

    Returns:
        list: A list of the string cluster labels selected by the user.

    Raises:
        KeyError: If the 'Cluster' column does not exist in the provided DataFrame.
    """
    if "Cluster" not in df.columns:
        raise KeyError("Cannot render cluster filters: 'Cluster' column is missing.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Cluster Deep-Dive")

    # Extract unique clusters and sort them for a clean UI dropdown
    available_clusters = sorted(df["Cluster"].unique())

    selected_clusters = st.sidebar.multiselect(
        "Select Clusters to Show:",
        options=available_clusters,
        default=available_clusters,
    )

    return selected_clusters
