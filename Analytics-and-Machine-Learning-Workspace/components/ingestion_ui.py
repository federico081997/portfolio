"""Streamlit UI components for dataset ingestion.

This module renders the sidebar controls used to load datasets into the
Analytics and Machine Learning Workspace. Users can upload a local CSV file or
provide a direct URL to a remote CSV resource.

Each data source is assigned a signature so that newly supplied datasets can be
detected across Streamlit reruns. When a new dataset is loaded, downstream
analytical session state is cleared while authentication status and essential
ingestion state are preserved.
"""

import streamlit as st

from core.data_loader import (
    load_data_from_file,
    load_data_from_url,
)


def clear_all_cache_except_essentials() -> None:
    """Clear downstream session state while preserving essential application data.

    Retains the active dataset signature, raw dataset, and authentication status.
    All other session-state entries are removed to prevent results from a
    previously loaded dataset from propagating into downstream modules.
    """
    essential_keys = {
        "current_data_signature",
        "raw_data",
        "password_correct",
    }

    for key in list(st.session_state.keys()):
        if key not in essential_keys:
            del st.session_state[key]


def render_data_ingestion_sidebar() -> None:
    """Render sidebar controls for loading and replacing datasets.

    Detects newly uploaded files or URL resources using data-source signatures.
    When a new source is detected, the dataset is loaded into
    ``st.session_state["raw_data"]`` and downstream analytical state is reset.
    """
    st.sidebar.subheader("📥 Data Ingestion Tools")

    data_source = st.sidebar.radio(
        "Choose the method to provide dataset:",
        ("Upload File", "Provide Link"),
    )

    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file:",
            type=["csv"],
        )

        if uploaded_file is not None:
            file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

            if st.session_state.get("current_data_signature") != file_signature:
                with st.spinner(
                    "Ingesting new dataset and clearing previous results..."
                ):
                    try:
                        st.session_state["current_data_signature"] = file_signature
                        st.session_state["raw_data"] = load_data_from_file(
                            uploaded_file
                        )

                        clear_all_cache_except_essentials()
                        st.rerun()

                    except Exception as error:
                        st.sidebar.error(f"Failed to load file: {error}")

    else:
        url = st.sidebar.text_input("Enter the URL to the CSV file:")

        if url and st.session_state.get("current_data_signature") != url:
            with st.spinner("Downloading dataset and clearing previous results..."):
                try:
                    st.session_state["current_data_signature"] = url
                    st.session_state["raw_data"] = load_data_from_url(url)

                    clear_all_cache_except_essentials()
                    st.rerun()

                except Exception as error:
                    st.sidebar.error(f"Failed to download URL: {error}")
