import streamlit as st
from core.data_loader import load_data_from_file, load_data_from_url


def render_data_ingestion_sidebar() -> None:
    """
    Renders the data upload controls in the Streamlit sidebar.

    Provides the user with a radio toggle to choose between uploading a local
    CSV file or fetching a dataset from a remote URL. Handles the file upload
    buffer and URL string inputs.

    Side Effects:
        Modifies `st.session_state['raw_data']`: Upon successful data ingestion,
        the resulting pandas DataFrame is stored in the global session state
        for downstream modules to access.
    """
    st.sidebar.subheader("📥 Data Ingestion Tools")

    data_source = st.sidebar.radio(
        "Choose the method to provide dataset:", ("Upload File", "Provide Link")
    )

    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type=["csv"])
        if uploaded_file is not None:
            # Load from buffer and store in session state
            st.session_state["raw_data"] = load_data_from_file(uploaded_file)

    else:
        url = st.sidebar.text_input("Enter the URL to the CSV file:")
        if url:
            # Fetch from remote URL and store in session state
            st.session_state["raw_data"] = load_data_from_url(url)
