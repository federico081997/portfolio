import streamlit as st


def render_main_navigation() -> str:
    """
    Renders the primary navigation menu in the Streamlit sidebar.

    Returns:
        str: The string value of the currently selected pipeline phase.
    """
    st.sidebar.title("Pipeline Navigation")

    active_phase = st.sidebar.radio(
        "Select Phase:",
        [
            "1. Data Ingestion",
            "2. ETL & Data Engineering",
            "3. Exploratory Data Analysis",
            "4. Geospatial Mapping",
            "5. Predictive Modeling",
        ],
    )

    st.sidebar.markdown("---")

    return active_phase
