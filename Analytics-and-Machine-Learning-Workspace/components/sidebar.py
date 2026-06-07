"""Sidebar navigation components for the Streamlit application.

This module contains the primary navigation control used by the Predictive
Analytics Dashboard. The selected sidebar option determines which pipeline
module is rendered in the main application area.
"""

import streamlit as st


def render_main_navigation() -> str:
    """Render the main sidebar navigation menu.

    Displays the available workflow phases and returns the currently selected
    phase so the main application can route the user to the corresponding
    module.

    Returns:
        The selected pipeline phase.
    """
    st.sidebar.title("Navigation")

    active_phase = st.sidebar.radio(
        "Select Phase:",
        [
            "1. Data Ingestion",
            "2. ETL & Data Engineering",
            "3. Exploratory Data Analysis",
            "4. Geospatial Mapping",
            "5. Machine Learning",
            "6. Deep Learning",
        ],
    )

    st.sidebar.markdown("---")

    return active_phase
