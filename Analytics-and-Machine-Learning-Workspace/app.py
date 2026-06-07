"""Main Streamlit application entry point.

This module configures and launches the Analytics and Machine Learning Workspace. It
coordinates authentication, initializes global session-state variables, and
routes the user interface to the appropriate pipeline module based on the
selected workflow phase.

The application is organized as a modular analytics workspace with separate
components for data ingestion, ETL and data engineering, exploratory data
analysis, geospatial visualization, and predictive modeling.
"""

import streamlit as st

from components.auth import check_password
from components.cleaning_ui import render_cleaning_module
from components.eda_ui import render_eda_module
from components.geospatial_ui import render_geospatial_module
from components.ingestion_ui import render_data_ingestion_sidebar
from components.ml_ui import render_ml_module
from components.dl_ui import render_dl_module
from components.sidebar import render_main_navigation

st.set_page_config(
    page_title="Analytics and Machine Learning Workspace",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize global session-state variables used across the application.

    Ensures that the main data containers exist before any pipeline module is
    rendered. These keys allow independent UI modules to share raw and cleaned
    datasets across Streamlit reruns.
    """
    if "raw_data" not in st.session_state:
        st.session_state["raw_data"] = None

    if "cleaned_data" not in st.session_state:
        st.session_state["cleaned_data"] = None


def main() -> None:
    """Run the Streamlit application and route users to the active module.

    Handles authentication, initializes shared application state, renders the
    main sidebar navigation, and dispatches the selected workflow phase to the
    corresponding UI module.
    """
    if not check_password():
        st.stop()

    initialize_session_state()

    active_phase = render_main_navigation()

    master_container = st.empty()

    with master_container.container():
        if active_phase == "1. Data Ingestion":
            render_data_ingestion_sidebar()

            st.title("Analytics & Machine Learning Workspace")
            st.markdown("""
                Welcome to the **Analytics & Machine Learning Workspace** — a
                modular, end-to-end data science platform for transforming raw
                tabular data into structured analyses, visual insights, and
                predictive modeling workflows.

                The platform is designed around a state-managed interface, where
                each stage of the machine learning lifecycle is handled by a
                dedicated module.

                ### Platform Architecture

                Use the sidebar to access the five core analytical modules:

                * **Data Ingestion:** Load local data files or retrieve datasets
                  from remote URLs into session memory.
                * **ETL & Data Engineering:** Clean, transform, format, and
                  prepare datasets for downstream analysis.
                * **Exploratory Data Analysis:** Investigate distributions,
                  missing values, statistical patterns, feature relationships,
                  and potential anomalies.
                * **Geospatial Mapping:** Visualize spatial data using
                  interactive maps, clustering, heatmaps, and time-aware
                  geospatial views.
                * **Predictive Modeling:** Build Scikit-Learn workflows for
                  supervised machine learning, model comparison, and diagnostic
                  evaluation.
                """)
            st.markdown("---")

            if (
                st.session_state["raw_data"] is not None
                and not st.session_state["raw_data"].empty
            ):
                st.success("Dataset loaded successfully!")
                st.subheader("Raw Dataset Preview")
                st.dataframe(st.session_state["raw_data"].head(20), width="stretch")
            else:
                st.info(
                    "👈 Please upload a CSV file or provide a URL in the sidebar "
                    "to initialize the pipeline."
                )

        elif active_phase == "2. ETL & Data Engineering":
            render_cleaning_module()

        elif active_phase == "3. Exploratory Data Analysis":
            render_eda_module()

        elif active_phase == "4. Geospatial Mapping":
            render_geospatial_module()

        elif active_phase == "5. Machine Learning":
            render_ml_module()

        elif active_phase == "6. Deep Learning":
            render_dl_module()


if __name__ == "__main__":
    main()
