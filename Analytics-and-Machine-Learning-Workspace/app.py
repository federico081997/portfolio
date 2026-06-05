import streamlit as st
from components.auth import check_password

# Import our modular UI components
from components.sidebar import render_main_navigation
from components.ingestion_ui import render_data_ingestion_sidebar
from components.cleaning_ui import render_cleaning_module
from components.eda_ui import render_eda_module
from components.geospatial_ui import render_geospatial_module
from components.ml_ui import render_ml_module

# 1. Configure the global page layout
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """
    Initializes global session state variables to persist data across UI reruns.

    Ensures that the memory dictionary contains the necessary keys for passing
    data between the completely decoupled modules of the pipeline.
    """
    if "raw_data" not in st.session_state:
        st.session_state["raw_data"] = None
    if "cleaned_data" not in st.session_state:
        st.session_state["cleaned_data"] = None


def main() -> None:
    """
    Main entry point and traffic router for the Streamlit application.

    Delegates the rendering of the primary navigation menu to the sidebar module.
    Reads the user's selection and dynamically routes traffic, swapping both
    the sidebar toolsets and the main content area based on the active
    machine learning pipeline phase.
    """
    # Show login page and check user credentials
    if not check_password():
        st.stop()

    initialize_session_state()

    # 1. Retrieve the current phase from the decoupled sidebar navigation module
    active_phase = render_main_navigation()

    # Create a master container that completely resets on every click
    master_container = st.empty()

    # --- DYNAMIC ROUTING LOGIC ---
    with master_container.container():
        if active_phase == "1. Data Ingestion":
            # Render the separated upload tools in the sidebar
            render_data_ingestion_sidebar()

            # Render the main welcome screen
            st.title("Analytics & Machine Learning Workspace")
            st.markdown("""
            Welcome to the **Analytics & Machine Learning Workspace**—a fully modular, end-to-end data science platform designed to transform raw tabular data into production-ready predictive models.

            Engineered with a focus on mathematical rigor, numerical stability, and robust software architecture, this environment provides complete control over the machine learning lifecycle through an interactive, state-managed UI.

            ### Platform Architecture
            Navigate through the sidebar to access the five core analytical engines:

            * **Data Ingestion:** Securely load local data files or stream datasets directly from remote URLs into the session memory.
            * **ETL & Data Engineering:** A stateful, zero-leakage preprocessing studio for structural formatting, cardinality reduction, and memory footprint optimization.
            * **Exploratory Data Analysis:** Discover statistical anomalies, multidimensional collinearity, and non-linear relationships through interactive visualizations.
            * **Geospatial Mapping:** Render heavy spatial datasets safely with DOM-protected WebGL heatmaps, interactive clustering, and dynamic time-lapse animations.
            * **Predictive Modeling:** Construct mathematically sound Scikit-Learn pipelines featuring algorithmic bias reduction, float-safe skewness transformations, and comprehensive test-set diagnostics.
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
                    "👈 Please upload a CSV file or provide a URL in the sidebar to initialize the pipeline."
                )

        elif active_phase == "2. ETL & Data Engineering":
            # Routes to our ETL pipeline module
            render_cleaning_module()

        elif active_phase == "3. Exploratory Data Analysis":
            # Routes to our tabbed Plotly charts module
            render_eda_module()

        elif active_phase == "4. Geospatial Mapping":
            render_geospatial_module()

        elif active_phase == "5. Predictive Modeling":
            render_ml_module()


if __name__ == "__main__":
    main()
