"""
Customer Segmentation & Persona Analytics Dashboard
===================================================

This Streamlit application serves as the main orchestrator for a B2C machine learning pipeline.
It allows users to ingest raw demographic and behavioral transaction data, automatically cleans
the dataset, and provisions a suite of interactive Plotly dashboards.

Architecture highlights:
    - Modular Design: Backend ML logic and Plotly rendering are decoupled from the UI layer.
    - Stateful Routing: Utilizes Streamlit's session_state to prevent expensive recalculations
      during UI interactions and maintain the user's current view.
    - Defensive Programming: Implements strict gatekeeping for required columns and gracefully
      handles state errors (e.g., preventing chart rendering before the model is trained).
    - Machine Learning: Integrates an interactive Elbow Method diagnostic plot to allow the
      user to guide the hyperparameter selection (k) for the K-Means clustering algorithm.

Dependencies:
    - pandas: Data manipulation and feature engineering
    - scikit-learn: Data standardization (StandardScaler) and clustering (KMeans)
    - plotly: Interactive, WebGL-accelerated data visualization
    - streamlit: UI framework and session management
"""

import streamlit as st

# Import UI Components
from sidebar import (
    render_file_uploader,
    render_analysis_controls,
    render_cluster_filters,
)

# Import Backend Logic
from data_processing import load_and_clean_data
from ml_models import prepare_clustering_data, calculate_wcss, train_kmeans_model
from visualizations import (
    plot_age_distribution,
    plot_gender_category_sunburst,
    plot_subscription_status,
    plot_elbow_curve,
    plot_cluster_sunburst,
    plot_seasonal_trends,
    plot_purchase_frequency,
)

# Import Styling
from styles import inject_custom_css

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation", layout="wide", initial_sidebar_state="expanded"
)

# Apply the custom structural styling
inject_custom_css()

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
# This prevents the app from reloading data/models every time a user interacts
if "df" not in st.session_state:
    st.session_state["df"] = None
if "model_built" not in st.session_state:
    st.session_state["model_built"] = False
if "current_file" not in st.session_state:
    st.session_state["current_file"] = None
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "overview"

# -----------------------------------------------------------------------------
# 3. Sidebar UI & Data Ingestion
# -----------------------------------------------------------------------------
uploaded_file = render_file_uploader()

# Handle a new file upload and reset the state
if uploaded_file is not None:
    if st.session_state["current_file"] != uploaded_file.name:
        with st.spinner("Validating and cleaning dataset..."):
            is_success, result = load_and_clean_data(uploaded_file)

            if is_success:
                st.session_state["df"] = result
                st.session_state["model_built"] = False
                st.session_state["current_file"] = uploaded_file.name
                st.session_state["current_view"] = "overview"
                st.sidebar.success("Dataset ready for analysis!")
            else:
                st.sidebar.error("Data Validation Failed")
                st.error(result)
                st.stop()
else:
    # If file is removed, clear the state completely
    st.session_state["df"] = None
    st.session_state["model_built"] = False
    st.session_state["current_file"] = None
    st.session_state["current_view"] = "overview"

# -----------------------------------------------------------------------------
# 4. Main Application Orchestration
# -----------------------------------------------------------------------------
if st.session_state["df"] is not None:
    df = st.session_state["df"]

    # --- Top-Level Dashboard Metrics ---
    st.title("Customer Segmentation Analysis")
    st.markdown("Analyze transaction history to identify high-value customer personas.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")

    unique_cust = (
        df["Customer ID"].nunique() if "Customer ID" in df.columns else len(df)
    )
    col2.metric("Unique Customers", f"{unique_cust:,}")

    col3.metric("Global Revenue", f"${df['Purchase Amount (USD)'].sum():,.2f}")

    locations = df["Location"].nunique() if "Location" in df.columns else 0
    col4.metric("Regions Reached", f"{locations}")

    st.markdown("---")

    # --- Analysis Controls & Global Filtering ---
    controls = render_analysis_controls()

    # Create the Global Filter here so it applies to ALL views
    if st.session_state["model_built"] and "Cluster" in df.columns:
        selected_clusters = render_cluster_filters(df)
        if selected_clusters:
            # Create a globally filtered dataframe
            filtered_df = df[df["Cluster"].isin(selected_clusters)].copy()
        else:
            # Fallback if the user unchecks all boxes
            filtered_df = df[df["Cluster"] == "NONE"].copy()
    else:
        # If the model isn't built yet, the filtered_df is just the raw df
        filtered_df = df
        selected_clusters = []

    # 1. Update the state when a button is clicked
    if controls["demographics"]:
        st.session_state["current_view"] = "demographics"
    elif controls["revenue"]:
        st.session_state["current_view"] = "revenue"
    elif controls["build_model"]:
        st.session_state["current_view"] = "build_model"

    # 2. Render the top-level UI based on the securely saved state
    if st.session_state["current_view"] == "overview":
        st.subheader("Dataset Preview")
        st.markdown(
            "Explore the cleaned B2C dataset. Scroll vertically and horizontally to view all features."
        )
        st.dataframe(df, width="stretch", height=400)

    elif st.session_state["current_view"] == "demographics":
        st.subheader("Demographic Breakdown by Persona")

        # Guard Clause for missing model or no selection
        if "Cluster" not in df.columns:
            st.warning(
                "⚠️ Please go to 'Build KMeans Model' in the sidebar and train the model first."
            )
        elif not selected_clusters:
            st.info("Please select at least one cluster from the sidebar.")
        else:
            st.plotly_chart(plot_age_distribution(filtered_df), width="stretch")

    elif st.session_state["current_view"] == "revenue":
        st.subheader("Revenue Drivers across Demographics")
        if st.session_state["model_built"] and not selected_clusters:
            st.info("Please select at least one cluster from the sidebar.")
        else:
            st.plotly_chart(plot_gender_category_sunburst(filtered_df), width="stretch")

    elif st.session_state["current_view"] == "build_model":
        st.subheader("Behavioral Segmentation (K-Means Clustering)")

        with st.spinner(
            "Extracting behavioral features and calculating the optimal k..."
        ):
            df_for_model, scaled_features = prepare_clustering_data(df)
            k_values, wcss = calculate_wcss(scaled_features)

        st.markdown("**Step 1: Analyze the Elbow Plot**")
        st.plotly_chart(plot_elbow_curve(k_values, wcss), width="stretch")

        st.markdown("**Step 2: Select 'k' and Train Final Model**")
        chosen_k = st.slider(
            "Select the number of clusters (k):",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
        )

        if st.button("Train Final Model with Selected k", type="secondary"):
            with st.spinner(f"Training model with k={chosen_k}..."):
                df = train_kmeans_model(
                    df_for_model, scaled_features, n_clusters=chosen_k
                )
                st.session_state["df"] = df
                st.session_state["model_built"] = True
                st.rerun()

    # --- Persistent Persona Deep-Dive View ---
    if st.session_state["model_built"]:
        st.markdown("---")
        st.subheader("Persona Deep-Dive Analysis")

        if selected_clusters:
            st.markdown("#### 1. Loyalty & Retention")
            st.plotly_chart(plot_subscription_status(filtered_df), width="stretch")
            st.plotly_chart(plot_purchase_frequency(filtered_df), width="stretch")

            st.markdown("#### 2. Purchasing Behavior")
            st.plotly_chart(plot_seasonal_trends(filtered_df), width="stretch")
            st.plotly_chart(
                plot_cluster_sunburst(filtered_df, selected_clusters), width="stretch"
            )
        else:
            st.info(
                "Please select at least one cluster from the sidebar to view the persona breakdown."
            )

else:
    # --- Landing State (Before Upload) ---
    st.title("Customer Persona Workspace")
    st.info(
        "Awaiting dataset. Please use the secure sidebar upload tool to ingest your B2C data."
    )

    st.markdown("""
    ### Expected Data Schema:
    * `Age`: Customer age
    * `Gender`: Customer gender
    * `Purchase Amount (USD)`: Transaction value
    * `Category` & `Item Purchased`: Product details
    * `Review Rating`: Customer satisfaction score
    * `Previous Purchases`: Historical loyalty metric
    """)
