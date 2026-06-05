import streamlit as st
from core.visualizer import (
    plot_distribution,
    plot_categorical_variance,
    plot_relationship,
    plot_time_series,
    plot_scatter_matrix,
    plot_correlation_heatmap,
    plot_hierarchical_sunburst,
)


def render_eda_module() -> None:
    """
    Renders the primary Exploratory Data Analysis (EDA) dashboard interface.

    This function securely pulls structurally cleaned data from the active session state (`st.session_state["cleaned_data"]`),
    ensuring that all generated Plotly visualizers accurately reflect the user's ETL preprocessing.
    The interface is strictly compartmentalized into 6 tabbed analytical workflows:
    1. 1D Distributions (Histograms & Bar Charts)
    2. Categorical Variance (Box & Violin Plots)
    3. 2D Relationships (Scatter plots with trendlines)
    4. Temporal Dynamics (Time Series with Moving Averages)
    5. High-Dimensionality (Correlation Heatmaps & Scatter Matrices)
    6. Segmentation (Hierarchical Sunburst Charts)

    It utilizes `st.form` to batch user input, preventing Streamlit from rerunning the entire script
    (and freezing the UI) every time a single dropdown is changed.

    Args:
        None. All data is retrieved dynamically via `st.session_state`.

    Returns:
        None. The function directly renders Streamlit UI components to the browser.

    Raises/Warnings:
        Warning (UI): Displays a warning if no cleaned data is found in the session state.
        Error (UI): Displays an error if the retrieved dataframe is unexpectedly empty.
        Info (UI): Displays contextual information if specific tabs lack the required data types
                   (e.g., trying to render a correlation matrix with fewer than 2 numerical columns).
    """

    st.empty()

    st.title("Exploratory Data Analysis (EDA) Studio")
    st.markdown("""
    Welcome to the **Exploratory Data Analysis (EDA) Studio**—a visual analytics engine designed for deep multidimensional data exploration and statistical pattern recognition.

    This module transforms raw data into actionable insights through specialized analytical workflows:
    * **Feature Distributions & Variance:** Identify statistical anomalies, skewness, and probability densities using histograms and violin plots.
    * **Relationship & Trend Analysis:** Uncover correlations and non-linear patterns using OLS and LOWESS trendlines with marginal distributions.
    * **Temporal Dynamics:** Track chronological evolution and extract underlying macroeconomic trends via rolling moving averages.
    * **High-Dimensional Topographies:** Map feature collinearity and spatial clusters using interactive correlation heatmaps and multidimensional scatter matrices.
    * **Hierarchical Segmentation:** Drill down into categorical taxonomies and market segments using dynamic sunburst architectures.
    """)
    st.markdown("---")

    # 1. Try to load the Cleaned Data first
    if (
        "cleaned_data" in st.session_state
        and st.session_state["cleaned_data"] is not None
    ):
        df = st.session_state["cleaned_data"]
        # Subtle indicator to the user
        st.caption(
            "✨ Using processed data from the Data Cleaning & Preprocessing Studio."
        )

    # 2. Fallback to Raw Data if Cleaning Studio was skipped
    elif "raw_data" in st.session_state and st.session_state["raw_data"] is not None:
        df = st.session_state["raw_data"]
        st.info(
            "ℹ️ **Notice:** Using the raw uploaded dataset. For optimal ML performance, consider processing this data in the Data Cleaning & Preprocessing Studio first."
        )

    # 3. Halt if absolutely no data exists
    else:
        st.warning("⚠️ No data available. Please upload a dataset first.")
        return

    # 4. Final safety check
    if df.empty:
        st.error(
            "The selected dataset is empty. Please check your data source or cleaning steps."
        )
        return

    # Parse Data Types for UI Routing
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(
        include=["float64", "int64", "float32", "int32"]
    ).columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    time_cols = df.select_dtypes(
        include=["datetime", "datetime64[ns]"]
    ).columns.tolist()

    t_dist, t_var, t_rel, t_time, t_multi, t_seg = st.tabs(
        [
            "📊 Distributions",
            "⚔️ Variance",
            "📈 Relationships",
            "⏱️ Time Series",
            "🌐 High-Dimensional",
            "🍩 Segmentation",
        ]
    )

    # --- TAB 1: Distributions ---
    with t_dist:
        st.subheader("Feature Distributions")

        # UX Improvement: Educational Info Box
        with st.expander("ℹ️ Guide: Understanding 1D Distributions"):
            st.markdown("""
                **How to read these charts:**
                * **Histograms (Numerical Data):** Show the frequency of data within specific ranges (bins). Look for normal curves (bell shapes) or skewness (long tails to the left or right).
                * **Box Plots (Numerical Marginal):** The box shows the middle 50% of the data (Interquartile Range). The line inside the box is the median. Dots outside the whiskers are mathematical outliers.
                * **Bar Charts (Categorical Data):** Show the exact count of rows belonging to specific text categories.
                * **Log Scale:** If a chart is heavily skewed (e.g., one bar is 1,000,000 and another is 10), checking the 'Logarithmic Y-Axis' box will compress the scale so you can see all bars clearly.
                """)

        with st.form("dist_form"):
            c1, c2 = st.columns(2)
            with c1:
                dist_col = st.selectbox("Analyze Feature:", all_cols)
                dist_color = st.selectbox("Group By (Color):", ["None"] + cat_cols)
            with c2:
                bins = st.slider("Number of Bins (for numerical features):", 5, 100, 30)
                log_scale = st.checkbox(
                    "Logarithmic Y-Axis", help="Useful for heavily skewed data."
                )

            submitted = st.form_submit_button(
                "Generate Plot", type="primary", use_container_width=True
            )

        if submitted:
            color_arg = None if dist_color == "None" else dist_color
            fig_dist = plot_distribution(df, dist_col, color_arg, bins, log_scale)
            st.session_state["saved_distribution_plot"] = fig_dist

        # Always render if it exists in state
        if "saved_distribution_plot" in st.session_state:
            st.plotly_chart(
                st.session_state["saved_distribution_plot"],
                use_container_width=True,
            )
        else:
            st.info("👆 Configure your parameters and click **Generate Plot**.")

    # --- TAB 2: Categorical Variance ---
    with t_var:
        st.subheader("Categorical Variance")
        with st.expander("ℹ️ Guide: Understanding Variance Plots"):
            st.markdown("""
                **How to read these charts:**
                * **Box Plots:** Excellent for spotting statistical outliers. The "box" represents where the middle 50% of the data lies. The line inside is the median.
                * **Violin Plots:** Excellent for seeing the "shape" of the data. Thicker areas mean a higher probability/concentration of events happening at that value.
                """)

        if not num_cols or not cat_cols:
            st.info("Requires at least one numerical and one categorical column.")
        else:
            with st.form("var_form"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    var_num = st.selectbox("Numerical Feature (Y-Axis):", num_cols)
                with c2:
                    var_cat = st.selectbox("Categorical Group (X-Axis):", cat_cols)
                with c3:
                    var_type = st.radio(
                        "Plot Geometry:",
                        [
                            "Box Plot (Quartiles & Outliers)",
                            "Violin Plot (Probability Density)",
                        ],
                    )

                submitted = st.form_submit_button(
                    "Generate Plot", type="primary", use_container_width=True
                )

            if submitted:
                v_type = "Box" if "Box" in var_type else "Violin"
                fig_var = plot_categorical_variance(df, var_cat, var_num, v_type)
                st.session_state["saved_variance_plot"] = fig_var

            # Always render if it exists in state
            if "saved_variance_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_variance_plot"],
                    use_container_width=True,
                )
            else:
                st.info("👆 Configure your parameters and click **Generate Plot**.")

    # --- TAB 3: Relationships ---
    with t_rel:
        st.subheader("Feature Relationships")

        with st.expander("ℹ️ Guide: Understanding 2D Relationships"):
            st.markdown("""
                **How to read these charts:**
                * **Scatter Plots:** Each dot represents a single row (event) in the dataset. They are used to find correlations (e.g., as X goes up, does Y go up?).
                * **Trendlines (OLS):** Ordinary Least Squares draws a straight line of best fit. If the line is flat, there is no linear relationship.
                * **Trendlines (LOWESS):** A localized curve that follows the "gravity" of the data. Excellent for spotting non-linear patterns or clusters.
                * **Marginal Distributions:** The mini-histograms on the top and right edges show the 1D distribution of the X and Y axes independently.
                """)

        if len(num_cols) < 2:
            st.info("Requires at least two numerical columns.")
        else:
            with st.form("rel_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_axis = st.selectbox("X-Axis (Predictor):", num_cols, index=0)
                    y_axis = st.selectbox(
                        "Y-Axis (Target):",
                        num_cols,
                        index=1 if len(num_cols) > 1 else 0,
                    )
                with c2:
                    scatter_color = st.selectbox("Color Segment:", ["None"] + cat_cols)
                    scatter_size = st.selectbox("Size Dimension:", ["None"] + num_cols)
                with c3:
                    trend = st.selectbox(
                        "Statistical Trendline:",
                        [
                            "None",
                            "Ordinary Least Squares (OLS)",
                            "Locally Weighted Smoothing (LOWESS)",
                        ],
                    )
                    show_marginals = st.checkbox(
                        "Show Marginal Distributions", value=True
                    )

                submitted = st.form_submit_button(
                    "Generate Plot", type="primary", use_container_width=True
                )

            if submitted:
                c_arg = None if scatter_color == "None" else scatter_color
                s_arg = None if scatter_size == "None" else scatter_size
                t_arg = (
                    None if trend == "None" else ("ols" if "OLS" in trend else "lowess")
                )

                fig_rel = plot_relationship(
                    df, x_axis, y_axis, c_arg, s_arg, t_arg, show_marginals
                )
                st.session_state["saved_relationships_plot"] = fig_rel

            # Always render if it exists in state
            if "saved_relationships_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_relationships_plot"],
                    use_container_width=True,
                )
            else:
                st.info("👆 Configure your parameters and click **Generate Plot**.")

    # --- TAB 4: Time Series ---
    with t_time:
        st.subheader("Temporal Dynamics")

        with st.expander("ℹ️ Guide: Understanding Time Series & Trends"):
            st.markdown("""
                **How to read these charts:**
                * **Time Series:** Tracks how a specific metric changes chronologically. Excellent for spotting seasonality (e.g., summer spikes) or macro-level growth and decline.
                * **Moving Average (MA):** Daily event data is incredibly noisy (a chaotic zigzag). A "Rolling Window" averages the data over *N* periods to draw a smooth, readable trendline through the chaos.
                """)

        if not time_cols:
            st.info(
                "No Datetime columns found. Use the ETL Studio to parse date features."
            )
        elif not num_cols:
            st.info("Requires at least one numerical column to plot against time.")
        else:
            with st.form("time_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    t_col = st.selectbox("Time Axis:", time_cols)
                with c2:
                    v_col = st.selectbox("Value to Track:", num_cols)
                with c3:
                    t_color = st.selectbox("Split by Category:", ["None"] + cat_cols)
                    roll_win = st.number_input(
                        "Rolling Moving Average (Periods):",
                        min_value=0,
                        max_value=365,
                        value=0,
                        step=1,
                    )

                submitted = st.form_submit_button(
                    "Generate Plot", type="primary", use_container_width=True
                )

            if submitted:
                tc_arg = None if t_color == "None" else t_color
                fig_time = plot_time_series(df, t_col, v_col, tc_arg, roll_win)
                st.session_state["saved_time_series_plot"] = fig_time

            # Always render if it exists in state
            if "saved_time_series_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_time_series_plot"],
                    use_container_width=True,
                )
            else:
                st.info("👆 Configure your parameters and click **Generate Plot**.")

    # --- TAB 5: High-Dimensional ---
    with t_multi:
        st.subheader("High-Dimensional Analysis")

        with st.expander("ℹ️ Guide: Understanding Multicollinearity & Matrices"):
            st.markdown("""
                **How to read these charts:**
                * **Correlation Heatmap:** Measures how strongly numerical variables are linked.
                    * **Deep Red (+1):** Perfect positive correlation (as X goes up, Y goes up).
                    * **Deep Blue (-1):** Perfect negative correlation (as X goes up, Y goes down).
                    * **White (0):** No mathematical relationship.
                * **Scatter Matrix:** A grid of 2D scatter plots crossing every selected variable against the others. Perfect for spotting spatial or temporal clusters.
                """)

        if len(num_cols) < 2:
            st.info("Requires multiple numerical columns.")
        else:
            st.markdown("**1. Correlation Heatmap**")
            with st.form("corr_form"):
                c1, _ = st.columns([1, 3])
                with c1:
                    corr_method = st.selectbox(
                        "Mathematical Method:",
                        ["Pearson (Linear)", "Spearman (Rank/Monotonic)"],
                    )
                submitted_corr = st.form_submit_button(
                    "Generate Heatmap", type="primary", use_container_width=True
                )

            if submitted_corr:
                c_meth = "pearson" if "Pearson" in corr_method else "spearman"
                fig_corr = plot_correlation_heatmap(df, c_meth)
                st.session_state["saved_heatmap"] = fig_corr

            # Always render if it exists in state
            if "saved_heatmap" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_heatmap"], use_container_width=True
                )
            else:
                st.info(
                    "👆 Select a mathematical method and click **Generate Heatmap**."
                )

            st.markdown("---")
            st.markdown("**2. Scatter Matrix (Pairplot)**")
            with st.form("matrix_form"):
                c3, c4 = st.columns([1, 3])
                with c3:
                    matrix_cols = st.multiselect(
                        "Select Dimensions:",
                        num_cols,
                        placeholder="Choose dimensions...",
                        default=num_cols[:3] if len(num_cols) >= 3 else num_cols,
                    )
                with c4:
                    matrix_color = st.selectbox("Color By:", ["None"] + cat_cols)
                submitted_mat = st.form_submit_button(
                    "Generate Matrix", type="primary", use_container_width=True
                )

            if submitted_mat:
                if len(matrix_cols) > 1:
                    mc_arg = None if matrix_color == "None" else matrix_color
                    fig_mat = plot_scatter_matrix(df, matrix_cols, mc_arg)
                    st.session_state["saved_matrix"] = fig_mat
                else:
                    st.error("Select at least two dimensions for the matrix.")

            # Always render if it exists in state
            if "saved_matrix" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_matrix"], use_container_width=True
                )
            elif not submitted_mat:
                st.info("👆 Configure your dimensions and click 'Generate Matrix'.")

    # --- TAB 6: Segmentation ---
    with t_seg:
        st.subheader("Hierarchical Sunburst")

        with st.expander("ℹ️ Guide: Understanding Hierarchical Segmentation"):
            st.markdown("""
                **How to read these charts:**
                * **The Sunburst:** The innermost circle is the top of the hierarchy. Moving outward breaks that category into smaller sub-segments.
                * **Interactivity:** You can **click** on any slice to drill down and make it the new center. Click the very center to zoom back out.
                """)

        if len(cat_cols) < 1:
            st.info("Requires categorical data for segmentation.")
        else:
            with st.form("seg_form"):
                c1, c2 = st.columns(2)
                with c1:
                    path_cols = st.multiselect(
                        "Select Hierarchy (Outer to Inner):",
                        cat_cols,
                        placeholder="Choose hierarchy levels...",
                        default=[cat_cols[0]],
                    )
                with c2:
                    sun_value = st.selectbox(
                        "Slice Size Determinant:", ["Row Count"] + num_cols
                    )

                submitted = st.form_submit_button(
                    "Generate Plot", type="primary", use_container_width=True
                )

            if submitted:
                if path_cols:
                    v_arg = None if sun_value == "Row Count" else sun_value
                    fig_seg = plot_hierarchical_sunburst(df, path_cols, v_arg)
                    st.session_state["saved_sunburst_plot"] = fig_seg
                else:
                    st.error("Select a hierarchy path.")

            # Always render if it exists in state
            if "saved_sunburst_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_sunburst_plot"],
                    use_container_width=True,
                )
            else:
                st.info("👆 Configure your hierarchy and click **Generate Plot**.")
