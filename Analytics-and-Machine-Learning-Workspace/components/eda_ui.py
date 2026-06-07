"""Streamlit UI for exploratory data analysis workflows.

This module renders the Exploratory Data Analysis Studio used by the Analytics
and Machine Learning Workspace. It provides tabbed analytical workflows for
distributions, categorical variance, feature relationships, time-series analysis,
high-dimensional visualization, and hierarchical segmentation.

The module uses the cleaned dataset from ``st.session_state`` when available and
falls back to the raw uploaded dataset when preprocessing has not been applied.
Generated Plotly figures are stored in session state so that charts persist
across Streamlit reruns.
"""

import streamlit as st

from core.eda_visualizations import (
    plot_categorical_variance,
    plot_correlation_heatmap,
    plot_distribution,
    plot_hierarchical_sunburst,
    plot_relationship,
    plot_scatter_matrix,
    plot_time_series,
)


def render_eda_module() -> None:
    """Render the exploratory data analysis interface.

    Displays the EDA workspace, selects the active dataset from session state,
    validates the available data types, and renders tabbed controls for the
    supported visualization workflows.
    """
    st.title("Exploratory Data Analysis (EDA) Studio")
    st.markdown("""
        Welcome to the **Exploratory Data Analysis (EDA) Studio** — an
        interactive visual analytics workspace for inspecting feature
        distributions, relationships, trends, and segmentation patterns.

        This module helps transform cleaned tabular data into exploratory
        insights through specialized analytical workflows:

        * **Feature Distributions & Variance:** Inspect numerical distributions,
          categorical counts, skewness, and group-level spread.
        * **Relationship & Trend Analysis:** Explore pairwise relationships,
          trendlines, and marginal distributions.
        * **Temporal Dynamics:** Track how numerical values evolve over time and
          apply rolling moving averages.
        * **High-Dimensional Analysis:** Examine multicollinearity through
          correlation heatmaps and scatter matrices.
        * **Hierarchical Segmentation:** Break down categorical structures using
          interactive sunburst charts.
        """)
    st.markdown("---")

    if (
        "cleaned_data" in st.session_state
        and st.session_state["cleaned_data"] is not None
    ):
        df = st.session_state["cleaned_data"]
        st.caption(
            "✨ Using processed data from the Data Cleaning & Preprocessing Studio."
        )

    elif "raw_data" in st.session_state and st.session_state["raw_data"] is not None:
        df = st.session_state["raw_data"]
        st.info(
            "ℹ️ **Notice:** Using the raw uploaded dataset. For optimal results, "
            "consider processing this data in the Data Cleaning & Preprocessing "
            "Studio first."
        )

    else:
        st.warning("⚠️ No data available. Please upload a dataset first.")
        return

    if df.empty:
        st.error(
            "❌ The selected dataset is empty. Please check your data source or "
            "cleaning steps."
        )
        return

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

    # -------------------------------------------------------------------------
    # Tab 1: feature distributions
    # -------------------------------------------------------------------------
    with t_dist:
        st.subheader("Feature Distributions")

        with st.expander("ℹ️ Guide: Understanding 1D Distributions"):
            st.markdown("""
                **How to read these charts:**

                * **Histograms:** Show the frequency of numerical values within
                  specific ranges. Use them to inspect skewness, spread, and
                  distribution shape.
                * **Box Plots:** Show the median, interquartile range, and
                  potential outliers for numerical features.
                * **Bar Charts:** Show row counts for categorical features.
                * **Log Scale:** Compresses large value ranges so smaller bars or
                  bins remain visible in heavily skewed charts.
                """)

        with st.form("dist_form"):
            c1, c2 = st.columns(2)

            with c1:
                dist_col = st.selectbox("Analyze Feature:", all_cols)
                dist_color = st.selectbox("Group By (Color):", ["None"] + cat_cols)

            with c2:
                bins = st.slider("Number of Bins (for numerical features):", 5, 100, 30)
                log_scale = st.checkbox(
                    "Logarithmic Y-Axis",
                    help="Useful for heavily skewed data.",
                )

            submitted = st.form_submit_button(
                "Generate Plot",
                type="primary",
                width="stretch",
            )

        if submitted:
            color_arg = None if dist_color == "None" else dist_color
            fig_dist = plot_distribution(df, dist_col, color_arg, bins, log_scale)
            st.session_state["saved_distribution_plot"] = fig_dist

        if "saved_distribution_plot" in st.session_state:
            st.plotly_chart(
                st.session_state["saved_distribution_plot"],
                width="stretch",
            )
        else:
            st.info("👆 Configure your parameters and click **Generate Plot**.")

    # -------------------------------------------------------------------------
    # Tab 2: categorical variance
    # -------------------------------------------------------------------------
    with t_var:
        st.subheader("Categorical Variance")

        with st.expander("ℹ️ Guide: Understanding Variance Plots"):
            st.markdown("""
                **How to read these charts:**

                * **Box Plots:** Compare medians, interquartile ranges, and
                  potential outliers across categorical groups.
                * **Violin Plots:** Show the distribution shape within each
                  category. Wider regions indicate higher data density.
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
                    "Generate Plot",
                    type="primary",
                    width="stretch",
                )

            if submitted:
                v_type = "Box" if "Box" in var_type else "Violin"
                fig_var = plot_categorical_variance(df, var_cat, var_num, v_type)
                st.session_state["saved_variance_plot"] = fig_var

            if "saved_variance_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_variance_plot"],
                    width="stretch",
                )
            else:
                st.info("👆 Configure your parameters and click **Generate Plot**.")

    # -------------------------------------------------------------------------
    # Tab 3: feature relationships
    # -------------------------------------------------------------------------
    with t_rel:
        st.subheader("Feature Relationships")

        with st.expander("ℹ️ Guide: Understanding 2D Relationships"):
            st.markdown("""
                **How to read these charts:**

                * **Scatter Plots:** Show pairwise relationships between two
                  numerical variables.
                * **OLS Trendlines:** Fit a linear relationship using ordinary
                  least squares.
                * **LOWESS Trendlines:** Fit a local smooth curve, useful for
                  non-linear patterns.
                * **Marginal Distributions:** Show the separate one-dimensional
                  distributions of the X and Y variables.
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
                        "Show Marginal Distributions",
                        value=True,
                    )

                submitted = st.form_submit_button(
                    "Generate Plot",
                    type="primary",
                    width="stretch",
                )

            if submitted:
                c_arg = None if scatter_color == "None" else scatter_color
                s_arg = None if scatter_size == "None" else scatter_size
                t_arg = (
                    None if trend == "None" else ("ols" if "OLS" in trend else "lowess")
                )

                fig_rel = plot_relationship(
                    df,
                    x_axis,
                    y_axis,
                    c_arg,
                    s_arg,
                    t_arg,
                    show_marginals,
                )
                st.session_state["saved_relationships_plot"] = fig_rel

            if "saved_relationships_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_relationships_plot"],
                    width="stretch",
                )
            else:
                st.info("👆 Configure your parameters and click **Generate Plot**.")

    # -------------------------------------------------------------------------
    # Tab 4: temporal dynamics
    # -------------------------------------------------------------------------
    with t_time:
        st.subheader("Temporal Dynamics")

        with st.expander("ℹ️ Guide: Understanding Time Series & Trends"):
            st.markdown("""
                **How to read these charts:**

                * **Time Series:** Track how a numerical value changes across
                  chronological observations.
                * **Moving Average:** Smooths noisy temporal data by averaging
                  values over a rolling window.
                """)

        if not time_cols:
            st.info(
                "No datetime columns found. Use the ETL Studio to parse date "
                "features."
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
                    "Generate Plot",
                    type="primary",
                    width="stretch",
                )

            if submitted:
                tc_arg = None if t_color == "None" else t_color
                fig_time = plot_time_series(df, t_col, v_col, tc_arg, roll_win)
                st.session_state["saved_time_series_plot"] = fig_time

            if "saved_time_series_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_time_series_plot"],
                    width="stretch",
                )
            else:
                st.info("👆 Configure your parameters and click **Generate Plot**.")

    # -------------------------------------------------------------------------
    # Tab 5: high-dimensional analysis
    # -------------------------------------------------------------------------
    with t_multi:
        st.subheader("High-Dimensional Analysis")

        with st.expander("ℹ️ Guide: Understanding Multicollinearity & Matrices"):
            st.markdown("""
                **How to read these charts:**

                * **Correlation Heatmap:** Measures relationships between
                  numerical variables.
                    * **+1:** Strong positive correlation.
                    * **-1:** Strong negative correlation.
                    * **0:** Weak or no linear/rank relationship.
                * **Scatter Matrix:** Compares multiple numerical variables
                  pairwise in a compact grid.
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
                    "Generate Heatmap",
                    type="primary",
                    width="stretch",
                )

            if submitted_corr:
                c_meth = "pearson" if "Pearson" in corr_method else "spearman"
                fig_corr = plot_correlation_heatmap(df, c_meth)
                st.session_state["saved_heatmap"] = fig_corr

            if "saved_heatmap" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_heatmap"],
                    width="stretch",
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
                    "Generate Matrix",
                    type="primary",
                    width="stretch",
                )

            if submitted_mat:
                if len(matrix_cols) > 1:
                    mc_arg = None if matrix_color == "None" else matrix_color
                    fig_mat = plot_scatter_matrix(df, matrix_cols, mc_arg)
                    st.session_state["saved_matrix"] = fig_mat
                else:
                    st.error("❌ Select at least two dimensions for the matrix.")

            if "saved_matrix" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_matrix"],
                    width="stretch",
                )
            elif not submitted_mat:
                st.info("👆 Configure your dimensions and click **Generate Matrix**.")

    # -------------------------------------------------------------------------
    # Tab 6: hierarchical segmentation
    # -------------------------------------------------------------------------
    with t_seg:
        st.subheader("Hierarchical Sunburst")

        with st.expander("ℹ️ Guide: Understanding Hierarchical Segmentation"):
            st.markdown("""
                **How to read these charts:**

                * **Sunburst Chart:** Shows nested categorical structure from
                  the center outward.
                * **Interactivity:** Click a slice to drill into that segment.
                  Click the center to zoom back out.
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
                        "Slice Size Determinant:",
                        ["Row Count"] + num_cols,
                    )

                submitted = st.form_submit_button(
                    "Generate Plot",
                    type="primary",
                    width="stretch",
                )

            if submitted:
                if path_cols:
                    v_arg = None if sun_value == "Row Count" else sun_value
                    fig_seg = plot_hierarchical_sunburst(df, path_cols, v_arg)
                    st.session_state["saved_sunburst_plot"] = fig_seg
                else:
                    st.error("❌ Select a hierarchy path.")

            if "saved_sunburst_plot" in st.session_state:
                st.plotly_chart(
                    st.session_state["saved_sunburst_plot"],
                    width="stretch",
                )
            else:
                st.info("👆 Configure your hierarchy and click **Generate Plot**.")
