"""Streamlit UI for the data cleaning and preprocessing workflow.

This module renders the interactive Data Cleaning & Preprocessing Studio used
inside the Predictive Analytics Dashboard. It provides sidebar controls for
structural cleaning, temporal feature engineering, text standardization,
cardinality reduction, memory optimization, and optional advanced analytical
transformations.

The module manages cleaned dataset versions through ``st.session_state`` so
that users can apply transformations sequentially, undo previous operations,
or reset the dataset to its original uploaded state.
"""

import pandas as pd
import streamlit as st

from core.preprocessor import (
    clean_text_features,
    drop_missing_targets,
    drop_selected_columns,
    encode_categorical_features,
    engineer_temporal_features,
    fix_numerical_skewness,
    handle_outliers,
    impute_categorical_features,
    impute_numerical_features,
    optimize_memory_usage,
    reduce_cardinality,
    remove_duplicate_rows,
    scale_numerical_features,
    summarize_cardinality,
    summarize_outliers,
    summarize_skewness,
)


def commit_transformation(new_df: pd.DataFrame) -> None:
    """Store a transformed dataset in the session-state history stack.

    Each committed DataFrame becomes the latest version of the cleaned dataset.
    The history stack allows users to apply preprocessing operations
    incrementally and undo previous transformations when needed.

    Args:
        new_df: The transformed dataset produced by a preprocessing operation.
    """
    st.session_state["data_history"].append(new_df)


def render_cleaning_module() -> None:
    """Render the data cleaning and preprocessing interface.

    Displays the full ETL workspace, including structural cleaning tools,
    datetime feature extraction, text cleaning, cardinality reduction, memory
    optimization, optional analytical transformations, dataset metrics, version
    controls, and a preview of the current transformed dataset.
    """
    raw_df = st.session_state.get("raw_data")

    st.title("Data Cleaning & Preprocessing Studio")
    st.markdown("""
        Welcome to the **Data Cleaning & Preprocessing Studio** — the
        foundational stage of the predictive analytics pipeline. This module
        helps structure, clean, transform, and optimize raw datasets before
        downstream exploratory analysis or machine learning.

        The workflow separates low-risk structural operations from advanced
        analytical transformations that may introduce data leakage if applied
        before a supervised machine learning split.

        * **Structural Cleaning:** Remove redundant columns, duplicate rows,
          and rows with missing values in selected target fields.
        * **Temporal Engineering:** Parse datetime columns and extract useful
          calendar-based features.
        * **Text & Cardinality Optimization:** Standardize text columns and
          reduce high-cardinality categorical features.
        * **Memory Optimization:** Downcast numeric data and convert suitable
          string columns into categorical types.
        * **Version Control:** Track transformation history and safely undo or
          reset preprocessing steps.
        """)
    st.markdown("---")

    if raw_df is None or raw_df.empty:
        st.warning("⚠️ No data available. Please upload a dataset first.")
        return

    if "data_history" not in st.session_state or not st.session_state["data_history"]:
        st.session_state["data_history"] = [raw_df.copy()]

    current_df = st.session_state["data_history"][-1]
    st.session_state["cleaned_data"] = current_df
    all_columns = current_df.columns.tolist()

    # -------------------------------------------------------------------------
    # Sidebar: structural cleaning controls
    # -------------------------------------------------------------------------
    st.sidebar.markdown("## 🟢 Structural Cleaning")
    st.sidebar.caption(
        "These operations reorganize the dataset structure and do not compute "
        "statistics from the full feature matrix."
    )

    st.sidebar.markdown("**1. Column & Row Management**")
    st.sidebar.caption(
        "Review missing values, remove unnecessary columns, drop duplicate rows, "
        "and remove incomplete records for selected fields."
    )

    missing_counts = current_df.isnull().sum()
    missing_data = pd.DataFrame(
        {
            "Missing Count": missing_counts,
            "Missing %": (missing_counts / len(current_df)) * 100,
        }
    )
    missing_data = missing_data[missing_data["Missing Count"] > 0].sort_values(
        by="Missing %",
        ascending=False,
    )

    if not missing_data.empty:
        st.sidebar.dataframe(
            missing_data.style.format({"Missing %": "{:.3f}%"}),
            width="stretch",
        )

    cols_to_drop = st.sidebar.multiselect(
        "Select Columns to Remove:",
        all_columns,
        placeholder="Choose columns...",
    )

    if st.sidebar.button("Remove Selected Columns", type="primary", width="stretch"):
        if cols_to_drop:
            new_state = drop_selected_columns(current_df, cols_to_drop)
            commit_transformation(new_state)
            st.rerun()
        else:
            st.sidebar.warning("Select columns to drop.")

    if st.sidebar.button("Remove Duplicate Rows", type="primary", width="stretch"):
        new_state = remove_duplicate_rows(current_df)
        commit_transformation(new_state)
        st.rerun()

    target_cols = st.sidebar.multiselect(
        "Remove Rows with Missing Values In:",
        all_columns,
        placeholder="Choose columns...",
    )

    if st.sidebar.button("Drop Incomplete Rows", type="primary", width="stretch"):
        if target_cols:
            new_state = drop_missing_targets(current_df, target_cols)
            commit_transformation(new_state)
            st.rerun()
        else:
            st.sidebar.warning("Select columns to check for missing values.")

    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Sidebar: temporal feature engineering controls
    # -------------------------------------------------------------------------
    st.sidebar.markdown("**2. Temporal Engineering**")
    st.sidebar.caption(
        "Extract time-based features such as year, quarter, month, weekday, "
        "hour, and weekend indicators from datetime columns."
    )

    possible_date_cols = current_df.select_dtypes(
        include=["object", "datetime", "datetime64[ns]"]
    ).columns.tolist()

    if not possible_date_cols:
        st.sidebar.info("⏳ No text or datetime columns detected.")
    else:
        date_target = st.sidebar.selectbox(
            "Select Datetime Column:",
            possible_date_cols,
        )

        format_mapping = {
            "Auto Detect (Slower)": "mixed",
            "ISO 8601 Format": "ISO8601",
            "YYYY-MM-DD HH:MM:SS (Global)": "%Y-%m-%d %H:%M:%S",
            "MM/DD/YYYY HH:MM:SS (US)": "%m/%d/%Y %H:%M:%S",
            "DD/MM/YYYY HH:MM:SS (EU)": "%d/%m/%Y %H:%M:%S",
            "YYYY-MM-DD HH:MM (Global)": "%Y-%m-%d %H:%M",
            "YYYY-MM-DD hh:mm:ss AM/PM": "%Y-%m-%d %I:%M:%S %p",
            "YYYY-MM-DD (Global Date)": "%Y-%m-%d",
            "DD.MM.YYYY (EU Dots)": "%d.%m.%Y",
        }

        selected_format_label = st.sidebar.selectbox(
            "Select Date Format:",
            list(format_mapping.keys()),
        )
        backend_format = format_mapping[selected_format_label]

        temporal_options = [
            "Year",
            "Quarter",
            "Month",
            "Day",
            "Hour",
            "Day of Week",
            "Is Weekend",
            "Is Month Start/End",
        ]

        selected_time_features = st.sidebar.multiselect(
            "Select Features to Extract:",
            options=temporal_options,
            default=["Month", "Day of Week"],
            placeholder="Choose date/time features...",
        )

        apply_cyclical = st.sidebar.checkbox(
            "Apply Cyclical Encoding Using Sine/Cosine",
            help=(
                "Transforms cyclical time variables into sine/cosine coordinates "
                "to preserve circular continuity."
            ),
        )
        sort_time = st.sidebar.checkbox(
            "Sort Rows Chronologically",
            help="Orders the dataset from oldest to newest.",
        )
        set_index = st.sidebar.checkbox(
            "Set Datetime Column as DataFrame Index",
            help="Useful for time-series workflows and forecasting libraries.",
        )

        if st.sidebar.button(
            "Apply Temporal Engineering",
            type="primary",
            width="stretch",
        ):
            try:
                new_state = engineer_temporal_features(
                    current_df,
                    date_column=date_target,
                    features_to_extract=selected_time_features,
                    datetime_format=backend_format,
                    apply_cyclical=apply_cyclical,
                    sort_chronological=sort_time,
                    set_as_index=set_index,
                )
                commit_transformation(new_state)
                st.rerun()
            except ValueError as error:
                st.sidebar.error(str(error))

    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Sidebar: text standardization controls
    # -------------------------------------------------------------------------
    st.sidebar.markdown("**3. Text Standardization**")
    st.sidebar.caption(
        "Normalize text columns by standardizing casing, spacing, punctuation, "
        "and numeric characters."
    )

    cat_cols = current_df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if cat_cols:
        cols_to_clean = st.sidebar.multiselect(
            "Select Text Columns:",
            cat_cols,
            placeholder="Choose columns...",
        )

        case_mode_ui = st.sidebar.selectbox(
            "Case Conversion:",
            ["None", "Lowercase", "Uppercase", "Title Case"],
        )
        case_map = {
            "None": "none",
            "Lowercase": "lower",
            "Uppercase": "upper",
            "Title Case": "title",
        }
        backend_case = case_map[case_mode_ui]

        strip_ws = st.sidebar.checkbox("Trim Leading/Trailing Spaces", value=True)
        collapse_ws = st.sidebar.checkbox("Collapse Multiple Spaces", value=True)
        rem_punct = st.sidebar.checkbox("Remove Punctuation", value=False)
        rem_nums = st.sidebar.checkbox("Remove Numbers", value=False)

        if st.sidebar.button(
            "Apply Text Standardization",
            type="primary",
            width="stretch",
        ):
            if cols_to_clean:
                new_state = clean_text_features(
                    current_df,
                    columns=cols_to_clean,
                    case_mode=backend_case,
                    remove_punctuation=rem_punct,
                    remove_numbers=rem_nums,
                    strip_whitespace=strip_ws,
                    collapse_spaces=collapse_ws,
                )
                commit_transformation(new_state)
                st.rerun()
            else:
                st.sidebar.warning("Select at least one text column to clean.")
    else:
        st.sidebar.info("No text columns available to standardize.")

    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Sidebar: cardinality reduction controls
    # -------------------------------------------------------------------------
    st.sidebar.markdown("**4. Cardinality Reduction**")
    st.sidebar.caption(
        "Inspect categorical features with many unique values and group rare "
        "categories into broader labels."
    )

    if cat_cols:
        card_summary = summarize_cardinality(current_df, cat_cols)
        st.sidebar.dataframe(card_summary, hide_index=True, width="stretch")

        card_cols_to_reduce = st.sidebar.multiselect(
            "Select Columns to Reduce:",
            cat_cols,
            placeholder="Choose columns...",
        )
        card_method_ui = st.sidebar.selectbox(
            "Reduction Strategy:",
            [
                "Frequency Threshold",
                "Keep Top-N Categories",
                "Substring/Pattern Match",
            ],
        )

        cardinality_method = "frequency"
        thresh_pct = 1.0
        top_n_val = 10
        sub_str = ""

        if card_method_ui == "Frequency Threshold":
            cardinality_method = "frequency"
            thresh_pct = st.sidebar.slider(
                "Keep Labels Appearing More Than (%):",
                0.1,
                5.0,
                1.0,
                0.1,
            )
        elif card_method_ui == "Keep Top-N Categories":
            cardinality_method = "top_n"
            top_n_val = st.sidebar.number_input(
                "Number of Top Categories to Keep:",
                min_value=1,
                max_value=100,
                value=10,
            )
        else:
            cardinality_method = "substring"
            sub_str = st.sidebar.text_input("Match Substring (e.g., 'Apple'):")

        other_label = st.sidebar.text_input("Replacement Label", value="Other")

        if st.sidebar.button(
            "Apply Cardinality Reduction",
            type="primary",
            width="stretch",
        ):
            if card_cols_to_reduce:
                if cardinality_method == "substring" and not sub_str:
                    st.sidebar.error("Please enter a substring to match.")
                else:
                    new_state = reduce_cardinality(
                        current_df,
                        columns=card_cols_to_reduce,
                        method=cardinality_method,
                        threshold_percent=(thresh_pct / 100.0),
                        top_n=top_n_val,
                        substring=sub_str,
                        replacement_label=other_label,
                    )
                    commit_transformation(new_state)
                    st.rerun()
            else:
                st.sidebar.warning("Select at least one column to reduce.")
    else:
        st.sidebar.info("No categorical columns available for reduction.")

    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Sidebar: memory optimization controls
    # -------------------------------------------------------------------------
    st.sidebar.markdown("**5. Memory Optimization**")
    st.sidebar.caption(
        "Reduce memory usage by downcasting numeric columns and converting "
        "suitable text columns to categorical data types."
    )

    mem_usage_mb = current_df.memory_usage(deep=True).sum() / 1024**2

    if mem_usage_mb > 300:
        st.sidebar.error(f"⚠️ Current Memory Usage: {mem_usage_mb:.2f} MB")
    else:
        st.sidebar.info(f"💾 Current Memory Usage: {mem_usage_mb:.2f} MB")

    opt_ints = st.sidebar.checkbox("Downcast Integers (e.g., int64 → int8)", value=True)
    opt_floats = st.sidebar.checkbox(
        "Downcast Floats (e.g., float64 → float32)",
        value=True,
    )
    opt_cats = st.sidebar.checkbox("Convert Strings to Categories", value=True)

    if st.sidebar.button("Optimize Memory Footprint", type="primary", width="stretch"):
        new_state = optimize_memory_usage(
            current_df,
            downcast_integers=opt_ints,
            downcast_floats=opt_floats,
            categorize_strings=opt_cats,
        )
        commit_transformation(new_state)
        st.rerun()

    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Sidebar: advanced analytical transformations
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### 🔴 Analytical Transformations")
    st.sidebar.caption(
        "Unlock mathematical imputation, outlier treatment, distribution "
        "transformations, scaling, and encoding."
    )

    show_advanced = st.sidebar.toggle("Unlock Advanced Analytical Tools", value=False)

    if show_advanced:
        st.sidebar.warning(
            "⚠️ **Leakage Warning:** Do not apply these tools if you plan to use "
            "this data directly in a machine learning pipeline. These operations "
            "compute statistics across the full current dataset. Use them only "
            "when preparing data for exploratory analysis."
        )

        num_cols = current_df.select_dtypes(
            include=["float64", "int64", "float32", "int32"]
        ).columns.tolist()

        # ---------------------------------------------------------------------
        # Sidebar: imputation controls
        # ---------------------------------------------------------------------
        st.sidebar.markdown("**6. Data Imputation**")
        st.sidebar.caption(
            "Fill missing values using numeric and categorical imputation methods."
        )

        num_missing = [col for col in num_cols if current_df[col].isna().sum() > 0]
        num_impute_cols = st.sidebar.multiselect(
            "Select Numeric Columns:",
            num_cols,
            placeholder="Choose columns...",
            default=num_missing,
        )
        num_strategy_ui = st.sidebar.selectbox(
            "Numeric Strategy:",
            ["Median", "Mean", "Most Frequent", "Zero"],
        )
        num_strategy_map = {
            "Median": "median",
            "Mean": "mean",
            "Most Frequent": "mode",
            "Zero": "zero",
        }

        if st.sidebar.button(
            "Apply Numeric Imputation",
            type="primary",
            width="stretch",
        ):
            if num_impute_cols:
                new_state = impute_numerical_features(
                    current_df,
                    num_impute_cols,
                    num_strategy_map[num_strategy_ui],
                )
                commit_transformation(new_state)
                st.rerun()

        cat_missing = [col for col in cat_cols if current_df[col].isna().sum() > 0]
        cat_impute_cols = st.sidebar.multiselect(
            "Select Categorical Columns:",
            cat_cols,
            placeholder="Choose columns...",
            default=cat_missing,
        )
        cat_strategy_ui = st.sidebar.selectbox(
            "Categorical Strategy:",
            ['Fill with "Unknown"', "Most Frequent", "Forward Fill"],
        )
        cat_strategy_map = {
            'Fill with "Unknown"': "unknown",
            "Most Frequent": "mode",
            "Forward Fill": "forward fill",
        }

        if st.sidebar.button(
            "Apply Categorical Imputation",
            type="primary",
            width="stretch",
        ):
            if cat_impute_cols:
                new_state = impute_categorical_features(
                    current_df,
                    cat_impute_cols,
                    cat_strategy_map[cat_strategy_ui],
                )
                commit_transformation(new_state)
                st.rerun()

        st.sidebar.markdown("---")

        # ---------------------------------------------------------------------
        # Sidebar: outlier treatment controls
        # ---------------------------------------------------------------------
        st.sidebar.markdown("**7. Outlier Treatment**")
        st.sidebar.caption(
            "Detect potential outliers in numeric columns and treat them by "
            "clipping, null replacement, or row removal."
        )

        if num_cols:
            outlier_cols = st.sidebar.multiselect(
                "Select Numeric Columns for Outliers:",
                num_cols,
                placeholder="Choose columns...",
                default=[num_cols[0]] if num_cols else None,
            )
            outlier_method_ui = st.sidebar.selectbox(
                "Outlier Detection Method:",
                ["IQR", "Z-Score", "Percentiles"],
            )

            outlier_method = "iqr"
            iqr_mult = 1.5
            z_thresh = 3.0
            pct_range = (0.01, 0.99)

            if outlier_method_ui == "IQR":
                outlier_method = "iqr"
                iqr_mult = st.sidebar.slider("IQR Multiplier:", 1.0, 5.0, 1.5, 0.1)
            elif outlier_method_ui == "Z-Score":
                outlier_method = "zscore"
                z_thresh = st.sidebar.slider(
                    "Z-Score Threshold:",
                    2.0,
                    5.0,
                    3.0,
                    0.1,
                )
            else:
                outlier_method = "percentile"
                col1, col2 = st.sidebar.columns(2)
                lower_pct = col1.number_input("Lower %", 0.0, 10.0, 1.0, 0.1)
                upper_pct = col2.number_input("Upper %", 90.0, 100.0, 99.0, 0.1)
                pct_range = (lower_pct / 100.0, upper_pct / 100.0)

            if outlier_cols:
                summary_df = summarize_outliers(
                    current_df,
                    outlier_cols,
                    outlier_method,
                    iqr_mult,
                    z_thresh,
                    pct_range,
                )
                st.sidebar.dataframe(
                    summary_df.style.format({"% of Data": "{:.1f}%"}),
                    hide_index=True,
                    width="stretch",
                )

            outlier_action_ui = st.sidebar.selectbox(
                "Treatment Method:",
                ["Cap/Clip Values", "Replace with Nulls", "Drop Rows"],
            )
            action_map = {
                "Cap/Clip Values": "cap",
                "Replace with Nulls": "nan",
                "Drop Rows": "drop",
            }

            if st.sidebar.button(
                "Apply Outlier Treatment",
                type="primary",
                width="stretch",
            ):
                if outlier_cols:
                    new_state = handle_outliers(
                        current_df,
                        columns=outlier_cols,
                        method=outlier_method,
                        action=action_map[outlier_action_ui],
                        iqr_multiplier=iqr_mult,
                        zscore_threshold=z_thresh,
                        percentile_range=pct_range,
                    )
                    commit_transformation(new_state)
                    st.rerun()

        st.sidebar.markdown("---")

        # ---------------------------------------------------------------------
        # Sidebar: skewness transformation controls
        # ---------------------------------------------------------------------
        st.sidebar.markdown("**8. Skewness Transformation**")
        st.sidebar.caption(
            "Analyze skewness in numeric features and apply transformations to "
            "reduce distribution asymmetry."
        )

        if num_cols:
            skew_summary = summarize_skewness(current_df, num_cols)

            if not skew_summary.empty:
                st.sidebar.dataframe(
                    skew_summary.style.format({"Skewness": "{:.2f}"}),
                    hide_index=True,
                    width="stretch",
                )

                highly_skewed_cols = skew_summary[
                    skew_summary["Diagnosis"] == "Highly Skewed"
                ]["Feature"].tolist()

                skew_cols = st.sidebar.multiselect(
                    "Select Columns to Transform:",
                    num_cols,
                    placeholder="Choose columns...",
                    default=highly_skewed_cols,
                )

                skew_method_ui = st.sidebar.selectbox(
                    "Transformation Method:",
                    ["Yeo-Johnson", "Log1p", "Square Root", "Box-Cox"],
                )
                skew_method_map = {
                    "Yeo-Johnson": "yeo-johnson",
                    "Log1p": "log1p",
                    "Square Root": "sqrt",
                    "Box-Cox": "box-cox",
                }

                if st.sidebar.button(
                    "Apply Distribution Transformation",
                    type="primary",
                    width="stretch",
                ):
                    if skew_cols:
                        try:
                            new_state = fix_numerical_skewness(
                                current_df,
                                skew_cols,
                                skew_method_map[skew_method_ui],
                            )
                            commit_transformation(new_state)
                            st.rerun()
                        except ValueError as error:
                            st.sidebar.error(str(error))

        st.sidebar.markdown("---")

        # ---------------------------------------------------------------------
        # Sidebar: scaling and encoding controls
        # ---------------------------------------------------------------------
        st.sidebar.markdown("**9. Scaling & Encoding**")
        st.sidebar.caption("Scale numeric features and encode categorical variables.")

        if num_cols:
            scale_cols = st.sidebar.multiselect(
                "Select Columns to Scale:",
                num_cols,
                placeholder="Choose columns...",
            )
            scale_method_ui = st.sidebar.selectbox(
                "Scaling Method:",
                ["Standard", "Min-Max", "Robust", "MaxAbs"],
            )
            scale_map = {
                "Standard": "standard",
                "Min-Max": "minmax",
                "Robust": "robust",
                "MaxAbs": "maxabs",
            }

            if st.sidebar.button(
                "Scale Selected Features",
                type="primary",
                width="stretch",
            ):
                if scale_cols:
                    new_state = scale_numerical_features(
                        current_df,
                        columns=scale_cols,
                        method=scale_map[scale_method_ui],
                    )
                    commit_transformation(new_state)
                    st.rerun()

        if cat_cols:
            encode_cols = st.sidebar.multiselect(
                "Select Columns to Encode:",
                cat_cols,
                placeholder="Choose columns...",
            )
            encode_method_ui = st.sidebar.selectbox(
                "Encoding Method:",
                ["One-Hot Encoding", "Label Encoding"],
            )
            encode_map = {
                "One-Hot Encoding": "onehot",
                "Label Encoding": "label",
            }

            if st.sidebar.button(
                "Encode Selected Features",
                type="primary",
                width="stretch",
            ):
                if encode_cols:
                    new_state = encode_categorical_features(
                        current_df,
                        columns=encode_cols,
                        method=encode_map[encode_method_ui],
                    )
                    commit_transformation(new_state)
                    st.rerun()

        else:
            st.sidebar.success(
                "✅ All categorical features have been successfully encoded into numerical formats."
            )

        st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Main page: dataset summary, version controls, and preview
    # -------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", f"{current_df.shape[0]:,}")
    col2.metric("Total Columns", f"{current_df.shape[1]:,}")
    col3.metric("Missing Values", f"{current_df.isna().sum().sum():,}")

    st.markdown("---")

    st.subheader("Data Version Control")
    col_undo, col_reset, _ = st.columns([1, 1, 2])

    with col_undo:
        history_length = len(st.session_state["data_history"])

        if st.button(
            f"↩️ Undo Action ({history_length - 1} steps)",
            disabled=(history_length <= 1),
            width="stretch",
        ):
            st.session_state["data_history"].pop()
            st.rerun()

    with col_reset:
        if st.button("⚠️ Reset All", type="primary", width="stretch"):
            st.session_state["data_history"] = [raw_df.copy()]
            st.rerun()

    st.markdown("### Current Dataset Preview")
    st.dataframe(current_df.head(50), width="stretch")
