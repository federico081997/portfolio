"""
Machine Learning Studio UI Module

Provides an interactive, portfolio-grade dashboard for configuring, training,
and evaluating predictive models. Exposes advanced preprocessing options
(outlier management, skewness correction, custom imputation) and dynamic
hyperparameter tuning while maintaining a clean, compartmentalized UX.
"""

import streamlit as st
from core.ml_engine import (
    build_preprocessor,
    get_model,
    train_and_evaluate,
    plot_confusion_matrix,
    plot_regression_residuals,
    plot_feature_importance,
)


def render_ml_module() -> None:
    """
    Renders the primary Predictive Modeling Studio.

    Manages state routing for machine learning tasks. Captures user configuration
    via a complex sidebar UI, triggers the Scikit-Learn pipeline execution in
    the backend, and dynamically renders diagnostic Plotly visualizations based
    on the mathematical nature of the selected task (Classification vs. Regression).
    """
    st.empty()
    st.title("Predictive Modeling & Machine Learning Studio")
    st.markdown("""
    Welcome to the **Predictive Modeling & Machine Learning Studio**—the final predictive stage of the analytics pipeline. This comprehensive framework is designed for automated pipeline construction and rigorous mathematical model evaluation.

    This module allows you to build a mathematically sound predictive architecture divided into four core execution stages:
    * **Advanced Preprocessing:** Implement leakage-free imputation, structural outlier management (IQR/Z-Scores), and dynamic distribution transformations (Box-Cox/Yeo-Johnson).
    * **Feature Engineering:** Standardize continuous numerical spaces and apply robust encoding strategies to categorical text variables.
    * **Model Architecture:** Train industry-standard algorithms (Random Forest, Gradient Boosting, SVM) with dynamic hyperparameter tuning and native bias-reduction (class weighting) capabilities.
    * **Performance Diagnostics:** Evaluate unseen test-set performance through real-time metrics (F1, RMSE, R²), interactive confusion matrices, residual scatter plots, and feature importance attribution.
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

    all_cols = df.columns.tolist()

    # ==========================================
    # SIDEBAR: PIPELINE CONFIGURATION
    # ==========================================

    st.sidebar.header("⚙️ Pipeline Configuration")

    # 1. Task Definition
    task_type = st.sidebar.radio(
        "Machine Learning Task:", ["Classification", "Regression"]
    )
    target_col = st.sidebar.selectbox(
        "Target Variable (Y):", all_cols, index=len(all_cols) - 1
    )

    # 2. Feature Selection
    feature_options = [c for c in all_cols if c != target_col]
    selected_features = st.sidebar.multiselect(
        "Predictor Features (X):",
        feature_options,
        placeholder="Choose features...",
        default=feature_options[:5] if len(feature_options) > 5 else feature_options,
    )

    # 3. Advanced Preprocessing
    st.sidebar.markdown("---")
    st.sidebar.header("Data Preprocessing")

    with st.sidebar.expander("**1. Missing Data Handling**", expanded=True):
        num_impute = st.selectbox(
            "Numerical Imputation:", ["Median", "Mean", "Most Frequent", "Zero"]
        )
        cat_impute = st.selectbox(
            "Categorical Imputation:",
            ['Fill with "Unknown"', "Most Frequent", "Forward Fill"],
        )

    with st.sidebar.expander("**2. Outlier Management**", expanded=False):
        outlier_method = st.selectbox(
            "Detection Method:", ["None", "IQR", "Z-Scores", "Percentiles"]
        )
        outlier_action = "None"
        if outlier_method != "None":
            outlier_action = st.selectbox(
                "Outlier Action:",
                ["Cap/Clip Values", "Replace with Nulls", "Drop Rows"],
            )

    with st.sidebar.expander("**3. Feature Transformation**", expanded=False):
        skew_transform = st.selectbox(
            "Skewness Correction:",
            ["None", "Yeo-Johnson", "Log1p", "Square Root", "Box-Cox"],
        )
        scaling = st.selectbox(
            "Numerical Scaling:", ["None", "Standard", "Min-Max", "Robust", "MaxAbs"]
        )
        encoding = st.selectbox(
            "Categorical Encoding:", ["One-Hot Encoding", "Label Encoding"]
        )

    # 4. Model Architecture & Training Parameters
    st.sidebar.markdown("---")
    st.sidebar.header("Model Architecture")

    if task_type == "Classification":
        model_name = st.sidebar.selectbox(
            "Algorithm:",
            [
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors",
            ],
        )

        # Algorithmic Bias Reduction (Only applicable for certain classifiers)
        reduce_bias = False
        if model_name in [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine (SVM)",
        ]:
            reduce_bias = st.sidebar.checkbox(
                "Apply Class Weighting (Reduce Bias)",
                value=False,
                help="Automatically applies heavier mathematical penalties to mistakes made on minority classes. Excellent for imbalanced datasets like Fraud or Rare Diseases.",
            )
    else:
        model_name = st.sidebar.selectbox(
            "Algorithm:",
            [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Random Forest",
                "Gradient Boosting",
                "Support Vector Regressor (SVR)",
                "K-Nearest Neighbors",
            ],
        )
        reduce_bias = False

    test_size = st.sidebar.slider("Test Set Size (%):", 10, 50, 20, step=5) / 100.0

    # Dynamic Hyperparameter UI
    params = {}
    with st.sidebar.expander("**Tune Hyperparameters**", expanded=True):
        if "Random Forest" in model_name or "Gradient Boosting" in model_name:
            params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100, step=10)
            params["max_depth"] = st.slider("Max Depth", 2, 50, 5)
        elif "Logistic Regression" in model_name:
            params["C"] = st.select_slider(
                "Regularization Strength (C)",
                options=[0.01, 0.1, 1.0, 10.0, 100.0],
                value=1.0,
            )
        elif "Neighbors" in model_name:
            params["n_neighbors"] = st.slider("Number of Neighbors (K)", 1, 50, 5)
        elif "SVM" in model_name or "SVR" in model_name:
            params["C"] = st.select_slider(
                "Regularization (C)", options=[0.1, 1.0, 10.0, 100.0], value=1.0
            )
            kernel_ui = st.selectbox(
                "Kernel", ["Radial Basis Function", "Linear", "Polynomial"]
            )
            kernel_map = {
                "Radial Basis Function": "rbf",
                "Linear": "linear",
                "Polynomial": "poly",
            }
            params["kernel"] = kernel_map[kernel_ui]
        elif "Ridge" in model_name or "Lasso" in model_name:
            params["alpha"] = st.select_slider(
                "Alpha (Penalty)", options=[0.01, 0.1, 1.0, 10.0], value=1.0
            )
        else:
            st.info("No primary hyperparameters to tune for this base model.")

    train_button = st.sidebar.button(
        "Train Pipeline", type="primary", use_container_width=True
    )

    # ==========================================
    # MAIN VIEW: EXECUTION & RESULTS
    # ==========================================
    if not selected_features:
        st.info(
            "👈 Please select at least one Predictor Feature in the sidebar to begin."
        )
        return

    if train_button:
        with st.spinner("Compiling mathematical pipeline and training model..."):
            try:
                # 1. Parse Data Types for the Preprocessor
                X_sample = df[selected_features]
                num_cols = X_sample.select_dtypes(
                    include=["float64", "int64", "float32", "int32"]
                ).columns.tolist()
                cat_cols = X_sample.select_dtypes(
                    include=["object", "category", "string", "bool"]
                ).columns.tolist()

                # 2. Build Pipeline Components
                preprocessor = build_preprocessor(
                    num_cols,
                    cat_cols,
                    num_impute,
                    cat_impute,
                    outlier_method,
                    outlier_action,
                    skew_transform,
                    scaling,
                    encoding,
                )
                model = get_model(task_type, model_name, params, reduce_bias)

                # 3. Execute Training
                results = train_and_evaluate(
                    df,
                    target_col,
                    selected_features,
                    task_type,
                    preprocessor,
                    model,
                    test_size,
                    outlier_method,
                    outlier_action,
                )

                # 4. Save to Session State
                st.session_state["ml_results"] = results
                st.session_state["ml_task"] = task_type
                st.session_state["trained_model_name"] = model_name

            except Exception as e:
                st.error(f"Model Training Failed: {str(e)}")
                st.info(
                    "Tip: If running Classification, ensure your Target Variable is categorical. If Regression, ensure it is numerical."
                )
                return

    # ==========================================
    # MAIN VIEW: DASHBOARD RENDERING
    # ==========================================
    if "ml_results" in st.session_state:
        res = st.session_state["ml_results"]
        metrics = res["metrics"]

        # System status & Drop report
        if res["rows_dropped"] > 0:
            st.warning(
                f"⚠️ **Pre-Split Outlier Filtering:** {res['rows_dropped']:,} outlier rows were physically dropped from the dataset prior to training."
            )

        st.success(
            f"✅ Pipeline executed successfully! Trained on {len(df) - len(res['X_test']) - res['rows_dropped']:,} rows. Evaluated on {len(res['X_test']):,} rows."
        )

        # 1. Top Level Metrics
        trained_model = st.session_state.get("trained_model_name", "Model")
        st.markdown(f"### {trained_model} Performance (Test Set)")
        if st.session_state["ml_task"] == "Classification":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
            c2.metric("F1 Score (Weighted)", f"{metrics['F1 Score']:.3f}")
            c3.metric("Precision", f"{metrics['Precision']:.3f}")
            c4.metric("Recall", f"{metrics['Recall']:.3f}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score (Explained Variance)", f"{metrics['R2 Score']:.3f}")
            c2.metric("RMSE (Root Mean Error)", f"{metrics['RMSE']:.3f}")
            c3.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.3f}")

        # 2. Diagnostic Visualizations
        st.markdown("---")
        t_diag, t_feat = st.tabs(["📊 Diagnostics", "🎯 Feature Importance"])

        with t_diag:
            if st.session_state["ml_task"] == "Classification":
                fig_cm = plot_confusion_matrix(
                    metrics["Confusion Matrix"], metrics["Classes"]
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                fig_res = plot_regression_residuals(res["y_test"], res["y_pred"])
                st.plotly_chart(fig_res, use_container_width=True)

        with t_feat:
            fig_feat = plot_feature_importance(res["pipeline"], selected_features)
            if fig_feat:
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.info(
                    "Feature importance is not natively supported by this algorithm (e.g., KNN, non-linear SVM). Select a Tree-based model or Linear/Logistic Regression to view importance weights."
                )
    else:
        st.info(
            "Configure your ML pipeline in the sidebar and click **Train Pipeline**."
        )
