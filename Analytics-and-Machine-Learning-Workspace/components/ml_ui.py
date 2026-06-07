"""Streamlit UI for predictive modeling workflows.

This module renders the Predictive Modeling & Machine Learning Studio used by
the Analytics and Machine Learning Workspace. It provides an interactive interface for
configuring preprocessing options, selecting supervised learning models,
tuning hyperparameters, training Scikit-Learn pipelines, and displaying model
diagnostics.

The module uses the cleaned dataset from ``st.session_state`` when available and
falls back to the raw uploaded dataset when preprocessing has not been applied.
Training results are stored in session state so that metrics and diagnostic
figures persist across Streamlit reruns.
"""

import streamlit as st

from core.ml_engine import (
    build_preprocessor,
    get_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_regression_residuals,
    train_and_evaluate,
)


def render_ml_module() -> None:
    """Render the predictive modeling and machine learning interface.

    Displays sidebar controls for task definition, feature selection,
    preprocessing configuration, model selection, hyperparameter tuning, and
    training execution. After training, renders test-set metrics, diagnostic
    plots, and feature-importance visualizations where supported.
    """
    st.title("Predictive Modeling & Machine Learning Studio")
    st.markdown("""
        Welcome to the **Predictive Modeling & Machine Learning Studio** — the
        final predictive stage of the analytics pipeline. This module helps
        configure, train, and evaluate supervised machine learning workflows
        using a structured Scikit-Learn pipeline.

        The workflow is organized around four main stages:

        * **Advanced Preprocessing:** Configure imputation, outlier handling,
          skewness correction, scaling, and categorical encoding.
        * **Feature Selection:** Choose predictor variables and define the target
          variable for classification or regression.
        * **Model Architecture:** Train common supervised learning algorithms
          with configurable hyperparameters.
        * **Performance Diagnostics:** Evaluate test-set performance using
          metrics, confusion matrices, residual plots, and feature importance
          where supported.
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
            "ℹ️ **Notice:** Using the raw uploaded dataset. For optimal ML "
            "performance, consider processing this data in the Data Cleaning & "
            "Preprocessing Studio first."
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

    # -------------------------------------------------------------------------
    # Sidebar: pipeline configuration
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Pipeline Configuration")

    task_type = st.sidebar.radio(
        "Machine Learning Task:",
        ["Classification", "Regression"],
    )
    target_col = st.sidebar.selectbox(
        "Target Variable (Y):",
        all_cols,
        index=len(all_cols) - 1,
    )

    feature_options = [col for col in all_cols if col != target_col]
    selected_features = st.sidebar.multiselect(
        "Predictor Features (X):",
        feature_options,
        placeholder="Choose features...",
        default=feature_options[:5] if len(feature_options) > 5 else feature_options,
    )

    if not selected_features:
        st.info("👈 Please select Predictor Features in the sidebar to begin.")
        return

    # -------------------------------------------------------------------------
    # Sidebar: preprocessing configuration
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Data Preprocessing")

    with st.sidebar.expander("**1. Missing Data Handling**", expanded=True):
        num_impute = st.selectbox(
            "Numerical Imputation:",
            ["Median", "Mean", "Most Frequent", "Zero"],
        )
        cat_impute = st.selectbox(
            "Categorical Imputation:",
            ['Fill with "Unknown"', "Most Frequent", "Forward Fill"],
        )

    with st.sidebar.expander("**2. Outlier Management**", expanded=False):
        outlier_method = st.selectbox(
            "Detection Method:",
            ["None", "IQR", "Z-Scores", "Percentiles"],
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
            "Numerical Scaling:",
            ["None", "Standard", "Min-Max", "Robust", "MaxAbs"],
        )
        encoding = st.selectbox(
            "Categorical Encoding:",
            ["One-Hot Encoding", "Label Encoding"],
        )

    # -------------------------------------------------------------------------
    # Sidebar: model architecture and training parameters
    # -------------------------------------------------------------------------
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

        reduce_bias = False

        if model_name in [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine (SVM)",
        ]:
            reduce_bias = st.sidebar.checkbox(
                "Apply Class Weighting (Reduce Bias)",
                value=False,
                help=(
                    "Applies heavier penalties to mistakes made on minority "
                    "classes. Useful for imbalanced classification datasets."
                ),
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

    test_size = (
        st.sidebar.slider(
            "Test Set Size (%):",
            10,
            50,
            20,
            step=5,
        )
        / 100.0
    )

    params = {}

    with st.sidebar.expander("**Tune Hyperparameters**", expanded=True):
        if "Random Forest" in model_name or "Gradient Boosting" in model_name:
            params["n_estimators"] = st.slider(
                "Number of Trees",
                10,
                500,
                100,
                step=10,
            )
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
                "Regularization (C)",
                options=[0.1, 1.0, 10.0, 100.0],
                value=1.0,
            )
            kernel_ui = st.selectbox(
                "Kernel",
                ["Radial Basis Function", "Linear", "Polynomial"],
            )
            kernel_map = {
                "Radial Basis Function": "rbf",
                "Linear": "linear",
                "Polynomial": "poly",
            }
            params["kernel"] = kernel_map[kernel_ui]

        elif "Ridge" in model_name or "Lasso" in model_name:
            params["alpha"] = st.select_slider(
                "Alpha (Penalty)",
                options=[0.01, 0.1, 1.0, 10.0],
                value=1.0,
            )

        else:
            st.info("No primary hyperparameters to tune for this base model.")

    train_button = st.sidebar.button(
        "Train Pipeline",
        type="primary",
        width="stretch",
    )

    # -------------------------------------------------------------------------
    # Main page: training execution
    # -------------------------------------------------------------------------
    if not selected_features:
        st.info(
            "👈 Please select at least one Predictor Feature in the sidebar to begin."
        )
        return

    if train_button:
        with st.spinner("Compiling mathematical pipeline and training model..."):
            try:
                x_sample = df[selected_features]
                num_cols = x_sample.select_dtypes(
                    include=["float64", "int64", "float32", "int32"]
                ).columns.tolist()
                cat_cols = x_sample.select_dtypes(
                    include=["object", "category", "string", "bool"]
                ).columns.tolist()

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

                st.session_state["ml_results"] = results
                st.session_state["ml_task"] = task_type
                st.session_state["trained_model_name"] = model_name

            except Exception as error:
                st.error(f"❌ Model Training Failed: {error}")
                st.info(
                    "Tip: If running Classification, ensure your target variable "
                    "is categorical. If running Regression, ensure it is numerical."
                )
                return

    # -------------------------------------------------------------------------
    # Main page: results dashboard
    # -------------------------------------------------------------------------
    if "ml_results" in st.session_state:
        results = st.session_state["ml_results"]
        metrics = results["metrics"]

        if results["rows_dropped"] > 0:
            st.warning(
                f"⚠️ **Pre-Split Outlier Filtering:** "
                f"{results['rows_dropped']:,} outlier rows were physically "
                "dropped from the dataset prior to training."
            )

        st.success(
            f"✅ Pipeline executed successfully! Trained on "
            f"{len(df) - len(results['X_test']) - results['rows_dropped']:,} rows. "
            f"Evaluated on {len(results['X_test']):,} rows."
        )

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

        st.markdown("---")
        t_diag, t_feat = st.tabs(["📊 Diagnostics", "🎯 Feature Importance"])

        with t_diag:
            if st.session_state["ml_task"] == "Classification":
                fig_cm = plot_confusion_matrix(
                    metrics["Confusion Matrix"],
                    metrics["Classes"],
                )
                st.plotly_chart(fig_cm, width="stretch")

            else:
                fig_res = plot_regression_residuals(
                    results["y_test"],
                    results["y_pred"],
                )
                st.plotly_chart(fig_res, width="stretch")

        with t_feat:
            fig_feat = plot_feature_importance(
                results["pipeline"],
                selected_features,
            )

            if fig_feat:
                st.plotly_chart(fig_feat, width="stretch")
            else:
                st.info(
                    "Feature importance is not natively supported by this "
                    "algorithm, such as KNN or non-linear SVM. Select a "
                    "tree-based model or a linear/logistic regression model to "
                    "view importance weights."
                )

    else:
        st.info(
            "👆 Configure your ML pipeline in the sidebar and click **Train Pipeline**."
        )
