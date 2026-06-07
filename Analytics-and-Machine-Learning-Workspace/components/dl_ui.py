"""Streamlit UI for deep learning model construction and evaluation.

This module renders the Deep Learning Studio used by the
analytics dashboard. It provides controls for selecting data, configuring
preprocessing, building neural network architectures layer by layer, selecting
optimization settings, monitoring live training progress, evaluating model
performance, and exporting training artifacts.

The module uses the cleaned dataset from ``st.session_state`` when available and
falls back to the raw uploaded dataset when preprocessing has not been applied.
Trained model results are stored in session state so that metrics, learning
curves, diagnostics, and architecture details persist across Streamlit reruns.
"""

import json

import pandas as pd
import streamlit as st

from core.dl_engine import plot_dl_learning_curves, train_neural_network
from core.ml_engine import (
    plot_confusion_matrix,
    plot_regression_residuals,
)


def initialize_layer_state() -> None:
    """Initialize the default neural network architecture in session state.

    Creates the default layer configuration only when no deep learning
    architecture has previously been stored in the current Streamlit session.
    """
    if "dl_layers" not in st.session_state:
        st.session_state["dl_layers"] = [
            {
                "type": "Dense",
                "units": 64,
                "activation": "relu",
            },
            {
                "type": "BatchNormalization",
            },
            {
                "type": "Dropout",
                "rate": 0.2,
            },
            {
                "type": "Dense",
                "units": 32,
                "activation": "relu",
            },
        ]


def render_dl_module() -> None:
    """Render the deep learning architecture and training interface.

    Displays controls for task selection, preprocessing, network construction,
    optimization, callbacks, and execution parameters. After training, renders
    test-set metrics, learning curves, prediction diagnostics, architecture
    details, and downloadable training artifacts.
    """
    st.title("Deep Learning Studio")
    st.markdown("""
        Welcome to the **Deep Learning Studio** — an interactive workspace for
        constructing, training, and evaluating fully connected neural networks.

        This module supports the complete deep learning workflow:

        * **Data Configuration:** Select classification or regression tasks,
          target variables, and predictor features.
        * **Preprocessing:** Configure missing-value treatment, outlier handling,
          skewness correction, scaling, and categorical encoding.
        * **Network Architecture:** Build a custom topology using Dense,
          Dropout, and Batch Normalization layers.
        * **Network Compilation:** Select an optimizer, learning rate, training
          callbacks, epoch limit, batch size, and validation split.
        * **Training Diagnostics:** Monitor live losses, evaluate unseen test
          data, inspect the compiled architecture, and export training artifacts.
        """)
    st.markdown("---")

    initialize_layer_state()

    # -------------------------------------------------------------------------
    # Data routing and validation
    # -------------------------------------------------------------------------
    if (
        "cleaned_data" in st.session_state
        and st.session_state["cleaned_data"] is not None
    ):
        df = st.session_state["cleaned_data"].copy()
        st.caption(
            "✨ Using processed data from the Data Cleaning & Preprocessing Studio."
        )

    elif "raw_data" in st.session_state and st.session_state["raw_data"] is not None:
        df = st.session_state["raw_data"].copy()
        st.info(
            "ℹ️ **Notice:** Using the raw uploaded dataset. For optimal deep "
            "learning performance, consider processing this data in the Data "
            "Cleaning & Preprocessing Studio first."
        )

    else:
        st.warning("⚠️ No data available. Please upload a dataset first.")
        return

    if df.empty or len(df) < 50:
        st.error(
            "❌ The dataset is too small for Deep Learning. Please provide a "
            "dataset with at least 50 rows."
        )
        return

    all_cols = df.columns.tolist()

    # -------------------------------------------------------------------------
    # Sidebar: task and feature configuration
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Pipeline Configuration")

    task_type = st.sidebar.selectbox(
        "Deep Learning Task:",
        [
            "Classification (Binary)",
            "Classification (Multi-Class)",
            "Regression",
        ],
    )
    target_col = st.sidebar.selectbox(
        "Target Variable (Y):",
        all_cols,
        index=len(all_cols) - 1,
    )

    df = df.dropna(subset=[target_col]).copy()

    if df.empty:
        st.error(
            "❌ The dataset is empty after removing rows with missing target values."
        )
        return

    feature_options = [col for col in all_cols if col != target_col]
    selected_features = st.sidebar.multiselect(
        "Predictor Features (X):",
        feature_options,
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
            ["Standard", "Min-Max", "Robust", "MaxAbs", "None"],
        )
        encoding = st.selectbox(
            "Categorical Encoding:",
            ["One-Hot Encoding", "Label Encoding", "None"],
        )

    # -------------------------------------------------------------------------
    # Sidebar: neural network topology
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Network Topology")

    with st.sidebar.expander("**Manage Network Layers**", expanded=True):
        for index, layer in enumerate(st.session_state["dl_layers"]):
            layer_description, remove_column = st.columns([4, 1])

            with layer_description:
                if layer["type"] == "Dense":
                    activation_name = (
                        layer["activation"].upper()
                        if layer["activation"] != "swish"
                        else "Swish"
                    )
                    st.markdown(
                        f"**L{index + 1}: Dense** | "
                        f"{layer['units']} units | {activation_name}"
                    )

                elif layer["type"] == "Dropout":
                    st.markdown(f"**L{index + 1}: Dropout** | Rate: {layer['rate']}")

                elif layer["type"] == "BatchNormalization":
                    st.markdown(f"**L{index + 1}: Batch Normalization**")

            with remove_column:
                if st.button(
                    "❌",
                    key=f"del_{index}",
                    help="Remove Layer",
                ):
                    st.session_state["dl_layers"].pop(index)
                    st.rerun()

        st.markdown("---")

        new_layer_type_ui = st.selectbox(
            "Select Layer Type:",
            ["Dense", "Dropout", "Batch Normalization"],
        )

        layer_type_map = {
            "Dense": "Dense",
            "Dropout": "Dropout",
            "Batch Normalization": "BatchNormalization",
        }
        backend_layer_type = layer_type_map[new_layer_type_ui]

        if backend_layer_type == "Dense":
            units_column, activation_column = st.columns(2)

            n_units = units_column.number_input(
                "Neurons",
                min_value=2,
                max_value=1024,
                value=64,
                step=8,
            )

            activation_map = {
                "ReLU (Rectified Linear)": "relu",
                "ELU (Exponential Linear)": "elu",
                "Tanh (Hyperbolic Tangent)": "tanh",
                "Sigmoid (Logistic)": "sigmoid",
                "Swish (Self-Gated)": "swish",
            }
            activation_label = activation_column.selectbox(
                "Activation Function",
                list(activation_map.keys()),
            )

            if st.button("➕ Add Dense Layer", width="stretch"):
                st.session_state["dl_layers"].append(
                    {
                        "type": "Dense",
                        "units": n_units,
                        "activation": activation_map[activation_label],
                    }
                )
                st.rerun()

        elif backend_layer_type == "Dropout":
            dropout_rate = st.slider(
                "Dropout Rate",
                min_value=0.05,
                max_value=0.8,
                value=0.2,
                step=0.05,
            )

            if st.button("➕ Add Dropout Layer", width="stretch"):
                st.session_state["dl_layers"].append(
                    {
                        "type": "Dropout",
                        "rate": dropout_rate,
                    }
                )
                st.rerun()

        elif backend_layer_type == "BatchNormalization":
            if st.button(
                "➕ Add Batch Normalization",
                width="stretch",
            ):
                st.session_state["dl_layers"].append(
                    {
                        "type": "BatchNormalization",
                    }
                )
                st.rerun()

    # -------------------------------------------------------------------------
    # Sidebar: compilation and training configuration
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Network Compilation")

    optimizer_map = {
        "Adam (Adaptive Moment Estimation)": "Adam",
        "Nadam (Nesterov-accelerated Adam)": "Nadam",
        "SGD (Stochastic Gradient Descent)": "SGD (with Momentum)",
        "RMSprop (Root Mean Square Prop)": "RMSprop",
    }

    optimizer_ui = st.sidebar.selectbox(
        "Optimization Algorithm:",
        list(optimizer_map.keys()),
    )
    optimizer_name = optimizer_map[optimizer_ui]

    learning_rate = st.sidebar.select_slider(
        "Initial Learning Rate:",
        options=[0.0001, 0.001, 0.005, 0.01, 0.1],
        value=0.001,
        help=(
            "Controls the gradient-descent step size. A value that is too high "
            "may cause instability, while a value that is too low may slow "
            "convergence."
        ),
    )

    st.sidebar.markdown("**Training Callbacks**")

    use_early_stopping = st.sidebar.checkbox(
        "Enable Early Stopping",
        value=True,
        help=(
            "Stops training when validation loss no longer improves, helping "
            "reduce overfitting."
        ),
    )
    use_lr_decay = st.sidebar.checkbox(
        "Enable Learning Rate Decay",
        value=True,
        help=(
            "Reduces the learning rate after a validation plateau to support "
            "finer convergence."
        ),
    )

    st.sidebar.markdown("**Execution Parameters**")

    epochs = st.sidebar.slider(
        "Maximum Training Epochs:",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
    )
    batch_size = st.sidebar.select_slider(
        "Batch Size:",
        options=[8, 16, 32, 64, 128, 256, 512],
        value=32,
        help="Number of samples processed before each gradient update.",
    )
    test_size = st.sidebar.slider(
        "Validation Split:",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )

    train_button = st.sidebar.button(
        "Compile & Train Network",
        type="primary",
        width="stretch",
    )

    # -------------------------------------------------------------------------
    # Main page: validation, execution, and live telemetry
    # -------------------------------------------------------------------------
    if train_button:
        active_features = df[selected_features]
        categorical_features_present = (
            active_features.select_dtypes(
                include=["object", "string", "category", "bool"]
            ).shape[1]
            > 0
        )

        if not st.session_state["dl_layers"]:
            st.error(
                "❌ Your network topology is empty. Please add at least one "
                "Dense layer in the sidebar."
            )
            return

        if categorical_features_present and encoding == "None":
            st.error(
                "❌ Deep learning tensors cannot process raw text values. Select "
                "a categorical encoding strategy in the preprocessing settings."
            )
            return

        estimated_train_rows = int(len(df) * (1 - test_size))

        if batch_size > estimated_train_rows:
            st.error(
                f"❌ Your batch size ({batch_size}) is larger than the available "
                f"training data ({estimated_train_rows} rows). Reduce the batch "
                "size or provide a larger dataset."
            )
            return

        target_series = df[target_col]
        target_is_text = target_series.dtype in [
            "object",
            "string",
            "category",
        ]
        unique_targets = target_series.nunique()

        if task_type == "Regression" and target_is_text:
            sample_value = target_series.iloc[0]
            st.error(
                f"❌ **Target Mismatch:** You selected Regression, but "
                f"'{target_col}' contains text values such as '{sample_value}'. "
                "Regression requires a continuous numerical target. Select a "
                "classification task instead."
            )
            return

        if task_type == "Classification (Binary)" and unique_targets > 2:
            st.error(
                f"❌ **Target Mismatch:** You selected binary classification, "
                f"but '{target_col}' contains {unique_targets} unique classes. "
                "Select multi-class classification instead."
            )
            return

        if task_type == "Classification (Multi-Class)" and unique_targets > 50:
            st.warning(
                f"⚠️ **High Cardinality Warning:** The target contains "
                f"{unique_targets} unique classes. The network may struggle to "
                "converge unless the dataset contains sufficient observations "
                "for each class."
            )

        st.markdown("### Live Training Telemetry")

        ui_progress_bar = st.progress(0)
        ui_status_text = st.empty()
        ui_live_chart = st.empty()

        st.markdown("---")

        def update_live_ui(
            epoch: int,
            total_epochs: int,
            logs: dict,
            live_history: dict,
        ) -> None:
            """Update live Streamlit training telemetry.

            Args:
                epoch: Zero-based index of the completed training epoch.
                total_epochs: Maximum number of epochs allocated to training.
                logs: Training metrics reported for the current epoch.
                live_history: Accumulated training and validation history.
            """
            progress_fraction = (epoch + 1) / total_epochs
            ui_progress_bar.progress(progress_fraction)

            validation_loss = logs.get("val_loss", 0)
            training_loss = logs.get("loss", 0)

            ui_status_text.markdown(
                f"**Epoch:** `{epoch + 1} / {total_epochs}` | "
                f"**Training Loss:** `{training_loss:.4f}` | "
                f"**Validation Loss:** `{validation_loss:.4f}`"
            )

            live_figure = plot_dl_learning_curves(live_history)
            ui_live_chart.plotly_chart(
                live_figure,
                width="stretch",
            )

        with st.spinner(
            f"Executing training graph on {estimated_train_rows:,} rows..."
        ):
            try:
                results = train_neural_network(
                    df,
                    target_col,
                    selected_features,
                    task_type,
                    st.session_state["dl_layers"],
                    optimizer_name,
                    learning_rate,
                    epochs,
                    batch_size,
                    test_size,
                    use_early_stopping,
                    use_lr_decay,
                    num_impute,
                    cat_impute,
                    outlier_method,
                    outlier_action,
                    skew_transform,
                    scaling,
                    encoding,
                    st_callback_func=update_live_ui,
                )

                st.session_state["dl_results"] = results
                st.session_state["dl_task"] = task_type

            except Exception as error:
                st.error(f"❌ TensorFlow Execution Failed: {error}")
                return

    # -------------------------------------------------------------------------
    # Main page: results dashboard
    # -------------------------------------------------------------------------
    if "dl_results" in st.session_state:
        results = st.session_state["dl_results"]
        metrics = results["metrics"]
        task = st.session_state["dl_task"]

        st.success(
            f"✅ Training complete! Convergence was reached at "
            f"**Epoch {results['epochs_run']}** out of the maximum allocated "
            f"{epochs}."
        )

        st.markdown("### Neural Network Performance (Test Set)")

        if "Classification" in task:
            metric_accuracy, metric_f1, metric_samples = st.columns(3)

            metric_accuracy.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.4f}",
            )
            metric_f1.metric(
                "F1 Score",
                f"{metrics['F1 Score']:.4f}",
            )
            metric_samples.metric(
                "Evaluated Samples",
                f"{len(results['y_test']):,}",
            )

        else:
            metric_r2, metric_rmse, metric_mae = st.columns(3)

            metric_r2.metric(
                "R² Score",
                f"{metrics['R2 Score']:.4f}",
            )
            metric_rmse.metric(
                "RMSE",
                f"{metrics['RMSE']:.4f}",
            )
            metric_mae.metric(
                "MAE",
                f"{metrics['MAE']:.4f}",
            )

        st.markdown("---")

        tab_curves, tab_diagnostics, tab_architecture = st.tabs(
            [
                "📉 Final Learning Curves",
                "📊 Output Diagnostics",
                "🧠 Final Architecture",
            ]
        )

        with tab_curves:
            history_figure = plot_dl_learning_curves(results["history"])
            st.plotly_chart(
                history_figure,
                width="stretch",
                key="final_dl_learning_curve_chart",
            )
            st.caption(
                "A validation loss that diverges from training loss may indicate "
                "overfitting."
            )

        with tab_diagnostics:
            if "Classification" in task:
                try:
                    source_df = st.session_state.get("raw_data", df)
                    original_labels = sorted(
                        source_df[target_col].dropna().unique().tolist()
                    )
                    class_names = [str(label) for label in original_labels]

                except Exception:
                    class_names = [
                        str(index) for index in range(len(metrics["Confusion Matrix"]))
                    ]

                confusion_figure = plot_confusion_matrix(
                    metrics["Confusion Matrix"],
                    class_names,
                )
                st.plotly_chart(
                    confusion_figure,
                    width="stretch",
                )

            else:
                residual_figure = plot_regression_residuals(
                    results["y_test"],
                    results["y_pred"],
                )
                st.plotly_chart(
                    residual_figure,
                    width="stretch",
                )

        with tab_architecture:
            st.markdown("### Compiled Computation Graph")

            metric_total, metric_trainable, metric_non_trainable = st.columns(3)

            metric_total.metric(
                "Total Parameters",
                f"{results['total_params']:,}",
            )
            metric_trainable.metric(
                "Trainable Parameters",
                f"{results['trainable_params']:,}",
            )
            metric_non_trainable.metric(
                "Non-Trainable",
                (f"{results['total_params'] - results['trainable_params']:,}"),
            )

            st.markdown("---")

            architecture_df = pd.DataFrame(results["architecture"])
            styled_architecture = architecture_df.style.format({"Parameters": "{:,}"})

            st.dataframe(
                styled_architecture,
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "The `None` value in an output shape represents the dynamic "
                "batch-size dimension."
            )

        st.markdown("---")
        st.markdown("### 📥 Export Artifacts")
        st.markdown(
            "Download the training history and architecture configuration for "
            "portfolio documentation or downstream deployment workflows."
        )

        history_column, architecture_column = st.columns(2)

        with history_column:
            history_df = pd.DataFrame(results["history"])
            history_df.index.name = "Epoch"
            csv_data = history_df.to_csv().encode("utf-8")

            st.download_button(
                label="📊 Download Training History (CSV)",
                data=csv_data,
                file_name="training_logs.csv",
                mime="text/csv",
                width="stretch",
            )

        with architecture_column:
            architecture_json = json.dumps(
                results["architecture"],
                indent=4,
            )

            st.download_button(
                label="⚙️ Download Architecture Profile (JSON)",
                data=architecture_json,
                file_name="model_architecture.json",
                mime="application/json",
                width="stretch",
            )

    else:
        st.info(
            "👆 Configure your parameters in the sidebar and click "
            "**Compile & Train Network**."
        )
