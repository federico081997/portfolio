"""Deep learning training engine for the analytics dashboard.

This module provides the backend logic for dynamically constructing, training,
and evaluating TensorFlow/Keras neural networks. It integrates the shared
Scikit-Learn preprocessing architecture with configurable dense layers,
regularization layers, optimizers, callbacks, and task-specific output layers.

The implementation includes reproducible random seeds, gradient clipping,
target validation, sparse-to-dense conversion, float32 enforcement, callback-
based Streamlit updates, and defensive extraction of model architecture details.
"""

import os
import logging
import warnings

# Reduce TensorFlow C++ logging before TensorFlow is imported.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Specifically silence TensorFlow deprecation warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# Silence all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Silence all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential

from core.ml_engine import build_preprocessor


class StreamlitLiveUpdate(tf.keras.callbacks.Callback):
    """Stream neural-network training progress to a Streamlit UI callback.

    Stores epoch-level metrics and calls a supplied UI update function after
    each completed training epoch.

    Args:
        ui_update_function: Callable used to update the Streamlit interface.
        total_epochs: Maximum number of configured training epochs.
    """

    def __init__(
        self,
        ui_update_function: Callable | None,
        total_epochs: int,
    ) -> None:
        """Initialize the live-training callback.

        Args:
            ui_update_function: Callable used to update the Streamlit interface.
            total_epochs: Maximum number of configured training epochs.
        """
        super().__init__()
        self.ui_update_function = ui_update_function
        self.total_epochs = total_epochs
        self.live_history: dict[str, list[Any]] = {}

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """Record epoch metrics and update the Streamlit interface.

        Args:
            epoch: Zero-based index of the completed epoch.
            logs: Metrics recorded by Keras for the completed epoch.
        """
        if logs is None:
            return

        for key, value in logs.items():
            self.live_history.setdefault(key, []).append(value)

        if self.ui_update_function:
            self.ui_update_function(
                epoch,
                self.total_epochs,
                logs,
                self.live_history,
            )


def build_dynamic_network(
    input_dim: int,
    task_type: str,
    num_classes: int,
    layer_config: list,
    optimizer_name: str,
    learning_rate: float,
) -> tf.keras.Model:
    """Build and compile a configurable Keras neural network.

    Creates a sequential neural network from the supplied layer configuration,
    appends a task-specific output layer, and compiles the model using the
    selected optimizer with gradient clipping.

    Args:
        input_dim: Number of features produced by the preprocessing pipeline.
        task_type: Neural-network task type.
        num_classes: Number of encoded classes for multiclass classification.
        layer_config: Ordered configuration of hidden network layers.
        optimizer_name: Name of the optimizer selected by the user.
        learning_rate: Learning rate supplied to the optimizer.

    Returns:
        A compiled TensorFlow/Keras model.

    Raises:
        KeyError: If ``optimizer_name`` is not present in the supported
            optimizer mapping.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for layer in layer_config:
        if layer["type"] == "Dense":
            model.add(
                Dense(
                    units=layer["units"],
                    activation=layer["activation"],
                )
            )
        elif layer["type"] == "Dropout":
            model.add(Dropout(rate=layer["rate"]))
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())

    if task_type == "Classification (Binary)":
        model.add(Dense(1, activation="sigmoid"))
        loss_function = "binary_crossentropy"
        training_metrics = ["accuracy"]

    elif task_type == "Classification (Multi-Class)":
        model.add(Dense(num_classes, activation="softmax"))
        loss_function = "sparse_categorical_crossentropy"
        training_metrics = ["accuracy"]

    else:
        model.add(Dense(1, activation="linear"))
        loss_function = "mse"
        training_metrics = ["mae"]

    optimizers = {
        "Adam": tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        "SGD (with Momentum)": tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            clipnorm=1.0,
        ),
        "RMSprop": tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        "Nadam": tf.keras.optimizers.Nadam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
    }

    model.compile(
        optimizer=optimizers[optimizer_name],
        loss=loss_function,
        metrics=training_metrics,
        jit_compile=True,
    )

    return model


def train_neural_network(
    df: pd.DataFrame,
    target: str,
    features: list,
    task_type: str,
    layer_config: list,
    optimizer_name: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    test_size: float,
    use_early_stopping: bool,
    use_lr_decay: bool,
    num_impute: str,
    cat_impute: str,
    outlier_method: str,
    outlier_action: str,
    skew_transform: str,
    scaling: str,
    encoding: str,
    st_callback_func: Callable | None = None,
) -> dict:
    """Train and evaluate a dynamically configured neural network.

    Validates and prepares the target variable, splits the data before fitting
    the preprocessing pipeline, converts processed features to dense float32
    arrays, trains the configured neural network, evaluates test-set
    performance, and extracts architecture metadata.

    Args:
        df: Input dataset.
        target: Target column name.
        features: Predictor column names.
        task_type: Deep-learning task type.
        layer_config: Ordered configuration of hidden network layers.
        optimizer_name: Name of the selected optimizer.
        learning_rate: Optimizer learning rate.
        epochs: Maximum number of training epochs.
        batch_size: Number of samples processed per gradient update.
        test_size: Fraction of data reserved for validation and testing.
        use_early_stopping: Whether to stop training when validation loss stops
            improving.
        use_lr_decay: Whether to reduce the learning rate when validation loss
            plateaus.
        num_impute: Numerical imputation strategy.
        cat_impute: Categorical imputation strategy.
        outlier_method: Outlier detection strategy.
        outlier_action: Outlier treatment strategy.
        skew_transform: Numerical skewness transformation.
        scaling: Numerical scaling strategy.
        encoding: Categorical encoding strategy.
        st_callback_func: Optional callback used to stream epoch metrics to the
            Streamlit interface.

    Returns:
        A dictionary containing architecture details, parameter counts, training
        history, evaluation metrics, targets, predictions, and epochs completed.

    Raises:
        ValueError: If no rows remain after target validation or if a regression
            target contains no valid numerical values.
        KeyError: If an unsupported optimizer or preprocessing option is passed.
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    work_df = df.dropna(subset=[target]).copy()

    if work_df.empty:
        raise ValueError(
            "Dataset is empty after removing rows with missing target values."
        )

    x_data = work_df[features].copy()
    y_data = work_df[target].copy()

    if "Classification" in task_type:
        y_data = y_data.astype(str)

        label_encoder = LabelEncoder()
        y_data = label_encoder.fit_transform(y_data)
        num_classes = len(np.unique(y_data))

        class_counts = pd.Series(y_data).value_counts()
        stratify_param = y_data if class_counts.min() > 1 else None

    else:
        y_data = pd.to_numeric(y_data, errors="coerce")
        valid_target_mask = y_data.notna()

        y_data = y_data[valid_target_mask]
        x_data = x_data[valid_target_mask]

        if len(y_data) == 0:
            raise ValueError(
                "No valid numeric values found in the target column for Regression."
            )

        y_data = np.asarray(y_data).astype(np.float32)
        num_classes = 1
        stratify_param = None

    num_cols = x_data.select_dtypes(
        include=["float64", "int64", "float32", "int32"]
    ).columns.tolist()
    cat_cols = x_data.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    if cat_cols:
        x_data[cat_cols] = x_data[cat_cols].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=42,
        stratify=stratify_param,
    )

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

    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    if scipy.sparse.issparse(x_train_processed):
        x_train_processed = x_train_processed.toarray()
        x_test_processed = x_test_processed.toarray()

    x_train_processed = np.asarray(x_train_processed).astype(np.float32)
    x_test_processed = np.asarray(x_test_processed).astype(np.float32)

    input_dimension = x_train_processed.shape[1]

    model = build_dynamic_network(
        input_dim=input_dimension,
        task_type=task_type,
        num_classes=num_classes,
        layer_config=layer_config,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
    )

    callbacks = []

    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=0,
            )
        )

    if use_lr_decay:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            )
        )

    if st_callback_func:
        callbacks.append(
            StreamlitLiveUpdate(
                st_callback_func,
                epochs,
            )
        )

    history = model.fit(
        x_train_processed,
        y_train,
        validation_data=(x_test_processed, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    predictions = model.predict(x_test_processed, verbose=0, steps=None, callbacks=None)
    metrics = {}

    if task_type == "Classification (Binary)":
        y_pred = (predictions > 0.5).astype(int).flatten()

        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["F1 Score"] = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )
        metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)

    elif task_type == "Classification (Multi-Class)":
        y_pred = np.argmax(predictions, axis=1)

        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["F1 Score"] = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)

    else:
        y_pred = predictions.flatten()

        metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics["R2 Score"] = r2_score(y_test, y_pred)

    architecture_details = []

    for layer in model.layers:
        try:
            output_shape = str(layer.output_shape)
        except AttributeError:
            try:
                output_shape = str(layer.output.shape)
            except AttributeError:
                output_shape = "Dynamic"

        architecture_details.append(
            {
                "Layer Name": layer.name,
                "Layer Type": layer.__class__.__name__,
                "Output Shape": output_shape,
                "Parameters": layer.count_params(),
            }
        )

    try:
        trainable_count = (
            int(
                np.sum(
                    [
                        tf.keras.backend.count_params(weight)
                        for weight in model.trainable_weights
                    ]
                )
            )
            if model.trainable_weights
            else 0
        )
    except Exception:
        try:
            trainable_count = sum(
                int(tf.size(variable)) for variable in model.trainable_variables
            )
        except Exception:
            trainable_count = 0

    return {
        "architecture": architecture_details,
        "total_params": model.count_params(),
        "trainable_params": trainable_count,
        "history": history.history,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "epochs_run": len(history.history["loss"]),
    }


def plot_dl_learning_curves(history: dict) -> go.Figure:
    """Create neural-network training and validation learning curves.

    Displays loss curves together with either accuracy or mean absolute error,
    depending on the task-specific metrics recorded during model training.

    Args:
        history: Keras history dictionary containing epoch-level metrics.

    Returns:
        A configured Plotly figure containing the available learning curves.
    """
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["loss"],
            mode="lines",
            name="Train Loss",
            line=dict(color="#EF553B"),
        )
    )

    if "val_loss" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                mode="lines",
                name="Validation Loss",
                line=dict(color="#EF553B", dash="dash"),
            )
        )

    metric_key = "accuracy" if "accuracy" in history else "mae"

    if metric_key in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history[metric_key],
                mode="lines",
                name=f"Train {metric_key.title()}",
                line=dict(color="#00CC96"),
            )
        )

        validation_metric = f"val_{metric_key}"

        if validation_metric in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[validation_metric],
                    mode="lines",
                    name=f"Validation {metric_key.title()}",
                    line=dict(color="#00CC96", dash="dash"),
                )
            )

    fig.update_layout(
        title="<b>Neural Network Learning Curves</b>",
        xaxis_title="Epoch",
        yaxis_title="Metric Value",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
