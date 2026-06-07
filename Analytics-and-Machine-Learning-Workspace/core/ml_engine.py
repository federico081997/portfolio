"""Machine learning pipeline engine for the analytics dashboard.

This module provides the backend logic for the Predictive Modeling & Machine
Learning Studio. It defines custom Scikit-Learn-compatible transformers,
constructs preprocessing pipelines, instantiates supervised learning models,
runs train/test evaluation, and generates diagnostic Plotly visualizations.

The design keeps preprocessing inside Scikit-Learn pipelines where possible so
that transformations are learned from the training data and applied consistently
to the test data.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR

DEFAULT_COLOR_PALETTE = px.colors.qualitative.Prism
DEFAULT_FONT = dict(family="system-ui, -apple-system, sans-serif", size=13)


# =============================================================================
# Custom Scikit-Learn transformers
# =============================================================================


class AdvancedCategoricalImputer(BaseEstimator, TransformerMixin):
    """Impute categorical features using configurable strategies.

    Supports most-frequent imputation, constant ``"Unknown"`` imputation, and
    forward/backward fill. The transformer is compatible with Scikit-Learn
    pipelines and preserves feature names for downstream preprocessing steps.

    Args:
        strategy: Imputation strategy. Supported values are ``"most_frequent"``,
            ``"unknown"``, and ``"ffill"``.
    """

    def __init__(self, strategy: str = "most_frequent"):
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X, y=None):
        """Learn categorical fill values from the training data.

        Args:
            X: Input feature matrix.
            y: Ignored target values, included for Scikit-Learn compatibility.

        Returns:
            The fitted transformer instance.
        """
        if self.strategy == "most_frequent":
            self.fill_values_ = pd.DataFrame(X).mode().iloc[0]

        return self

    def transform(self, X):
        """Apply categorical imputation to a feature matrix.

        Args:
            X: Input feature matrix.

        Returns:
            A NumPy array with missing categorical values imputed.
        """
        x_df = pd.DataFrame(X).copy()

        if self.strategy == "most_frequent":
            x_df = x_df.fillna(self.fill_values_)
        elif self.strategy == "unknown":
            x_df = x_df.fillna("Unknown")
        elif self.strategy == "ffill":
            x_df = x_df.ffill().bfill()

        return x_df.values

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for pipeline introspection.

        Args:
            input_features: Input feature names provided by Scikit-Learn.

        Returns:
            The unchanged input feature names.
        """
        return input_features


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """Learn and apply numerical outlier treatment inside a pipeline.

    Outlier boundaries are learned from the training data during ``fit()`` and
    then applied consistently during ``transform()``. This prevents test-set
    statistics from leaking into the preprocessing stage.

    Args:
        method: Outlier detection method. Supported values are ``"iqr"``,
            ``"z-scores"``, ``"percentiles"``, and ``"none"``.
        action: Outlier handling action. Supported values are ``"clip"`` and
            ``"null"``.
    """

    def __init__(self, method: str = "iqr", action: str = "clip"):
        self.method = method
        self.action = action
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        """Learn outlier bounds from the training data.

        Args:
            X: Input numerical feature matrix.
            y: Ignored target values, included for Scikit-Learn compatibility.

        Returns:
            The fitted transformer instance.
        """
        if self.method == "none":
            return self

        x_df = pd.DataFrame(X)

        if self.method == "iqr":
            q1 = x_df.quantile(0.25)
            q3 = x_df.quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_ = q1 - 1.5 * iqr
            self.upper_bounds_ = q3 + 1.5 * iqr

        elif self.method == "z-scores":
            mean = x_df.mean()
            std = x_df.std()
            self.lower_bounds_ = mean - 3 * std
            self.upper_bounds_ = mean + 3 * std

        elif self.method == "percentiles":
            self.lower_bounds_ = x_df.quantile(0.01)
            self.upper_bounds_ = x_df.quantile(0.99)

        return self

    def transform(self, X):
        """Apply learned outlier treatment to a feature matrix.

        Args:
            X: Input numerical feature matrix.

        Returns:
            A NumPy array with outliers clipped or replaced with missing values.
        """
        if self.method == "none":
            return X

        x_df = pd.DataFrame(X).copy()

        for index, col in enumerate(x_df.columns):
            lower = self.lower_bounds_[index]
            upper = self.upper_bounds_[index]

            if self.action == "clip":
                x_df[col] = x_df[col].clip(lower=lower, upper=upper)

            elif self.action == "null":
                mask = (x_df[col] < lower) | (x_df[col] > upper)
                x_df.loc[mask, col] = np.nan

        return x_df.values

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for pipeline introspection.

        Args:
            input_features: Input feature names provided by Scikit-Learn.

        Returns:
            The unchanged input feature names.
        """
        return input_features


class SafeSkewTransformer(BaseEstimator, TransformerMixin):
    """Apply safe power transformations to numerical features.

    Wraps Scikit-Learn's ``PowerTransformer`` with additional safeguards for
    Box-Cox positivity requirements and floating-point overflow handling.

    Args:
        method: Power transformation method. Supported values are
            ``"yeo-johnson"`` and ``"box-cox"``.
    """

    def __init__(self, method: str = "yeo-johnson"):
        self.method = method
        self.pt = PowerTransformer(method=method, standardize=False)
        self.shift_ = None

    def fit(self, X, y=None):
        """Fit the power transformer on training data.

        Args:
            X: Input numerical feature matrix.
            y: Ignored target values, included for Scikit-Learn compatibility.

        Returns:
            The fitted transformer instance.
        """
        x_df = pd.DataFrame(X)

        if self.method == "box-cox":
            min_values = x_df.min()
            self.shift_ = np.where(min_values <= 0, np.abs(min_values) + 1e-5, 0)
            self.pt.fit(x_df + self.shift_)
        else:
            self.pt.fit(x_df)

        return self

    def transform(self, X):
        """Apply the fitted power transformation.

        Args:
            X: Input numerical feature matrix.

        Returns:
            A transformed NumPy array with invalid floating-point values replaced.
        """
        x_df = pd.DataFrame(X).copy()

        if self.method == "box-cox":
            x_df = x_df + self.shift_
            x_df = x_df.clip(lower=1e-5)

        x_transformed = self.pt.transform(x_df)
        x_transformed = np.nan_to_num(
            x_transformed,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        return x_transformed

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for pipeline introspection.

        Args:
            input_features: Input feature names provided by Scikit-Learn.

        Returns:
            The unchanged input feature names.
        """
        return input_features


# =============================================================================
# Pre-pipeline data cleaning
# =============================================================================


def drop_outliers_safely(
    df: pd.DataFrame,
    num_cols: list,
    method: str,
) -> pd.DataFrame:
    """Remove rows containing outliers before train/test splitting.

    This function is used only when the selected outlier action is physical row
    removal. It is applied before ``train_test_split`` so that the feature matrix
    and target vector remain aligned.

    Args:
        df: Input dataset.
        num_cols: Numerical columns to inspect for outliers.
        method: Outlier detection method. Supported values are ``"IQR"``,
            ``"Z-Scores"``, ``"Percentiles"``, and ``"None"``.

    Returns:
        A DataFrame with outlier rows removed.
    """
    if method == "None" or not num_cols:
        return df

    df_clean = df.copy()

    for col in num_cols:
        if method == "IQR":
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        elif method == "Z-Scores":
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower = mean - 3 * std
            upper = mean + 3 * std

        elif method == "Percentiles":
            lower = df_clean[col].quantile(0.01)
            upper = df_clean[col].quantile(0.99)

        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean


# =============================================================================
# Pipeline architecture
# =============================================================================


def build_preprocessor(
    num_cols: list,
    cat_cols: list,
    num_impute: str,
    cat_impute: str,
    outlier_method: str,
    outlier_action: str,
    skew_transform: str,
    scaling: str,
    encoding: str,
) -> ColumnTransformer:
    """Build a dynamic preprocessing pipeline.

    Constructs separate numerical and categorical pipelines based on the user's
    selected preprocessing configuration.

    Args:
        num_cols: Numerical predictor columns.
        cat_cols: Categorical predictor columns.
        num_impute: Numerical imputation strategy selected in the UI.
        cat_impute: Categorical imputation strategy selected in the UI.
        outlier_method: Outlier detection method selected in the UI.
        outlier_action: Outlier action selected in the UI.
        skew_transform: Skewness transformation selected in the UI.
        scaling: Numerical scaling method selected in the UI.
        encoding: Categorical encoding method selected in the UI.

    Returns:
        A configured Scikit-Learn ``ColumnTransformer``.
    """
    transformers = []

    if num_cols:
        num_steps = []

        if num_impute == "Zero":
            num_steps.append(
                ("imputer", SimpleImputer(strategy="constant", fill_value=0))
            )
        else:
            strategy_map = {
                "Median": "median",
                "Mean": "mean",
                "Most Frequent": "most_frequent",
            }
            num_steps.append(
                ("imputer", SimpleImputer(strategy=strategy_map[num_impute]))
            )

        if outlier_method != "None" and outlier_action != "Drop Rows":
            method_map = {
                "IQR": "iqr",
                "Z-Scores": "z-scores",
                "Percentiles": "percentiles",
            }
            action_map = {
                "Cap/Clip Values": "clip",
                "Replace with Nulls": "null",
            }

            num_steps.append(
                (
                    "outliers",
                    OutlierTreatment(
                        method=method_map[outlier_method],
                        action=action_map[outlier_action],
                    ),
                )
            )

            if action_map[outlier_action] == "null":
                num_steps.append(("re_imputer", SimpleImputer(strategy="median")))

        if skew_transform != "None":
            if skew_transform == "Yeo-Johnson":
                num_steps.append(("skew", SafeSkewTransformer(method="yeo-johnson")))

            elif skew_transform == "Box-Cox":
                num_steps.append(("skew", SafeSkewTransformer(method="box-cox")))

            elif skew_transform == "Log1p":
                num_steps.append(
                    (
                        "skew",
                        FunctionTransformer(
                            lambda x: np.log1p(np.maximum(x, 0)),
                            feature_names_out="one-to-one",
                        ),
                    )
                )

            elif skew_transform == "Square Root":
                num_steps.append(
                    (
                        "skew",
                        FunctionTransformer(
                            lambda x: np.sqrt(np.maximum(x, 0)),
                            feature_names_out="one-to-one",
                        ),
                    )
                )

        if scaling != "None":
            scale_map = {
                "Standard": StandardScaler(),
                "Min-Max": MinMaxScaler(),
                "Robust": RobustScaler(),
                "MaxAbs": MaxAbsScaler(),
            }
            num_steps.append(("scaler", scale_map[scaling]))

        transformers.append(("num", Pipeline(num_steps), num_cols))

    if cat_cols:
        cat_steps = []

        cat_impute_map = {
            'Fill with "Unknown"': "unknown",
            "Most Frequent": "most_frequent",
            "Forward Fill": "ffill",
        }
        cat_steps.append(
            (
                "imputer",
                AdvancedCategoricalImputer(strategy=cat_impute_map[cat_impute]),
            )
        )

        if encoding == "One-Hot Encoding":
            cat_steps.append(
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )

        elif encoding == "Label Encoding":
            cat_steps.append(
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            )

        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_model(
    task_type: str,
    model_name: str,
    params: dict,
    reduce_bias: bool = False,
) -> object | None:
    """Instantiate a supervised learning model.

    Creates a classifier or regressor based on the selected task type and model
    name. For supported classifiers, optional class weighting can be added to
    reduce bias toward majority classes in imbalanced datasets.

    Args:
        task_type: Machine learning task type. Supported values are
            ``"Classification"`` and ``"Regression"``.
        model_name: Name of the selected model.
        params: Hyperparameters passed to the estimator constructor.
        reduce_bias: Whether to apply balanced class weights where supported.

    Returns:
        A configured Scikit-Learn estimator, or ``None`` if no supported model
        matches the requested configuration.
    """
    if reduce_bias and task_type == "Classification":
        if model_name in [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine (SVM)",
        ]:
            params["class_weight"] = "balanced"

    if task_type == "Classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=2000, **params)

        if model_name == "Random Forest":
            return RandomForestClassifier(random_state=42, **params)

        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=42, **params)

        if model_name == "Support Vector Machine (SVM)":
            return SVC(probability=True, **params)

        if model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(**params)

    else:
        if model_name == "Linear Regression":
            return LinearRegression(**params)

        if model_name == "Ridge Regression":
            return Ridge(**params)

        if model_name == "Lasso Regression":
            return Lasso(**params)

        if model_name == "Random Forest":
            return RandomForestRegressor(random_state=42, **params)

        if model_name == "Gradient Boosting":
            return GradientBoostingRegressor(random_state=42, **params)

        if model_name == "Support Vector Regressor (SVR)":
            return SVR(**params)

        if model_name == "K-Nearest Neighbors":
            return KNeighborsRegressor(**params)

    return None


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    features: list,
    task_type: str,
    preprocessor: ColumnTransformer,
    model,
    test_size: float,
    outlier_method: str = "None",
    outlier_action: str = "None",
) -> dict:
    """Train a Scikit-Learn pipeline and evaluate test-set performance.

    Optionally removes outlier rows before splitting, builds a full
    preprocessing/model pipeline, trains it on the training subset, and computes
    classification or regression metrics on the held-out test subset.

    Args:
        df: Input dataset.
        target: Target variable name.
        features: Predictor feature names.
        task_type: Machine learning task type. Supported values are
            ``"Classification"`` and ``"Regression"``.
        preprocessor: Configured preprocessing transformer.
        model: Scikit-Learn estimator to train.
        test_size: Fraction of data reserved for testing.
        outlier_method: Optional pre-split outlier detection method.
        outlier_action: Optional pre-split outlier action.

    Returns:
        A dictionary containing the fitted pipeline, test features, test target,
        predictions, metrics, and number of rows dropped.
    """
    work_df = df.copy()
    rows_dropped = 0

    if outlier_method != "None" and outlier_action == "Drop Rows":
        num_cols = (
            df[features]
            .select_dtypes(include=["float64", "int64", "float32", "int32"])
            .columns.tolist()
        )
        work_df = drop_outliers_safely(work_df, num_cols, outlier_method)
        rows_dropped = len(df) - len(work_df)

    x_data = work_df[features]
    y_data = work_df[target]

    stratify_param = y_data if task_type == "Classification" else None

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=42,
        stratify=stratify_param,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = {}

    if task_type == "Classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["Precision"] = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        metrics["Recall"] = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        metrics["F1 Score"] = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
        metrics["Classes"] = pipeline.classes_

    else:
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics["R2 Score"] = r2_score(y_test, y_pred)

    return {
        "pipeline": pipeline,
        "X_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "rows_dropped": rows_dropped,
    }


# =============================================================================
# Diagnostic visualizations
# =============================================================================


def plot_confusion_matrix(cm, classes) -> go.Figure:
    """Create a confusion-matrix heatmap.

    Args:
        cm: Confusion matrix values.
        classes: Class labels used for x-axis and y-axis tick labels.

    Returns:
        A configured Plotly heatmap figure.
    """
    fig = px.imshow(
        cm,
        text_auto=True,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        title="<b>Confusion Matrix</b>",
        template="plotly_white",
        font=DEFAULT_FONT,
    )

    return fig


def plot_regression_residuals(y_test, y_pred) -> go.Figure:
    """Create an actual-vs-predicted regression diagnostic plot.

    Args:
        y_test: True target values from the test set.
        y_pred: Predicted target values from the model.

    Returns:
        A configured Plotly scatter figure.
    """
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        opacity=0.6,
        labels={"x": "Actual", "y": "Predicted"},
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="Red", dash="dash"),
    )
    fig.update_layout(
        title="<b>Actual vs. Predicted</b>",
        template="plotly_white",
        font=DEFAULT_FONT,
    )

    return fig


def plot_feature_importance(pipeline, feature_names) -> go.Figure | None:
    """Create a feature-importance bar chart where supported.

    Extracts feature importances from tree-based estimators or absolute
    coefficients from linear models. Feature names are retrieved from the fitted
    preprocessing pipeline.

    Args:
        pipeline: Fitted Scikit-Learn pipeline containing ``preprocessor`` and
            ``model`` steps.
        feature_names: Original predictor feature names.

    Returns:
        A configured Plotly bar figure, or ``None`` if the fitted model does not
        expose feature importances or coefficients.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        importances = (
            np.abs(model.coef_).mean(axis=0)
            if model.coef_.ndim > 1
            else np.abs(model.coef_).flatten()
        )

    else:
        return None

    final_features = preprocessor.get_feature_names_out(feature_names)

    importance_df = (
        pd.DataFrame(
            {
                "Feature": final_features,
                "Importance": importances,
            }
        )
        .sort_values(by="Importance", ascending=True)
        .tail(15)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=(
            f"<b>Top {len(importance_df) if len(importance_df) < 15 else 15} "
            "Feature Importances</b>"
        ),
        template="plotly_white",
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
    )
    fig.update_layout(font=DEFAULT_FONT)

    return fig
