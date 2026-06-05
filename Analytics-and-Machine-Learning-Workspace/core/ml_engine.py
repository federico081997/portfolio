"""
Advanced Machine Learning Engine Module

Provides an enterprise-grade, dataset-agnostic pipeline architecture.
Features custom Scikit-Learn transformers for safe, leakage-free outlier
management, skewness transformation, advanced imputation algorithms, and
algorithmic bias reduction techniques.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    FunctionTransformer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CUSTOM SCIKIT-LEARN TRANSFORMERS
# ==========================================


class AdvancedCategoricalImputer(BaseEstimator, TransformerMixin):
    """Custom imputer allowing Forward Fill and 'Unknown' constants safely."""

    def __init__(self, strategy="most_frequent"):
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X, y=None):
        if self.strategy == "most_frequent":
            self.fill_values_ = pd.DataFrame(X).mode().iloc[0]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if self.strategy == "most_frequent":
            X_df = X_df.fillna(self.fill_values_)
        elif self.strategy == "unknown":
            X_df = X_df.fillna("Unknown")
        elif self.strategy == "ffill":
            X_df = X_df.ffill().bfill()
        return X_df.values

    # THE FIX: Allow pipeline to trace feature names
    def get_feature_names_out(self, input_features=None):
        return input_features


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """
    Learns outlier boundaries on the training set and safely applies
    capping/clipping or nullification to the test set without data leakage.
    """

    def __init__(self, method="iqr", action="clip"):
        self.method = method  # 'iqr', 'z-scores', 'percentiles', 'none'
        self.action = action  # 'clip', 'null'
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        if self.method == "none":
            return self

        X_df = pd.DataFrame(X)
        if self.method == "iqr":
            Q1 = X_df.quantile(0.25)
            Q3 = X_df.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_ = Q1 - 1.5 * IQR
            self.upper_bounds_ = Q3 + 1.5 * IQR
        elif self.method == "z-scores":
            mean = X_df.mean()
            std = X_df.std()
            self.lower_bounds_ = mean - 3 * std
            self.upper_bounds_ = mean + 3 * std
        elif self.method == "percentiles":
            self.lower_bounds_ = X_df.quantile(0.01)
            self.upper_bounds_ = X_df.quantile(0.99)
        return self

    def transform(self, X):
        if self.method == "none":
            return X

        X_df = pd.DataFrame(X).copy()
        for i, col in enumerate(X_df.columns):
            lower, upper = self.lower_bounds_[i], self.upper_bounds_[i]
            if self.action == "clip":
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
            elif self.action == "null":
                mask = (X_df[col] < lower) | (X_df[col] > upper)
                X_df.loc[mask, col] = np.nan
        return X_df.values

    # THE FIX: Allow pipeline to trace feature names
    def get_feature_names_out(self, input_features=None):
        return input_features


class SafeSkewTransformer(BaseEstimator, TransformerMixin):
    """
    Safely applies Power Transformations (Box-Cox or Yeo-Johnson).
    Prevents float64 overflow crashes by clipping extreme outputs,
    turning off internal standardization, and catching Infinity/NaNs.
    """

    def __init__(self, method="yeo-johnson"):
        self.method = method
        # standardize=False prevents the mean(inf) == NaN cascade
        self.pt = PowerTransformer(method=method, standardize=False)
        self.shift_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        if self.method == "box-cox":
            # Shift data to strictly > 0 for Box-Cox
            min_vals = X_df.min()
            self.shift_ = np.where(min_vals <= 0, np.abs(min_vals) + 1e-5, 0)
            self.pt.fit(X_df + self.shift_)
        else:
            self.pt.fit(X_df)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        if self.method == "box-cox":
            X_df = X_df + self.shift_
            # Protect against extreme test-set minimums
            X_df = X_df.clip(lower=1e-5)

        # 1. Apply Transformation
        X_trans = self.pt.transform(X_df)

        # 2. Catch Float64 Overflows!
        X_trans = np.nan_to_num(X_trans, nan=0.0, posinf=1e6, neginf=-1e6)

        return X_trans

    def get_feature_names_out(self, input_features=None):
        return input_features


# ==========================================
# 2. PRE-PIPELINE DATA CLEANING
# ==========================================


def drop_outliers_safely(df: pd.DataFrame, num_cols: list, method: str) -> pd.DataFrame:
    """
    Physically removes outlier rows from the DataFrame.
    MUST be run BEFORE train_test_split to ensure X and y remain aligned.
    """
    if method == "None" or not num_cols:
        return df

    df_clean = df.copy()
    for col in num_cols:
        if method == "IQR":
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        elif method == "Z-Scores":
            mean, std = df_clean[col].mean(), df_clean[col].std()
            lower, upper = mean - 3 * std, mean + 3 * std
        elif method == "Percentiles":
            lower, upper = df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)

        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


# ==========================================
# 3. PIPELINE ARCHITECTURE
# ==========================================


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
    """Constructs the master preprocessing pipeline dynamically."""
    transformers = []

    # --- NUMERICAL PIPELINE ---
    if num_cols:
        num_steps = []

        # 1. Imputation
        if num_impute == "Zero":
            num_steps.append(
                ("imputer", SimpleImputer(strategy="constant", fill_value=0))
            )
        else:
            strat_map = {
                "Median": "median",
                "Mean": "mean",
                "Most Frequent": "most_frequent",
            }
            num_steps.append(("imputer", SimpleImputer(strategy=strat_map[num_impute])))

        # 2. Outliers (Only run here if capping or nulling. 'Drop Rows' is handled pre-split)
        if outlier_method != "None" and outlier_action != "Drop Rows":
            meth_map = {
                "IQR": "iqr",
                "Z-Scores": "z-scores",
                "Percentiles": "percentiles",
            }
            act_map = {"Cap/Clip Values": "clip", "Replace with Nulls": "null"}
            num_steps.append(
                (
                    "outliers",
                    OutlierTreatment(
                        method=meth_map[outlier_method], action=act_map[outlier_action]
                    ),
                )
            )

            # If replaced with nulls, we must impute the new nulls again
            if act_map[outlier_action] == "null":
                num_steps.append(("re_imputer", SimpleImputer(strategy="median")))

        # 3. Skewness
        # 3. Skewness
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

        # 4. Scaling
        if scaling != "None":
            scale_map = {
                "Standard": StandardScaler(),
                "Min-Max": MinMaxScaler(),
                "Robust": RobustScaler(),
                "MaxAbs": MaxAbsScaler(),
            }
            num_steps.append(("scaler", scale_map[scaling]))

        transformers.append(("num", Pipeline(num_steps), num_cols))

    # --- CATEGORICAL PIPELINE ---
    if cat_cols:
        cat_steps = []

        # 1. Imputation
        cat_imp_map = {
            'Fill with "Unknown"': "unknown",
            "Most Frequent": "most_frequent",
            "Forward Fill": "ffill",
        }
        cat_steps.append(
            ("imputer", AdvancedCategoricalImputer(strategy=cat_imp_map[cat_impute]))
        )

        # 2. Encoding
        if encoding == "One-Hot Encoding":
            cat_steps.append(
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )
        elif encoding == "Label Encoding":
            cat_steps.append(
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                )
            )

        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )


def get_model(task_type: str, model_name: str, params: dict, reduce_bias: bool = False):
    """
    Instantiates models safely.
    If reduce_bias is True, applies dynamic class weighting to penalize the
    algorithm for ignoring minority/underrepresented classes.
    """

    # Apply algorithmic Bias Reduction (Class Weighting) if supported
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
        elif model_name == "Random Forest":
            return RandomForestClassifier(random_state=42, **params)
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=42, **params)
        elif model_name == "Support Vector Machine (SVM)":
            return SVC(probability=True, **params)
        elif model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(**params)
    else:
        if model_name == "Linear Regression":
            return LinearRegression(**params)
        elif model_name == "Ridge Regression":
            return Ridge(**params)
        elif model_name == "Lasso Regression":
            return Lasso(**params)
        elif model_name == "Random Forest":
            return RandomForestRegressor(random_state=42, **params)
        elif model_name == "Gradient Boosting":
            return GradientBoostingRegressor(random_state=42, **params)
        elif model_name == "Support Vector Regressor (SVR)":
            return SVR(**params)
        elif model_name == "K-Nearest Neighbors":
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
    """Executes the full machine learning lifecycle including pre-split dropping."""

    # 1. Pre-Split Drop (If requested)
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

    # 2. Split Data (Stratification naturally reduces sampling bias)
    X = work_df[features]
    y = work_df[target]

    stratify_param = y if task_type == "Classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_param
    )

    # 3. Build & Train
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = pipeline.predict(X_test)
    metrics = {}

    if task_type == "Classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["Precision"] = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        metrics["Recall"] = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        metrics["F1 Score"] = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
        metrics["Classes"] = pipeline.classes_
    else:
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics["R2 Score"] = r2_score(y_test, y_pred)

    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "rows_dropped": rows_dropped,
    }


# ==========================================
# 4. DIAGNOSTIC VISUALIZATIONS
# ==========================================


def plot_confusion_matrix(cm, classes) -> go.Figure:
    """Generates a Plotly heatmap visualization of a classification confusion matrix."""
    fig = px.imshow(
        cm,
        text_auto=True,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(title="<b>Confusion Matrix</b>", template="plotly_white")
    return fig


def plot_regression_residuals(y_test, y_pred) -> go.Figure:
    """Generates an Actual vs. Predicted scatter plot for regression diagnostics."""
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        opacity=0.6,
        labels={"x": "Actual", "y": "Predicted"},
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="Red", dash="dash"),
    )
    fig.update_layout(title="<b>Actual vs. Predicted</b>", template="plotly_white")
    return fig


def plot_feature_importance(pipeline, feature_names) -> go.Figure:
    """
    Extracts and visualizes feature importances with robust name retrieval
    from the Scikit-Learn pipeline.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # 1. Get coefficients or importances
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

    # 2. Extract accurate feature names from the preprocessor
    final_features = preprocessor.get_feature_names_out(feature_names)

    # 3. Create DataFrame
    importance_df = (
        pd.DataFrame({"Feature": final_features, "Importance": importances})
        .sort_values(by="Importance", ascending=True)
        .tail(15)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"<b>Top {len(importance_df) if len(importance_df) < 15 else 15} Feature Importances</b>",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    return fig
