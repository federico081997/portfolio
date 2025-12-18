"""
modeling.py

Machine learning utilities for the SpaceX Launch Dashboard.

This module prepares the dashboard dataset for modelling, trains a set of
baseline classifiers using grid search, and returns precomputed evaluation
metrics and confusion matrices for display in the UI.

The main entry point is :func:`execute_models`, which returns a lightweight
cache dictionary suitable for storing in memory at application startup.
"""

from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# Model definitions and hyperparameter grids
# =============================================================================

# Each entry maps a display name to (estimator, param_grid).
# The keys in param_grid refer to estimator parameters.
MODELS = {
    "Logistic Regression": (
        LogisticRegression(random_state=42, max_iter=5000),
        {
            "C": [0.01, 0.1, 1.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
    ),
    "Support Vector Machines": (
        SVC(random_state=42),
        {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],  # only applies to the RBF kernel
        },
    ),
    "Decision Trees": (
        DecisionTreeClassifier(random_state=42),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 8, None],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt", None],
        },
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2],  # 1=Manhattan, 2=Euclidean
        },
    ),
}


# =============================================================================
# Internal helper
# =============================================================================


def _prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare modelling features (X) and target (y) from the dashboard dataset.

    The target variable encodes landing success as:
    - 1 if the "Landing Outcome" string contains "True"
    - 0 otherwise

    Several columns are excluded from modelling either because they are not
    intended as predictors for this project, or because they would introduce
    leakage/instability for the dashboard use case.

    :param df: Master DataFrame produced by the data-wrangling pipeline.
    :type df: pd.DataFrame
    :return: Tuple (X, y) where X is the feature DataFrame and y is the target series.
    :rtype: Tuple[pd.DataFrame, pd.Series]
    """
    # Target variable (1 = successful landing, 0 = failure).
    # Note: landing outcome is stored as a string like "True <landing_type>" or "False <landing_type>".
    y = df["Landing Outcome"].apply(
        lambda x: 1 if (x is True or "true" in str(x).lower()) else 0
    )

    # Remove columns that must not be used as predictors.
    # - Landing Outcome: target variable
    # - Launch Date: time-based information is excluded in this dashboard
    # - Booster Name: excluded by design (could be treated as leakage-like proxy)
    # - Coordinates: excluded from modelling and kept for mapping only
    X = df.drop(
        columns=[
            "Landing Outcome",
            "Launch Date",
            "Booster Name",
            "Launch Site Latitude",
            "Launch Site Longitude",
        ]
    )

    return X, y


# =============================================================================
# Public API
# =============================================================================


def execute_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and evaluate multiple baseline models using GridSearchCV.

    The pipeline:
    - splits the data into a stratified train/test set,
    - preprocesses numeric features with scaling and categoricals with one-hot encoding,
    - runs a grid search (F1 scoring) for each candidate model,
    - evaluates the best estimator on the held-out test set,
    - returns formatted metric strings and confusion matrices for the dashboard.

    :param df: Master DataFrame produced by the data-wrangling pipeline.
    :type df: pd.DataFrame
    :return: Model evaluation cache containing a results table and confusion matrices.
    :rtype: Dict[str, Any]
    """
    # Prepare features and target.
    X, y = _prepare_data(df)

    # Stratified split to preserve class balance between train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Determine feature types from the training DataFrame.
    numerical_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    # Preprocessor applied inside the modelling pipeline.
    # - Scale numeric features.
    # - One-hot encode categorical features; ignore unknowns to be robust to new categories.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    results_rows = []
    cms = {}

    # Train and evaluate each candidate model.
    for name, (estimator, param_grid) in MODELS.items():
        # Pipeline ensures preprocessing is performed consistently within cross-validation.
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

        # Prefix parameters for the pipeline step "model".
        grid = {f"model__{k}": v for k, v in param_grid.items()}

        # Grid search for best hyperparameters using F1 score (balanced emphasis on precision/recall).
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1",
            cv=5,
            n_jobs=1,
            refit=True,
        )

        gs.fit(X_train, y_train)

        # Evaluate best estimator on the held-out test set.
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)

        results_rows.append(
            {
                "Model": name,
                "Accuracy": f"{accuracy_score(y_test, y_pred) * 100:.2f}%",
                "Precision": f"{precision_score(y_test, y_pred, zero_division=0) * 100:.2f}%",
                "Recall": f"{recall_score(y_test, y_pred, zero_division=0) * 100:.2f}%",
                "F1": f"{f1_score(y_test, y_pred, zero_division=0) * 100:.2f}%",
            }
        )

        # Convert to a plain nested list for easy JSON-serialisation and storage in Dash.
        cms[name] = confusion_matrix(y_test, y_pred).tolist()

    return {"Results": results_rows, "Confusion Matrix": cms}
