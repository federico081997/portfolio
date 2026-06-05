import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_confusion_matrix(cm, classes) -> go.Figure:
    """
    Generates a Plotly heatmap visualization of a classification confusion matrix.

    Args:
        cm (array-like): A 2D array representing the confusion matrix counts
            (typically output from sklearn.metrics.confusion_matrix).
        classes (list): A list of string labels representing the unique class names,
            ordered to match the confusion matrix indices.

    Returns:
        go.Figure: A configured Plotly heatmap figure.
    """
    fig = px.imshow(
        cm,
        text_auto=True,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
    )
    fig.update_layout(title="<b>Confusion Matrix</b>", template="plotly_white")
    return fig


def plot_regression_residuals(y_test, y_pred) -> go.Figure:
    """
    Generates an Actual vs. Predicted scatter plot for regression diagnostics.

    Visually evaluates the performance of a continuous model by plotting the
    predicted values against the true target values. Includes a dashed reference
    line representing perfect prediction (y = x) to easily identify systematic
    bias, variance, or heteroscedasticity in the model's errors.

    Args:
        y_test (array-like): The true, observed target values from the test set.
        y_pred (array-like): The target values predicted by the model.

    Returns:
        go.Figure: A configured Plotly scatter plot figure.
    """
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        opacity=0.6,
        labels={"x": "Actual Values", "y": "Predicted Values"},
        color_discrete_sequence=px.colors.qualitative.Prism,
    )

    # Add a perfect prediction line
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
    fig.update_layout(title="<b>Actual vs. Predicted</b>", template="plotly_white")
    return fig


def plot_feature_importance(pipeline, feature_names) -> go.Figure:
    """
    Extracts and visualizes tree-based feature importances from a fitted pipeline.

    Dynamically parses the ColumnTransformer within the pipeline to map
    post-transformation features (e.g., newly generated One-Hot Encoded columns)
    back to readable names. Safely handles estimators that lack native importance
    attributes (such as KNN or SVR) by returning None.

    Args:
        pipeline (sklearn.pipeline.Pipeline): A fitted scikit-learn pipeline
            containing a 'preprocessor' step and a 'model' step.
        feature_names (list): The original list of predictor column names passed
            into the pipeline before preprocessing.

    Returns:
        go.Figure or None: A Plotly horizontal bar chart displaying the top 15
        most heavily weighted features, or None if the estimator does not support
        native feature importance.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    # Get feature names after preprocessing (crucial for One-Hot Encoding)
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        final_features = preprocessor.get_feature_names_out(feature_names)
        # Clean up sklearn's default prefixing (e.g., 'num__Age' -> 'Age')
        final_features = [f.split("__")[-1] for f in final_features]
    except:
        final_features = [
            f"Feature {i}" for i in range(len(model.feature_importances_))
        ]

    importance_df = (
        pd.DataFrame(
            {"Feature": final_features, "Importance": model.feature_importances_}
        )
        .sort_values(by="Importance", ascending=True)
        .tail(15)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="<b>Top 15 Feature Importances</b>",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    return fig
