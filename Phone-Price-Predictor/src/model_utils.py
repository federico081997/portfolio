import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, EXPECTED_COLUMNS_PATH


@st.cache_resource
def load_model_artifacts():
    """
    Loads the trained model and expected feature columns.

    Returns
    -------
    tuple
        Trained model and expected feature column list.
    """

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not EXPECTED_COLUMNS_PATH.exists():
        st.error(f"Expected columns file not found: {EXPECTED_COLUMNS_PATH}")
        st.stop()

    trained_model = joblib.load(MODEL_PATH)
    expected_cols = joblib.load(EXPECTED_COLUMNS_PATH)

    return trained_model, expected_cols


def get_actual_phone_rows(source_df, brand_model_pairs):
    """
    Pulls the real rows from the dataset for the selected phones.

    Parameters
    ----------
    source_df : pd.DataFrame
        Original dataset.

    brand_model_pairs : list[tuple]
        Selected brand-model pairs.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the selected real phone rows.
    """
    selected_rows = []

    for brand, model_name in brand_model_pairs:
        matching_rows = source_df[
            (source_df["Company Name"] == brand)
            & (source_df["Model Name"] == model_name)
        ]

        if not matching_rows.empty:
            selected_rows.append(matching_rows.iloc[0])

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def prepare_features_for_prediction(input_df, expected_columns):
    """
    Encodes the selected real phone rows and aligns the prediction
    dataframe with the model's expected training schema.

    Parameters
    ----------
    input_df : pd.DataFrame
        Raw phone dataframe.

    expected_columns : list
        Feature columns expected by the trained model.

    Returns
    -------
    pd.DataFrame
        Encoded and schema-aligned prediction matrix.
    """
    input_encoded = pd.get_dummies(
        input_df,
        columns=["Company Name"],
    )

    columns_to_drop = [
        "Model Name",
        "Price_USD",
        "Predicted Price (USD)",
    ]

    X_predict = input_encoded.drop(
        columns=[col for col in columns_to_drop if col in input_encoded.columns]
    )

    X_predict = X_predict.reindex(
        columns=expected_columns,
        fill_value=0,
    )

    return X_predict


def add_predictions(input_df, trained_model, expected_columns):
    """
    Adds ML predictions to the selected phone dataframe.

    Parameters
    ----------
    input_df : pd.DataFrame
        Selected phone dataframe.

    trained_model
        Fitted machine learning model.

    expected_columns : list
        Feature columns expected by the trained model.

    Returns
    -------
    pd.DataFrame
        Phone dataframe with predicted prices.
    """
    X_predict = prepare_features_for_prediction(
        input_df=input_df,
        expected_columns=expected_columns,
    )

    output_df = input_df.copy()
    output_df["Predicted Price (USD)"] = trained_model.predict(X_predict)

    return output_df
