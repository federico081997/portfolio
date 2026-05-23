from src.data_loader import load_data
from src.model_utils import load_model_artifacts, add_predictions, get_actual_phone_rows
from src.ui_components import render_sidebar, render_phone_card, show_error_table
from src.plots import plot_accuracy_breakdown
from src.styles import apply_custom_css

import streamlit as st

st.set_page_config(page_title="Phone Price Predictor", page_icon="📱", layout="wide")

apply_custom_css()

df = load_data()
model, expected_columns = load_model_artifacts()

selected_models_by_brand = render_sidebar(df)

st.title("📱 Phone Pricing Predictive Dashboard")

st.markdown(
    """
    <p class="section-subtitle">
    Select specific phone models from the sidebar to compare their real market prices
    against the machine learning model's predictions.
    </p>
    """,
    unsafe_allow_html=True,
)

run_button = st.button("Run Market Comparison", type="primary")

if run_button:
    brand_model_pairs = [
        (brand, model_name)
        for brand, models in selected_models_by_brand.items()
        for model_name in models
    ]

    if not brand_model_pairs:
        st.warning("Please select at least one model from the sidebar.")
        st.stop()

    with st.spinner("Fetching actual hardware specs and running inference..."):
        input_df = get_actual_phone_rows(df, brand_model_pairs)

        if input_df.empty:
            st.error("No matching phone rows were found in the dataset.")
            st.stop()

        results_df = add_predictions(
            input_df=input_df,
            trained_model=model,
            expected_columns=expected_columns,
        )

    has_real_price = "Price_USD" in results_df.columns

    if not has_real_price:
        st.warning(
            "The column `Price_USD` was not found in your dataset. "
            "The app can show predictions, but it cannot compare them against real prices."
        )

    st.markdown("---")
    st.subheader("Algorithm Accuracy Breakdown")
    plot_accuracy_breakdown(results_df, has_real_price)

    st.markdown("---")
    st.subheader("Market Comparison Results")

    for start in range(0, len(results_df), 3):
        chunk = results_df.iloc[start : start + 3]
        cols = st.columns(3)

        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                render_phone_card(row, has_real_price)

        st.markdown("<br>", unsafe_allow_html=True)

    if has_real_price:
        st.markdown("---")
        st.subheader("Prediction Error Table")
        show_error_table(results_df)
