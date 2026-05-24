from data_loader import load_data
from model_utils import (
    load_model_artifacts,
    add_predictions,
    get_actual_phone_rows,
    predict_custom_phone,
)
from ui_components import (
    render_market_sidebar,
    render_custom_sidebar,
    render_phone_card,
    show_error_table,
)
from plots import plot_accuracy_breakdown
from styles import apply_custom_css

import streamlit as st

st.set_page_config(page_title="Phone Price Predictor", page_icon="📱", layout="wide")

apply_custom_css()

df = load_data()
model, expected_columns = load_model_artifacts()

st.sidebar.header("Analysis Mode")

app_mode = st.sidebar.radio(
    "Analysis Mode",
    label_visibility="collapsed",
    options=["📊 Market Analytics", "🛠️ Custom Price Predictor"],
)

st.sidebar.markdown("---")

if app_mode == "📊 Market Analytics":
    selected_models_by_brand = render_market_sidebar(df)

    st.title("📊 Market Analytics")

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

elif app_mode == "🛠️ Custom Price Predictor":
    selected_custom_payload = render_custom_sidebar(df)

    # Main Page Header Layout
    st.title("🛠️ Custom Price Predictor")
    st.markdown(
        "Review the estimated fair-market value of your custom smartphone "
        "configuration based on historical macro pricing trends."
    )

    # Visual layout showing the specs currently selected in the sidebar
    st.subheader("Current Prototype Blueprint")

    blueprint_col1, blueprint_col2, blueprint_col3 = st.columns(3)
    with blueprint_col1:
        st.markdown(f"**Brand:** {selected_custom_payload['Company Name']}")
        st.markdown(f"**Launch Year:** {selected_custom_payload['Launched Year']}")
    with blueprint_col2:
        st.markdown(f"**RAM:** {selected_custom_payload['RAM_GB']} GB")
        st.markdown(f"**Battery:** {selected_custom_payload['Battery_mAh']} mAh")
    with blueprint_col3:
        st.markdown(f"**Screen:** {selected_custom_payload['Screen_Size_inches']}\"")
        st.markdown(
            f"**Optics:** {selected_custom_payload['Back_Camera_MP']}MP / {selected_custom_payload['Front_Camera_MP']}MP"
        )

    st.markdown("---")

    # Main Action Trigger Button
    run_custom_prediction = st.button("Calculate Prototype Valuation", type="primary")

    if run_custom_prediction:
        with st.spinner("Processing structural encoding maps and running inference..."):
            predicted_price = predict_custom_phone(
                selected_custom_payload=selected_custom_payload,
                trained_model=model,
                expected_columns=expected_columns,
            )

        # Render output UI layout cards
        st.success("🎉 Valuation Simulation Complete!")

        result_container_1, result_container_2 = st.columns([1, 2])

        with result_container_1:
            st.metric(label="Estimated Value (USD)", value=f"${predicted_price:,.0f}")

        with result_container_2:
            # Contextual business analysis based on the output prediction
            if predicted_price < 200:
                tier, description = (
                    "Budget Tier",
                    "Optimized for entry-level market affordability.",
                )
            elif predicted_price < 600:
                tier, description = (
                    "Mid-Ranger",
                    "Balanced hardware performance-to-cost ratio.",
                )
            else:
                tier, description = (
                    "Premium Flagship",
                    "High-tier specification profile commanding a market premium.",
                )

            st.markdown(f"**Model Classification:** `{tier}`")
            st.markdown(f"*Design Note:* {description}")
