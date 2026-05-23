import pandas as pd
import streamlit as st

from src.config import DEFAULT_BRANDS
from src.image_scraper import get_phone_image_bytes, image_bytes_to_base64


def render_sidebar(df):
    """
    Renders the Streamlit sidebar used to select phone brands and models.

    The sidebar first allows the user to select one or more phone brands.
    For each selected brand, it then creates an independent multiselect
    dropdown containing only the models that belong to that brand. This
    prevents invalid brand-model combinations, such as pairing an Apple
    model with Samsung.

    Parameters
    ----------
    df : pd.DataFrame
        Phone specification dataframe. It must contain at least the columns
        `"Company Name"` and `"Model Name"`.

    Returns
    -------
    dict
        Dictionary where each key is a selected brand and each value is a
        list of selected models for that brand. Brands with no selected
        models are not included.

        Example:
        {
            "Apple": ["iPhone 11 Pro Max 64GB"],
            "Samsung": ["Galaxy A14 128GB"]
        }
    """
    st.sidebar.title("Market Analytics")
    st.sidebar.header("Select Phones to Compare")

    all_brands = sorted(df["Company Name"].dropna().unique())

    default_brands = [brand for brand in DEFAULT_BRANDS if brand in all_brands]

    selected_brands = st.sidebar.multiselect(
        "Select Brand(s)",
        options=all_brands,
        default=default_brands,
    )

    selected_models_by_brand = {}

    if selected_brands:
        st.sidebar.subheader("Select Models by Brand")

        for brand in selected_brands:
            brand_models = sorted(
                df.loc[df["Company Name"] == brand, "Model Name"].dropna().unique()
            )

            selected_models = st.sidebar.multiselect(
                f"Models for {brand}",
                options=brand_models,
                key=f"models_for_{brand}",
            )

            if selected_models:
                selected_models_by_brand[brand] = selected_models

    else:
        st.sidebar.info("Select at least one brand.")

    return selected_models_by_brand


def format_price(value):
    """
    Safely formats a price value.

    Parameters
    ----------
    value : float | int | None
        Price value.

    Returns
    -------
    str
        Formatted price string.
    """
    if value is None or pd.isna(value):
        return "N/A"

    return f"${value:,.0f}"


def render_phone_image(brand, model_name):
    """
    Renders the phone image inside a fixed-height container so all
    cards align vertically.

    Parameters
    ----------
    brand : str
        Phone brand.

    model_name : str
        Phone model name.
    """
    image_bytes = get_phone_image_bytes(brand, model_name)

    if image_bytes is not None:
        image_base64 = image_bytes_to_base64(image_bytes)

        st.markdown(
            f"""
            <div class="phone-image-box">
                <img src="data:image/jpeg;base64,{image_base64}" alt="{brand} {model_name}">
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="phone-image-box">
                <img src="https://placehold.co/300x300?text=Image+Unavailable" alt="Image unavailable">
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_phone_card(row, has_real_price):
    """
    Renders one aligned phone result card.

    Parameters
    ----------
    row : pd.Series
        Phone row.

    has_real_price : bool
        Whether the dataframe contains a real price column.
    """
    brand = row["Company Name"]
    model_name = row["Model Name"]

    st.markdown('<div class="phone-card-wrapper">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="phone-card-title">
            {brand} {model_name}
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_phone_image(brand, model_name)

    real_price = row["Price_USD"] if has_real_price else None
    pred_price = row["Predicted Price (USD)"]

    metric_col_1, metric_col_2 = st.columns(2)

    with metric_col_1:
        st.metric(
            label="Real Price",
            value=format_price(real_price),
        )

    with metric_col_2:
        if has_real_price and real_price is not None and not pd.isna(real_price):
            error_margin = pred_price - real_price

            st.metric(
                label="Predicted",
                value=format_price(pred_price),
                delta=f"{error_margin:,.0f} error",
                delta_color="inverse",
            )
        else:
            st.metric(
                label="Predicted",
                value=format_price(pred_price),
            )

    with st.expander("View Hardware Specs"):
        st.write(f"**Brand:** {brand}")
        st.write(f"**Model:** {model_name}")
        st.write(f"**RAM:** {row['RAM_GB']} GB")
        st.write(f"**Battery:** {row['Battery_mAh']} mAh")
        st.write(f"**Screen:** {row['Screen_Size_inches']} inches")
        st.write(f"**Weight:** {row['Weight_g']} g")
        st.write(f"**Back Camera:** {row['Back_Camera_MP']} MP")
        st.write(f"**Front Camera:** {row['Front_Camera_MP']} MP")

        if "Launched Year" in row.index:
            st.write(f"**Launched Year:** {row['Launched Year']}")

    st.markdown("</div>", unsafe_allow_html=True)


def show_error_table(results_df):
    """
    Shows prediction error table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with actual and predicted prices.
    """
    error_df = results_df.copy()

    error_df["Absolute Error"] = (
        error_df["Predicted Price (USD)"] - error_df["Price_USD"]
    ).abs()

    error_df["Percentage Error"] = (
        error_df["Absolute Error"] / error_df["Price_USD"]
    ) * 100

    error_table = error_df[
        [
            "Company Name",
            "Model Name",
            "Price_USD",
            "Predicted Price (USD)",
            "Absolute Error",
            "Percentage Error",
        ]
    ].copy()

    error_table = error_table.rename(
        columns={
            "Company Name": "Brand",
            "Model Name": "Model",
            "Price_USD": "Actual Price",
            "Predicted Price (USD)": "Predicted Price",
        }
    )

    st.dataframe(
        error_table.style.format(
            {
                "Actual Price": "${:,.0f}",
                "Predicted Price": "${:,.0f}",
                "Absolute Error": "${:,.0f}",
                "Percentage Error": "{:.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )
