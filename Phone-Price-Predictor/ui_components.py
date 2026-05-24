import pandas as pd
import streamlit as st

from config import DEFAULT_BRANDS
from image_scraper import get_phone_image_bytes, image_bytes_to_base64


def render_market_sidebar(df):
    """
    Renders the Streamlit sidebar used to select phone brands and models
    for the Market Analytics mode.

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
    st.sidebar.header("Select Phones to Compare")

    all_brands = sorted(df["Company Name"].dropna().unique())

    default_brands = [brand for brand in DEFAULT_BRANDS if brand in all_brands]

    selected_brands = st.sidebar.multiselect(
        "Brand(s)",
        options=all_brands,
        default=default_brands,
    )

    selected_models_by_brand = {}

    if selected_brands:
        for brand in selected_brands:
            brand_models = sorted(
                df.loc[df["Company Name"] == brand, "Model Name"].dropna().unique()
            )

            selected_models = st.sidebar.multiselect(
                f"{brand} Models",
                options=brand_models,
                key=f"models_for_{brand}",
            )

            if selected_models:
                selected_models_by_brand[brand] = selected_models

    else:
        st.sidebar.info("Select at least one brand.")

    return selected_models_by_brand


def render_custom_sidebar(df):
    """
    Renders the hardware configuration control panel in the sidebar
    for simulating a custom smartphone prototype.

    Parameters
    ----------
    df : pd.DataFrame
        Master phone specification dataframe used to extract valid
        min/max dynamic constraints for the control sliders.

    Returns
    -------
    dict
        A dictionary containing the exact specification states selected
        by the user, ready to be converted into a predictive feature row.
    """
    st.sidebar.header("Configure Specifications")

    # Categorical Brand Dropdown
    all_brands = sorted(df["Company Name"].dropna().unique())
    selected_brand = st.sidebar.selectbox(
        "Manufacturer / Brand",
        options=all_brands,
        index=0,
    )

    st.sidebar.markdown("---")

    # Extracting dynamic limits from the dataset to keep boundaries realistic
    ram_min, ram_max = int(df["RAM_GB"].min()), int(df["RAM_GB"].max())
    bat_min, bat_max = int(df["Battery_mAh"].min()), int(df["Battery_mAh"].max())
    scr_min, scr_max = float(df["Screen_Size_inches"].min()), float(
        df["Screen_Size_inches"].max()
    )
    back_min, back_max = int(df["Back_Camera_MP"].min()), int(
        df["Back_Camera_MP"].max()
    )
    front_min, front_max = int(df["Front_Camera_MP"].min()), int(
        df["Front_Camera_MP"].max()
    )

    # Sliders for Core Internal Specs
    st.sidebar.subheader("Core Internals")
    selected_ram = st.sidebar.slider(
        "RAM Capacity (GB)",
        min_value=ram_min,
        max_value=ram_max,
        value=max(ram_min, min(8, ram_max)),  # Defaults safely to 8GB if within bounds
        step=1,
    )

    selected_battery = st.sidebar.slider(
        "Battery Size (mAh)",
        min_value=bat_min,
        max_value=bat_max,
        value=max(bat_min, min(4000, bat_max)),  # Defaults safely to 4000 mAh
        step=100,
    )

    # 4. Sliders for External Hardware
    st.sidebar.subheader("Display & Optics")
    selected_screen = st.sidebar.slider(
        "Screen Size (Inches)",
        min_value=scr_min,
        max_value=scr_max,
        value=max(scr_min, min(6.1, scr_max)),  # Defaults safely to 6.1 inches
        step=0.1,
        format="%.1f",
    )

    selected_back_cam = st.sidebar.slider(
        "Rear Camera (MP)",
        min_value=back_min,
        max_value=back_max,
        value=max(back_min, min(48, back_max)),  # Defaults safely to 48 MP
        step=1,
    )

    selected_front_cam = st.sidebar.slider(
        "Selfie Camera (MP)",
        min_value=front_min,
        max_value=front_max,
        value=max(front_min, min(12, front_max)),  # Defaults safely to 12 MP
        step=1,
    )

    st.sidebar.markdown("---")

    # Temporal Features (Launched Year)
    st.sidebar.subheader("Market Timing")

    year_min, year_max = int(df["Launched Year"].min()), int(df["Launched Year"].max())

    # We use a selectbox or number_input as a "picker" for years
    selected_year = st.sidebar.selectbox(
        "Target Launch Year",
        options=list(range(year_max, year_min - 1, -1)),
        index=0,
    )

    # Bundle all selections into a clean feature payload dictionary
    custom_specs_payload = {
        "Company Name": selected_brand,
        "RAM_GB": selected_ram,
        "Battery_mAh": selected_battery,
        "Screen_Size_inches": selected_screen,
        "Back_Camera_MP": selected_back_cam,
        "Front_Camera_MP": selected_front_cam,
        "Launched Year": selected_year,
        # Default fillers for columns dropped during inference pipeline prep
        "Model Name": "Custom Prototype",
        "Price_USD": 0,
    }

    return custom_specs_payload


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
