"""
Geospatial Analytics UI Module

Provides the interactive Streamlit dashboard interface for mapping physical data.
Includes automatic spatial column detection and mandatory data downsampling
to protect browser memory during heavy Folium/Leaflet rendering. Utilizes
session state persistence to maintain map rendering across user interactions.
"""

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from core.mapping_visualizations import (
    plot_interactive_clusters,
    plot_density_heatmap,
    plot_animated_map,
)


def detect_spatial_columns(df: pd.DataFrame) -> tuple:
    """
    Heuristically scans the dataframe for columns representing geographical coordinates.

    Automatically identifies columns named 'lat', 'longitude', 'x', 'y', etc.,
    allowing the UI to intelligently pre-select coordinate parameters for the user.

    Args:
        df (pd.DataFrame): The dataset to scan.

    Returns:
        tuple: A pair of strings (lat_col, lon_col). If a coordinate is not found,
            its respective position in the tuple will be None.
    """
    lat_col, lon_col = None, None

    for col in df.columns:
        c_lower = col.lower()
        if c_lower in ["lat", "latitude", "y"]:
            lat_col = col
        elif c_lower in ["lon", "long", "longitude", "lng", "x"]:
            lon_col = col

    return lat_col, lon_col


def clear_spatial_cache() -> None:
    """
    Deletes compiled map objects from the Streamlit session state.

    This acts as a cache invalidation callback. It is triggered whenever the user
    alters global spatial variables (like switching the target Latitude column or
    changing the downsampling limits). This prevents st_folium from crashing by
    attempting to render old HTML geometries with new data parameters.

    Side Effects:
        Removes "saved_cluster_map", "saved_heat_map", and "saved_anim_map"
        from `st.session_state` if they exist.
    """
    keys_to_clear = ["saved_cluster_map", "saved_heat_map", "saved_anim_map"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def render_geospatial_module() -> None:
    """
    Renders the primary Geospatial Analytics Studio dashboard in Streamlit.

    Retrieves structurally sound data from the ETL pipeline (`st.session_state["cleaned_data"]`)
    and routes it into three advanced spatial workflows: Marker Clustering, Density Heatmaps,
    and WebGL Time-Lapse Animation. Enforces mathematical downsampling limits to protect
    the end-user's browser architecture from DOM memory overloads.

    Raises/Warnings:
        Warning (UI): Displays a warning if no cleaned data is found in the session state.
        Error (UI): Displays an error if the dataset lacks numerical coordinate columns.
    """

    st.empty()
    st.title("Geospatial Analytics & Mapping Studio")
    st.markdown("""
    Welcome to the **Geospatial Analytics & Mapping Studio**—a high-performance rendering engine for spatial data visualization and coordinate-based intelligence.

    This module leverages interactive Leaflet and WebGL architectures to map physical environments, featuring built-in DOM memory protection via intelligent data downsampling:
    * **Interactive Clustering:** Aggregate high-density coordinates into dynamic, drill-down cluster geometries to prevent visual overload.
    * **Spatial Density Heatmaps:** Translate discrete physical events into continuous probability gradients to identify hotspots and areas of extreme concentration.
    * **Time-Lapse Animation:** Track the geographical spread of data over chronological periods using frame-by-frame animated spatial evolution.
    * **Adaptive Geometries:** Automatically detects coordinate fields (Lat/Lon) and allows on-the-fly toggling of base-map aesthetics (Dark Matter, Positron, OpenStreetMap).
    """)
    st.markdown("---")

    # 1. Try to load the Cleaned Data first
    if (
        "cleaned_data" in st.session_state
        and st.session_state["cleaned_data"] is not None
    ):
        df = st.session_state["cleaned_data"]
        # Subtle indicator to the user
        st.caption(
            "✨ Using processed data from the Data Cleaning & Preprocessing Studio."
        )

    # 2. Fallback to Raw Data if Cleaning Studio was skipped
    elif "raw_data" in st.session_state and st.session_state["raw_data"] is not None:
        df = st.session_state["raw_data"]
        st.info(
            "ℹ️ **Notice:** Using the raw uploaded dataset. For optimal ML performance, consider processing this data in the Data Cleaning & Preprocessing Studio first."
        )

    # 3. Halt if absolutely no data exists
    else:
        st.warning("⚠️ No data available. Please upload a dataset first.")
        return

    # 4. Final safety check
    if df.empty:
        st.error(
            "The selected dataset is empty. Please check your data source or cleaning steps."
        )
        return

    auto_lat, auto_lon = detect_spatial_columns(df)

    num_cols = df.select_dtypes(
        include=["float64", "int64", "float32", "int32"]
    ).columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    time_cols = df.select_dtypes(
        include=["datetime", "datetime64[ns]"]
    ).columns.tolist()

    if not num_cols:
        st.error("No numerical columns available to act as coordinates.")
        return

    with st.expander("⚙️ Spatial Coordinates & Memory Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            lat = st.selectbox(
                "Latitude Column:",
                num_cols,
                index=num_cols.index(auto_lat) if auto_lat in num_cols else 0,
                on_change=clear_spatial_cache,
            )
        with c2:
            lon = st.selectbox(
                "Longitude Column:",
                num_cols,
                index=(
                    num_cols.index(auto_lon)
                    if auto_lon in num_cols
                    else 1 if len(num_cols) > 1 else 0
                ),
                on_change=clear_spatial_cache,
            )
        with c3:
            sample_size = st.number_input(
                "Data Sample Limit (Rows):",
                min_value=100,
                max_value=25000,
                value=5000,
                step=100,
                help="Caps the maximum points rendered to prevent WebGL/DOM overloads.",
                on_change=clear_spatial_cache,
            )

    map_df = df.dropna(subset=[lat, lon]).copy()
    if len(map_df) > sample_size:
        st.info(
            f"🛡️ **System Protection:** Dataset randomly downsampled from {len(map_df):,} to {sample_size:,} rows for safe rendering."
        )
        map_df = map_df.sample(n=sample_size, random_state=42)

    t_cluster, t_heat, t_anim = st.tabs(
        ["📌 Clustered Pins", "🔥 Density Heatmap", "🎬 Animated Time-Lapse"]
    )

    # --- TAB 1: Clustering ---
    with t_cluster:
        st.subheader("Interactive Marker Clusters")

        with st.expander("ℹ️ Guide: Understanding Clustered Mapping"):
            st.markdown("""
                **How to use this map:**
                * **The Engine:** Grouping nearby coordinates into a single numbered cluster prevents the map from becoming unreadable.
                * **Drill-Down:** Click on any numbered cluster to physically zoom the map into that specific boundary and break the cluster apart.
                * **Popups:** Select specific dataset columns to attach to the final pins. Once zoomed in, click a specific location marker to read the data.
                """)

        with st.form("cluster_form"):
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                folium_ui_options = [
                    "Carto Positron (Light)",
                    "Carto Dark Matter (Dark)",
                    "OpenStreetMap",
                ]
                folium_style_map = {
                    "Carto Positron (Light)": "CartoDB positron",
                    "Carto Dark Matter (Dark)": "CartoDB dark_matter",
                    "OpenStreetMap": "OpenStreetMap",
                }
                tile_style = st.selectbox(
                    "Base Map Aesthetic:",
                    folium_ui_options,
                )
            with c_col2:
                marker_color = st.selectbox("Color Segment By:", ["None"] + cat_cols)
            with c_col3:
                popups = st.multiselect(
                    "Data to Show on Click:",
                    df.columns.tolist(),
                    placeholder="Choose data...",
                    max_selections=5,
                )

            submit_cluster = st.form_submit_button(
                "Generate Map", type="primary", use_container_width=True
            )

        if submit_cluster:
            with st.spinner("Compiling Leaflet geometry..."):
                color_arg = None if marker_color == "None" else marker_color
                folium_map_style = folium_style_map[tile_style]
                cluster_map = plot_interactive_clusters(
                    map_df, lat, lon, popups, color_arg, folium_map_style
                )
                st.session_state["saved_cluster_map"] = cluster_map

        if "saved_cluster_map" in st.session_state:
            components.html(
                st.session_state["saved_cluster_map"]._repr_html_(), height=600
            )
        elif not submit_cluster:
            st.info("👆 Configure your parameters and click **Generate Map**.")

    # --- TAB 2: Heatmap ---
    with t_heat:
        st.subheader("Spatial Density Heatmap")

        with st.expander("ℹ️ Guide: Understanding Heatmaps"):
            st.markdown("""
                **How to use this map:**
                * **The Engine:** Converts thousands of discrete data points into a continuous probability gradient. Red areas signify intense concentration/density.
                * **Aesthetics:** Heatmaps look significantly better on dark backgrounds. `CartoDB dark_matter` is recommended.
                * **Radius Tuning:** If the heatmap looks like a bunch of isolated dots, increase the Gradient Radius slider to blur them together.
                """)

        with st.form("heat_form"):
            h_col1, h_col2, h_col3 = st.columns(3)
            with h_col1:
                heat_ui_options = [
                    "Carto Dark Matter (Dark)",
                    "Carto Positron (Light)",
                    "OpenStreetMap",
                ]
                heat_style_map = {
                    "Carto Dark Matter (Dark)": "CartoDB dark_matter",
                    "Carto Positron (Light)": "CartoDB positron",
                    "OpenStreetMap": "OpenStreetMap",
                }
                heat_tile = st.selectbox("Base Map Aesthetic:", heat_ui_options)
            with h_col2:
                weight = st.selectbox(
                    "Weight By (Gravity):", ["Density (Equal Weight)"] + num_cols
                )
            with h_col3:
                radius_slider = st.slider(
                    "Gradient Radius:", min_value=5, max_value=30, value=15
                )

            submit_heat = st.form_submit_button(
                "Generate Heatmap", type="primary", use_container_width=True
            )

        if submit_heat:
            with st.spinner("Rendering WebGL gradient..."):
                weight_arg = None if weight == "Density (Equal Weight)" else weight
                folium_heat_style = heat_style_map[heat_tile]
                heat_map = plot_density_heatmap(
                    map_df,
                    lat,
                    lon,
                    weight_arg,
                    folium_heat_style,
                    radius=radius_slider,
                    blur=radius_slider,
                )
                st.session_state["saved_heat_map"] = heat_map

        if "saved_heat_map" in st.session_state:
            components.html(
                st.session_state["saved_heat_map"]._repr_html_(), height=600
            )
        elif not submit_heat:
            st.info("👆 Configure your parameters and click **Generate Heatmap**.")

    # --- TAB 3: Animated Time-Lapse ---
    with t_anim:
        st.subheader("Dynamic Spatial Evolution")

        with st.expander("ℹ️ Guide: Understanding Animated Maps"):
            st.markdown("""
                **How to use this tool:**
                * **The Playback:** Press **Play** to watch the geographical spread of data evolve chronologically. Drag the slider to scrub to a specific frame.
                * **Frame Selection:** Plotly creates one animation frame for every unique value in your `Time Frame` column.
                * **Warning:** Do not use continuous datetimes (e.g., exact seconds). Generating more than 100 frames will freeze the application. Always use aggregated/binned time features (like `Hour`, `Month`, or `Year`).
                """)

        time_options = time_cols + num_cols + cat_cols

        with st.form("anim_form"):
            a_col1, a_col2, a_col3 = st.columns(3)
            with a_col1:
                anim_time = st.selectbox(
                    "Time Frame (Animation Step):",
                    time_options,
                    help="Select an aggregated feature like 'Hour' or 'Year'.",
                )
            with a_col2:
                anim_color = st.selectbox("Color By:", ["None"] + cat_cols)
            with a_col3:
                ui_map_options = [
                    "Carto Positron (Light)",
                    "Carto Dark Matter (Dark)",
                    "OpenStreetMap",
                ]
                anim_style_selection = st.selectbox("Map Style:", ui_map_options)
                anim_style_map = {
                    "Carto Positron (Light)": "carto-positron",
                    "Carto Dark Matter (Dark)": "carto-darkmatter",
                    "OpenStreetMap": "open-street-map",
                }

            submit_anim = st.form_submit_button(
                "Generate Animation", type="primary", use_container_width=True
            )

        if submit_anim:
            unique_frames = map_df[anim_time].nunique()
            if unique_frames > 500:
                st.error(
                    f"⚠️ Your chosen Time Frame has {unique_frames} unique values. Generating more than 500 animation frames might crash the browser. Please use the ETL studio to extract a discrete column like 'Hour' or 'Year'."
                )
            else:
                with st.spinner("Compiling WebGL animation frames..."):
                    c_arg = None if anim_color == "None" else anim_color
                    plotly_map_style = anim_style_map[anim_style_selection]
                    anim_map = plot_animated_map(
                        map_df, lat, lon, anim_time, c_arg, plotly_map_style
                    )
                    st.session_state["saved_anim_map"] = anim_map

        if "saved_anim_map" in st.session_state:
            st.plotly_chart(
                st.session_state["saved_anim_map"],
                use_container_width=True,
                key="plotly_anim_map_view",
            )
        elif not submit_anim:
            st.info("👆 Configure your parameters and click **Generate Animation**.")
