"""Streamlit UI for geospatial analytics and mapping workflows.

This module renders the Geospatial Analytics & Mapping Studio used by the
Analytics and Machine Learning Workspace. It provides interactive controls for coordinate
selection, browser-safe downsampling, clustered marker maps, density heatmaps,
and animated spatial time-lapse visualizations.

The module uses the cleaned dataset from ``st.session_state`` when available and
falls back to the raw uploaded dataset when preprocessing has not been applied.
Generated map objects are stored in session state so they persist across
Streamlit reruns.
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.geospatial_visualizations import (
    plot_animated_map,
    plot_density_heatmap,
    plot_interactive_clusters,
)


def detect_spatial_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Detect likely latitude and longitude columns in a DataFrame.

    The function scans column names for common coordinate labels such as
    ``"lat"``, ``"latitude"``, ``"lon"``, ``"longitude"``, ``"x"``, and ``"y"``.

    Args:
        df: Dataset to inspect.

    Returns:
        A tuple containing the detected latitude and longitude column names.
        Either value may be ``None`` if no suitable column is found.
    """
    lat_col = None
    lon_col = None

    for col in df.columns:
        col_lower = col.lower()

        if col_lower in ["lat", "latitude", "y"]:
            lat_col = col
        elif col_lower in ["lon", "long", "longitude", "lng", "x"]:
            lon_col = col

    return lat_col, lon_col


def clear_spatial_cache() -> None:
    """Remove cached geospatial map objects from session state.

    Clears previously generated cluster, heatmap, and animation outputs so that
    maps are regenerated after coordinate or sampling configuration changes.
    """
    keys_to_clear = ["saved_cluster_map", "saved_heat_map", "saved_anim_map"]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def render_geospatial_module() -> None:
    """Render the geospatial analytics and mapping interface.

    Displays coordinate configuration controls, applies browser-safe
    downsampling, and renders tabbed workflows for clustered maps, density
    heatmaps, and animated spatial evolution.
    """
    st.title("Geospatial Analytics & Mapping Studio")
    st.markdown("""
        Welcome to the **Geospatial Analytics & Mapping Studio** — an
        interactive workspace for visualizing coordinate-based datasets and
        spatial patterns.

        This module uses Leaflet/Folium and Plotly map rendering to support
        multiple geospatial workflows:

        * **Interactive Clustering:** Group nearby coordinates into dynamic
          marker clusters to reduce visual clutter.
        * **Spatial Density Heatmaps:** Convert discrete point locations into
          continuous density gradients for hotspot detection.
        * **Time-Lapse Animation:** Track how spatial patterns evolve across
          discrete time or category frames.
        * **Adaptive Map Configuration:** Detect likely coordinate columns,
          select base-map styles, and apply safe downsampling limits.
        """)
    st.markdown("---")

    if (
        "cleaned_data" in st.session_state
        and st.session_state["cleaned_data"] is not None
    ):
        df = st.session_state["cleaned_data"]
        st.caption(
            "✨ Using processed data from the Data Cleaning & Preprocessing Studio."
        )

    elif "raw_data" in st.session_state and st.session_state["raw_data"] is not None:
        df = st.session_state["raw_data"]
        st.info(
            "ℹ️ **Notice:** Using the raw uploaded dataset. For optimal results, "
            "consider processing this data in the Data Cleaning & Preprocessing "
            "Studio first."
        )

    else:
        st.warning("⚠️ No data available. Please upload a dataset first.")
        return

    if df.empty:
        st.error(
            "❌ The selected dataset is empty. Please check your data source or "
            "cleaning steps."
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
        st.error("❌ No numerical columns available to act as coordinates.")
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
                help="Caps the maximum points rendered to prevent browser overload.",
                on_change=clear_spatial_cache,
            )

    map_df = df.dropna(subset=[lat, lon]).copy()

    if len(map_df) > sample_size:
        st.info(
            f"🛡️ **System Protection:** Dataset randomly downsampled from "
            f"{len(map_df):,} to {sample_size:,} rows for safe rendering."
        )
        map_df = map_df.sample(n=sample_size, random_state=42)

    t_cluster, t_heat, t_anim = st.tabs(
        ["📌 Clustered Pins", "🔥 Density Heatmap", "🎬 Animated Time-Lapse"]
    )

    # -------------------------------------------------------------------------
    # Tab 1: clustered marker map
    # -------------------------------------------------------------------------
    with t_cluster:
        st.subheader("Interactive Marker Clusters")

        with st.expander("ℹ️ Guide: Understanding Clustered Mapping"):
            st.markdown("""
                **How to use this map:**

                * **Clustering:** Nearby coordinates are grouped into numbered
                  clusters to keep the map readable.
                * **Drill-Down:** Click a cluster to zoom into that region and
                  reveal smaller clusters or individual markers.
                * **Popups:** Select dataset columns to show when individual
                  markers are clicked.
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
                "Generate Map",
                type="primary",
                width="stretch",
            )

        if submit_cluster:
            with st.spinner("Compiling Leaflet geometry..."):
                color_arg = None if marker_color == "None" else marker_color
                folium_map_style = folium_style_map[tile_style]

                cluster_map = plot_interactive_clusters(
                    map_df,
                    lat,
                    lon,
                    popups,
                    color_arg,
                    folium_map_style,
                )
                st.session_state["saved_cluster_map"] = cluster_map

        if "saved_cluster_map" in st.session_state:
            components.html(
                st.session_state["saved_cluster_map"]._repr_html_(),
                height=600,
            )
        elif not submit_cluster:
            st.info("👆 Configure your parameters and click **Generate Map**.")

    # -------------------------------------------------------------------------
    # Tab 2: density heatmap
    # -------------------------------------------------------------------------
    with t_heat:
        st.subheader("Spatial Density Heatmap")

        with st.expander("ℹ️ Guide: Understanding Heatmaps"):
            st.markdown("""
                **How to use this map:**

                * **Density:** Discrete coordinates are converted into a
                  continuous intensity gradient.
                * **Aesthetics:** Heatmaps are usually easier to read on darker
                  base maps.
                * **Radius:** Increase the gradient radius if the heatmap appears
                  as isolated dots rather than smooth density regions.
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
                    "Weight By (Gravity):",
                    ["Density (Equal Weight)"] + num_cols,
                )

            with h_col3:
                radius_slider = st.slider(
                    "Gradient Radius:",
                    min_value=5,
                    max_value=30,
                    value=15,
                )

            submit_heat = st.form_submit_button(
                "Generate Heatmap",
                type="primary",
                width="stretch",
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
                st.session_state["saved_heat_map"]._repr_html_(),
                height=600,
            )
        elif not submit_heat:
            st.info("👆 Configure your parameters and click **Generate Heatmap**.")

    # -------------------------------------------------------------------------
    # Tab 3: animated time-lapse map
    # -------------------------------------------------------------------------
    with t_anim:
        st.subheader("Dynamic Spatial Evolution")

        with st.expander("ℹ️ Guide: Understanding Animated Maps"):
            st.markdown("""
                **How to use this tool:**

                * **Playback:** Press play to watch spatial patterns evolve
                  across the selected frame variable.
                * **Frame Selection:** Plotly creates one animation frame for
                  each unique value in the selected time-frame column.
                * **Performance:** Avoid continuous timestamps with many unique
                  values. Prefer aggregated fields such as hour, month, or year.
                """)

        time_options = time_cols + num_cols + cat_cols

        with st.form("anim_form"):
            a_col1, a_col2, a_col3 = st.columns(3)

            with a_col1:
                anim_time = st.selectbox(
                    "Time Frame (Animation Step):",
                    time_options,
                    help="Select an aggregated feature such as Hour, Month, or Year.",
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
                "Generate Animation",
                type="primary",
                width="stretch",
            )

        if submit_anim:
            unique_frames = map_df[anim_time].nunique()

            if unique_frames > 500:
                st.error(
                    f"❌ Your chosen Time Frame has {unique_frames} unique values. "
                    "Generating more than 500 animation frames might crash the "
                    "browser. Please use the ETL Studio to extract a discrete "
                    "column like 'Hour' or 'Year'."
                )

            else:
                with st.spinner("Compiling WebGL animation frames..."):
                    color_arg = None if anim_color == "None" else anim_color
                    plotly_map_style = anim_style_map[anim_style_selection]

                    anim_map = plot_animated_map(
                        map_df,
                        lat,
                        lon,
                        anim_time,
                        color_arg,
                        plotly_map_style,
                    )
                    st.session_state["saved_anim_map"] = anim_map

        if "saved_anim_map" in st.session_state:
            st.plotly_chart(
                st.session_state["saved_anim_map"],
                width="stretch",
                key="plotly_anim_map_view",
            )
        elif not submit_anim:
            st.info("👆 Configure your parameters and click **Generate Animation**.")
