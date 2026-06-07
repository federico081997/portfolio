"""Geospatial visualization utilities for the analytics dashboard.

This module contains reusable Folium and Plotly functions for rendering
coordinate-based datasets. It supports interactive clustered marker maps,
spatial density heatmaps, and animated spatial time-lapse visualizations.

The functions are used by the Geospatial Analytics & Mapping Studio and return
fully configured map or figure objects ready for Streamlit rendering.
"""

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap, MarkerCluster

DEFAULT_COLOR_PALETTE = px.colors.qualitative.Prism
DEFAULT_FONT = dict(family="system-ui, -apple-system, sans-serif", size=13)

FOLIUM_MARKER_COLORS = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "lightred",
    "beige",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "pink",
    "lightblue",
    "lightgreen",
    "gray",
    "black",
]


def plot_interactive_clusters(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    popup_cols: list | None = None,
    color_col: str | None = None,
    base_map: str = "CartoDB positron",
) -> folium.Map:
    """Create an interactive Folium marker-cluster map.

    Nearby coordinate points are grouped into dynamic marker clusters that
    expand as the user zooms into the map. Optional popup fields and categorical
    marker colors can be added for richer spatial inspection.

    Args:
        df: Dataset containing latitude and longitude columns.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        popup_cols: Optional columns to display inside marker popups.
        color_col: Optional categorical column used to color markers.
        base_map: Folium tile layer used as the base map.

    Returns:
        A configured Folium map object.
    """
    center = [df[lat_col].median(), df[lon_col].median()]
    map_object = folium.Map(location=center, zoom_start=12, tiles=base_map)

    marker_cluster = MarkerCluster(
        name="Clustered Data",
        overlay=True,
        control=True,
        icon_create_function=None,
    )

    color_map = {}

    if color_col:
        unique_categories = df[color_col].dropna().unique()

        for index, category in enumerate(unique_categories):
            color_map[category] = FOLIUM_MARKER_COLORS[
                index % len(FOLIUM_MARKER_COLORS)
            ]

    for _, row in df.iterrows():
        html_popup = "<div style='font-family: sans-serif; font-size: 12px;'>"
        html_popup += "<b>📍 Location Data:</b><br><hr style='margin: 5px 0;'>"

        if popup_cols:
            for col in popup_cols:
                html_popup += f"<b>{col}:</b> {row[col]}<br>"

        html_popup += "</div>"

        marker_color = color_map.get(row[color_col], "blue") if color_col else "blue"

        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=folium.Popup(html_popup, max_width=300),
            icon=folium.Icon(color=marker_color, icon="info-sign"),
        ).add_to(marker_cluster)

    marker_cluster.add_to(map_object)
    folium.LayerControl().add_to(map_object)

    return map_object


def plot_density_heatmap(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str | None = None,
    base_map: str = "CartoDB dark_matter",
    radius: int = 15,
    blur: int = 15,
) -> folium.Map:
    """Create a Folium spatial density heatmap.

    Converts coordinate points into a continuous density layer. If a valid
    numerical weight column is provided, the heat intensity is weighted by that
    column after normalization.

    Args:
        df: Dataset containing latitude and longitude columns.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        weight_col: Optional numerical column used to weight heat intensity.
        base_map: Folium tile layer used as the base map.
        radius: Radius of influence for each heatmap point.
        blur: Smoothing factor applied to the heatmap gradient.

    Returns:
        A configured Folium map object.
    """
    center = [df[lat_col].median(), df[lon_col].median()]
    map_object = folium.Map(location=center, zoom_start=12, tiles=base_map)

    if weight_col and pd.api.types.is_numeric_dtype(df[weight_col]):
        max_value = df[weight_col].max()
        normalized_weight = df[weight_col] / max_value if max_value > 0 else 1

        heat_data = [
            [row[lat_col], row[lon_col], row["weight"]]
            for _, row in pd.concat(
                [df[[lat_col, lon_col]], normalized_weight.rename("weight")],
                axis=1,
            ).iterrows()
        ]

    else:
        heat_data = [[row[lat_col], row[lon_col]] for _, row in df.iterrows()]

    HeatMap(
        heat_data,
        name="Spatial Density",
        radius=radius,
        blur=blur,
        min_opacity=0.3,
        gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 1.0: "red"},
    ).add_to(map_object)

    folium.LayerControl().add_to(map_object)

    return map_object


def plot_animated_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    time_col: str,
    color_col: str | None = None,
    map_style: str = "carto-positron",
) -> go.Figure:
    """Create an animated Plotly spatial map.

    Builds a frame-based map animation using the selected temporal or
    categorical column as the animation index. Each frame shows the spatial
    distribution of points for one unique value of the frame column.

    Args:
        df: Dataset containing spatial and frame-index columns.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        time_col: Column used to define animation frames.
        color_col: Optional feature used for point coloring.
        map_style: Plotly map tile style.

    Returns:
        A configured Plotly figure with animation controls.
    """
    plot_df = df.sort_values(by=time_col).copy()
    plot_df[time_col] = plot_df[time_col].astype(str)

    center_lat = plot_df[lat_col].median()
    center_lon = plot_df[lon_col].median()

    fig = px.scatter_mapbox(
        plot_df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        animation_frame=time_col,
        zoom=11,
        center=dict(lat=center_lat, lon=center_lon),
        mapbox_style=map_style,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        opacity=0.75,
        size_max=15,
        hover_data=[time_col],
    )

    fig.update_layout(
        title=(
            f"<b>Dynamic Spatial Evolution</b><br>"
            f"<sup>Frame Index: {time_col}</sup>"
        ),
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        font=DEFAULT_FONT,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.15,
                x=0.05,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[dict(y=-0.15, x=0.2, len=0.75)],
    )

    fig.update_traces(marker=dict(size=6))

    return fig
