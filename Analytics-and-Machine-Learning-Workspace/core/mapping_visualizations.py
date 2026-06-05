import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_interactive_clusters(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    popup_cols: list = None,
    color_col: str = None,
    base_map: str = "CartoDB positron",
) -> folium.Map:
    """
    Generates a geospatial map utilizing Leaflet's MarkerCluster engine.

    Mathematically groups nearby coordinates into interactive nodes that
    dynamically break apart as the user zooms in, preventing browser DOM crashes
    associated with rendering thousands of raw HTML pins.

    Args:
        df (pd.DataFrame): The active dataset containing spatial coordinates.
        lat_col (str): The column name for Latitude.
        lon_col (str): The column name for Longitude.
        popup_cols (list, optional): Data columns to display when a final pin is clicked.
        color_col (str, optional): Categorical column used to color-code the markers.
        base_map (str, optional): The tile layer aesthetic. Defaults to "CartoDB positron".

    Returns:
        folium.Map: A configured Folium map object.
    """
    center = [df[lat_col].median(), df[lon_col].median()]
    m = folium.Map(location=center, zoom_start=12, tiles=base_map)

    marker_cluster = MarkerCluster(
        name="Clustered Data", overlay=True, control=True, icon_create_function=None
    )

    color_map = {}
    if color_col:
        unique_cats = df[color_col].dropna().unique()
        available_colors = [
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
        for i, cat in enumerate(unique_cats):
            color_map[cat] = available_colors[i % len(available_colors)]

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

    marker_cluster.add_to(m)
    folium.LayerControl().add_to(m)

    return m


def plot_density_heatmap(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str = None,
    base_map: str = "CartoDB dark_matter",
    radius: int = 15,
    blur: int = 15,
) -> folium.Map:
    """
    Generates a continuous spatial probability density function (Heatmap).

    Calculates the spatial concentration of data points. Highly effective for
    massive datasets as it translates discrete points into a WebGL gradient,
    saving memory and highlighting macro-level geographic trends.

    Args:
        df (pd.DataFrame): The active dataset containing spatial coordinates.
        lat_col (str): The column name for Latitude.
        lon_col (str): The column name for Longitude.
        weight_col (str, optional): Numerical column defining the intensity/gravity
            of a specific coordinate. Defaults to None (equal density).
        base_map (str, optional): The tile layer aesthetic. Defaults to "CartoDB dark_matter".
        radius (int, optional): The radius of influence for each data point.
        blur (int, optional): The smoothing factor for the gradient.

    Returns:
        folium.Map: A configured Folium map object.
    """
    center = [df[lat_col].median(), df[lon_col].median()]
    m = folium.Map(location=center, zoom_start=12, tiles=base_map)

    if weight_col and pd.api.types.is_numeric_dtype(df[weight_col]):
        max_val = df[weight_col].max()
        norm_weight = df[weight_col] / max_val if max_val > 0 else 1
        heat_data = [
            [row[lat_col], row[lon_col], row["weight"]]
            for _, row in pd.concat(
                [df[[lat_col, lon_col]], norm_weight.rename("weight")], axis=1
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
    ).add_to(m)

    folium.LayerControl().add_to(m)

    return m


def plot_animated_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    time_col: str,
    color_col: str = None,
    map_style: str = "carto-positron",
) -> go.Figure:
    """
    Generates a time-lapse animated spatial map using WebGL.

    Provides a dynamic visualization of geographical patterns evolving over a
    chronological index. The data is discretized into frames based on the
    specified time column, allowing playback and scrubbing.

    Args:
        df (pd.DataFrame): The dataset containing spatial and temporal features.
        lat_col (str): The column name containing latitude coordinates.
        lon_col (str): The column name containing longitude coordinates.
        time_col (str): The temporal or categorical column to use as animation frames.
        color_col (str, optional): Feature for color mapping. Defaults to None.
        map_style (str, optional): The base map tile style. Defaults to "carto-positron".

    Returns:
        go.Figure: A configured Plotly Mapbox Figure with animation controls.
    """
    color_palette = px.colors.qualitative.Prism

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
        color_discrete_sequence=color_palette,
        opacity=0.75,
        size_max=15,
        hover_data=[time_col],
    )

    fig.update_layout(
        title=f"<b>Dynamic Spatial Evolution</b><br><sup>Frame Index: {time_col}</sup>",
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
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
