"""
figures.py

Plotting and visualisation helpers for the SpaceX Launch Dashboard.

This module centralises the creation of Plotly figures and Folium maps used
throughout the dashboard so that styling and behaviour remain consistent
across tabs and callbacks.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster

from constants import PALETTE


# =============================================================================
# Public APIs
# =============================================================================


def plot(
    kind: str,
    df: pd.DataFrame,
    x: Any,
    y: Any,
    px_kwargs: Optional[Dict[str, Any]] = None,
    layout_kwargs: Optional[Dict[str, Any]] = None,
    trace_kwargs: Optional[Dict[str, Any]] = None,
    note_kwargs: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Create a Plotly figure of a supported type and apply common configuration.

    The function currently supports:
    - ``scatter`` via :func:`plotly.express.scatter`
    - ``pie`` via :func:`plotly.express.pie`

    :param kind: Plot type to create (e.g. ``"scatter"`` or ``"pie"``).
    :type kind: str
    :param df: DataFrame used as the data source for the plot.
    :type df: pd.DataFrame
    :param x: X-axis column/series for scatter, or values for pie.
    :type x: Any
    :param y: Y-axis column/series for scatter, or labels for pie.
    :type y: Any
    :param px_kwargs: Keyword arguments forwarded to the Plotly Express constructor.
    :type px_kwargs: Dict[str, Any], optional
    :param layout_kwargs: Keyword arguments forwarded to :meth:`go.Figure.update_layout`.
    :type layout_kwargs: Dict[str, Any], optional
    :param trace_kwargs: Keyword arguments forwarded to :meth:`go.Figure.update_traces`.
    :type trace_kwargs: Dict[str, Any], optional
    :param note_kwargs: Keyword arguments forwarded to :meth:`go.Figure.add_annotation`.
    :type note_kwargs: Dict[str, Any], optional
    :return: A configured Plotly figure.
    :rtype: plotly.graph_objects.Figure
    :raises ValueError: If ``kind`` is not a supported chart type.
    """
    # Avoid mutable default arguments and normalise "no kwargs" to empty dicts.
    px_kwargs = px_kwargs or {}
    layout_kwargs = layout_kwargs or {}
    trace_kwargs = trace_kwargs or {}
    note_kwargs = note_kwargs or {}

    # Construct the base figure using Plotly Express.
    match kind:
        case "scatter":
            fig = px.scatter(df, x=x, y=y, **px_kwargs)
        case "pie":
            fig = px.pie(df, values=x, names=y, **px_kwargs)
        case _:
            raise ValueError(
                f"Unsupported chart type: '{kind}'. Expected one of ['scatter', 'pie']."
            )

    # Apply optional post-configuration in a consistent order.
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    if trace_kwargs:
        fig.update_traces(**trace_kwargs)
    if note_kwargs:
        fig.add_annotation(**note_kwargs)

    return fig


def display_map(
    df: pd.DataFrame,
    location: List[float],
    zoom_start: int = 5,
    cluster: Optional[pd.DataFrame] = None,
) -> str:
    """
    Create a Folium map (HTML) showing launch sites and optional clustered markers.

    The `df` input is expected to contain rows of the form:
    (name, latitude, longitude). Each site is rendered as a circular marker with
    a popup label.

    If `cluster` is provided and non-empty, it is expected to contain rows of the form:
    (latitude, longitude, success_flag). Markers are added to a MarkerCluster with a
    popup label.

    :param df: DataFrame containing launch site name and coordinates.
    :type df: pd.DataFrame
    :param location: Initial map centre as ``[lat, lon]``.
    :type location: List[float]
    :param zoom_start: Initial zoom level for the map.
    :type zoom_start: int
    :param cluster: Optional DataFrame containing clustered marker inputs.
    :type cluster: pd.DataFrame, optional
    :return: Full HTML representation of the Folium map.
    :rtype: str
    """
    # Create map instance
    folium_map = folium.Map(location=location, zoom_start=zoom_start)

    # Mark each launch site position with a circular marker and a popup.
    for name, lat, lon in df.values:
        circle_text = f"""
        <div style="
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            font-size: 14px;
            font-weight: 700;
            padding: 6px 10px;
            white-space: nowrap;
        ">
            {name}
        </div>
        """

        folium.Circle(
            location=(lat, lon),
            radius=100,
            color=PALETTE[4],
            fill=True,
        ).add_child(
            folium.Popup(folium.Html(circle_text, script=True), max_width=220)
        ).add_to(
            folium_map
        )

    # Optional marker clustering layer (used for success/failure markers).
    if cluster is not None and not cluster.empty:
        marker_cluster = MarkerCluster()
        folium_map.add_child(marker_cluster)

        for lat, lon, success in cluster.values:
            # Support both booleans and string-like success values.
            is_success = success is True or "true" in str(success).lower()

            popup_text = f"""
            <div style="
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                font-size: 14px;
                padding: 6px 10px;
                white-space: nowrap;
            ">
                <div><b>Landing Outcome:</b><br>{"Success" if is_success else "Failure"}</div>
            </div>
            """

            # Folium expects icon_color as a single string, not a list.
            marker = folium.Marker(
                location=(lat, lon),
                popup=folium.Popup(folium.Html(popup_text, script=True), max_width=250),
                icon=folium.Icon(
                    color="white",
                    icon_color=PALETTE[2] if is_success else PALETTE[5],
                ),
            )
            marker_cluster.add_child(marker)

    return folium_map.get_root().render()


def display_confusion_matrix(cm: List[List[int]]) -> go.Figure:
    """
    Create a Plotly heatmap confusion matrix with readable annotations.

    :param cm: Confusion matrix as a 2×2 nested list in the form [[tn, fp], [fn, tp]].
    :type cm: list[list[int]]
    :return: A Plotly figure containing the confusion matrix heatmap.
    :rtype: plotly.graph_objects.Figure
    """
    labels = ["Not landed", "Landed"]

    # Guard against unexpected empty inputs.
    zmax = max(map(max, cm)) if cm else 1
    threshold = 0.6 * zmax  # Used to switch annotation colour for readability.

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
            xgap=2,
            ygap=2,
            hovertemplate=(
                "<b>True:</b> %{y}<br>"
                "<b>Predicted:</b> %{x}<br>"
                "<b>Count:</b> %{z}<extra></extra>"
            ),
        )
    )

    # Add numeric cell annotations with contrast-aware text colour.
    for i, y in enumerate(labels):
        for j, x in enumerate(labels):
            v = cm[i][j]
            text_color = "white" if v >= threshold else "black"
            fig.add_annotation(
                x=x,
                y=y,
                text=str(v),
                showarrow=False,
                font=dict(size=16, color=text_color),
            )

    # Layout and axis styling for a clean dashboard look.
    fig.update_layout(
        xaxis_title="Predicted labels",
        yaxis_title="True labels",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Add a border around the matrix area.
    fig.update_xaxes(
        showline=True,
        mirror=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        mirror=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
    )

    return fig
