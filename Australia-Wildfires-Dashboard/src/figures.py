"""
figures.py

Figure builders for the dashboard charts.

This module provides two helpers that construct Plotly figures from
pre-aggregated data:

- `generate_donut(df, total_area)`: donut chart of monthly average estimated fire area.
- `generate_bar(df, total_pixels)`: bar chart of monthly average pixel counts.

Both functions assume that the input dataframe is already filtered to a single
region/year and is aggregated at the monthly level. They return fully configured
Plotly figures ready to be assigned to Dash `dcc.Graph(figure=...)`.
"""

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


def generate_donut(df: pd.DataFrame, total_area: float) -> go.Figure:
    """
    Build the donut chart from pre-aggregated data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least:
        - 'Month_name' (month labels in desired order),
        - 'Estimated_fire_area' (monthly mean values).
        The function uses `df['Month_name'].unique()` to establish label order.
    total_area : float
        Numeric total to display in the donut hole. This is the sum of
        monthly means for the selected region/year and is formatted
        to two decimals inside the annotation.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly donut figure with:
        - categorical order aligned to `df['Month_name'].unique()`,
        - percent labels on slices,
        - a centered annotation showing the total,
        - a styled legend.
        - hover text shows value (km²) and percentage
    """
    # Build the donut: derive names and category order directly from the data
    fig = px.pie(
        data_frame=df,
        values="Estimated_fire_area",
        names=df["Month_name"].unique(),
        category_orders={"Month_name": list(df["Month_name"].unique())},
        hole=0.4,
    )

    # Layout: keep legend readable and visually separated
    fig.update_layout(
        legend=dict(
            title=dict(text="Month", font_weight=1000, side="top center"),
            bordercolor="black",
            borderwidth=1,
            xanchor="left",
            x=-0.3,
            y=0.5,
        ),
    )

    # Traces: stable slice ordering, formatted hover, soft qualitative palette
    fig.update_traces(
        sort=False,
        hovertemplate="<b>%{label}:</b><br>%{value:.2f} km²<br>(%{percent:.2%})<extra></extra>",
        marker=dict(colors=px.colors.qualitative.Pastel),
        texttemplate="%{percent:.2%}",
    )

    # Center annotation: present the overall total inside the donut hole
    fig.add_annotation(
        text=f"<b>Total</b>:<br>{total_area:.2f} km²",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig


def generate_bar(df: pd.DataFrame, total_pixels: int | float) -> go.Figure:
    """
    Build the bar chart from pre-aggregated data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least:
        - 'Month_name' (month labels in desired order),
        - 'Count' (monthly mean pixel counts; can be float or rounded upstream).
    total_pixels : int | float
        Numeric total to display as an annotation at the top of the plot area.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly bar figure with:
        - x-axis labeled “Month” (with grid enabled),
        - y-axis labeled “Pixel Count”,
        - a hover template that shows integer values,
        - a top-left annotation that displays the overall total.
    """
    # Bars: categories on x-axis, numeric measure on y-axis
    fig = px.bar(
        data_frame=df,
        x="Month_name",
        y="Count",
    )

    # Axes styling: subtle x-grid for vertical guidance; standard y title
    fig.update_layout(
        xaxis=dict(title="Month", title_standoff=0.2, showgrid=True),
        yaxis=dict(title="Pixel Count"),
    )

    # Hover: label + value
    fig.update_traces(
        hovertemplate="<b>%{x}:</b><br>%{y} pixels<extra></extra>",
    )

    # Annotation: communicate overall total at the top left of the figure
    fig.add_annotation(
        text=f"<b>Total number of pixels</b>: {int(total_pixels)} pixels",
        x=0,
        y=1.1,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig
