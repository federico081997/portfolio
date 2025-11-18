"""
figures.py

Utility module for constructing Plotly figures used throughout the dashboard.

This module provides a set of lightweight, configurable helper functions for
building common chart types—line charts, bar charts, and pie/donut charts.
Each helper accepts pre-aggregated data along with optional customization
dictionaries:

- `layout_kwargs`: parameters forwarded to `fig.update_layout(...)`,
  governing global figure properties such as titles, axes, margins,
  background, or legend configuration.

- `trace_kwargs`: parameters forwarded to `fig.update_traces(...)`,
  controlling stylistic properties of the graphical elements such as
  colors, markers, text labels, and hover templates.

- `note_kwargs`: parameters forwarded to `fig.add_annotation(...)`,
  enabling optional annotations, totals, labels, or contextual notes.

The functions themselves impose no constraints on the shape or content of
the provided DataFrame. They simply build the base Plotly Express figure
and apply any additional styling or annotations required by the caller.
All returned objects are fully configured `plotly.graph_objects.Figure`
instances suitable for use in Dash via `dcc.Graph(figure=...)`.
"""

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


def plot(
    kind: str,
    df: pd.DataFrame,
    x: str | pd.DataFrame,
    y: str | pd.DataFrame,
    px_kwargs: dict = {},
    layout_kwargs: dict = {},
    trace_kwargs: dict = {},
    note_kwargs: dict = {},
) -> go.Figure:
    """
    Construct and return a line chart.

    Parameters
    ----------
    kind : str
        The type of plot that has to be plotted (i.e. `line`, `bar`, `pie`).
    df : pd.DataFrame
        Pre-aggregated data used as the source for the plot.
    x : str | pd.DataFrame
        Column name or series to place on the x-axis or, in case of a pie chart,
        the magnitude of each slice.
    y : str | pd.DataFrame
        Column name or series to place on the y-axis or, in case of a pie chart,
        the categorical labels.
    px_kwargs : dict, optional
        Parameters forwarded to `px.line/bar/pie()`.
    layout_kwargs : dict, optional
        Parameters forwarded to `Figure.update_layout()`.
    trace_kwargs : dict, optional
        Parameters forwarded to `Figure.update_traces()`.
    note_kwargs : dict, optional
        Parameters forwarded to `Figure.add_annotation()`.

    Returns
    -------
    go.Figure
        A configured Plotly chart of a specific type.
    """

    match kind:
        case 'line':               
            fig = px.line(df, x=x, y=y, **px_kwargs)
        case 'bar':
            fig = px.bar(df, x=x, y=y, **px_kwargs)  
        case 'pie':
            fig = px.pie(df, values=x, names=y, **px_kwargs)
        case _:
            raise ValueError(f"Unsupported chart type: '{kind}'. Expected one of ['line', 'bar', 'pie'].")    

    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    if trace_kwargs:
        fig.update_traces(**trace_kwargs)

    if note_kwargs:
        fig.add_annotation(**note_kwargs)

    return fig


