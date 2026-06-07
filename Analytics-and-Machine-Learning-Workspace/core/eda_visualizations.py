"""Plotly visualization utilities for exploratory data analysis.

This module contains reusable plotting functions for the Exploratory Data
Analysis Studio. Each function accepts a pandas DataFrame and returns a Plotly
figure configured for interactive Streamlit rendering.

The visualizations cover feature distributions, categorical variance,
two-dimensional relationships, time-series trends, scatter matrices,
correlation heatmaps, and hierarchical sunburst charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_COLOR_PALETTE = px.colors.qualitative.Prism
DEFAULT_FONT = dict(family="system-ui, -apple-system, sans-serif", size=13)
DEFAULT_MARGIN = dict(t=70, b=40, l=40, r=40)
DEFAULT_GRID_COLOR = "#E5E5E5"


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    color_col: str | None = None,
    bins: int = 30,
    log_y: bool = False,
) -> go.Figure:
    """Create a distribution plot for a selected feature.

    Numerical features are rendered as histograms with an aligned marginal box
    plot. Non-numerical features are rendered as categorical frequency bar
    charts. Optional color segmentation can be applied when a grouping column is
    provided.

    Args:
        df: Input dataset.
        column: Feature to plot.
        color_col: Optional categorical column used for color segmentation.
        bins: Number of histogram bins for numerical features.
        log_y: Whether to use a logarithmic y-axis.

    Returns:
        A configured Plotly figure.
    """
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

        hist = px.histogram(
            df,
            x=column,
            color=color_col,
            nbins=bins,
            log_y=log_y,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
            opacity=0.85,
        )
        hist.update_traces(marker_line_width=1, marker_line_color="white")

        for trace in hist.data:
            fig.add_trace(trace, row=1, col=1)

        box = px.box(
            df,
            x=column,
            color=color_col,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )

        for trace in box.data:
            trace.showlegend = False
            fig.add_trace(trace, row=2, col=1)

        fig.update_layout(
            title=f"<b>Distribution of {column}</b>",
            template="plotly_white",
            barmode="overlay" if color_col else "relative",
            hovermode="x unified",
            legend_title_text=color_col if color_col else "",
        )

        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=1,
            showgrid=True,
            gridcolor=DEFAULT_GRID_COLOR,
            type="log" if log_y else "linear",
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            row=2,
            col=1,
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(title_text=column, row=2, col=1)

    else:
        if color_col:
            counts = df.groupby([column, color_col]).size().reset_index(name="Count")
            fig = px.bar(
                counts,
                x=column,
                y="Count",
                color=color_col,
                log_y=log_y,
                title=(
                    f"<b>Frequency of {column}</b><br>"
                    f"<sup>Segmented by {color_col}</sup>"
                ),
                template="plotly_white",
                barmode="stack",
                color_discrete_sequence=DEFAULT_COLOR_PALETTE,
                opacity=0.9,
            )

        else:
            counts = df[column].value_counts().reset_index()
            counts.columns = [column, "Count"]

            fig = px.bar(
                counts,
                x=column,
                y="Count",
                color=column,
                log_y=log_y,
                title=f"<b>Frequency of {column}</b>",
                template="plotly_white",
                color_discrete_sequence=DEFAULT_COLOR_PALETTE,
                opacity=0.9,
            )

        fig.update_layout(
            xaxis={"categoryorder": "total descending"},
            hovermode="x unified",
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_yaxes(
            title_text="Count",
            showgrid=True,
            gridcolor=DEFAULT_GRID_COLOR,
        )
        fig.update_xaxes(title_text="")

    fig.update_layout(
        bargap=0.1,
        font=DEFAULT_FONT,
        margin=DEFAULT_MARGIN,
    )

    return fig


def plot_categorical_variance(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    plot_type: str = "Box",
) -> go.Figure:
    """Create a box or violin plot grouped by category.

    The categorical axis is sorted by the median value of the numerical feature
    so that group-level comparisons are easier to interpret.

    Args:
        df: Input dataset.
        cat_col: Categorical feature used to define groups.
        num_col: Numerical feature plotted on the y-axis.
        plot_type: Plot geometry to use. Supported values are ``"Box"`` and
            ``"Violin"``.

    Returns:
        A configured Plotly figure.
    """
    medians = df.groupby(cat_col)[num_col].median().sort_values(ascending=False)
    category_order = medians.index.tolist()

    if plot_type == "Box":
        fig = px.box(
            df,
            x=cat_col,
            y=num_col,
            color=cat_col,
            category_orders={cat_col: category_order},
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        title_text = f"<b>{num_col} Variance</b><br><sup>Segmented by {cat_col}</sup>"

    else:
        fig = px.violin(
            df,
            x=cat_col,
            y=num_col,
            color=cat_col,
            box=True,
            category_orders={cat_col: category_order},
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        title_text = f"<b>{num_col} Density</b><br><sup>Segmented by {cat_col}</sup>"

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        showlegend=False,
        font=DEFAULT_FONT,
        margin=DEFAULT_MARGIN,
    )
    fig.update_yaxes(
        title_text=num_col,
        showgrid=True,
        gridcolor=DEFAULT_GRID_COLOR,
    )
    fig.update_xaxes(title_text="", tickangle=-45)

    return fig


def plot_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    trendline: str | None = None,
    marginals: bool = False,
) -> go.Figure:
    """Create a two-dimensional scatter plot for feature relationships.

    The plot supports optional color segmentation, marker sizing, statistical
    trendlines, and marginal distributions for the selected x and y features.

    Args:
        df: Input dataset.
        x_col: Numerical feature plotted on the x-axis.
        y_col: Numerical feature plotted on the y-axis.
        color_col: Optional categorical feature used for color segmentation.
        size_col: Optional numerical feature used to scale marker size.
        trendline: Optional trendline type. Supported values are ``"ols"`` and
            ``"lowess"``.
        marginals: Whether to show marginal histograms on the x and y axes.

    Returns:
        A configured Plotly figure.
    """
    marginal_plot = "histogram" if marginals else None
    trendline_arg = trendline if trendline in ["ols", "lowess"] else None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        trendline=trendline_arg,
        marginal_x=marginal_plot,
        marginal_y=marginal_plot,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        opacity=0.6,
        size_max=35,
        hover_data=df.columns[:5],
    )

    title_text = f"<b>Relationship: {x_col} vs {y_col}</b>"

    if color_col or size_col:
        subtitle_parts = []

        if color_col:
            subtitle_parts.append(f"Color: {color_col}")

        if size_col:
            subtitle_parts.append(f"Size: {size_col}")

        title_text += f"<br><sup>{' | '.join(subtitle_parts)}</sup>"

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=DEFAULT_FONT,
        margin=DEFAULT_MARGIN,
        legend_title_text=color_col if color_col else "",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=DEFAULT_GRID_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=DEFAULT_GRID_COLOR,
        zeroline=False,
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    if trendline_arg:
        fig.update_traces(line=dict(width=3), selector=dict(mode="lines"))

    return fig


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    val_col: str,
    color_col: str | None = None,
    rolling_window: int = 0,
) -> go.Figure:
    """Create a time-series plot with optional moving-average smoothing.

    When a rolling window is provided, the raw series is faded and the moving
    average is emphasized to make long-term trends easier to inspect.

    Args:
        df: Input dataset.
        time_col: Datetime feature plotted on the x-axis.
        val_col: Numerical feature plotted on the y-axis.
        color_col: Optional categorical feature used to split the time series.
        rolling_window: Number of periods used for moving-average smoothing.
            Use ``0`` to disable smoothing.

    Returns:
        A configured Plotly figure.
    """
    plot_df = df.sort_values(by=time_col).copy()
    ma_label = f"{rolling_window}-Period MA"

    if rolling_window > 0:
        if color_col:
            plot_df[ma_label] = plot_df.groupby(color_col)[val_col].transform(
                lambda series: series.rolling(rolling_window, min_periods=1).mean()
            )
        else:
            plot_df[ma_label] = (
                plot_df[val_col].rolling(rolling_window, min_periods=1).mean()
            )

        plot_df = plot_df.melt(
            id_vars=[time_col] + ([color_col] if color_col else []),
            value_vars=[val_col, ma_label],
            var_name="Trend",
            value_name="Value",
        )

        fig = px.line(
            plot_df,
            x=time_col,
            y="Value",
            color=color_col if color_col else "Trend",
            line_dash="Trend" if color_col else None,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )

        for trace in fig.data:
            if ma_label in trace.name:
                trace.line.width = 3
                trace.opacity = 1.0
            else:
                trace.line.width = 1
                trace.opacity = 0.35

    else:
        fig = px.line(
            plot_df,
            x=time_col,
            y=val_col,
            color=color_col,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        fig.update_traces(line=dict(width=2))

    title_text = f"<b>{val_col} over Time</b>"
    subtitle_parts = []

    if color_col:
        subtitle_parts.append(f"Segmented by {color_col}")

    if rolling_window > 0:
        subtitle_parts.append(f"Smoothing: {rolling_window}-Period MA")

    if subtitle_parts:
        title_text += f"<br><sup>{' | '.join(subtitle_parts)}</sup>"

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        hovermode="x unified",
        font=DEFAULT_FONT,
        margin=DEFAULT_MARGIN,
        legend_title_text="",
    )
    fig.update_xaxes(
        title_text="",
        showgrid=False,
        rangeslider_visible=True,
    )
    fig.update_yaxes(
        title_text=val_col,
        showgrid=True,
        gridcolor=DEFAULT_GRID_COLOR,
        zeroline=False,
    )

    return fig


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: list,
    color_col: str | None = None,
) -> go.Figure:
    """Create a scatterplot matrix for selected numerical features.

    The matrix compares all selected numerical dimensions pairwise and can
    optionally color points by a categorical feature.

    Args:
        df: Input dataset.
        columns: Numerical features to include in the matrix.
        color_col: Optional categorical feature used for color segmentation.

    Returns:
        A configured Plotly figure.
    """
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
    )

    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=True,
        showlowerhalf=True,
        marker=dict(
            size=3,
            opacity=0.4,
            line=dict(width=0),
        ),
    )

    title_text = "<b>Multivariate Scatter Matrix</b>"

    if color_col:
        title_text += f"<br><sup>Segmented by {color_col}</sup>"

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=DEFAULT_FONT,
        margin=DEFAULT_MARGIN,
        hovermode="closest",
        legend_title_text=color_col if color_col else "",
    )

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
) -> go.Figure | None:
    """Create a correlation heatmap for numerical features.

    Computes a Pearson or Spearman correlation matrix for all numerical columns
    and visualizes it using a fixed ``[-1, 1]`` color scale.

    Args:
        df: Input dataset.
        method: Correlation method. Supported values are ``"pearson"`` and
            ``"spearman"``.

    Returns:
        A configured Plotly figure, or ``None`` if fewer than two numerical
        columns are available.
    """
    num_df = df.select_dtypes(include=["float64", "int64", "float32", "int32"])

    if num_df.empty or len(num_df.columns) < 2:
        return None

    corr_matrix = np.round(num_df.corr(method=method), 2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )

    relationship_type = "linear" if method == "pearson" else "monotonic"
    title_text = (
        f"<b>{method.capitalize()} Correlation Heatmap</b>"
        f"<br><sup>Identifies {relationship_type} multicollinearity</sup>"
    )

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=DEFAULT_FONT,
        margin=DEFAULT_MARGIN,
        coloraxis_colorbar=dict(
            title="<b>Correlation</b>",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def plot_hierarchical_sunburst(
    df: pd.DataFrame,
    path_columns: list,
    value_col: str | None = None,
) -> go.Figure:
    """Create a hierarchical sunburst chart.

    Missing values in hierarchy columns are removed before plotting. When a
    value column is provided, rows with non-positive values are filtered out to
    prevent invalid sunburst geometry.

    Args:
        df: Input dataset.
        path_columns: Ordered categorical columns defining the hierarchy.
        value_col: Optional numerical column used to size each slice. If not
            provided, row count is used.

    Returns:
        A configured Plotly figure.
    """
    clean_df = df.dropna(subset=path_columns).copy()

    if value_col:
        clean_df = clean_df[clean_df[value_col] > 0]

    fig = px.sunburst(
        clean_df,
        path=path_columns,
        values=value_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
    )

    title_text = "<b>Hierarchical Segmentation</b>"
    title_text += f"<br><sup>Path: {' ➔ '.join(path_columns)}</sup>"
    title_text += f"<br><sup>Size: {value_col if value_col else 'Row Count'}</sup>"

    fig.update_traces(
        textinfo="label+percent parent",
        insidetextorientation="radial",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count/Value: %{value}<br>"
            "Parent Contribution: %{percentParent:.1%}"
            "<extra></extra>"
        ),
        marker=dict(line=dict(color="white", width=1.5)),
    )

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=DEFAULT_FONT,
        margin=dict(t=90, b=40, l=40, r=40),
    )

    return fig
