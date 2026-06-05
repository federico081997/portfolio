import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    color_col: str = None,
    bins: int = 30,
    log_y: bool = False,
) -> go.Figure:
    """
    Generates a statistical distribution plot for a specified feature.

    Automatically detects the data type. For numerical features, it renders a
    histogram with an aligned marginal box plot. For categorical features,
    it renders a frequency bar chart, safely grouping secondary color columns if provided.

    Args:
        df (pd.DataFrame): The input dataset containing the features to visualize.
        column (str): The name of the primary feature to plot on the x-axis.
        color_col (str, optional): The name of a categorical column used to segment
            and color-code the distribution. Defaults to None.
        bins (int, optional): The number of bins to divide the data into. This
            argument is only applied to numerical histograms. Defaults to 30.
        log_y (bool, optional): If True, applies a logarithmic scale to the
            y-axis. Highly recommended for severely skewed distributions (e.g., income,
            transaction amounts). Defaults to False.

    Returns:
        go.Figure: A fully configured Plotly Figure object ready for rendering.
    """
    # Define a clean, professional color palette
    color_palette = px.colors.qualitative.Prism

    if pd.api.types.is_numeric_dtype(df[column]):
        # Create a layout with 2 rows, sharing the X-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

        # Add Histogram (Row 1)
        hist = px.histogram(
            df,
            x=column,
            color=color_col,
            nbins=bins,
            log_y=log_y,
            color_discrete_sequence=color_palette,
            opacity=0.85,
        )
        # Add a subtle border to the histogram bars for crispness
        hist.update_traces(marker_line_width=1, marker_line_color="white")

        for trace in hist.data:
            fig.add_trace(trace, row=1, col=1)

        # Add Box Plot (Row 2)
        box = px.box(
            df, x=column, color=color_col, color_discrete_sequence=color_palette
        )
        for trace in box.data:
            # Prevent the box plot from generating duplicate legend items
            trace.showlegend = False
            fig.add_trace(trace, row=2, col=1)

        fig.update_layout(
            title=f"<b>Distribution of {column}</b>",
            template="plotly_white",
            barmode="overlay" if color_col else "relative",
            hovermode="x unified",  # Cleaner, single-column hover tooltip
            legend_title_text=color_col if color_col else "",
        )

        # Clean up axes
        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=1,
            showgrid=True,
            gridcolor="#E5E5E5",
            type="log" if log_y else "linear",
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=2, col=1
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)  # Hide top X-axis labels
        fig.update_xaxes(title_text=column, row=2, col=1)

    else:
        # Handle categorical grouping safely
        if color_col:
            counts = df.groupby([column, color_col]).size().reset_index(name="Count")
            fig = px.bar(
                counts,
                x=column,
                y="Count",
                color=color_col,
                log_y=log_y,
                title=f"<b>Frequency of {column}</b><br><sup>Segmented by {color_col}</sup>",
                template="plotly_white",
                barmode="stack",
                color_discrete_sequence=color_palette,
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
                color_discrete_sequence=color_palette,
                opacity=0.9,
            )

        # Force Plotly to sort the categorical bars from highest to lowest total count
        fig.update_layout(
            xaxis={"categoryorder": "total descending"}, hovermode="x unified"
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_yaxes(title_text="Count", showgrid=True, gridcolor="#E5E5E5")
        fig.update_xaxes(
            title_text=""
        )  # Remove redundant X-axis title if categories are self-explanatory

    fig.update_layout(
        bargap=0.1,
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(t=70, b=40, l=40, r=40),
    )

    return fig


def plot_categorical_variance(
    df: pd.DataFrame, cat_col: str, num_col: str, plot_type: str = "Box"
) -> go.Figure:
    """
    Evaluates how a numerical distribution shifts across different categorical groups.

    Generates either a Box plot (to visualize interquartile ranges and explicit outliers)
    or a Violin plot (to visualize the Kernel Density Estimation of the probability shape).
    Automatically sorts the categories on the X-axis by their median values in descending
    order to make statistical comparisons intuitive.

    Args:
        df (pd.DataFrame): The active dataset containing the features.
        cat_col (str): The categorical feature defining the groups (X-axis).
        num_col (str): The numerical feature to measure (Y-axis).
        plot_type (str, optional): The geometry to render. Accepts 'Box' or 'Violin'.
            Defaults to "Box".

    Returns:
        go.Figure: A fully configured and styled Plotly Figure object.
    """
    color_palette = px.colors.qualitative.Prism

    # Mathematical Sorting: Find the median for each category and sort descending
    # This prevents a chaotic, jumbled chart and creates a clean trend line
    medians = df.groupby(cat_col)[num_col].median().sort_values(ascending=False)
    category_order = medians.index.tolist()

    if plot_type == "Box":
        fig = px.box(
            df,
            x=cat_col,
            y=num_col,
            color=cat_col,
            category_orders={cat_col: category_order},
            color_discrete_sequence=color_palette,
        )
        title_text = f"<b>{num_col} Variance</b><br><sup>Segmented by {cat_col}</sup>"
    else:
        fig = px.violin(
            df,
            x=cat_col,
            y=num_col,
            color=cat_col,
            box=True,  # Overlays a miniature box plot inside the violin
            category_orders={cat_col: category_order},
            color_discrete_sequence=color_palette,
        )
        title_text = f"<b>{num_col} Density</b><br><sup>Segmented by {cat_col}</sup>"

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        showlegend=False,  # The X-axis already has the labels; the legend is redundant
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(t=70, b=40, l=40, r=40),
    )

    fig.update_yaxes(title_text=num_col, showgrid=True, gridcolor="#E5E5E5")

    fig.update_xaxes(title_text="", tickangle=-45)

    return fig


def plot_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    size_col: str = None,
    trendline: str = None,
    marginals: bool = False,
) -> go.Figure:
    """
    Generates a parameterized 2D Scatter plot to evaluate feature interactions.

    Capable of plotting up to 4 dimensions simultaneously (X, Y, Color, Size) while
    overlaying statistical trendlines to detect linear or non-linear patterns.
    Optimized for large datasets by employing opacity blending and marker outlines
    to prevent dense data clusters from rendering as a solid, unreadable blob.

    Args:
        df (pd.DataFrame): The active dataset.
        x_col (str): The independent/predictor numerical variable.
        y_col (str): The dependent/target numerical variable.
        color_col (str, optional): Categorical variable for cluster segmentation.
        size_col (str, optional): Numerical variable for data point scaling.
        trendline (str, optional): 'ols' for Ordinary Least Squares (linear) or
            'lowess' for Locally Weighted Scatterplot Smoothing (non-linear).
        marginals (bool, optional): Whether to display 1D distributions on the axes.

    Returns:
        go.Figure: A configured Plotly Figure object.
    """
    color_palette = px.colors.qualitative.Prism
    marg = "histogram" if marginals else None
    tl = trendline if trendline in ["ols", "lowess"] else None

    # Base Plot Generation
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        trendline=tl,
        marginal_x=marg,
        marginal_y=marg,
        color_discrete_sequence=color_palette,
        opacity=0.6,
        size_max=35,
        hover_data=df.columns[:5],
    )

    # Dynamic HTML Title Generation
    title_text = f"<b>Relationship: {x_col} vs {y_col}</b>"
    if color_col or size_col:
        subs = []
        if color_col:
            subs.append(f"Color: {color_col}")
        if size_col:
            subs.append(f"Size: {size_col}")
        title_text += f"<br><sup>{' | '.join(subs)}</sup>"

    # Global Layout & Styling
    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(t=70, b=40, l=40, r=40),
        legend_title_text=color_col if color_col else "",
    )

    # Axis Cleanup
    fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5", zeroline=False)

    # Trace Polish
    # Add a subtle dark outline to all scatter dots so overlapping points remain distinct
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    # If a trendline was requested, make it slightly thicker so it stands out against the noise
    if tl:
        fig.update_traces(line=dict(width=3), selector=dict(mode="lines"))

    return fig


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    val_col: str,
    color_col: str = None,
    rolling_window: int = 0,
) -> go.Figure:
    """
    Visualizes temporal dynamics and trends over a chronological time index.

    Optionally calculates and overlays a Moving Average (MA). To ensure readability,
    if an MA is applied, the noisy raw data is visually suppressed (faded and thinned)
    so the underlying long-term trend becomes the focal point of the chart. Includes
    an interactive range slider for chronological zooming.

    Args:
        df (pd.DataFrame): The active dataset.
        time_col (str): The datetime feature for the X-axis.
        val_col (str): The numerical metric to track on the Y-axis.
        color_col (str, optional): Categorical feature for tracking multiple series.
        rolling_window (int, optional): Number of periods for the moving average calculation.
            Set to 0 to disable. Defaults to 0.

    Returns:
        go.Figure: A fully configured Plotly Figure object.
    """
    color_palette = px.colors.qualitative.Prism

    # Ensure strict chronological ordering
    plot_df = df.sort_values(by=time_col).copy()

    # Moving Average & DataFrame Melting
    ma_label = f"{rolling_window}-Period MA"

    if rolling_window > 0:
        if color_col:
            plot_df[ma_label] = plot_df.groupby(color_col)[val_col].transform(
                lambda x: x.rolling(rolling_window, min_periods=1).mean()
            )
        else:
            plot_df[ma_label] = (
                plot_df[val_col].rolling(rolling_window, min_periods=1).mean()
            )

        # Melt to format data for Plotly's coloring/dashing engine
        plot_df = plot_df.melt(
            id_vars=[time_col] + ([color_col] if color_col else []),
            value_vars=[val_col, ma_label],
            var_name="Trend",
            value_name="Value",
        )

        # Render Multi-Line Chart
        fig = px.line(
            plot_df,
            x=time_col,
            y="Value",
            color=color_col if color_col else "Trend",
            line_dash="Trend" if color_col else None,
            color_discrete_sequence=color_palette,
        )

        # 4. Establish Visual Hierarchy (Fade raw data, highlight MA)
        for trace in fig.data:
            if ma_label in trace.name:
                trace.line.width = 3
                trace.opacity = 1.0
            else:
                trace.line.width = 1
                trace.opacity = 0.35

    else:
        # Render Standard Line Chart
        fig = px.line(
            plot_df,
            x=time_col,
            y=val_col,
            color=color_col,
            color_discrete_sequence=color_palette,
        )
        fig.update_traces(line=dict(width=2))

    # 5. Dynamic HTML Title Generation
    title_text = f"<b>{val_col} over Time</b>"
    subs = []
    if color_col:
        subs.append(f"Segmented by {color_col}")
    if rolling_window > 0:
        subs.append(f"Smoothing: {rolling_window}-Period MA")
    if subs:
        title_text += f"<br><sup>{' | '.join(subs)}</sup>"

    # 6. Global Layout & Axes Polish
    fig.update_layout(
        title=title_text,
        template="plotly_white",
        hovermode="x unified",
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(t=70, b=40, l=40, r=40),
        legend_title_text="",
    )

    # Remove the redundant X-axis title and add an interactive scrubbing slider
    fig.update_xaxes(title_text="", showgrid=False, rangeslider_visible=True)
    fig.update_yaxes(
        title_text=val_col, showgrid=True, gridcolor="#E5E5E5", zeroline=False
    )

    return fig


def plot_scatter_matrix(
    df: pd.DataFrame, columns: list, color_col: str = None
) -> go.Figure:
    """
    Generates a scatterplot matrix (pairplot) for high-dimensional analysis.

    Provides a dense grid of 2D scatter plots for all combinations of the selected
    features, allowing rapid visual identification of collinearity and clusters.
    Optimized for large datasets by employing extremely small, semi-transparent
    markers to mitigate severe overplotting in the matrix grid.

    Args:
        df (pd.DataFrame): The active dataset.
        columns (list): List of numerical features to cross-plot.
        color_col (str, optional): Categorical feature for hue segmentation.

    Returns:
        go.Figure: A configured Plotly Figure object.
    """
    color_palette = px.colors.qualitative.Prism

    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color_col,
        color_discrete_sequence=color_palette,
    )

    # Trace Polish
    # We must hide diagonals (redundant) and make markers tiny & transparent
    # to prevent the grid from turning into a solid, unreadable blob of color.
    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=True,
        showlowerhalf=True,
        marker=dict(
            size=3,  # Extremely small markers
            opacity=0.4,  # High transparency for density blending
            line=dict(width=0),  # No borders, as they clutter small matrices
        ),
    )

    # Dynamic HTML Title
    title_text = "<b>Multivariate Scatter Matrix</b>"
    if color_col:
        title_text += f"<br><sup>Segmented by {color_col}</sup>"

    # Global Layout & Styling
    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(t=70, b=40, l=40, r=40),
        hovermode="closest",
        legend_title_text=color_col if color_col else "",
    )

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> go.Figure:
    """
    Calculates the statistical collinearity matrix and visualizes it as a heatmap.

    Evaluates the strength of linear (Pearson) or monotonic (Spearman) relationships
    between all numerical variables. The color scale is strictly locked from -1 to 1
    to ensure red always signifies positive correlation and blue signifies negative.

    Args:
        df (pd.DataFrame): The active dataset.
        method (str, optional): 'pearson' for measuring standard linear correlation,
            or 'spearman' for rank-based monotonic correlation (resistant to outliers).
            Defaults to 'pearson'.

    Returns:
        go.Figure: A configured Plotly Figure object, or None if insufficient numeric columns.
    """
    num_df = df.select_dtypes(include=["float64", "int64", "float32", "int32"])

    # Need at least 2 numerical columns to calculate correlation
    if num_df.empty or len(num_df.columns) < 2:
        return None

    # Calculate correlation and round it to 2 decimals for clean rendering
    corr_matrix = np.round(num_df.corr(method=method), 2)

    # 1. Generate Heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",  # Red (+), White (0), Blue (-)
        zmin=-1,
        zmax=1,
    )

    # 2. Global Layout & Styling
    title_text = f"<b>{method.capitalize()} Correlation Heatmap</b>"
    title_text += f"<br><sup>Identifies {'linear' if method == 'pearson' else 'monotonic'} multicollinearity</sup>"

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(t=70, b=40, l=40, r=40),
        coloraxis_colorbar=dict(
            title="<b>Correlation</b>",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
        ),
    )

    # Clean up gridlines so the colored blocks sit flush against each other
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def plot_hierarchical_sunburst(
    df: pd.DataFrame, path_columns: list, value_col: str = None
) -> go.Figure:
    """
    Generates a radial treemap (Sunburst chart) to visualize hierarchical categorical logic.

    Allows users to drill down through categorical segments. Automatically handles
    missing data in the path and protects the rendering engine by filtering out zero
    or negative weights that would otherwise cause a geometric calculation crash.

    Args:
        df (pd.DataFrame): The active dataset.
        path_columns (list): Ordered list of categorical features defining the drill-down
            hierarchy (e.g., ['Region', 'Country', 'City']).
        value_col (str, optional): Numerical column defining the size/weight of each
            radial slice. If None, uses raw row count.

    Returns:
        go.Figure: A fully configured and styled Plotly Figure object.
    """
    color_palette = px.colors.qualitative.Prism

    # Data Prep & Safety Filters
    clean_df = df.dropna(subset=path_columns).copy()

    # Sunbursts will fatally crash if values are <= 0
    if value_col:
        clean_df = clean_df[clean_df[value_col] > 0]

    # Render Base Plot
    fig = px.sunburst(
        clean_df,
        path=path_columns,
        values=value_col,
        color_discrete_sequence=color_palette,
    )

    # Dynamic HTML Title
    title_text = "<b>Hierarchical Segmentation</b>"
    title_text += f"<br><sup>Path: {' ➔ '.join(path_columns)}</sup>"
    title_text += f"<br><sup>Size: {value_col if value_col else 'Row Count'}</sup>"

    # Trace Polish
    fig.update_traces(
        textinfo="label+percent parent",  # Shows what % it takes up of its direct parent
        insidetextorientation="radial",  # Curves the text along the slice geometry
        hovertemplate="<b>%{label}</b><br>Count/Value: %{value}<br>Parent Contribution: %{percentParent:.1%}<extra></extra>",
        marker=dict(line=dict(color="white", width=1.5)),  # Crisp slice separation
    )

    # Global Layout & Styling
    fig.update_layout(
        title=title_text,
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(t=90, b=40, l=40, r=40),
    )

    return fig
