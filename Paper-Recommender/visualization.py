import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc

# --------------------------------
# GLOBAL VISUAL THEME
# --------------------------------

PRIMARY = "#4C6EF5"
SECONDARY = "#ADB5BD"
BACKGROUND = "#F8F9FA"
PAGE_BACKGROUND = "#BBD5F6"
HEADER_BG = "#DFF0F8"
SURFACE = "#FFFFFF"
TEXT = "#212529"
MUTED_TEXT = "#6C757D"
GRID = "#E9ECEF"
BORDER = "#DEE2E6"

CLUSTER_COLORS = px.colors.qualitative.Set2

# --------------------------------
# FIGURES
# --------------------------------


def apply_plot_style(
    fig: go.Figure,
    height: int = 500,
    show_x_grid: bool = True,
    show_y_grid: bool = True,
) -> go.Figure:
    """
    Apply a consistent visual theme to Plotly figures.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(
            family="Inter, Arial, sans-serif",
            size=12,
            color=TEXT,
        ),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=TEXT),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER,
            font_size=12,
            font_family="Inter, Arial, sans-serif",
            font_color=TEXT,
        ),
    )

    fig.update_xaxes(
        showgrid=show_x_grid,
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=TEXT),
    )

    fig.update_yaxes(
        showgrid=show_y_grid,
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=TEXT),
    )

    return fig


def build_topic_scatter_plot(
    df: pd.DataFrame, selected_cluster_id: int | str | None = None
) -> go.Figure:
    """
    Build an interactive 2D topic map using UMAP coordinates.

    Behavior:
    - No selected cluster: show all papers colored by cluster
    - selected cluster:
        * Selected cluster is hoverable
        * Other papers are faint and not hoverable
        * Plot zooms to the relevant cluster
    """
    # Prepare the dataframe for the figure
    plot_df = df.copy().reset_index(drop=True)
    plot_df["paper_id"] = plot_df.index
    plot_df["cluster_str"] = plot_df["cluster_id"].astype(str)
    plot_df["published_date"] = pd.to_datetime(
        plot_df["published_date"], errors="coerce"
    )

    if selected_cluster_id is None:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="cluster_str",
            color_discrete_sequence=CLUSTER_COLORS,
            hover_name="title",
            hover_data={"cluster_str": False, "x": False, "y": False},
            custom_data=["paper_id"],
            render_mode="webgl",
        )

        fig.update_traces(
            marker=dict(size=4, opacity=0.4, line=dict(width=0)),
        )

        fig.update_layout(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            showlegend=False,
        )

        return apply_plot_style(fig)

    selected_cluster_id = str(selected_cluster_id)

    selected_df = plot_df[plot_df["cluster_str"] == selected_cluster_id].copy()
    other_df = plot_df[plot_df["cluster_str"] != selected_cluster_id].copy()

    fig = go.Figure()

    # Trace 1: Background papers, non-hoverable
    fig.add_trace(
        go.Scattergl(
            x=other_df["x"],
            y=other_df["y"],
            mode="markers",
            marker=dict(size=4, color=SECONDARY, opacity=0.06, line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Trace 2: selected cluster, hoverable and clickable
    fig.add_trace(
        go.Scattergl(
            x=selected_df["x"],
            y=selected_df["y"],
            mode="markers",
            marker=dict(size=5, color=PRIMARY, opacity=0.92, line=dict(width=0)),
            text=selected_df["title"],
            customdata=selected_df[["paper_id"]].to_numpy(),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        )
    )

    # Compute zoom window for selected cluster
    if not selected_df.empty:
        x_min = selected_df["x"].min()
        x_max = selected_df["x"].max()
        y_min = selected_df["y"].min()
        y_max = selected_df["y"].max()

        # Padding so points do not touch plot edges
        x_pad = max((x_max - x_min) * 0.10, 0.5)
        y_pad = max((y_max - y_min) * 0.10, 0.5)

        fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad])
        fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad])

    fig.update_layout(
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        showlegend=False,
        transition=dict(duration=700, easing="cubic-in-out"),
    )

    return apply_plot_style(fig)


def build_cluster_size_bar_chart(
    cluster_summary: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """
    Build a horizontal bar chart of the largest clusters.

    Expected columns:
    - cluster_id
    - cluster_label
    - cluster_size
    """
    # Prepare the dataframe for the figure
    plot_df = cluster_summary.copy()
    plot_df["cluster_str"] = plot_df["cluster_id"].astype(str)
    plot_df = plot_df.head(top_n).sort_values("cluster_size", ascending=True)

    fig = px.bar(
        plot_df,
        x="cluster_size",
        y="cluster_label",
        orientation="h",
        custom_data=["cluster_id"],
        labels={
            "cluster_size": "Number of Papers",
            "cluster_label": "Cluster",
        },
    )

    fig.update_traces(
        marker=dict(color=PRIMARY, line=dict(width=0)),
        hovertemplate=("<b>%{y}</b><br>" "Papers: %{x}<br>" "<extra></extra>"),
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title=None,
    )

    return apply_plot_style(fig, show_y_grid=False)


def build_cluster_size_histogram(cluster_summary: pd.DataFrame) -> go.Figure:
    """Cluster size histogram"""

    fig = px.histogram(
        cluster_summary,
        x="cluster_size",
        nbins=30,
        labels={"cluster_size": "Cluster Size"},
    )

    fig.update_layout(
        xaxis_title="Cluster Size",
        yaxis_title="Number of Clusters",
    )

    fig.update_traces(
        marker=dict(color=PRIMARY, line=dict(width=1, color=BACKGROUND)),
        hovertemplate=(
            "<b>Range:</b> %{x}<br>" "<b>Clusters:</b> %{y}<br>" "<extra></extra>"
        ),
    )

    return apply_plot_style(fig)


def build_category_distribution_bar_chart(
    df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:

    # Prepare data for the figure
    category_counts = df["category"].value_counts().head(top_n).reset_index()
    category_counts.columns = ["category", "paper_count"]
    category_counts = category_counts.sort_values("paper_count", ascending=True)

    fig = px.bar(
        category_counts,
        x="paper_count",
        y="category",
        orientation="h",
        labels={
            "paper_count": "Number of Papers",
            "category": "Category",
        },
    )

    fig.update_layout(showlegend=False, yaxis_title=None)

    fig.update_traces(
        marker=dict(color=PRIMARY, line=dict(width=0)),
        hovertemplate=(
            "<b>Category:</b> %{y}<br>" "<b>Papers:</b> %{x}<br>" "<extra></extra>"
        ),
    )

    return apply_plot_style(fig)


def build_publication_trend_line_chart(df: pd.DataFrame) -> go.Figure:

    # Prepare the data
    plot_df = df.copy()
    plot_df["published_date"] = pd.to_datetime(
        plot_df["published_date"], errors="coerce"
    )
    plot_df["year"] = plot_df["published_date"].dt.year

    yearly_counts = (
        plot_df.groupby("year")
        .size()
        .reset_index(name="paper_count")
        .sort_values("year")
    )

    fig = px.line(
        yearly_counts,
        x="year",
        y="paper_count",
        markers=True,
        labels={
            "year": "Year",
            "paper_count": "Number of Papers",
        },
    )

    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Papers")

    fig.update_traces(
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(color=PRIMARY, size=6),
        hovertemplate=(
            "<b>Year:</b> %{x}<br>" "<b>Papers:</b> %{y}<br>" "<extra></extra>"
        ),
    )

    return apply_plot_style(fig)
