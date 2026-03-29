from pathlib import Path
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

from visualization import (
    build_topic_scatter_plot,
    build_cluster_size_bar_chart,
    build_cluster_size_histogram,
    build_category_distribution_bar_chart,
    build_publication_trend_line_chart,
    PAGE_BACKGROUND,
    PRIMARY,
    TEXT,
)
from components import (
    build_graph_card,
    build_stats_card,
    build_default_paper_details_card,
    build_tabs_card,
    build_search_bar,
    build_select,
)

# Load preprocessed datasets used across the dashboard.
project_root = Path(__file__).parent
cluster_data_path = project_root / "data" / "processed" / "papers_clustered.csv"
cluster_summary_path = project_root / "data" / "processed" / "cluster_summary.csv"
df = pd.read_csv(cluster_data_path)
cluster_summary = pd.read_csv(cluster_summary_path)


# Build static figures
# These figures are computed once at startup and reused.
topic_fig = build_topic_scatter_plot(df)

cluster_bar_fig = build_cluster_size_bar_chart(cluster_summary)
cluster_hist_fig = build_cluster_size_histogram(cluster_summary)
category_fig = build_category_distribution_bar_chart(df)
publication_fig = build_publication_trend_line_chart(df)


# Default placeholder card shown before any paper is selected.
default_paper_card = build_default_paper_details_card(
    "Click a paper in the topic map to see its information here."
)


def build_explore_layout():
    """
    Build the main "Explore" tab layout.

    The layout includes:
    - animated statistic cards (top row)
    - interactive topic map + paper details panel
    - cluster analysis charts
    - category and publication trend charts

    Returns:
        html.Div: Fully composed Dash layout for the Explore view.
    """
    return html.Div(
        [
            # Interval component used to animate or update stats dynamically.
            dcc.Interval(
                id="counter-interval",
                interval=20,
                n_intervals=0,
                max_intervals=100,
            ),
            # Top statistics cards
            dbc.Row(
                [
                    dbc.Col(
                        build_stats_card(
                            "Total Papers",
                            "total-papers-value",
                        ),
                        xs=12,
                        md=4,
                        className="mb-4",
                    ),
                    dbc.Col(
                        build_stats_card(
                            "Total Clusters",
                            "total-clusters-value",
                        ),
                        xs=12,
                        md=4,
                        className="mb-4",
                    ),
                    dbc.Col(
                        build_stats_card(
                            "Top Category",
                            "top-category-value",
                            value=str(df["category"].mode().iloc[0]),
                        ),
                        xs=12,
                        md=4,
                        className="mb-4",
                    ),
                ],
            ),
            # Topic map + paper details
            dbc.Row(
                [
                    dbc.Col(
                        build_graph_card(
                            "Topic Map",
                            dcc.Graph(
                                figure=topic_fig,
                                id="topic-map-graph",
                                config={"displayModeBar": False},
                                style={"height": "75vh"},
                            ),
                        ),
                        xs=12,
                        lg=8,
                        className="mb-4",
                    ),
                    dbc.Col(
                        # Container dynamically updated when a paper is clicked.
                        html.Div(id="paper-details-container"),
                        xs=12,
                        lg=4,
                        className="mb-4",
                    ),
                ],
            ),
            # Cluster analysis
            dbc.Row(
                [
                    dbc.Col(
                        build_graph_card(
                            "Top Clusters",
                            dcc.Graph(
                                figure=cluster_bar_fig,
                                config={"displayModeBar": False},
                                style={"height": "75vh"},
                            ),
                        ),
                        xs=12,
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        build_graph_card(
                            "Cluster Size Distribution",
                            dcc.Graph(
                                figure=cluster_hist_fig,
                                config={"displayModeBar": False},
                                style={"height": "75vh"},
                            ),
                        ),
                        xs=12,
                        lg=6,
                        className="mb-4",
                    ),
                ],
            ),
            # Category + publication trends
            dbc.Row(
                [
                    dbc.Col(
                        build_graph_card(
                            "Top Categories",
                            dcc.Graph(
                                figure=category_fig,
                                config={"displayModeBar": False},
                                style={"height": "75vh"},
                            ),
                        ),
                        xs=12,
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        build_graph_card(
                            "Publication Trend",
                            dcc.Graph(
                                figure=publication_fig,
                                config={"displayModeBar": False},
                                style={"height": "75vh"},
                            ),
                        ),
                        xs=12,
                        lg=6,
                        className="mb-4",
                    ),
                ],
            ),
        ],
        style={
            "padding": "16px 16px 24px 16px",
        },
    )


def build_search_layout():
    """
    Build the layout for the semantic search tab.

    The layout includes:
    - a search bar
    - a top-k results selector
    - a category filter
    - a results panel
    - a paper details panel

    Returns:
        html.Div: Dash layout for the search view.
    """
    # Build category dropdown options dynamically from the dataset.
    category_options = [{"label": "All Categories", "value": "ALL"}]
    category_options += [
        {"label": cat, "value": cat}
        for cat in sorted(df["category"].astype(str).unique())
    ]

    return html.Div(
        [
            # Search controls.
            dbc.Row(
                [
                    dbc.Col(
                        build_search_bar(
                            input_placeholder="Search topics, papers, or keywords...",
                            button_text="Search",
                            input_id="search-input",
                            input_type="text",
                            button_id="search-button",
                        ),
                        xs=12,
                        lg=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        build_select(
                            options=[
                                {"label": "Top 10", "value": 10},
                                {"label": "Top 20", "value": 20},
                                {"label": "Top 50", "value": 50},
                            ],
                            value=10,
                            dropdown_id="top-k-filter",
                        ),
                        xs=12,
                        lg=2,
                        className="mb-3",
                    ),
                    dbc.Col(
                        build_select(
                            options=category_options,
                            value="ALL",
                            dropdown_id="category-filter",
                        ),
                        xs=12,
                        lg=3,
                        className="mb-4",
                    ),
                ],
                className="g-2",
            ),
            # Search results and selected paper details.
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="search-results-container"),
                        xs=12,
                        lg=8,
                        className="mb-4",
                    ),
                    dbc.Col(
                        html.Div(id="search-paper-details"),
                        xs=12,
                        lg=4,
                        className="mb-4",
                    ),
                ]
            ),
        ],
        style={
            "padding": "16px 16px 24px 16px",
            "overflow": "visible",
        },
    )


def build_recommendations_layout():
    """
    Build the layout for the paper-to-paper recommendation tab.

    The layout includes:
    - a paper selector
    - a top-k recommendations selector
    - a category filter
    - a recommendations results panel
    - a recommended paper details panel

    Returns:
        html.Div: Dash layout for the recommendations view.
    """
    # Build category dropdown options dynamically from the dataset.
    category_options = [{"label": "All Categories", "value": "ALL"}]
    category_options += [
        {"label": cat, "value": cat}
        for cat in sorted(df["category"].astype(str).unique())
    ]

    return html.Div(
        [
            # Recommendation controls.
            dbc.Row(
                [
                    dbc.Col(
                        build_select(
                            options=[
                                {"label": row["title"], "value": i}
                                for i, row in df.sample(n=100).iterrows()
                            ],
                            dropdown_id="paper-select",
                            placeholder="Select a paper...",
                        ),
                        xs=12,
                        lg=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        build_select(
                            options=[
                                {"label": "Top 10", "value": 10},
                                {"label": "Top 20", "value": 20},
                                {"label": "Top 50", "value": 50},
                            ],
                            value=10,
                            dropdown_id="recommend-top-k-filter",
                        ),
                        xs=12,
                        lg=2,
                        className="mb-3",
                    ),
                    dbc.Col(
                        build_select(
                            options=category_options,
                            value="ALL",
                            dropdown_id="recommend-category-filter",
                        ),
                        xs=12,
                        lg=3,
                        className="mb-4",
                    ),
                ],
                className="g-2",
            ),
            # Recommendation results and selected paper details.
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="recommendation-results-container"),
                        xs=12,
                        lg=8,
                        className="mb-4",
                    ),
                    dbc.Col(
                        html.Div(id="recommended-paper-details"),
                        xs=12,
                        lg=4,
                        className="mb-4",
                    ),
                ]
            ),
        ]
    )


def build_main_layout():
    """
    Build the main application layout.

    This is the top-level page structure and includes:
    - the dashboard title
    - a short project description
    - a visual separator
    - the main tabbed interface

    Returns:
        dbc.Container: Full app layout container.
    """
    # Define the main application tabs.
    tabs = [
        {"label": "Explore", "value": "explore"},
        {"label": "Search", "value": "search"},
        {"label": "Recommendations", "value": "recommendations"},
    ]

    return dbc.Container(
        [
            # Main dashboard title.
            html.H2(
                "Technical Paper Recommendation System",
                className="mt-4 mb-2",
                style={
                    "color": TEXT,
                    "fontWeight": "600",
                    "marginBottom": "12px",
                    "textAlign": "center",
                },
            ),
            # Short subtitle describing the dashboard purpose.
            html.P(
                """
                Discover, explore, and analyze research papers using semantic similarity,
                interactive topic visualization, and a hybrid recommendation system
                combining embeddings, keywords, and recency.""",
                style={
                    "color": TEXT,
                    "fontSize": "0.95rem",
                    "marginBottom": "20px",
                    "maxWidth": "600px",
                    "marginLeft": "auto",
                    "marginRight": "auto",
                    "textAlign": "center",
                },
            ),
            # Decorative divider below the heading.
            html.Hr(
                style={
                    "borderTop": f"1px solid {PRIMARY}",
                    "marginTop": "10px",
                    "marginBottom": "30px",
                    "width": "60%",
                    "marginLeft": "auto",
                    "marginRight": "auto",
                    "textAlign": "center",
                },
            ),
            # Main tabbed content card.
            build_tabs_card(
                tabs=tabs,
                card_id="main-tabs",
                content_id="main-tab-content",
                default_tab="explore",
            ),
        ],
        fluid=True,
        style={
            "backgroundColor": PAGE_BACKGROUND,
            "minHeight": "100vh",
            "padding": "16px",
        },
    )
