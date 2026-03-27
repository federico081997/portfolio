from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

from visualization import (
    build_topic_scatter_plot,
    build_cluster_size_bar_chart,
    build_cluster_size_histogram,
    build_category_distribution_bar_chart,
    build_publication_trend_line_chart,
)
from components import (
    build_graph_card,
    build_stats_card,
    build_default_paper_details_card,
    build_tabs_card,
    build_search_bar,
    build_select,
    build_top_k_results_card,
)
from visualization import (
    PAGE_BACKGROUND,
    PRIMARY,
    SURFACE,
    BORDER,
    TEXT,
    HEADER_BG,
    MUTED_TEXT,
    BACKGROUND,
)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = pd.read_csv("data/processed/papers_clustered.csv")
cluster_summary = pd.read_csv("data/processed/cluster_summary.csv")

# -------------------------------------------------------------------
# Build figures
# -------------------------------------------------------------------
topic_fig = build_topic_scatter_plot(df)

cluster_bar_fig = build_cluster_size_bar_chart(cluster_summary)
cluster_hist_fig = build_cluster_size_histogram(cluster_summary)
category_fig = build_category_distribution_bar_chart(df)
publication_fig = build_publication_trend_line_chart(df)

default_paper_card = build_default_paper_details_card(
    "Click a paper in the topic map to see its information here."
)


def build_explore_layout():
    return html.Div(
        [
            dcc.Interval(
                id="counter-interval",
                interval=20,
                n_intervals=0,
                max_intervals=100,
            ),
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
                        html.Div(
                            id="paper-details-container",
                        ),
                        xs=12,
                        lg=4,
                        className="mb-4",
                    ),
                ],
            ),
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
        style={"padding": "16px 16px 24px 16px"},
    )


def build_search_layout():
    category_options = [{"label": "All Categories", "value": "ALL"}]
    category_options += [
        {"label": cat, "value": cat}
        for cat in sorted(df["category"].astype(str).unique())
    ]

    return html.Div(
        [
            # Search controls
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
            # Results + Details
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

    category_options = [{"label": "All Categories", "value": "ALL"}]
    category_options += [
        {"label": cat, "value": cat}
        for cat in sorted(df["category"].astype(str).unique())
    ]
    return html.Div(
        [
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
    Main app layout with tabs card.
    """

    tabs = [
        {"label": "Explore", "value": "explore"},
        {"label": "Search", "value": "search"},
        {"label": "Recommendations", "value": "recommendations"},
    ]

    return (
        dbc.Container(
            [
                # Title
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
                html.P(
                    "Discover, explore, and analyze research papers through semantic search and interactive topic visualization.",
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
                # Tabs inside card
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
        ),
    )
