import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc

from visualization import PRIMARY, TEXT, SURFACE, BORDER, HEADER_BG, MUTED_TEXT


# --------------------------------
# CARDS (UI)
# --------------------------------


def build_stats_card(title: str, value_id: str, value: str | None = None) -> dbc.Card:
    """Build a small summary card with optional value"""

    display_value = value if value is not None else "0"

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    title,
                    className="text-muted",
                    style={"fontSize": "0.85rem", "lineHeight": 1.2},
                ),
                html.H3(
                    display_value,
                    id=value_id,
                    className="mt-2 mb-0",
                    style={
                        "color": TEXT,
                        "fontWeight": 600,
                        "fontSize": "1.4rem",
                    },
                ),
            ]
        ),
        className="card-hover h-100",
        style={
            "borderRadius": "12px",
            "border": f"1px solid {BORDER}",
            "backgroundColor": SURFACE,
        },
    )


def build_graph_card(title: str, graph_component) -> dbc.Card:
    """
    Wrap a graph inside a Bootstrap card.
    Plot titles should not be inside figures
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    title,
                    className="mb-0",
                    style={"color": TEXT, "fontweight": 600, "fontSize": "1rem"},
                ),
                style={
                    "backgroundColor": HEADER_BG,
                    "borderBottom": f"1px solid {BORDER}",
                    "borderTopLeftRadius": "12px",
                    "borderTopRightRadius": "12px",
                    "padding": "0.85rem 1rem",
                },
            ),
            dbc.CardBody(
                graph_component,
                style={
                    "backgroundColor": SURFACE,
                    "padding": "0.75rem",
                },
            ),
        ],
        className="card-hover h-100",
        style={
            "borderRadius": "12px",
            "backgroundColor": SURFACE,
            "maxHeight": "680px",
            "border": f"1px solid {BORDER}",
            "overflow": "hidden",
        },
    )


def build_default_paper_details_card(default_text: str) -> dbc.Card:
    """Default paper details card shown before a paper is clicked"""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(
                    "Paper Details",
                    className="mb-0",
                    style={"color": TEXT, "fontweight": "600", "fontSize": "1rem"},
                ),
                style={
                    "backgroundColor": HEADER_BG,
                    "borderBottom": f"1px solid {BORDER}",
                    "borderTopLeftRadius": "12px",
                    "borderTopRightRadius": "12px",
                    "padding": "0.9rem 1rem",
                },
            ),
            dbc.CardBody(
                [
                    html.P(
                        default_text,
                        className="mb-0",
                        style={
                            "color": MUTED_TEXT,
                            "fontSize": "0.95rem",
                            "lineHeight": "1.5",
                        },
                    ),
                ],
                style={"BackgroundColor": SURFACE, "padding": "1rem"},
            ),
        ],
        className="card-hover h-100",
        style={
            "borderRadius": "12px",
            "backgroundColor": SURFACE,
            "border": f"1px solid {BORDER}",
            "overflow": "hidden",
        },
    )


def build_paper_details_card(df: pd.DataFrame, paper_id: int) -> dbc.Card:
    """Build details card for a clicked paper"""
    # Prepare dataframe
    df = df.copy().reset_index(drop=True)
    df["paper_id"] = df.index
    df["cluster_str"] = df["cluster_id"].astype(str)
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    # Extract information from row with paper_id
    row = df.loc[df["paper_id"] == paper_id].iloc[0]

    title = row.get("title", "Untitled")
    category = row.get("category", "Unknown")
    cluster_label = row.get("cluster_label", "Unknown")
    cluster_id = row.get("cluster_id", "Unknown")
    abstract = row.get("abstract", "No abstract available")
    authors = row.get("authors", "Unknown")
    x_val = row.get("x", "N/A")
    y_val = row.get("y", "N/A")
    published_date = row.get("published_date", "Unknown").strftime("%Y-%m-%d")

    # Define rows to display
    meta_rows = [
        meta_row("Category", str(category)),
        meta_row("Cluster Label", str(cluster_label)),
        meta_row("Cluster ID", str(cluster_id)),
        meta_row("Published Date", str(published_date)),
        meta_row("Authors", str(authors)),
        meta_row("UMAP", f"({x_val}, {y_val})"),
    ]

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(
                    "Paper Details",
                    className="mb-0",
                    style={
                        "color": TEXT,
                        "fontWeight": "600",
                        "fontSize": "1.05rem",
                    },
                ),
                style={
                    "backgroundColor": HEADER_BG,
                    "borderBottom": f"1px solid {BORDER}",
                    "borderTopLeftRadius": "12px",
                    "borderTopRightRadius": "12px",
                    "padding": "0.9rem 1rem",
                },
            ),
            dbc.CardBody(
                [
                    html.H5(
                        title,
                        className="mb-3",
                        style={
                            "color": TEXT,
                            "fontWeight": "600",
                            "fontSize": "1rem",
                            "lineHeight": "1.4",
                        },
                    ),
                    html.Div(meta_rows),
                    html.Hr(style={"borderColor": BORDER, "margin": "0.9rem 0"}),
                    html.H6(
                        "Abstract",
                        className="mb-2",
                        style={
                            "color": TEXT,
                            "fontWeight": "600",
                            "fontSize": "0.95rem",
                        },
                    ),
                    html.P(
                        abstract,
                        className="mb-0",
                        style={
                            "whiteSpace": "pre-wrap",
                            "lineHeight": "1.6",
                            "color": TEXT,
                            "fontSize": "0.92rem",
                        },
                    ),
                ],
                style={
                    "backgroundColor": SURFACE,
                    "padding": "1rem",
                    "overflowY": "auto",
                    "scrollBehavior": "smooth",
                },
            ),
        ],
        className="card-hover h-100",
        style={
            "borderRadius": "12px",
            "backgroundColor": SURFACE,
            "maxHeight": "680px",
            "overflow": "hidden",
            "border": f"1px solid {BORDER}",
        },
    )


def build_tabs_card(
    tabs: list[dict],
    card_id: str = "tabs",
    content_id: str = "tab-content",
    default_tab: str | None = None,
) -> dbc.Card:
    """
    Build a reusable tabs card.

    Parameters
    ----------
    tabs : list of dict
        Example:
        [
            {"label": "Explore", "value": "explore"},
            {"label": "Search", "value": "search"},
        ]

    card_id : str
        ID of the Tabs component

    content_id : str
        ID of the content container (used in callback)

    default_tab : str
        Default active tab (if None, uses first tab)
    """

    if not tabs:
        raise ValueError("tabs list cannot be empty")

    if default_tab is None:
        default_tab = tabs[0]["value"]

    return dbc.Card(
        [
            # Tabs in header
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(
                            label=tab["label"],
                            tab_id=tab["value"],
                        )
                        for tab in tabs
                    ],
                    id=card_id,
                    active_tab=default_tab,
                    className="card-header-tabs",
                ),
                style={
                    "backgroundColor": HEADER_BG,
                    "borderBottom": f"1px solid {BORDER}",
                },
            ),
            # Dynamic content
            dbc.CardBody(
                html.Div(
                    html.Div(id=content_id),
                    style={
                        "backgroundColor": SURFACE,
                        "padding": "1rem",
                        "overflow": "visible",
                    },
                ),
                style={
                    "backgroundColor": "transparent",
                    "padding": "1rem",
                    "overflow": "visible",
                },
            ),
        ],
        style={
            "backgroundColor": SURFACE,
            "border": f"1px solid {BORDER}",
            "overflow": "visible",
        },
    )


def build_search_bar(
    input_placeholder: str,
    button_text: str,
    input_id: str = "search-input",
    input_type: str = "text",
    button_id: str = "search-button",
) -> dbc.InputGroup:
    """Build a search bar with a button"""

    return dbc.InputGroup(
        [
            dbc.Input(
                id=input_id,
                placeholder=input_placeholder,
                type=input_type,
                style={"backgroundColor": SURFACE, "height": "38px"},
            ),
            dbc.Button(
                button_text,
                id=button_id,
                className="custom-btn",
                style={
                    "backgroundColor": PRIMARY,
                    "borderColor": PRIMARY,
                    "color": "white",
                    "height": "38px",
                },
            ),
        ],
    )


def build_select(
    options: list,
    value: int | str | None = None,
    dropdown_id: str = "dropdown",
    placeholder: str | None = None,
) -> dcc.Dropdown:
    """Build a dropdown menu"""

    return dbc.Select(
        id=dropdown_id,
        options=options,
        value=value,
        placeholder=placeholder,
        className="custom-dropdown",
        style={"backgroundColor": SURFACE},
    )


def build_search_result_button(row) -> dbc.Button:
    """
    Clickable search result card.
    """
    title = str(row.get("title", "Untitled"))
    category = str(row.get("category", "Unknown"))
    authors = str(row.get("authors", "Unknown"))
    year = str(row.get("year", "Unknown"))
    explanation = str(row.get("explanation", ""))
    score = row.get("final_score", row.get("semantic_similarity", None))
    paper_id = int(row["paper_id"])
    rank = int(row.get("rank", 0))
    score_text = f"{score:.3f}" if score is not None else "N/A"

    return dbc.Button(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(
                                f"#{rank}",
                                style={
                                    "fontWeight": "700",
                                    "fontSize": "0.85rem",
                                    "color": MUTED_TEXT,
                                    "marginRight": "0.5rem",
                                },
                            ),
                            html.Span(
                                f"Score: {score_text}",
                                style={
                                    "fontSize": "0.85rem",
                                    "color": MUTED_TEXT,
                                },
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.H6(
                        title,
                        className="mb-2",
                        style={
                            "fontWeight": "600",
                            "fontSize": "1rem",
                            "lineHeight": "1.35",
                            "textAlign": "left",
                        },
                    ),
                    html.Div(
                        f"{category} • {year}",
                        className="mb-1",
                        style={
                            "fontSize": "0.85rem",
                            "color": MUTED_TEXT,
                            "textAlign": "left",
                        },
                    ),
                    html.Div(
                        authors,
                        className="mb-2",
                        style={
                            "fontSize": "0.85rem",
                            "color": MUTED_TEXT,
                            "textAlign": "left",
                        },
                    ),
                    html.Div(
                        explanation,
                        style={
                            "fontSize": "0.85rem",
                            "color": TEXT,
                            "textAlign": "left",
                            "lineHeight": "1.45",
                        },
                    ),
                ]
            ),
            className="card-hover",
            style={
                "borderRadius": "12px",
                "backgroundColor": SURFACE,
                "borderBottom": f"1px solid {BORDER}",
            },
        ),
        id={"type": "search-result-card", "index": paper_id},
        color="link",
        className="w-100 p-0 text-decoration-none mb-4 result-btn",
        style={"border": "none"},
    )


def build_top_k_results_card() -> dbc.Card:
    """Build card of top-k search results"""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(
                    "Search Results",
                    className="mb-0",
                    style={
                        "color": TEXT,
                        "fontWeight": "600",
                        "fontSize": "1.05rem",
                    },
                ),
                style={
                    "backgroundColor": HEADER_BG,
                    "borderBottom": f"1px solid {BORDER}",
                    "borderTopLeftRadius": "12px",
                    "borderTopRightRadius": "12px",
                    "padding": "0.9rem 1rem",
                },
            ),
            dbc.CardBody(
                [
                    html.Div(
                        id="search-results-header",
                        children="",
                        className="mb-3",
                        style={
                            "fontSize": "0.95rem",
                            "color": MUTED_TEXT,
                        },
                    ),
                    html.Div(
                        id="search-results",
                        style={
                            "flex": 1,
                            "overflowY": "auto",
                            "padding": "16px 16px 8px 16px",
                        },
                    ),
                ],
                style={
                    "backgroundColor": SURFACE,
                    "display": "flex",
                    "flexDirection": "column",
                    "minHeight": 0,
                    "overflow": "hidden",
                },
            ),
        ],
        className="card-hover",
        style={
            "borderRadius": "12px",
            "backgroundColor": SURFACE,
            "border": f"1px solid {BORDER}",
            "display": "flex",
            "flexDirection": "column",
            "height": "680px",
            "overflow": "hidden",
        },
    )


def build_no_results_card(query: str, text: str) -> dbc.Card:
    """Build card for no search results"""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(
                    "Search Results",
                    className="mb-0",
                    style={
                        "color": TEXT,
                        "fontWeight": "600",
                        "fontSize": "1.05rem",
                    },
                ),
                style={
                    "backgroundColor": HEADER_BG,
                    "borderBottom": f"1px solid {BORDER}",
                    "borderTopLeftRadius": "12px",
                    "borderTopRightRadius": "12px",
                    "padding": "0.9rem 1rem",
                },
            ),
            dbc.CardBody(
                [
                    html.Div(
                        f'No results found for "{query}"',
                        style={
                            "color": TEXT,
                            "fontWeight": "600",
                            "marginBottom": "0.5rem",
                        },
                    ),
                    html.Div(
                        text,
                        style={
                            "color": MUTED_TEXT,
                            "fontSize": "0.95rem",
                        },
                    ),
                ],
                style={
                    "backgroundColor": SURFACE,
                },
            ),
        ],
        className="card-hover",
        style={
            "borderRadius": "12px",
            "backgroundColor": SURFACE,
            "border": f"1px solid {BORDER}",
            "overflow": "hidden",
        },
    )


# --------------------------------
# HELPERS
# --------------------------------


def meta_row(label: str, value) -> html.Div:
    """
    Compact metadata row that wraps well on phone.
    """
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "fontWeight": "600",
                    "color": TEXT,
                    "fontSize": "0.85rem",
                    "marginBottom": "0.15rem",
                },
            ),
            html.Div(
                value,
                style={
                    "color": MUTED_TEXT if isinstance(value, str) else TEXT,
                    "fontSize": "0.92rem",
                    "lineHeight": "1.45",
                    "wordBreak": "break-word",
                },
            ),
        ],
        style={"marginBottom": "0.75rem"},
    )
