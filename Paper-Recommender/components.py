import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

from visualization import (
    BORDER,
    HEADER_BG,
    MUTED_TEXT,
    PRIMARY,
    SURFACE,
    TEXT,
)


# --------------------------------
# CARDS (UI)
# --------------------------------


def build_stats_card(title: str, value_id: str, value: str | None = None) -> dbc.Card:
    """
    Build a small summary card with an optional displayed value.

    Args:
        title: Label shown at the top of the card.
        value_id: Component ID for the displayed value.
        value: Initial value shown in the card. Defaults to "0" if None.

    Returns:
        dbc.Card: Styled summary card component.
    """
    display_value = value if value is not None else "0"

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    title,
                    className="text-muted",
                    style={
                        "fontSize": "0.85rem",
                        "lineHeight": 1.2,
                    },
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
    Wrap a graph component inside a styled Bootstrap card.

    Plot titles are kept outside the figure itself and displayed in the card
    header instead.

    Args:
        title: Card title shown in the header.
        graph_component: Dash graph component or any renderable UI component.

    Returns:
        dbc.Card: Styled graph card component.
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    title,
                    className="mb-0",
                    style={
                        "color": TEXT,
                        "fontweight": 600,
                        "fontSize": "1rem",
                    },
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
    """
    Build the default paper details card shown before a paper is selected.

    Args:
        default_text: Placeholder message displayed in the card body.

    Returns:
        dbc.Card: Styled placeholder details card.
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(
                    "Paper Details",
                    className="mb-0",
                    style={
                        "color": TEXT,
                        "fontweight": "600",
                        "fontSize": "1rem",
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
                style={
                    "BackgroundColor": SURFACE,
                    "padding": "1rem",
                },
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
    """
    Build a detailed information card for a selected paper.

    The card includes metadata such as category, cluster information,
    publication date, authors, UMAP coordinates, and the abstract.

    Args:
        df: DataFrame containing paper metadata.
        paper_id: Integer identifier of the selected paper.

    Returns:
        dbc.Card: Styled paper details card.
    """
    # Prepare a copy so the original dataframe is not modified.
    df = df.copy().reset_index(drop=True)
    df["paper_id"] = df.index
    df["cluster_str"] = df["cluster_id"].astype(str)
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    # Extract the selected paper row.
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

    # Build the metadata rows shown above the abstract.
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
                    html.Hr(
                        style={
                            "borderColor": BORDER,
                            "margin": "0.9rem 0",
                        }
                    ),
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
    Build a reusable tabs card with dynamic content.

    Args:
        tabs: List of tab definitions, for example:
            [
                {"label": "Explore", "value": "explore"},
                {"label": "Search", "value": "search"},
            ]
        card_id: ID assigned to the Tabs component.
        content_id: ID assigned to the dynamic content container.
        default_tab: Default active tab. If None, the first tab is used.

    Returns:
        dbc.Card: Styled tabs card component.

    Raises:
        ValueError: If the tabs list is empty.
    """
    if not tabs:
        raise ValueError("tabs list cannot be empty")

    if default_tab is None:
        default_tab = tabs[0]["value"]

    return dbc.Card(
        [
            # Tabs are placed in the card header.
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
            # The body contains a placeholder for callback-driven content.
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
    """
    Build a search bar with a text input and a button.

    Args:
        input_placeholder: Placeholder text shown inside the input field.
        button_text: Text displayed on the search button.
        input_id: ID assigned to the input component.
        input_type: HTML input type for the input field.
        button_id: ID assigned to the button component.

    Returns:
        dbc.InputGroup: Styled input group containing the search input and button.
    """
    return dbc.InputGroup(
        [
            dbc.Input(
                id=input_id,
                placeholder=input_placeholder,
                type=input_type,
                style={
                    "backgroundColor": SURFACE,
                    "height": "38px",
                },
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
    """
    Build a styled select dropdown component.

    Args:
        options: List of dropdown options.
        value: Initially selected value.
        dropdown_id: ID assigned to the dropdown component.
        placeholder: Placeholder text shown when no value is selected.

    Returns:
        dcc.Dropdown: Styled dropdown/select component.
    """
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
    Build a clickable search result card wrapped in a button.

    The displayed information includes:
    - result rank
    - score
    - title
    - category and year
    - authors
    - explanation for the recommendation

    Args:
        row: Row-like object containing result metadata.

    Returns:
        dbc.Button: Clickable result card component.
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
                    # Top row with rank and score.
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
                    # Paper title.
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
                    # Category and year.
                    html.Div(
                        f"{category} • {year}",
                        className="mb-1",
                        style={
                            "fontSize": "0.85rem",
                            "color": MUTED_TEXT,
                            "textAlign": "left",
                        },
                    ),
                    # Authors.
                    html.Div(
                        authors,
                        className="mb-2",
                        style={
                            "fontSize": "0.85rem",
                            "color": MUTED_TEXT,
                            "textAlign": "left",
                        },
                    ),
                    # Recommendation explanation.
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
    """
    Build the container card used to display top-k search results.

    The card includes:
    - a header title
    - a small results summary area
    - a scrollable container for result cards

    Returns:
        dbc.Card: Styled search results card.
    """
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
                    # Small summary text above the search results list.
                    html.Div(
                        id="search-results-header",
                        children="",
                        className="mb-3",
                        style={
                            "fontSize": "0.95rem",
                            "color": MUTED_TEXT,
                        },
                    ),
                    # Scrollable search results container.
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
    """
    Build a card shown when a search returns no results.

    Args:
        query: User search query.
        text: Additional explanatory message displayed below the main notice.

    Returns:
        dbc.Card: Styled no-results card.
    """
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
                    # Main no-results message.
                    html.Div(
                        f'No results found for "{query}"',
                        style={
                            "color": TEXT,
                            "fontWeight": "600",
                            "marginBottom": "0.5rem",
                        },
                    ),
                    # Additional helper text.
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
    Build a compact metadata row for paper details.

    This helper is designed to wrap cleanly on smaller screens.

    Args:
        label: Metadata field label.
        value: Metadata value to display.

    Returns:
        html.Div: Styled metadata row.
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
        style={
            "marginBottom": "0.75rem",
        },
    )
