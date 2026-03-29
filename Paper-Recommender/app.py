import logging
import os
import warnings
from pathlib import Path

# Suppress verbose TensorFlow backend messages before importing TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="keras")
warnings.filterwarnings("ignore", module="tf_keras")

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from dash import ALL, Dash, Input, Output, State, ctx, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import tensorflow as tf

from components import (
    build_default_paper_details_card,
    build_no_results_card,
    build_paper_details_card,
    build_search_result_button,
    build_top_k_results_card,
)
from layout import (
    build_explore_layout,
    build_main_layout,
    build_recommendations_layout,
    build_search_layout,
)
from recommender import get_similar_by_paper, get_similar_by_query, load_artifacts
from visualization import MUTED_TEXT, TEXT

tf.get_logger().setLevel("ERROR")


# -------------------------------------------------------------------
# LOAD DATA AND ARTIFACTS
# -------------------------------------------------------------------

# Load recommendation artifacts used by the search and recommendation callbacks.
_, embeddings, faiss_index = load_artifacts()

# Load dashboard data files.
project_root = Path(__file__).parent
df = pd.read_csv(project_root / "data" / "processed" / "papers_clustered.csv")
cluster_summary = pd.read_csv(
    project_root / "data" / "processed" / "cluster_summary.csv"
)


# -------------------------------------------------------------------
# APP INITIALIZATION
# -------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.layout = build_main_layout()


# -------------------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------------------


@app.callback(
    Output("main-tab-content", "children"),
    Input("main-tabs", "active_tab"),
)
def render_tab_content(active_tab):
    """
    Render the content associated with the selected main tab.

    Args:
        active_tab: Currently active tab identifier.

    Returns:
        html.Div: A Dash component containing the selected tab layout.
    """
    if active_tab == "explore":
        return build_explore_layout()

    if active_tab == "search":
        return build_search_layout()

    if active_tab == "recommendations":
        return build_recommendations_layout()

    return html.Div("No content available")


@app.callback(
    Output("total-papers-value", "children"),
    Output("total-clusters-value", "children"),
    Input("counter-interval", "n_intervals"),
)
def animate_counters(n):
    """
    Animate the summary counters shown in the Explore tab.

    The animation uses an easing curve to make the counter growth feel smoother.

    Args:
        n: Current interval tick count.

    Returns:
        tuple: Animated total paper count and total cluster count.
    """
    total_papers = len(df)
    total_clusters = df["cluster_id"].nunique()

    steps = 100

    # Normalize interval progress to the range [0, 1].
    progress = min(n / steps, 1)

    # Apply easing for a smoother animation.
    progress = 3 * progress**2 - 2 * progress**3
    progress = 1 - (1 - progress) ** 4

    papers_value = int(progress * total_papers)
    clusters_value = int(progress * total_clusters)

    return f"{papers_value:,}", f"{clusters_value:,}"


@app.callback(
    Output("paper-details-container", "children"),
    Input("topic-map-graph", "clickData"),
)
def update_topic_map_paper_details(clickData):
    """
    Update the paper details panel when a paper is clicked on the topic map.

    Args:
        clickData: Plotly click payload from the topic map graph.

    Returns:
        dbc.Card: A paper details card, or the default placeholder card if no valid paper
        is selected.
    """
    if not clickData or "points" not in clickData or not clickData["points"]:
        return build_default_paper_details_card(
            "Click a paper in the topic map to see its information here."
        )

    try:
        paper_id = clickData["points"][0]["customdata"][0]
        return build_paper_details_card(df, paper_id)
    except (KeyError, IndexError, TypeError, ValueError):
        return build_default_paper_details_card(
            "Click a paper in the topic map to see its information here."
        )


@app.callback(
    Output("search-results-container", "children"),
    Output("search-paper-details", "children"),
    Input("search-button", "n_clicks"),
    State("search-input", "value"),
    State("top-k-filter", "value"),
    State("category-filter", "value"),
    prevent_initial_call=True,
)
def run_search(n_clicks, query, top_k, category_filter):
    """
    Run semantic search from a free-text query and update the results panel.

    Args:
        n_clicks: Number of times the search button has been clicked.
        query: User search query.
        top_k: Number of results requested.
        category_filter: Optional category filter value.

    Returns:
        tuple: Search results card and paper details card.
    """
    # Ignore empty or whitespace-only queries.
    if not query or not query.strip():
        return no_update, no_update

    query = query.strip()
    top_k = int(top_k)

    # Run semantic search.
    results = get_similar_by_query(
        query=query,
        df=df,
        faiss_index=faiss_index,
        top_k=top_k,
    ).copy()

    # Apply category filtering after retrieval.
    if category_filter and category_filter != "ALL":
        results = results[results["category"] == category_filter].copy()

    # Show a no-results card if nothing remains after filtering.
    if results.empty:
        return (
            build_no_results_card(
                query,
                "Try a broader query or remove the category filter.",
            ),
            no_update,
        )

    # Prepare result metadata for display.
    results = results.head(top_k).copy()
    results["rank"] = range(1, len(results) + 1)
    results["paper_id"] = results["paper_index"].astype(str)

    # Build clickable result cards.
    result_cards = [build_search_result_button(row) for _, row in results.iterrows()]

    # Create the small header shown above the result list.
    header = [
        html.Span(
            f"Top {len(results)} result/s",
            style={
                "color": TEXT,
                "fontWeight": "600",
            },
        ),
        html.Span(
            f' for "{query}"',
            style={
                "color": MUTED_TEXT,
                "marginLeft": "0.25rem",
            },
        ),
    ]

    # Reuse the existing top-k results container and inject dynamic content.
    results_card = build_top_k_results_card()
    results_card.children[1].children[0].children = header
    results_card.children[1].children[1].children = result_cards

    # Show the first result in the details panel by default.
    first_paper_id = int(results.iloc[0]["paper_id"])
    details_card = build_paper_details_card(df, first_paper_id)

    return results_card, details_card


@app.callback(
    Output("search-paper-details", "children", allow_duplicate=True),
    Input({"type": "search-result-card", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_search_paper_details(n_clicks_list):
    """
    Update the search details panel when a search result card is clicked.

    Args:
        n_clicks_list: List of click counts for all dynamic result cards.

    Returns:
        dbc.Card: Updated paper details card, or no_update if nothing triggered.
    """
    if not ctx.triggered_id:
        return no_update

    paper_id = int(ctx.triggered_id["index"])
    return build_paper_details_card(df, paper_id)


@app.callback(
    Output("recommendation-results-container", "children"),
    Output("recommended-paper-details", "children"),
    Input("paper-select", "value"),
    Input("recommend-top-k-filter", "value"),
    Input("recommend-category-filter", "value"),
)
def run_recommendations(selected_paper_id, top_k, category_filter):
    """
    Run paper-to-paper recommendations and update the recommendation view.

    Args:
        selected_paper_id: Selected source paper ID from the dropdown.
        top_k: Number of recommendations requested.
        category_filter: Optional category filter value.

    Returns:
        tuple: Recommendation results card and recommended paper details card.
    """
    # Do nothing until a source paper is selected.
    if selected_paper_id is None:
        return no_update, no_update

    # Convert UI values to expected types.
    paper_idx = int(selected_paper_id)
    top_k = int(top_k)

    # Fetch candidate recommendations.
    results = get_similar_by_paper(
        paper_idx=paper_idx,
        df=df,
        embeddings=embeddings,
        faiss_index=faiss_index,
        top_k=top_k + 1,
    ).copy()

    # Remove the source paper itself if it appears in the results.
    results = results[results["paper_index"] != paper_idx].copy()

    # Apply category filtering after retrieval.
    if category_filter and category_filter != "ALL":
        results = results[results["category"] == category_filter].copy()

    selected_title = df.iloc[paper_idx]["title"]

    # Show a no-results card if nothing remains after filtering.
    if results.empty:
        return (
            build_no_results_card(
                selected_title,
                "Try another paper or remove the category filter.",
            ),
            no_update,
        )

    # Prepare recommendation metadata for display.
    results = results.head(top_k).copy()
    results["rank"] = range(1, len(results) + 1)
    results["paper_id"] = results["paper_index"].astype(int)

    # Build clickable recommendation cards.
    result_cards = [build_search_result_button(row) for _, row in results.iterrows()]

    header = [
        html.Span(
            f"Top {len(results)} recommendation/s",
            style={
                "color": TEXT,
                "fontWeight": "600",
            },
        ),
        html.Span(
            f' based on "{selected_title}"',
            style={
                "color": MUTED_TEXT,
                "marginLeft": "0.25rem",
            },
        ),
    ]

    # Reuse the existing results card layout and inject recommendation content.
    results_card = build_top_k_results_card()
    results_card.children[1].children[0].children = header
    results_card.children[1].children[1].children = result_cards

    # Show the first recommended paper in the details panel by default.
    first_paper_id = int(results.iloc[0]["paper_index"])
    details_card = build_paper_details_card(df, first_paper_id)

    return results_card, details_card


@app.callback(
    Output("recommended-paper-details", "children", allow_duplicate=True),
    Input({"type": "search-result-card", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_recommended_paper_details(n_clicks_list):
    """
    Update the recommendation details panel when a recommendation card is clicked.

    Args:
        n_clicks_list: List of click counts for all dynamic result cards.

    Returns:
        dbc.Card: Updated paper details card, or no_update if nothing triggered.
    """
    if not ctx.triggered_id:
        return no_update

    paper_id = int(ctx.triggered_id["index"])
    return build_paper_details_card(df, paper_id)


if __name__ == "__main__":
    app.run(debug=True)
