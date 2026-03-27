import os
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="keras")
warnings.filterwarnings("ignore", module="tf_keras")

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from dash import Dash, html, Input, Output, State, ctx, no_update, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import faiss

from layout import (
    build_explore_layout,
    build_search_layout,
    build_recommendations_layout,
    build_main_layout,
)
from components import (
    build_default_paper_details_card,
    build_paper_details_card,
    build_search_result_button,
    build_no_results_card,
    build_top_k_results_card,
)
from recommender import load_artifacts, get_similar_by_query, get_similar_by_paper
from visualization import BORDER, MUTED_TEXT, SURFACE, TEXT, HEADER_BG

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
_, embeddings, faiss_index = load_artifacts()
df = pd.read_csv("data/processed/papers_clustered.csv")
cluster_summary = pd.read_csv("data/processed/cluster_summary.csv")

# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.layout = build_main_layout()

# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------


@app.callback(
    Output("main-tab-content", "children"),
    Input("main-tabs", "active_tab"),
)
def render_tab_content(active_tab):

    if active_tab == "explore":
        return build_explore_layout()

    elif active_tab == "search":
        return build_search_layout()

    elif active_tab == "recommendations":
        return build_recommendations_layout()

    return html.Div("No content available")


@app.callback(
    Output("total-papers-value", "children"),
    Output("total-clusters-value", "children"),
    Input("counter-interval", "n_intervals"),
)
def animate_counters(n):

    total_papers = len(df)
    total_clusters = df["cluster_id"].nunique()

    steps = 100

    progress = min(n / steps, 1)
    progress = 3 * progress**2 - 2 * progress**3
    progress = 1 - (1 - progress) ** 4

    papers_value = int(progress * total_papers)
    clusters_value = int(progress * total_clusters)

    return f"{papers_value:,}", f"{clusters_value:,}"


@app.callback(
    Output("paper-details-container", "children"),
    Input("topic-map-graph", "clickData"),
)
def update_paper_details(clickData):
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

    # ---- 1. Handle empty query ----
    if not query or not query.strip():
        return no_update, no_update

    query = query.strip()
    top_k = int(top_k)

    # ---- 2. Run search ----
    results = get_similar_by_query(
        query=query,
        df=df,
        faiss_index=faiss_index,
        top_k=top_k,
    ).copy()

    # ---- 3. Apply category filter ----
    if category_filter and category_filter != "ALL":
        results = results[results["category"] == category_filter].copy()

    # ---- 4. Handle no results ----
    if results.empty:
        return (
            build_no_results_card(
                query, "Try a broader query or remove the category filter."
            ),
            no_update,
        )

    # ---- 5. Prepare results ----
    results = results.head(top_k).copy()
    results["rank"] = range(1, len(results) + 1)
    results["paper_id"] = results["paper_index"].astype(str)

    # ---- 6. Build result buttons ----
    result_cards = [build_search_result_button(row) for _, row in results.iterrows()]

    # ---- 7. Header ----
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

    # ---- 8. Use existing card ----
    results_card = build_top_k_results_card()

    # Inject content
    results_card.children[1].children[0].children = header
    results_card.children[1].children[1].children = result_cards

    # ---- 9. Default details ----
    first_paper_id = int(results.iloc[0]["paper_id"])
    details_card = build_paper_details_card(df, first_paper_id)

    return results_card, details_card


@app.callback(
    Output("search-paper-details", "children", allow_duplicate=True),
    Input({"type": "search-result-card", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_paper_details(n_clicks_list):

    # Nothing clicked
    if not ctx.triggered_id:
        return no_update

    # Get clicked paper_id
    paper_id = int(ctx.triggered_id["index"])

    # Build details card
    return build_paper_details_card(df, paper_id)


@app.callback(
    Output("recommendation-results-container", "children"),
    Output("recommended-paper-details", "children"),
    Input("paper-select", "value"),
    Input("recommend-top-k-filter", "value"),
    Input("recommend-category-filter", "value"),
)
def run_recommendations(selected_paper_id, top_k, category_filter):
    # ---- 1. Handle no selected paper ----
    if selected_paper_id is None:
        return no_update, no_update

    # Convert dropdown value to integer row position
    paper_idx = int(selected_paper_id)
    top_k = int(top_k)

    # ---- 2. Get recommendations ----
    results = get_similar_by_paper(
        paper_idx=paper_idx,
        df=df,
        embeddings=embeddings,
        faiss_index=faiss_index,
        top_k=top_k + 1,
    ).copy()

    # ---- 3. Remove selected paper itself ----
    results = results[results["paper_index"] != paper_idx].copy()

    # ---- 4. Apply category filter ----
    if category_filter and category_filter != "ALL":
        results = results[results["category"] == category_filter].copy()

    # ---- 5. Selected paper title ----
    selected_title = df.iloc[paper_idx]["title"]

    # ---- 6. Handle no results ----
    if results.empty:
        return (
            build_no_results_card(
                selected_title, "Try another paper or remove the category filter."
            ),
            no_update,
        )

    # ---- 7. Keep only top_k ----
    results = results.head(top_k).copy()
    results["rank"] = range(1, len(results) + 1)
    results["paper_id"] = results["paper_index"].astype(int)

    # ---- 8. Build result cards/buttons ----
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

    # ---- 9. Build results card ----
    results_card = build_top_k_results_card()
    results_card.children[1].children[0].children = header
    results_card.children[1].children[1].children = result_cards

    # ---- 10. Default details card = first recommended paper ----
    first_paper_id = int(results.iloc[0]["paper_index"])
    details_card = build_paper_details_card(df, first_paper_id)

    return results_card, details_card


@app.callback(
    Output("recommended-paper-details", "children", allow_duplicate=True),
    Input({"type": "search-result-card", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_recommended_paper_details(n_clicks_list):

    if not ctx.triggered_id:
        return no_update

    paper_id = int(ctx.triggered_id["index"])

    return build_paper_details_card(df, paper_id)


if __name__ == "__main__":
    app.run(debug=True)
