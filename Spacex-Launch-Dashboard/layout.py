"""
layout.py

Layout builder for the SpaceX Launch Dashboard.

This module defines the top-level Dash layout used by the application. The main
entry point is :func:`build_layout`, which returns the root `html.Div` containing:

- A page title and short description.
- A decorative separator line.
- A main card component with tab navigation and a dynamic content area.

The dynamic content area (``main-card-content``) is populated by callbacks based
on the active tab selection.
"""

from dash import html

from components import display_card, display_tabs, display_text, display_title

# =============================================================================
# Dashboard description
# =============================================================================

# General description displayed at the top of the dashboard.
DESCRIPTION = (
    "The dashboard provides an interactive overview of SpaceX Falcon 9 launches, "
    "combining a clean launch dataset, exploratory visualisations, and a simple predictive modelling summary. "
    "Use the tabs to review the curated dataset, analyse launch outcomes by site and payload mass, "
    "and compare baseline machine learning models trained on mission and booster attributes."
)


# =============================================================================
# Public API
# =============================================================================


def build_layout() -> html.Div:
    """
    Construct the top-level Dash layout for the application.

    :return: Root layout container for the dashboard.
    :rtype: dash.html.Div
    """
    # Main page container. Child components are arranged vertically in display order.
    return html.Div(
        children=[
            # Page header: title + short description for context.
            display_title("SpaceX Falcon 9 Launch Dashboard"),
            display_text(DESCRIPTION),
            # Decorative separator between header and content.
            html.Div(html.Hr(className="line")),
            # Main card: tab navigation in the header and a callback-driven content area.
            display_card(
                html.Div(id="main-card-content"),
                header=display_tabs(
                    labels=[
                        "SpaceX Dataset",
                        "Exploratory Analysis",
                        "Predictive Modeling",
                    ],
                    id="master-tabs",
                ),
            ),
        ]
    )
