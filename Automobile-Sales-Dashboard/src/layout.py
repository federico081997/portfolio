"""
layout.py

Defines the Dash application layout for the Automobile Sales Statistics Dashboard.
This module constructs the overall page structure, including the title section,
control panel, and grid-based visualization area.
"""

from .components import (
    display_title,
    display_text,
    display_card,
    display_dropdown,
    create_grid,
)
from dash import html, dcc

# General description displayed at the top of the dashboard.
page_description = (
    "This dashboard presents a concise analytical overview of automobile sales "
    "between 1980 and 2013, highlighting both yearly trends and the distinct patterns that emerged during "
    "recession periods. Through interactive visualizations built with Plotly and Dash, "
    "it examines shifts in total sales, vehicle-type performance, advertising expenditure, and the effects of "
    "unemployment on consumer demand."
)


def build_layout(report_types: tuple[str], years: tuple[int]) -> html.Div:
    """
    Build and return the root layout for the Dash application.

    Parameters
    ----------
    report_types : tuple[str]
        Dropdown options representing available report types (e.g., yearly, recession).
    years : tuple[int]
        Dropdown options listing all available years in the dataset.

    Returns
    -------
    dash.html.Div
        The complete Dash layout containing titles, controls, and plot containers.
    """

    # Root container: includes title, description, controls, and plot grid.
    return html.Div(
        children=[
            # Page header
            display_title(text="Automobile Sales Statistics"),
            display_text(text=page_description),
            # Separator line
            html.Div(html.Hr(className="line")),
            # Control panel (report type + conditional year selector)
            display_card(
                create_grid(
                    rows=1,
                    cols=2,
                    items=[
                        [
                            display_text(
                                text="Select Report Type:",
                                className="controls-title",
                            ),
                            display_dropdown(
                                report_types,
                                id="report-dropdown",
                                placeholder="Select a report type",
                            ),
                        ],
                        [
                            # Year selector; hidden until appropriate report is chosen.
                            html.Div(
                                children=[
                                    display_text(
                                        "Select Year:", className="controls-title"
                                    ),
                                    display_dropdown(
                                        years,
                                        id="year-dropdown",
                                        placeholder="Select a year",
                                    ),
                                ],
                                id="year-control",
                                style={"display": "none"},
                            )
                        ],
                    ],
                ),
                header_text=display_text(text="Control Panel", className="card-title"),
            ),
            # Grid containing four plot cards; hidden until a report is selected.
            html.Div(
                create_grid(
                    rows=2,
                    cols=2,
                    items=[
                        display_card(
                            html.Div(dcc.Graph(id=f"plot-{i}")),
                            header_text=display_text(
                                id=f"plot-title-{i}", className="card-title"
                            ),
                        )
                        for i in range(1, 5)
                    ],
                ),
                id="plot-control",
                style={"display": "none"},
            ),
        ]
    )
