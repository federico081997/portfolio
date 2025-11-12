"""
callbacks.py

Dash callbacks: figure updates and title updates.

This module registers all application callbacks on a provided Dash app instance.
"""

import pandas as pd
import dash
from dash.dependencies import Input, Output
from .logic import compute_info
from .figures import generate_donut, generate_bar


def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:
    """
    Register all Dash callbacks on the given app.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance on which callbacks are declared.
    df : pandas.DataFrame
        Full dataset used by callbacks. The dataframe is expected to
        include columns: 'Region', 'Year', 'Month_name',
        'Estimated_fire_area', and 'Count'.
    """

    # Figure-generating callback
    # --------------------------
    # When the user changes region or year, recompute the aggregates and
    # rebuild both the donut and bar figures. The two Output objects
    # correspond to the two dcc.Graph components in the layout.
    @app.callback(
        [
            Output(component_id="pie-chart", component_property="figure"),
            Output(component_id="bar-chart", component_property="figure"),
        ],
        [
            Input(component_id="region-items", component_property="value"),
            Input(component_id="year-items", component_property="value"),
        ],
    )
    def get_graph(entered_region, entered_year):
        # Compute all monthly aggregates required by both figures
        avg_fire_area, total_area, avg_count_pixels, total_pixels = compute_info(
            df, entered_region, entered_year
        )

        # Return figures in the order matching the Output list above
        return [
            generate_donut(avg_fire_area, total_area),
            generate_bar(avg_count_pixels, total_pixels),
        ]

    # Title-generating callback
    # -------------------------
    # Updates the H3 titles above the charts based on current inputs. This
    # keeps the textual context aligned with the visualizations.
    @app.callback(
        [
            Output(component_id="pie-title", component_property="children"),
            Output(component_id="bar-title", component_property="children"),
        ],
        [
            Input(component_id="region-items", component_property="value"),
            Input(component_id="year-items", component_property="value"),
        ],
    )
    def get_title(entered_region, entered_year):
        pie_title = f"{entered_region}: Monthly Average Estimated Fire Area in Year {str(entered_year)}."
        bar_title = f"{entered_region}: Monthly Average Count of Pixels for Presumed Vegetation Fires in Year {str(entered_year)}."

        return pie_title, bar_title
