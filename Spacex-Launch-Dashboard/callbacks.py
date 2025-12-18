"""
callbacks.py

Dash callbacks for the SpaceX Launch Dashboard.

This module defines and registers all callbacks that drive dashboard interactivity.
Callbacks are registered through :func:`register_callbacks`, which binds callback
functions to the provided Dash application instance.
"""

from typing import Tuple, Dict, Any

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, no_update
from dash.dependencies import Input, Output

from components import (
    create_grid,
    display_card,
    display_dropdown,
    display_radioItems,
    display_range_slider,
    display_table,
    display_text,
    display_title,
)
from constants import PALETTE
from figures import display_confusion_matrix, display_map, plot


# =============================================================================
# Public API
# =============================================================================


def register_callbacks(
    app: dash.Dash, df: pd.DataFrame, model_cache: Dict[str, Any]
) -> None:
    """
    Register all dashboard callbacks on the provided Dash `app` instance.

    The callback functions defined inside this function close over the supplied
    dataset and model cache. This keeps callbacks fast and avoids recomputing
    model results on every user interaction.

    :param app: Dash application on which callbacks are registered.
    :type app: dash.Dash
    :param df: Master dataset used to compute all analytics and visualisations.
    :type df: pd.DataFrame
    :param model_cache: Precomputed model results produced by the modelling layer.
    :type model_cache: Dict[str, Any]
    :return: None
    :rtype: None
    """

    # -------------------------------------------------------------------------
    # Tab switcher: renders the contents of the main card based on the active tab
    # -------------------------------------------------------------------------
    @app.callback(
        Output("main-card-content", "children"),
        Input("master-tabs", "active_tab"),
    )
    def switch_tab(active_tab) -> dash.html.Div:
        """
        Render the main content area based on the selected tab.

        :param active_tab: Active tab identifier from `dbc.Tabs` (e.g. "tab-0").
        :type active_tab: str
        :return: A Dash component tree for the selected tab.
        :rtype: dash.html.Div
        """
        if active_tab == "tab-0":
            # Dataset view: display the full master table.
            return display_table(df, cell_selectable=True)

        if active_tab == "tab-1":
            # Exploratory view: controls + charts + embedded map.
            dropdown_options = [{"label": "All Sites", "value": "ALL"}] + [
                {"label": site, "value": site}
                for site in sorted(df["Launch Site"].unique())
            ]

            return html.Div(
                children=[
                    display_title("Launch Site Selection", className="controls-title"),
                    display_dropdown(
                        options=dropdown_options,
                        placeholder="Select a launch site",
                        value="ALL",
                        id="site-dropdown",
                        searchable=True,
                    ),
                    html.Br(),
                    display_title(
                        "Payload Mass Range (kg)", className="controls-title"
                    ),
                    display_range_slider(
                        min=0,
                        max=16000,
                        step=2000,
                        value=[0, 16000],
                        id="payload-slider",
                    ),
                    html.Br(),
                    *create_grid(
                        rows=1,
                        cols=2,
                        items=[
                            display_card(
                                html.Div(
                                    dcc.Graph(id="success-pie-chart", className="graph")
                                ),
                                header=display_text(
                                    id="pie-chart-header",
                                    className="card-title",
                                ),
                                className="card-style",
                            ),
                            display_card(
                                html.Div(
                                    dcc.Graph(
                                        id="payload-scatter-chart", className="graph"
                                    )
                                ),
                                header=display_text(
                                    id="scatter-chart-header",
                                    className="card-title",
                                ),
                                className="card-style",
                            ),
                        ],
                    ),
                    display_card(
                        html.Div(
                            html.Iframe(
                                id="launch-site-map",
                                className="map-style",
                            )
                        ),
                        header=display_text("Launch Site Map", className="card-title"),
                    ),
                ]
            )

        if active_tab == "tab-2":
            # Modelling view: model picker + confusion matrix + summary metrics.
            radioItem_options = [
                "Logistic Regression",
                "Support Vector Machines",
                "Decision Trees",
                "K-Nearest Neighbors",
            ]

            return html.Div(
                children=[
                    display_title("Model Type", className="controls-title"),
                    display_radioItems(
                        options=radioItem_options,
                        value=radioItem_options[0],
                        id="model-radioitems",
                        inline=True,
                        className="radio-style",
                    ),
                    html.Br(),
                    *create_grid(
                        rows=1,
                        cols=2,
                        items=[
                            display_card(
                                dcc.Graph(id="confusion-matrix"),
                                header=display_text(
                                    id="confusion-matrix-header",
                                    className="card-title",
                                ),
                            ),
                            display_card(
                                html.Div(id="model-results-table"),
                                header=display_text(
                                    "Summary of Results",
                                    className="card-title",
                                ),
                            ),
                        ],
                    ),
                ]
            )

        # Defensive fallback: if an unexpected tab ID appears, show nothing.
        return html.Div()

    # -------------------------------------------------------------------------
    # Exploratory tab: update dynamic chart titles based on selected site
    # -------------------------------------------------------------------------

    @app.callback(
        Output("pie-chart-header", "children"),
        Output("scatter-chart-header", "children"),
        Input("site-dropdown", "value"),
    )
    def update_site_headers(selected_site) -> Tuple[str, str]:
        """
        Update chart headers to reflect the selected launch site.

        :param selected_site: Selected site value (either "ALL" or a site name).
        :type selected_site: str
        :return: (pie_header_text, scatter_header_text)
        :rtype: Tuple[str, str]
        """
        if selected_site == "ALL":
            return (
                "Total Success Launches for All Sites",
                "Payload vs. Success for All Sites",
            )

        return (
            f"Total Success Launches for Site {selected_site}",
            f"Payload vs. Success for Site {selected_site}",
        )

    # -------------------------------------------------------------------------
    # Exploratory tab: update pie and scatter charts based on filters
    # -------------------------------------------------------------------------
    @app.callback(
        Output("success-pie-chart", "figure"),
        Output("payload-scatter-chart", "figure"),
        Input("site-dropdown", "value"),
        Input("payload-slider", "value"),
    )
    def update_charts(selected_site, payload_range) -> Tuple[go.Figure, go.Figure]:
        """
        Update the exploratory pie and scatter charts when filters change.

        :param selected_site: Selected site value (either "ALL" or a site name).
        :type selected_site: str
        :param payload_range: Selected payload range as [min, max].
        :type payload_range: List[int]
        :return: (pie_chart_figure, scatter_chart_figure)
        :rtype: Tuple[go.Figure, go.Figure]
        """
        min_payload, max_payload = payload_range

        # Filter by payload mass range first.
        filtered_df = df[
            (df["Payload Mass (kg)"] >= min_payload)
            & (df["Payload Mass (kg)"] <= max_payload)
        ][["Launch Site", "Payload Mass (kg)", "Landing Outcome"]].copy()

        # Convert landing outcome into a numeric success flag (1=success, 0=failure).
        filtered_df["Landing Outcome"] = filtered_df["Landing Outcome"].apply(
            lambda x: 1 if (x is True or "true" in str(x).lower()) else 0
        )

        if selected_site == "ALL":
            # Pie chart: total successes per site.
            pie_chart = plot(
                kind="pie",
                df=filtered_df.groupby("Launch Site").sum().reset_index(),
                x="Landing Outcome",
                y="Launch Site",
                px_kwargs={
                    "category_orders": {
                        "Launch Site": sorted(filtered_df["Launch Site"].unique())
                    },
                },
                layout_kwargs={
                    "legend": dict(
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.2,
                        yanchor="top",
                        title_text="Launch Site",
                    ),
                },
                trace_kwargs={
                    "hovertemplate": (
                        "<b>Launch Site:</b> %{label}<br>"
                        "<b>Successes:</b> %{value}<extra></extra>"
                    ),
                    "texttemplate": "%{percent:.2%}",
                    "marker": {"colors": PALETTE},
                },
            )

            # Scatter chart: payload vs success (0/1), coloured by site.
            scatter_chart = plot(
                kind="scatter",
                df=filtered_df,
                x="Payload Mass (kg)",
                y="Landing Outcome",
                px_kwargs={
                    "category_orders": {
                        "Launch Site": sorted(filtered_df["Launch Site"].unique())
                    },
                    "color": "Launch Site",
                    "color_discrete_sequence": PALETTE,
                    "custom_data": [
                        filtered_df["Launch Site"],
                        filtered_df["Landing Outcome"].apply(
                            lambda x: "Success" if x == 1 else "Failure"
                        ),
                    ],
                },
                layout_kwargs={
                    "yaxis": dict(
                        tickmode="array",
                        tickvals=[0, 1],
                        ticktext=["Failure", "Success"],
                    ),
                    "legend": dict(
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.2,
                        yanchor="top",
                        title_text="Launch Site",
                    ),
                },
                trace_kwargs={
                    "marker": {"size": 5},
                    "hovertemplate": (
                        "<b>Launch Site:</b> %{customdata[0]}<br>"
                        "<b>Payload Mass:</b> %{x} kg<br>"
                        "<b>Landing Outcome:</b> %{customdata[1]}<extra></extra>"
                    ),
                },
            )

            return pie_chart, scatter_chart

        # Site-specific charts.
        site_df = filtered_df[filtered_df["Launch Site"] == selected_site].copy()
        site_df["Landing Outcome"] = site_df["Landing Outcome"].apply(
            lambda x: "Success" if x == 1 else "Failure"
        )

        # Pie chart: distribution of success/failure for the selected site.
        pie_chart = plot(
            kind="pie",
            df=site_df.groupby("Landing Outcome").size().reset_index(name="Count"),
            x="Count",
            y="Landing Outcome",
            px_kwargs={
                "category_orders": {
                    "Landing Outcome": sorted(
                        site_df["Landing Outcome"].unique(), reverse=True
                    )
                },
            },
            layout_kwargs={
                "legend": dict(
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.2,
                    yanchor="top",
                    title_text="Outcome",
                ),
            },
            trace_kwargs={
                "hovertemplate": (
                    "<b>Landing Outcome:</b> %{label}<br>"
                    "<b>Count:</b> %{value}<extra></extra>"
                ),
                "texttemplate": "%{percent:.2%}",
                "marker": {"colors": [PALETTE[2], PALETTE[5]]},
            },
        )

        # Scatter chart: payload vs. categorical outcome for the selected site.
        scatter_chart = plot(
            kind="scatter",
            df=site_df,
            x="Payload Mass (kg)",
            y="Landing Outcome",
            px_kwargs={
                "category_orders": {
                    "Landing Outcome": sorted(
                        site_df["Landing Outcome"].unique(), reverse=True
                    )
                },
                "color": "Landing Outcome",
                "color_discrete_sequence": [PALETTE[2], PALETTE[5]],
            },
            layout_kwargs={
                "legend": dict(
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.2,
                    yanchor="top",
                    title_text="Outcome",
                ),
            },
            trace_kwargs={
                "marker": {"size": 5},
                "hovertemplate": (
                    "<b>Payload Mass:</b> %{x} kg<br>"
                    "<b>Landing Outcome:</b> %{y}<extra></extra>"
                ),
            },
        )

        return pie_chart, scatter_chart

    # -------------------------------------------------------------------------
    # Exploratory tab: update the embedded map only when the tab is active
    # -------------------------------------------------------------------------
    @app.callback(
        Output("launch-site-map", "srcDoc"),
        Input("master-tabs", "active_tab"),
    )
    def update_map(active_tab) -> str:
        """
        Render the Folium map HTML when the exploratory tab is active.

        :param active_tab: Currently active tab identifier.
        :type active_tab: str
        :return: HTML string for the iframe `srcDoc`.
        :rtype: str
        """
        if active_tab != "tab-1":
            # Clear srcDoc when not visible to avoid unnecessary client rendering.
            return ""

        # One row per launch site for the base site markers.
        filtered_df = (
            df.groupby("Launch Site")[["Launch Site Latitude", "Launch Site Longitude"]]
            .first()
            .reset_index()
        )

        # Per-launch rows used for marker clustering (success/failure colour coding).
        cluster_df = df[
            ["Launch Site Latitude", "Launch Site Longitude", "Landing Outcome"]
        ]

        # Map centred roughly over the continental US.
        usa_coords = (39.45107477305616, -101.33058847306438)

        return display_map(filtered_df, usa_coords, zoom_start=4, cluster=cluster_df)

    # -------------------------------------------------------------------------
    # Modelling tab: update confusion matrix header based on selected model
    # -------------------------------------------------------------------------
    @app.callback(
        Output("confusion-matrix-header", "children"),
        Input("model-radioitems", "value"),
    )
    def update_model_header(selected_model) -> str:
        """
        Update the confusion matrix card header when a different model is selected.

        :param selected_model: Name of the selected model.
        :type selected_model: str
        :return: Header text.
        :rtype: str
        """
        return f"Confusion Matrix for {selected_model}"

    # -------------------------------------------------------------------------
    # Modelling tab: update confusion matrix and results table
    # -------------------------------------------------------------------------
    @app.callback(
        Output("confusion-matrix", "figure"),
        Output("model-results-table", "children"),
        Input("master-tabs", "active_tab"),
        Input("model-radioitems", "value"),
    )
    def update_model_tab(active_tab, selected_model) -> Tuple[go.Figure, dash.html.Div]:
        """
        Update modelling outputs when the modelling tab is active and the model changes.

        :param active_tab: Currently active tab identifier.
        :type active_tab: str
        :param selected_model: Name of the selected model (matches keys in model_cache).
        :type selected_model: str
        :return: (confusion_matrix_figure, results_table_component)
        :rtype: Tuple[go.Figure, dash.html.Div]
        """
        # Avoid updating hidden components when the user is not on the modelling tab.
        if active_tab != "tab-2":
            return no_update, no_update

        cm = model_cache["Confusion Matrix"].get(selected_model)

        # Results table is the same for all selected models; the radio changes only the matrix.
        table = display_table(
            pd.DataFrame(model_cache["Results"]), cell_selectable=True
        )

        fig = display_confusion_matrix(cm)
        return fig, table
