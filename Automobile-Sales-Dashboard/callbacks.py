"""
callbacks.py

This module defines and registers all Dash callbacks used throughout the
application. It handles dynamic rendering of figures, conditional display
of interface elements, and generation of plot titles based on user input.

Callbacks include:
    - Conditional visibility of the year selector based on the selected report
    - Construction and updating of all analytical figures
    - Context-aware updates of plot titles

All callbacks are bound to the Dash application instance passed to
`register_callbacks`.
"""

from typing import Any, Dict, Tuple
import pandas as pd
import dash
from dash.dependencies import Input, Output
from logic import compute_yearly_info, compute_recession_info, wrap_label
from figures import plot
from constants import MONTHS, YEARS, VEHICLE_TYPES


def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:
    """
    Register all application callbacks on the provided Dash `app` instance.

    Parameters
    ----------
    app : dash.Dash
        The Dash application on which the callbacks are to be registered.

    df : pandas.DataFrame
        The dataset used to compute all analytics and visualizations.
    """

    # ------------------------------------------------------------
    # Callback: Conditional display of year selector
    # ------------------------------------------------------------
    @app.callback(
        [
            Output(component_id="year-control", component_property="style"),
            Output(component_id="year-dropdown", component_property="value"),
        ],
        Input(component_id="report-dropdown", component_property="value"),
        prevent_initial_call=True,
    )
    def display_year_dropdown(input_report_type: str) -> Tuple[Dict[str, str], None]:
        """
        Show or hide the year selector depending on the chosen report type.
        """
        if input_report_type == "Yearly Statistics":
            return {"display": "block"}, None
        else:
            return {"display": "none"}, None

    # ------------------------------------------------------------
    # Callback: Main figure generation for all report types
    # ------------------------------------------------------------
    @app.callback(
        [
            Output(component_id="plot-1", component_property="figure"),
            Output(component_id="plot-2", component_property="figure"),
            Output(component_id="plot-3", component_property="figure"),
            Output(component_id="plot-4", component_property="figure"),
            Output(component_id="plot-control", component_property="style"),
        ],
        [
            Input(component_id="report-dropdown", component_property="value"),
            Input(component_id="year-dropdown", component_property="value"),
        ],
    )
    def get_graphs(
        input_report_type: str, input_year: int
    ) -> Tuple[Any, Any, Any, Any, Dict[str, str]]:
        """
        Construct and return all analytical plots based on the selected
        report type and (if applicable) the selected year.
        """
        # --- Recession Period Statistics -------------------------------------
        if input_report_type == "Recession Period Statistics":
            # Calculate data for Recession Period Statistics plots
            avg_by_year, avg_by_type, total_ads_by_type, avg_sales_by_rate_type = (
                compute_recession_info(df)
            )

            # Plot 1: Average Automobile Sales During Recession Years
            line_plot = plot(
                kind="line",
                df=avg_by_year,
                x="Year",
                y="Automobile_Sales",
                layout_kwargs={
                    "xaxis_title": "Year",
                    "yaxis_title": "Average Automobile Sales",
                    "xaxis": dict(
                        showgrid=False,
                        tickmode="array",
                        tickvals=avg_by_year["Year"],
                        tickangle=75,
                        ticks="outside",
                    ),
                    "shapes": [
                        dict(
                            type="line",
                            x0=year,
                            x1=year,
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            line=dict(width=1, color="grey", dash="dash"),
                        )
                        for year in avg_by_year["Year"]
                    ],
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "mode": "lines+markers",
                    "marker": dict(
                        size=8,
                        symbol="circle",
                        color="#4B6CB7",
                        line=dict(width=0.5, color="white"),
                    ),
                    "line": dict(width=2, color="#4B6CB7"),
                    "hovertemplate": "<b>Year:</b> %{x}<br><b>Sales:</b> %{y:.0f}<extra></extra>",
                },
            )

            # Plot 2: Average Automobile Sales by Vehicle Type
            bar_plot_1 = plot(
                kind="bar",
                df=avg_by_type,
                x="Vehicle_Type",
                y="Automobile_Sales",
                px_kwargs={
                    "color": "Vehicle_Type",
                    "color_discrete_map": VEHICLE_TYPES,
                    "category_orders": {"Vehicle_Type": list(VEHICLE_TYPES.keys())},
                    "custom_data": ["Vehicle_Type"],
                },
                layout_kwargs={
                    "xaxis_title": "Vehicle Type",
                    "yaxis_title": "Average Automobile Sales",
                    "xaxis": dict(
                        tickmode="array",
                        tickvals=avg_by_type["Vehicle_Type"],
                        ticktext=[
                            wrap_label(vehicle)
                            for vehicle in avg_by_type["Vehicle_Type"]
                        ],
                        ticks="outside",
                    ),
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "marker": dict(
                        line=dict(width=1, color="black"),
                    ),
                    "showlegend": False,
                    "hovertemplate": "<b>Type:</b> %{customdata[0]}<br><b>Sales:</b> %{y:.0f}<extra></extra>",
                },
            )

            # Plot 3: Advertising Expenditure Distribution by Vehicle Type
            pie_plot = plot(
                kind="pie",
                df=total_ads_by_type,
                x="Advertising_Expenditure",
                y="Vehicle_Type",
                px_kwargs={
                    "color": "Vehicle_Type",
                    "color_discrete_map": VEHICLE_TYPES,
                    "category_orders": {"Vehicle_Type": list(VEHICLE_TYPES.keys())},
                },
                layout_kwargs={
                    "legend": dict(
                        title_text="Vehicle Type<br>",
                    ),
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "marker": dict(
                        line=dict(width=1, color="black"),
                    ),
                    "hovertemplate": (
                        "<b>Vehicle Type:</b> %{label}<br>"
                        "<b>Advertisement Cost:</b> $%{value:,.0f}<br>"
                        "<b>Share:</b> %{percent:.2%}<br>"
                    ),
                    "texttemplate": "%{percent:.2%}",
                },
            )

            # Plot 4: Automobile Sales vs. Unemployment Rate by Vehicle Type
            bar_plot_2 = plot(
                kind="bar",
                df=avg_sales_by_rate_type,
                x="Unemployment_Rate",
                y="Automobile_Sales",
                px_kwargs={
                    "color": "Vehicle_Type",
                    "color_discrete_map": VEHICLE_TYPES,
                    "category_orders": {"Vehicle_Type": list(VEHICLE_TYPES.keys())},
                },
                layout_kwargs={
                    "xaxis_title": "Unemployment Rate (%)",
                    "yaxis_title": "Average Automobile Sales",
                    "xaxis": dict(
                        ticks="outside",
                    ),
                    "legend": dict(
                        title_text="Vehicle Type",
                        x=0.5,
                        y=-0.38,
                        xanchor="center",
                        yanchor="bottom",
                        orientation="h",
                    ),
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "marker": dict(
                        line=dict(width=1, color="black"),
                    ),
                    "hovertemplate": (
                        "<b>Vehicle Type:</b> %{fullData.name}<br>"
                        "<b>Unemployment Rate:</b> %{x}<br>"
                        "<b>Sales:</b> %{y:.0f}<extra></extra>"
                    ),
                },
            )

            return line_plot, bar_plot_1, pie_plot, bar_plot_2, {"display": "block"}

        # --- Yearly Statistics: No year selected -----------------------------
        elif not input_report_type or (
            input_report_type == "Yearly Statistics" and not input_year
        ):
            return None, None, None, None, {"display": "none"}

        # --- Yearly Statistics: Valid year provided --------------------------
        elif input_year and input_report_type == "Yearly Statistics":
            total_by_year, total_by_month, avg_by_vehicle, total_ads_by_type = (
                compute_yearly_info(df, input_year)
            )

            # Plot 1: Automobile Sales Over the Years
            line_plot_1 = plot(
                kind="line",
                df=total_by_year,
                x="Year",
                y="Automobile_Sales",
                layout_kwargs={
                    "xaxis_title": "Year",
                    "yaxis_title": "Total Automobile Sales",
                    "xaxis": dict(
                        tickmode="array",
                        tickvals=list(YEARS[:-2:5]) + [total_by_year["Year"].iloc[-1]],
                        tickangle=90,
                        ticks="outside",
                    ),
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "mode": "lines+markers",
                    "marker": dict(
                        size=8,
                        symbol="circle",
                        color="#4B6CB7",
                        line=dict(width=0.5, color="white"),
                    ),
                    "line": dict(width=2, color="#4B6CB7"),
                    "hovertemplate": "<b>Year:</b> %{x}<br><b>Sales:</b> %{y:.0f}<extra></extra>",
                },
            )

            # Plot 2: Monthly Automobile Sales in {input year}
            line_plot_2 = plot(
                kind="line",
                df=total_by_month,
                x="Month",
                y="Automobile_Sales",
                px_kwargs={
                    "custom_data": [total_by_month["Month"].map(MONTHS)],
                },
                layout_kwargs={
                    "xaxis_title": "Year",
                    "yaxis_title": "Total Automobile Sales",
                    "xaxis": dict(
                        tickmode="array",
                        tickvals=list(total_by_month["Month"]),
                        tickangle=90,
                        ticks="outside",
                    ),
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "mode": "lines+markers",
                    "marker": dict(
                        size=8,
                        symbol="circle",
                        color="#4B6CB7",
                        line=dict(width=0.5, color="white"),
                    ),
                    "line": dict(width=2, color="#4B6CB7"),
                    "hovertemplate": "<b>Month:</b> %{customdata}<br><b>Sales:</b> %{y:.0f}<extra></extra>",
                },
            )

            # Plot 3: Average Automobile Sales by Vehicle Type in {input year}
            bar_plot = plot(
                kind="bar",
                df=avg_by_vehicle,
                x="Vehicle_Type",
                y="Automobile_Sales",
                px_kwargs={
                    "color": "Vehicle_Type",
                    "color_discrete_map": VEHICLE_TYPES,
                    # "category_orders": {"Vehicle_Type": list(VEHICLE_TYPES.keys())},
                    "custom_data": ["Vehicle_Type"],
                },
                layout_kwargs={
                    "xaxis_title": "Vehicle Type",
                    "yaxis_title": "Average Automobile Sales",
                    "xaxis": dict(
                        tickmode="array",
                        tickvals=avg_by_vehicle["Vehicle_Type"],
                        ticktext=[
                            wrap_label(vehicle)
                            for vehicle in avg_by_vehicle["Vehicle_Type"]
                        ],
                        ticks="outside",
                    ),
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "marker": dict(
                        line=dict(width=1, color="black"),
                    ),
                    "showlegend": False,
                    "hovertemplate": "<b>Type:</b> %{customdata}<br><b>Sales:</b> %{y:.0f}<extra></extra>",
                },
            )

            # Plot 4: Advertising Expenditure Distribution by Vehicle Type in {input year}
            pie_plot = plot(
                kind="pie",
                df=total_ads_by_type,
                x="Advertising_Expenditure",
                y="Vehicle_Type",
                px_kwargs={
                    "color": "Vehicle_Type",
                    "color_discrete_map": VEHICLE_TYPES,
                    "category_orders": {"Vehicle_Type": list(VEHICLE_TYPES.keys())},
                },
                layout_kwargs={
                    "legend": dict(
                        title_text="Vehicle Type<br>",
                    ),
                    "margin": dict(r=20, l=20, t=20),
                },
                trace_kwargs={
                    "marker": dict(
                        line=dict(width=1, color="black"),
                    ),
                    "hovertemplate": (
                        "<b>Vehicle Type:</b> %{label}<br>"
                        "<b>Advertisement Cost:</b> $%{value:,.0f}<br>"
                        "<b>Share:</b> %{percent:.2%}<br>"
                    ),
                    "texttemplate": "%{percent:.2%}",
                },
            )

            return line_plot_1, line_plot_2, bar_plot, pie_plot, {"display": "block"}

    # ------------------------------------------------------------
    # Callback: Update plot titles dynamically
    # ------------------------------------------------------------
    @app.callback(
        [
            Output(component_id="plot-title-1", component_property="children"),
            Output(component_id="plot-title-2", component_property="children"),
            Output(component_id="plot-title-3", component_property="children"),
            Output(component_id="plot-title-4", component_property="children"),
        ],
        [
            Input(component_id="report-dropdown", component_property="value"),
            Input(component_id="year-dropdown", component_property="value"),
        ],
    )
    def get_plot_title(
        input_report_type: str, input_year: int
    ) -> Tuple[str, str, str, str]:
        """
        Return context-appropriate titles for all plots.
        """
        if input_report_type == "Recession Period Statistics":
            return (
                "Average Automobile Sales During Recession Years",
                "Average Automobile Sales by Vehicle Type",
                "Advertising Expenditure Distribution by Vehicle Type",
                "Automobile Sales vs. Unemployment Rate by Vehicle Type",
            )
        else:
            return (
                "Automobile Sales Over the Years",
                f"Monthly Automobile Sales in {input_year}",
                f"Average Automobile Sales by Vehicle Type in {input_year}",
                f"Advertising Expenditure Distribution by Vehicle Type in {input_year}",
            )
