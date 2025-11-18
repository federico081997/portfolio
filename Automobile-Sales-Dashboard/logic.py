"""
logic.py

Domain logic for the Automobile Sales Statistics Dashboard.

This module provides filtering and aggregation utilities for both yearly
statistics and recession-period statistics. Each function returns pre-processed
dataframes used by the dashboard's visualization layer.

It also provides a utility function that automatically inserts line breaks into Plotly axis
labels when they are too long, improving readability.
"""

import pandas as pd
from constants import MONTHS

# Define month order for the dataframes
MONTH_ORDER = {m: i for i, m in enumerate(MONTHS.keys())}


def compute_yearly_info(
    df: pd.DataFrame,
    input_year: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute yearly statistics for a selected year.

    Parameters
    ----------
    df : pd.DataFrame
        The full automobile sales dataset.
    input_year : str
        The year for which statistics are computed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - total_by_year: Total automobile sales aggregated by year.
        - total_by_month: Monthly sales totals for the selected year.
        - avg_by_vehicle: Average sales by vehicle type for the selected year.
        - total_ads_by_type: Total advertising expenditure by vehicle type.
    """

    # Filter data for the selected year
    yearly_data = df[df["Year"] == input_year]

    # Total automobile sales aggregated across all available years
    total_by_year = df.groupby("Year")["Automobile_Sales"].sum().reset_index()

    # Monthly sales totals for the selected year
    total_by_month = (
        yearly_data.groupby("Month", sort=False)["Automobile_Sales"]
        .sum()
        .reset_index()
        .sort_values("Month", key=lambda s: s.map(MONTH_ORDER))
    )

    # Average sales by vehicle type for the selected year
    avg_by_vehicle = (
        yearly_data.groupby("Vehicle_Type")["Automobile_Sales"].mean().reset_index()
    )

    # Total advertising expenditure by vehicle type for the selected year
    total_ads_by_type = (
        yearly_data.groupby("Vehicle_Type")["Advertising_Expenditure"]
        .sum()
        .reset_index()
    )

    return total_by_year, total_by_month, avg_by_vehicle, total_ads_by_type


def compute_recession_info(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute aggregated statistics specifically for recession periods.

    Parameters
    ----------
    df : pd.DataFrame
        The full automobile sales dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - avg_by_year: Average sales during recession years.
        - avg_by_type: Average sales by vehicle type during recession years.
        - total_ads_by_type: Total advertising expenditure by vehicle type.
        - avg_sales_by_rate_type: Average sales grouped by unemployment rate
          and vehicle type.
    """

    # Filter data for recession periods
    recession_data = df[df["Recession"] == 1]

    # Average sales during recession years
    avg_by_year = (
        recession_data.groupby("Year")["Automobile_Sales"].mean().round(0).reset_index()
    )

    # Average sales by vehicle type during recession years
    avg_by_type = (
        recession_data.groupby("Vehicle_Type")["Automobile_Sales"]
        .mean()
        .round(0)
        .reset_index()
    )

    # Total advertising expenditures by vehicle type during recession years
    total_ads_by_type = (
        recession_data.groupby("Vehicle_Type")["Advertising_Expenditure"]
        .sum()
        .reset_index()
    )

    # Average sales by unemployment rate and vehicle type
    avg_sales_by_rate_type = (
        recession_data.groupby(["Unemployment_Rate", "Vehicle_Type"])[
            "Automobile_Sales"
        ]
        .mean()
        .reset_index()
    )

    return avg_by_year, avg_by_type, total_ads_by_type, avg_sales_by_rate_type


def wrap_label(text: str, max_len: int = 10) -> str:
    """
    Inserts a line break into long tick labels for improved readability.

    This function attempts to split the input string into two lines when its
    length exceeds `max_len`. The split is performed at the nearest whitespace
    character before `max_len`; if none is found, it searches for the nearest
    whitespace after `max_len`. If the string contains no whitespace at all,
    a hard split is performed at `max_len`. The line break is inserted using
    an HTML <br> tag, which is compatible with Plotly axis tick labels.

    Parameters
    ----------
    text : str
        The label text to process.
    max_len : int
        The maximum allowed length for the first line before attempting a split.
        Defaults to 10.

    Returns
    -------
    str
        The processed label, potentially containing an HTML line break.
    """
    if len(text) <= max_len:
        return text

    # Attempt to break at a whitespace character before `max_len`
    break_pos = text.rfind(" ", 0, max_len)
    if break_pos == -1:
        # If not found, attempt to break at a whitespace after `max_len`
        break_pos = text.find(" ", max_len)
        if break_pos == -1:
            # If no whitespace exists at all, fall back to a hard split
            break_pos = max_len

    return text[:break_pos] + "<br>" + text[break_pos + 1 :]
