"""
logic.py

Domain logic: filtering and aggregation for the Australia Wildfires Dashboard.

This module provides a single orchestration function, `compute_info`, which:
1) filters the dataset by region and year;
2) computes monthly averages for the estimated fire area and pixel counts; and
3) returns both the monthly aggregates and their totals.
"""

import pandas as pd


def compute_info(
    data: pd.DataFrame,
    entered_region: str,
    entered_year: str,
) -> tuple[pd.DataFrame, float, pd.DataFrame, float]:
    """
    Filter and aggregate wildfire metrics for a given region and year.

    Parameters
    ----------
    data : pandas.DataFrame
        The full dataset. Must include columns:
        ['Region', 'Year', 'Month_name', 'Estimated_fire_area', 'Count'].
    entered_region : str
        Region code selected by the user (e.g., 'NSW', 'QLD', 'VIC').
    entered_year : str
        Calendar year selected by the user. Cast to `int` for comparison.

    Returns
    -------
    avg_fire_area : pandas.DataFrame
        Monthly mean of 'Estimated_fire_area'.
        Columns: ['Month_name', 'Estimated_fire_area'].
    total_area : float
        Sum of the monthly means of 'Estimated_fire_area' (for display).
    avg_count_pixels : pandas.DataFrame
        Monthly mean of 'Count', rounded to 0 decimals.
        Columns: ['Month_name', 'Count'].
    total_pixels : float
        Sum of the rounded monthly means of 'Count' (for display).
    """

    # Filter to the selected region and year (year cast to int to match column dtype)
    df = data[(data["Region"] == entered_region) & (data["Year"] == int(entered_year))]

    # Compute the average estimated fire area per month
    # Grouping uses the existing 'Month_name' column and preserves first-encounter order (sort=False).
    avg_fire_area = (
        df.groupby(df["Month_name"], sort=False)["Estimated_fire_area"]
        .mean()
        .reset_index()
    )
    total_area = avg_fire_area["Estimated_fire_area"].sum(axis=0)

    # Compute the average pixel count per month and round to 0 decimals for display parity
    avg_count_pixels = (
        df.groupby(df["Month_name"], sort=False)["Count"].mean().round(0).reset_index()
    )
    total_pixels = avg_count_pixels["Count"].sum(axis=0)

    return avg_fire_area, total_area, avg_count_pixels, total_pixels
