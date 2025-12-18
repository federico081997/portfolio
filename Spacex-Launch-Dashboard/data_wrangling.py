"""
data_wrangling.py

Utilities for transforming raw SpaceX launch data into a clean, analysis-ready
dataset for the dashboard.

This module sits above :mod:`spacex_data_loader`. It converts the raw JSON
payloads (launches, launchpads, rockets, payloads, cores) into normalised
pandas DataFrames and then merges them into a single “master” table that the
dashboard can consume efficiently.

Design notes
------------
- The dashboard assumes *one core* and *one payload* per launch. Multi-core and
  multi-payload launches are therefore excluded to keep the data model simple
  and consistent with the visualisations and ML pipeline.
- Nested core fields from the launches endpoint are flattened into explicit
  columns (landing outcome, reuse flags, etc.).
- Basic missing-value imputation is applied at the end of the pipeline.
"""

from typing import List, Dict, Any, Iterable, Optional

import pandas as pd

from spacex_data_loader import load_spacex_raw_data


# =============================================================================
# Internal helpers
# =============================================================================


def _json_to_dataframe(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of JSON dictionaries into a pandas DataFrame.

    :param json_data: A list of JSON objects as returned by the SpaceX API.
    :type json_data: List[Dict[str, Any]]
    :return: A DataFrame constructed from the JSON data.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(json_data)


def _prepare_launches_data(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare the launches table and enforce one-core / one-payload per launch.

    The SpaceX launches endpoint contains nested lists for `cores` and `payloads`.
    The dashboard and ML pipeline assume a one-to-one relationship, therefore:
    - launches with multiple cores are excluded, and the remaining rows have their
      single core dict flattened into columns;
    - launches with multiple payloads are excluded, and the remaining rows keep
      only the single payload ID.

    In addition, `date_utc` is converted to a date (no time-of-day), and selected
    core-level attributes are extracted into dedicated columns.

    :param json_data: Raw launch JSON data.
    :type json_data: List[Dict[str, Any]]
    :return: Filtered launches DataFrame with one core and one payload per launch.
    :rtype: pd.DataFrame
    """
    # Convert JSON to a DataFrame and select only fields used by the dashboard.
    data = _json_to_dataframe(json_data)[
        ["flight_number", "date_utc", "rocket", "payloads", "launchpad", "cores"]
    ]

    # Enforce single-core launches (exclude side-booster missions), then flatten.
    data = data[data["cores"].apply(lambda cores: len(cores) == 1)]
    data["cores"] = data["cores"].apply(lambda cores: cores[0])

    # Enforce single-payload launches, then keep the payload ID.
    data = data[data["payloads"].apply(lambda payloads: len(payloads) == 1)]
    data["payloads"] = data["payloads"].apply(lambda payloads: payloads[0])

    # Flatten the relevant nested core metadata into explicit columns.
    data["core_id"] = data["cores"].apply(lambda core: core["core"])
    data["outcome"] = data["cores"].apply(
        lambda core: f'{core["landing_success"]} {core["landing_type"]}'
    )
    data["flights"] = data["cores"].apply(lambda core: core["flight"])
    data["gridFins"] = data["cores"].apply(lambda core: core["gridfins"])
    data["reused"] = data["cores"].apply(lambda core: core["reused"])
    data["legs"] = data["cores"].apply(lambda core: core["legs"])

    # Drop the original nested structure once flattened.
    data = data.drop(columns=["cores"])

    # Normalise date format.
    data["date_utc"] = pd.to_datetime(data["date_utc"]).dt.date

    return data


def _prepare_launchpads_data(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare the launchpads table with the columns used by the dashboard.

    :param json_data: Raw launchpad JSON data.
    :type json_data: List[Dict[str, Any]]
    :return: Launchpads DataFrame restricted to id, name, and coordinates.
    :rtype: pd.DataFrame
    """
    data = _json_to_dataframe(json_data)[["id", "name", "latitude", "longitude"]]
    return data


def _prepare_rockets_data(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare the rockets table restricted to Falcon 9.

    :param json_data: Raw rocket JSON data.
    :type json_data: List[Dict[str, Any]]
    :return: Rockets DataFrame containing only Falcon 9 entries (id and name).
    :rtype: pd.DataFrame
    """
    data = _json_to_dataframe(json_data)[["id", "name"]]

    # Dashboard scope is Falcon 9 launches only.
    data = data[data["name"] == "Falcon 9"]

    return data


def _prepare_payloads_data(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare the payloads table with relevant attributes.

    :param json_data: Raw payload JSON data.
    :type json_data: List[Dict[str, Any]]
    :return: Payloads DataFrame containing id, mass_kg, and orbit.
    :rtype: pd.DataFrame
    """
    data = _json_to_dataframe(json_data)[["id", "mass_kg", "orbit"]]
    return data


def _prepare_cores_data(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare the cores table with relevant attributes.

    :param json_data: Raw core JSON data.
    :type json_data: List[Dict[str, Any]]
    :return: Cores DataFrame containing id, block, serial, and reuse_count.
    :rtype: pd.DataFrame
    """
    data = _json_to_dataframe(json_data)[["id", "block", "serial", "reuse_count"]]
    return data


def _fix_missing_values(
    df: pd.DataFrame,
    exclude_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Impute missing values with simple, readable rules.

    - Numeric columns: fill missing values with the column mean, then round to 2 decimals.
    - Object columns: fill missing values with the column mode.

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param exclude_cols: Columns to exclude from imputation.
    :type exclude_cols: Iterable[str], optional
    :return: The same DataFrame instance with missing values handled.
    :rtype: pd.DataFrame
    """
    # Normalise exclusions to a list to make subsequent operations deterministic.
    exclude = list(exclude_cols) if exclude_cols is not None else []

    # Numerical imputation: mean fill + simple formatting for display.
    numerical_cols = df.select_dtypes(include=["number"]).columns.difference(exclude)
    for col in numerical_cols:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value).round(2)

    # Categorical imputation: mode fill.
    # Note: `mode(dropna=True)` returns a Series; take the first mode value.
    categorical_cols = df.select_dtypes(include=["object"]).columns.difference(exclude)
    for col in categorical_cols:
        # Guard against all-NaN columns: if mode is empty, leave values as-is.
        modes = df[col].mode(dropna=True)
        if modes.empty:
            continue
        mode_value = modes.iloc[0]
        df.loc[df[col].isna(), col] = mode_value

    return df


# =============================================================================
# Public API
# =============================================================================


def assemble_master_data() -> pd.DataFrame:
    """
    Assemble the fully merged and cleaned SpaceX launch dataset.

    Pipeline overview
    -----------------
    1) Load raw JSON entities from :func:`load_spacex_raw_data`.
    2) Prepare entity-level DataFrames (launches, launchpads, rockets, payloads, cores).
    3) Merge into a single table, keeping only Falcon 9 launches via an inner join
       between launches and rockets.
    4) Rename and order columns for clear display in the dashboard.
    5) Apply basic missing-value imputation.

    :return: Consolidated, cleaned SpaceX launch dataset.
    :rtype: pd.DataFrame
    """
    # Load raw data (API-first, with local snapshot fallback implemented in the loader).
    raw_data = load_spacex_raw_data()

    # Prepare entity-specific tables (normalised, minimal columns).
    launches = _prepare_launches_data(raw_data["launches"])
    launchpads = _prepare_launchpads_data(raw_data["launchpads"])
    rockets = _prepare_rockets_data(raw_data["rockets"])
    payloads = _prepare_payloads_data(raw_data["payloads"])
    cores = _prepare_cores_data(raw_data["cores"])

    # Merge data into a single master DataFrame.
    # 1) Rockets × launches (inner join keeps only Falcon 9 launches).
    merged_df = pd.merge(
        launches,
        rockets,
        how="inner",
        left_on="rocket",
        right_on="id",
    ).drop(columns=["rocket", "id"])

    # 2) Payload metadata (mass, orbit).
    merged_df = pd.merge(
        merged_df,
        payloads,
        how="inner",
        left_on="payloads",
        right_on="id",
    ).drop(columns=["payloads", "id"])

    # 3) Launchpad metadata (name and coordinates).
    merged_df = pd.merge(
        merged_df,
        launchpads,
        how="inner",
        left_on="launchpad",
        right_on="id",
    ).drop(columns=["launchpad", "id"])

    # 4) Core metadata (block/serial/reuse count).
    merged_df = pd.merge(
        merged_df,
        cores,
        how="inner",
        left_on="core_id",
        right_on="id",
    ).drop(columns=["core_id", "id"])

    # Rename columns to match dashboard labels and improve readability.
    rename_map = {
        "flight_number": "Flight Number",
        "date_utc": "Launch Date",
        "name_y": "Launch Site",
        "orbit": "Orbit",
        "mass_kg": "Payload Mass (kg)",
        "name_x": "Booster Name",
        "block": "Booster Block Version",
        "serial": "Core Serial",
        "reuse_count": "Core Reuse Count",
        "flights": "Number of Flights",
        "reused": "Reused Core",
        "gridFins": "Grid Fins",
        "legs": "Landing Legs",
        "outcome": "Landing Outcome",
        "latitude": "Launch Site Latitude",
        "longitude": "Launch Site Longitude",
    }

    # Explicit column order guarantees stable table display.
    column_order = [
        "Flight Number",
        "Launch Date",
        "Launch Site",
        "Orbit",
        "Payload Mass (kg)",
        "Booster Name",
        "Booster Block Version",
        "Core Serial",
        "Core Reuse Count",
        "Number of Flights",
        "Reused Core",
        "Grid Fins",
        "Landing Legs",
        "Landing Outcome",
        "Launch Site Latitude",
        "Launch Site Longitude",
    ]

    merged_df = merged_df.rename(columns=rename_map)[column_order]

    # Reset Flight Number to a simple 1..N sequence for clean dashboard display.
    merged_df["Flight Number"] = range(1, len(merged_df) + 1)

    # Handle missing values (exclude coordinates to avoid inventing geographic locations).
    merged_df = _fix_missing_values(
        merged_df,
        exclude_cols=["Launch Site Latitude", "Launch Site Longitude"],
    )

    return merged_df
