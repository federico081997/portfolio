"""
spacex_data_loader.py

Utilities for retrieving raw SpaceX data for the application.

This module provides a single interface to access SpaceX launch-related data
either from the public SpaceX REST API or from local static JSON snapshots.
It is intended to be the single source of truth for data acquisition, so that
other parts of the project do not need to know whether the data originates
from the live API or an offline fallback.

The main entry point is :func:`load_spacex_raw_data`, which returns a dictionary
containing the raw JSON payloads for launches, launchpads, rockets, payloads
and cores. Internally, the module:

* Tries to fetch all datasets from the SpaceX API.
* On failure (network errors, HTTP errors, JSON decoding issues), falls back
  to reading pre-downloaded JSON files from the ``static_data`` directory.

The structure of the returned dictionary is:

    {
        "launches": [...],
        "launchpads": [...],
        "rockets": [...],
        "payloads": [...],
        "cores": [...],
    }

with each value being a list of dictionaries as returned by the SpaceX API
(or the corresponding static JSON files).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import requests

# =============================================================================
# API configuration
# =============================================================================

# Base URL for the public SpaceX REST API.
SPACEX_API_BASE = "https://api.spacexdata.com"

# Endpoints used by the dashboard. Keys must match the expected output dict keys.
ENDPOINTS = {
    "launches": "/v5/launches/",
    "launchpads": "/v4/launchpads/",
    "rockets": "/v4/rockets/",
    "payloads": "/v4/payloads/",
    "cores": "/v4/cores/",
}

# =============================================================================
# Static fallback configuration
# =============================================================================

# Location of local JSON snapshots used when the API is not reachable.
# The directory is resolved relative to this module to keep behaviour consistent
# across different working directories.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "static_data"

# =============================================================================
# Internal helpers
# =============================================================================


def _fetch_json(url: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """
    Send a GET request to the given URL and return the decoded JSON payload.

    This function raises an exception if the request fails, if the server returns a
    non-successful HTTP status code, or if the response body cannot be decoded as JSON.

    :param url: Endpoint URL expected to return a JSON response.
    :type url: str
    :param timeout: Maximum time to wait for a response, in seconds.
    :type timeout: int, optional
    :return: Parsed JSON payload returned by the endpoint.
    :rtype: List[Dict[str, Any]]
    :raises requests.HTTPError: If the response status code indicates an error.
    :raises requests.RequestException: If a network-related error occurs.
    :raises ValueError: If the response body cannot be decoded as JSON.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # `requests` raises ValueError for JSON decoding failures via response.json()
    return response.json()


def _load_static_data(filename: str) -> List[Dict[str, Any]]:
    """
    Load static JSON data from a file in the ``static_data`` directory.

    :param filename: Name of the JSON file to load.
    :type filename: str
    :return: Parsed JSON data from the file.
    :rtype: List[Dict[str, Any]]
    :raises FileNotFoundError: If the specified file does not exist.
    :raises json.JSONDecodeError: If the file content cannot be decoded as JSON.
    """
    filepath = DATA_PATH / filename

    # Fail loudly if the snapshot is missing; this indicates a packaging/deployment issue.
    if not filepath.exists():
        raise FileNotFoundError(f"Static data file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_all_from_api() -> Dict[str, Any]:
    """
    Fetch all required SpaceX datasets from the live API.

    :return: Dictionary containing all fetched datasets.
    :rtype: Dict[str, Any]
    """
    data = {}

    # Fetch each dataset independently so the output dict shape is stable.
    for key, endpoint in ENDPOINTS.items():
        url = SPACEX_API_BASE + endpoint
        data[key] = _fetch_json(url)

    return data


def _load_all_from_static() -> Dict[str, Any]:
    """
    Load all required SpaceX datasets from local static JSON files.

    :return: Dictionary containing all loaded datasets.
    :rtype: Dict[str, Any]
    """
    data = {}

    # File names are expected to be "<key>.json" for each key in ENDPOINTS.
    for key in ENDPOINTS.keys():
        filename = f"{key}.json"
        data[key] = _load_static_data(filename)

    return data


# =============================================================================
# Public API
# =============================================================================


def load_spacex_raw_data() -> Dict[str, Any]:
    """
    Load SpaceX raw data from the live API, with a static-file fallback.

    The function attempts to fetch all datasets from the SpaceX API. If any
    request fails due to HTTP/network/JSON-decoding issues, it falls back to
    local JSON snapshots stored under ``static_data``.

    :return: Dictionary containing all SpaceX raw data with the following structure:
        {
            "launches": [...],
            "launchpads": [...],
            "rockets": [...],
            "payloads": [...],
            "cores": [...],
        }
    :rtype: Dict[str, Any]
    """
    # Prefer live data when possible to keep the dashboard current.
    try:
        return _fetch_all_from_api()

    # Fall back to static snapshots if *any* API fetch fails.
    # This keeps the application usable offline and makes demos reproducible.
    except (requests.HTTPError, requests.RequestException, ValueError):
        return _load_all_from_static()
