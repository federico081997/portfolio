"""
data_io.py

Lightweight I/O utilities for the Automobile Sales Statistics Dashboard.

This module is exposes a single function, `read_data`, which loads
the cleaned automobile sales CSV into a pandas DataFrame.
"""

from pathlib import Path
import pandas as pd


def read_data(csv_path: str | Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    csv_path : str | pathlib.Path
        Path to the CSV file (e.g., 'data/Automobile_Sales.csv').
        Both `str` paths and `Path` objects are supported.
    encoding : str, default "ISO-8859-1"
        Text encoding used to decode the file. The default matches the
        original dataset’s encoding.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the full dataset. No mutation or
        filtering is performed at load time.
    """
    return pd.read_csv(csv_path, encoding=encoding)
