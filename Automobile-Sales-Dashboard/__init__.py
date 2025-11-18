"""
Automobile Sales Statistics Dashboard Package

This package provides a structured framework for loading data, constructing
the application layout, computing analytical results, and registering all
callbacks for the interactive dashboard.

"""

__version__ = "0.1.0"

# Package metadata
__all__ = [
    "REPORT_TYPES",
    "YEARS",
    "MONTHS",
    "VEHICLE_TYPES",
    "display_title",
    "display_text",
    "display_card",
    "display_dropdown",
    "create_grid",
    "read_data",
    "build_layout",
    "compute_yearly_info",
    "compute_recession_info",
    "wrap_label",
    "register_callbacks",
    "plot",
]

# Export used symbols for convenient top-level imports
from .constants import REPORT_TYPES, YEARS, MONTHS, VEHICLE_TYPES  # noqa: F401
from .components import (
    display_title,
    display_text,
    display_card,
    display_dropdown,
    create_grid,
)  # noqa: F401
from .data_io import read_data  # noqa: F401
from .layout import build_layout  # noqa: F401
from .logic import compute_yearly_info, compute_recession_info, wrap_label  # noqa: F401
from .callbacks import register_callbacks  # noqa: F401
from .figures import plot
