"""
Australia Wildfires Dashboard (package)

This package contains the modular components of the Dash application, including:
- Constants and UI options (`constants`)
- Data loading utilities (`data_io`)
- Domain logic for filtering/aggregation (`logic`)
- Figure builders for Plotly charts (`figures`)
- Layout factory for the Dash UI (`layout`)
- Callback registrations wiring inputs to outputs (`callbacks`)

The package exposes a small, curated public API to make imports from `app.py`
and external scripts straightforward and readable. Importing from the package
root is supported, for example:

    from src import (
        REGIONS,
        YEARS,
        read_data,
        build_layout,
        compute_info,
        generate_donut,
        generate_bar,
        register_callbacks,
    )
"""

# Package metadata
__all__ = [
    "REGIONS",
    "YEARS",
    "read_data",
    "build_layout",
    "compute_info",
    "generate_donut",
    "generate_bar",
    "register_callbacks",
]
__version__ = "0.1.0"

# Export used symbols for convenient top-level imports
from .constants import REGIONS, YEARS                # noqa: F401
from .data_io import read_data                       # noqa: F401
from .layout import build_layout                     # noqa: F401
from .logic import compute_info                      # noqa: F401
from .figures import generate_donut, generate_bar      # noqa: F401
from .callbacks import register_callbacks            # noqa: F401