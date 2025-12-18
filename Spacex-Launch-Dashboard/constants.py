"""
constants.py

Shared constants for the SpaceX Launch Dashboard.

This module centralises small, stable values that are reused across multiple
parts of the application (layout, figures, callbacks). Keeping them here
avoids duplication and ensures visual and semantic consistency.
"""

# -----------------------------------------------------------------------------
# Colour palette
# -----------------------------------------------------------------------------
# Discrete colour palette used across charts and maps.
PALETTE = [
    "#1F77B4",  # blue
    "#17BECF",  # teal
    "#2CA02C",  # green (success)
    "#BCBD22",  # olive
    "#FF7F0E",  # orange
    "#D62728",  # red (failure)
    "#9467BD",  # purple
    "#7F7F7F",  # gray
]
