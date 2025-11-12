"""
constants.py

Static configuration constants for the Australia Wildfires Dashboard.

This module stores immutable values used across the application, such as
UI options for regions and available years in the dataset.

These constants are not meant to be modified at runtime. If your dataset
changes, update these values accordingly.
"""

# ------------------------------------------------------------
# Region Options
# ------------------------------------------------------------
# Each region is defined as a dictionary containing:
#   - "label": Display name in the user interface.
#   - "value": Internal code used for filtering and callbacks.
# These values correspond to Australian states and territories.
# ------------------------------------------------------------

REGIONS = (
    {"label": "New South Wales", "value": "NSW"},
    {"label": "Northern Territory", "value": "NT"},
    {"label": "Queensland", "value": "QLD"},
    {"label": "South Australia", "value": "SA"},
    {"label": "Tasmania", "value": "TAS"},
    {"label": "Victoria", "value": "VIC"},
    {"label": "Western Australia", "value": "WA"},
)


# ------------------------------------------------------------
# Year Options
# ------------------------------------------------------------
# Fixed set of years corresponding to the available dataset range.
# If additional data is added later, this list should be updated.
# ------------------------------------------------------------

YEARS = tuple(range(2005, 2021))
