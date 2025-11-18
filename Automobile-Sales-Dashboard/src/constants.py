"""
constants.py

Static configuration constants for the Automobile Sales Statistics Dashboard.

This module stores immutable values used across the application, such as
UI options for report type and available years in the dataset.

These constants are not meant to be modified at runtime. If your dataset
changes, update these values accordingly.
"""

# ------------------------------------------------------------
# Report Type Options
# ------------------------------------------------------------
# Fixed set of options that the user can select from the
# dropdown menu
# ------------------------------------------------------------

REPORT_TYPES = ("Yearly Statistics", "Recession Period Statistics")


# ------------------------------------------------------------
# Year Options
# ------------------------------------------------------------
# Fixed set of years corresponding to the available dataset range.
# If additional data is added later, this list should be updated.
# ------------------------------------------------------------

YEARS = tuple(range(1980, 2024))

# ------------------------------------------------------------
# Month Options
# ------------------------------------------------------------
# Fixed set of months.
# ------------------------------------------------------------

MONTHS = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}


# ------------------------------------------------------------
# Vehicle Types
# ------------------------------------------------------------
# Fixed set of vehicle types with the corresponding colors.
# If additional data is added later, this list should be updated.
# ------------------------------------------------------------

VEHICLE_TYPES = {
    "Executive car": "#4B6CB7",
    "Medium family car": "#FF7F0E",
    "Small family car": "#2CA02C",
    "Sports car": "#D62728",
    "Supermini car": "#9467BD",
}
