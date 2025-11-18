"""
app.py

Entry point for the Automobile Sales Statistics Dashboard.

This module initializes the Dash application, loads the underlying dataset,
constructs the layout, and registers all callbacks required to drive the
interactive visualizations.
"""

from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from constants import REPORT_TYPES, YEARS
from data_io import read_data
from layout import build_layout
from callbacks import register_callbacks

# Resolve data path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Automobile_Sales.csv"

# Load data
automobile_data = read_data(DATA_PATH)

# Create app and layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.layout = build_layout(REPORT_TYPES, YEARS)

# Wire callbacks
register_callbacks(app, automobile_data)

if __name__ == "__main__":
    app.run(debug=True)