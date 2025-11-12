"""
app.py 

Australian Wildfires Dashboard Application.

Responsibilities
---------------
- Read the cleaned wildfire dataset from disk.
- Build the Dash layout using pre-defined UI options.
- Register all callbacks that connect inputs to outputs.
- Launch the Dash development server.

Notes
-----
- The data path is resolved relative to the project root so the app runs
  consistently when cloned from GitHub.
"""

from pathlib import Path
import dash
from .constants import REGIONS, YEARS
from .data_io import read_data
from .layout import build_layout
from .callbacks import register_callbacks

# Resolve data path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Historical_Wildfires.csv"

# Load data
wildfire_data = read_data(DATA_PATH)

# Create app and layout
app = dash.Dash(__name__)
app.layout = build_layout(REGIONS, YEARS)

# Wire callbacks
register_callbacks(app, wildfire_data)

if __name__ == "__main__":
    app.run()