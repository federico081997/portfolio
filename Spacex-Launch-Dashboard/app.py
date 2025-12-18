"""
app.py

Entry point for the SpaceX Launch Dashboard.

This module is responsible for:
- assembling the master SpaceX dataset used throughout the dashboard,
- precomputing machine learning model results at startup (for responsiveness),
- creating the Dash application instance,
- building the UI layout, and
- registering all callbacks that drive interactivity.
"""

import dash
import dash_bootstrap_components as dbc

from callbacks import register_callbacks
from data_wrangling import assemble_master_data
from layout import build_layout
from modeling import execute_models


# =============================================================================
# Main app entry point
# =============================================================================


def create_app() -> dash.Dash:
    """
    Create and configure the Dash application.

    :return: A fully configured Dash app instance ready to be served.
    :rtype: dash.Dash
    """
    # Step 1: Build the dataset once at startup.
    # This keeps callbacks fast and ensures the app operates on a consistent snapshot.
    spacex_data = assemble_master_data()

    # Step 2: Train / evaluate models once at startup.
    # Model training can be slow; caching results avoids recomputation on user interaction.
    model_cache = execute_models(spacex_data)

    # Step 3: Create the Dash application instance.
    # Bootstrap theme is provided via external stylesheets.
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.LUX],
    )

    # Step 4: Assemble the UI tree (tabs, cards, placeholders for dynamic content).
    app.layout = build_layout()

    # Step 5: Register all interactive callbacks.
    # The callbacks receive the dataset and model cache so they can render quickly.
    register_callbacks(app, spacex_data, model_cache)

    return app


# Expose a module-level `app` for Dash servers (and deployment platforms) to discover.
app = create_app()

if __name__ == "__main__":
    # Local development / direct execution entry point.
    app.run(debug=False)
