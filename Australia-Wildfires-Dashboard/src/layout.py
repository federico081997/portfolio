"""
layout.py

Dash layout factory.

This module exposes a single factory function, `build_layout`, which returns the
root Dash layout for the Australian Wildfires Dashboard. The function is pure
(i.e., it has no side effects) and relies solely on the `regions` and `years`
collections provided by the caller.
"""

from dash import html, dcc


def build_layout(regions, years):
    """
    Create and return the root Dash layout.

    Parameters
    ----------
    regions : list of `dict`
        Options for the `dcc.RadioItems` control. Each element should be a dict
        with keys compatible with Dash options, e.g.:
        `{"label": "New South Wales", "value": "NSW"}`.
    years : list of `int`
        Options for the `dcc.Dropdown` control: a list of unique
        year values (e.g., `[2018, 2019, 2020, ...]`). The last element is
        used as the default selection.

    Returns
    -------
    dash.html.Div
        The root layout component to be assigned to `app.layout`.

    Notes
    -----
    - The component IDs referenced here (`region-items`, `year-items`,
      `pie-title`, `bar-title`, `pie-chart`, `bar-chart`) are used by callbacks
      in `callbacks.py`. If you change an ID in this file, update the
      corresponding callbacks accordingly.
    """

    # Root container: page title, controls pane, and charts pane.
    return html.Div(
        children=[
            # Page title (centered)
            html.H1(
                "Australian Wildfires Dashboard",
                style={"textAlign": "center", "color": "#503D36", "font-size": 30},
            ),
            # Controls pane: region radio + year dropdown
            html.Div(
                children=[
                    html.Label("Region", style={"font-size": 20}),
                    html.Br(),
                    html.Br(),
                    dcc.RadioItems(
                        id="region-items",
                        options=regions,
                        value=regions[0]["value"],
                        labelStyle={
                            "display": "inline-flex",
                            "margin-right": "16px",
                            "align-items": "center",
                        },
                        inputStyle={"margin-right": "8px", "verticalAlign": "middle"},
                    ),
                    html.Br(),
                    html.Label("Year", style={"font-size": 20}),
                    html.Br(),
                    html.Br(),
                    dcc.Dropdown(
                        id="year-items",
                        options=years,
                        value=years[-1],
                        style={"width": "50%"},
                    ),
                ]
            ),
            # Charts pane: two columns, each with a title and a graph
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H3(
                                id="pie-title",
                                style={"textAlign": "center", "marginBottom": "8px"},
                            ),
                            dcc.Graph(id="pie-chart"),
                        ],
                        style={"width": "50%", "padding": "16px"},
                    ),
                    html.Div(
                        children=[
                            html.H3(
                                id="bar-title",
                                style={"textAlign": "center", "marginBottom": "8px"},
                            ),
                            dcc.Graph(id="bar-chart"),
                        ],
                        style={"width": "50%", "padding": "16px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "flex-start",
                    "gap": "16px",
                },
            ),
        ]
    )
