"""
components.py

Reusable UI components for the SpaceX Launch Dashboard.

This module provides helper functions that construct common layout elements
such as titles, text blocks, cards, dropdowns, tables, sliders, radio items,
and responsive grid structures.

The goal is to keep layout/callback code concise by centralising common UI
patterns in small, well-documented builders.
"""

from typing import Any, List, Optional, Sequence

import pandas as pd
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc


# =============================================================================
# Public APIs
# =============================================================================


def display_title(
    text: Optional[str] = None,
    className: str = "page-title",
    **kwargs: Any,
) -> html.H1:
    """
    Create a level-one title component.

    :param text: The heading text to display.
    :type text: str, optional
    :param className: CSS class applied for styling.
    :type className: str
    :param kwargs: Additional keyword arguments passed to `html.H1`.
    :type kwargs: Any
    :return: A styled `html.H1` element.
    :rtype: dash.html.H1
    """
    return html.H1(text, className=className, **kwargs)


def display_text(
    text: Optional[str] = None,
    className: str = "page-description",
    **kwargs: Any,
) -> html.P:
    """
    Create a paragraph text component.

    :param text: Content displayed in the text block.
    :type text: str, optional
    :param className: CSS class applied for styling.
    :type className: str
    :param kwargs: Additional keyword arguments passed to `html.P`.
    :type kwargs: Any
    :return: A styled `html.P` element.
    :rtype: dash.html.P
    """
    return html.P(text, className=className, **kwargs)


def display_card(
    card_body: Any = None,
    header: Any = None,
    footer: Any = None,
    className: str = "card-style",
    **kwargs: Any,
) -> dbc.Card:
    """
    Create a Bootstrap card with optional header and footer.

    :param card_body: Main content of the card.
    :type card_body: Any
    :param header: Optional header content (placed inside `dbc.CardHeader`).
    :type header: Any
    :param footer: Optional footer content (placed inside `dbc.CardFooter`).
    :type footer: Any
    :param className: CSS class applied to the card for styling.
    :type className: str
    :param kwargs: Additional keyword arguments passed to `dbc.Card`.
    :type kwargs: Any
    :return: A `dbc.Card` component with the specified structure.
    :rtype: dash_bootstrap_components.Card
    """
    children = []

    # Header and footer are genuinely optional; omit them entirely when absent.
    if header is not None:
        children.append(dbc.CardHeader(header))

    children.append(dbc.CardBody(card_body))

    if footer is not None:
        children.append(dbc.CardFooter(footer))

    return dbc.Card(children=children, className=className, **kwargs)


def display_dropdown(
    options: Sequence[Any],
    placeholder: Optional[str] = None,
    value: Any = None,
    className: str = "dropdown-style",
    **kwargs: Any,
) -> dcc.Dropdown:
    """
    Create a styled dropdown selection component.

    :param options: Dropdown options accepted by `dcc.Dropdown`.
    :type options: Sequence[Any]
    :param placeholder: Text shown when no option is selected.
    :type placeholder: str, optional
    :param value: Initial selected value.
    :type value: Any
    :param className: CSS class for consistent styling.
    :type className: str
    :param kwargs: Additional keyword arguments passed to `dcc.Dropdown`.
    :type kwargs: Any
    :return: A configured `dcc.Dropdown` component.
    :rtype: dash.dcc.Dropdown
    """
    return dcc.Dropdown(
        options=options,
        placeholder=placeholder,
        value=value,
        className=className,
        **kwargs,
    )


def create_grid(rows: int, cols: int, items: List[Any]) -> List[dbc.Row]:
    """
    Arrange components into a responsive grid (rows × cols) using Bootstrap.

    Missing items are filled with empty `html.Div()` placeholders; extra items
    are discarded. Each grid cell is rendered as a `dbc.Col` with:
    - `xs=12` for full width on small screens
    - `md=12 // cols` for equal splits on medium+ screens

    :param rows: Number of grid rows.
    :type rows: int
    :param cols: Number of grid columns.
    :type cols: int
    :param items: Components to place in the grid (row-major order).
    :type items: List[Any]
    :return: A list of `dbc.Row` elements representing the grid.
    :rtype: List[dash_bootstrap_components.Row]
    :raises ValueError: If rows or cols are not positive integers.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("create_grid requires positive 'rows' and 'cols' values.")

    total_cells = rows * cols

    # Copy to avoid mutating the caller's list.
    cells = list(items)

    # Ensure we have exactly the number of cells required.
    if len(cells) < total_cells:
        cells.extend([html.Div()] * (total_cells - len(cells)))
    else:
        cells = cells[:total_cells]

    # Ensure each column has a valid Bootstrap width (1..12).
    md_width = max(1, 12 // cols)

    grid = []
    idx = 0

    for _ in range(rows):
        row_container = []
        for _ in range(cols):
            row_container.append(dbc.Col(cells[idx], xs=12, md=md_width))
            idx += 1
        grid.append(dbc.Row(row_container))

    return grid


def display_tabs(
    labels: List[Any],
    active_tab: int = 0,
    className: str = "tabs-style",
    **kwargs: Any,
) -> dbc.Tabs:
    """
    Create a Bootstrap tabs component.

    Tabs are assigned stable tab IDs of the form `tab-0`, `tab-1`, ... which
    are used by callbacks to control which content is displayed.

    :param labels: A list of tab labels (strings or Dash components).
    :type labels: List[Any]
    :param active_tab: Index of the initially active tab.
    :type active_tab: int
    :param className: CSS class applied for styling.
    :type className: str
    :param kwargs: Additional keyword arguments passed to `dbc.Tabs`.
    :type kwargs: Any
    :return: A configured `dbc.Tabs` component.
    :rtype: dash_bootstrap_components.Tabs
    """
    # Clamp the active tab index to a valid range so the UI never breaks if a
    # caller passes an out-of-range value.
    if labels:
        active_tab = max(0, min(active_tab, len(labels) - 1))
    else:
        active_tab = 0

    tab_children = [
        dbc.Tab(label=label, tab_id=f"tab-{i}") for i, label in enumerate(labels)
    ]

    return dbc.Tabs(
        tab_children,
        active_tab=f"tab-{active_tab}",
        className=className,
        **kwargs,
    )


def display_table(
    df: pd.DataFrame, className: str = "table-wrap", **kwargs: Any
) -> html.Div:
    """
    Create a Dash DataTable wrapped in a scrollable container.

    The surrounding `html.Div` uses by default the `table-wrap` CSS class to keep large
    tables readable within cards by allowing horizontal scrolling.

    :param df: DataFrame to display.
    :type df: pd.DataFrame
    :param className: CSS class applied for styling.
    :type className: str
    :param kwargs: Additional keyword arguments passed to `dash_table.DataTable`.
    :type kwargs: Any
    :return: A table container with a `dash_table.DataTable` inside.
    :rtype: dash.html.Div
    """
    return html.Div(
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in df.columns],
            **kwargs,
        ),
        className=className,
    )


def display_range_slider(
    min: int,
    max: int,
    step: int,
    value: Any = None,
    **kwargs: Any,
) -> dcc.RangeSlider:
    """
    Create a range slider component.

    Note: the parameter names `min` and `max` shadow Python built-ins; they are
    kept to match Dash's component API and to avoid changing calling code.

    :param min: Minimum value for the slider.
    :type min: int
    :param max: Maximum value for the slider.
    :type max: int
    :param step: Step size for the slider.
    :type step: int
    :param value: Initial selected range, typically a two-item list like [low, high].
    :type value: Any
    :param kwargs: Additional keyword arguments passed to `dcc.RangeSlider`.
    :type kwargs: Any
    :return: A configured `dcc.RangeSlider`.
    :rtype: dash.dcc.RangeSlider
    """
    return dcc.RangeSlider(
        min=min,
        max=max,
        step=step,
        value=value,
        **kwargs,
    )


def display_radioItems(options: Any, value: Any, **kwargs: Any) -> dcc.RadioItems:
    """
    Create a radio-items selection component.

    :param options: Options accepted by `dcc.RadioItems`.
    :type options: Any
    :param value: Initial selected value.
    :type value: Any
    :param kwargs: Additional keyword arguments passed to `dcc.RadioItems`.
    :type kwargs: Any
    :return: A configured `dcc.RadioItems` component.
    :rtype: dash.dcc.RadioItems
    """
    return dcc.RadioItems(options=options, value=value, **kwargs)
