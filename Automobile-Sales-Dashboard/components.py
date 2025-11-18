"""
components.py

Reusable UI components for the Automobile Sales Statistics Dashboard.

This module provides helper functions that construct common layout elements
such as titles, text blocks, cards, dropdowns, and responsive grid structures.
"""

from typing import Any, Iterable, List
from dash import html, dcc
import dash_bootstrap_components as dbc


def display_title(
    text: str = "",
    className: str = "page-title",
    id: str = None,
) -> html.H1:
    """Create a level-one title component.

    Parameters:
        text: The heading text to display.
        className: CSS class for styling.
        id: Component ID for Dash interactivity.

    Returns:
        An `html.H1` element styled as a page title.
    """
    # Build props dynamically so empty IDs are not passed to Dash
    props = {"className": className}
    if id is not None:
        props["id"] = id

    return html.H1(text, **props)


def display_text(
    text: str = "",
    className: str = "page-description",
    id: str = None,
) -> html.P:
    """Create a paragraph component.

    Parameters:
        text: Content displayed in the text block.
        className: CSS class for styling.
        id: Component ID for Dash interactivity.

    Returns:
        An `html.P` element containing arbitrary descriptive text.
    """
    props = {"className": className}
    if id is not None:
        props["id"] = id

    return html.P(text, **props)


def display_card(
    card_body: str,
    header_text: str = None,
    footer_text: str = None,
    className: str = "card-style",
) -> dbc.Card:
    """Create a card with optional header and footer sections.

    Parameters:
        card_body: Main content of the card.
        header_text: Header component.
        footer_text: Footer component.
        className: CSS class applied to the card for styling.

    Returns:
        A `dbc.Card` component with the specified structure and style.
    """
    # Conditional header/footer avoids adding empty components
    return dbc.Card(
        [
            dbc.CardHeader(header_text) if header_text is not None else None,
            dbc.CardBody(card_body),
            dbc.CardFooter(footer_text) if footer_text is not None else None,
        ],
        className=className,
    )


def display_dropdown(
    options: Iterable[Any],
    placeholder: str = None,
    id: str = None,
    value: Any = None,
    className: str = "dropdown-style",
) -> dcc.Dropdown:
    """Create a styled dropdown selection component.

    Parameters:
        options: Iterable of dropdown options.
        placeholder: Text shown when no item is selected.
        id: Component ID used by callbacks.
        value: Initial default value.
        className: CSS class for consistent styling.

    Returns:
        A `dcc.Dropdown` element for user selection.
    """
    return dcc.Dropdown(
        options=options,
        id=id,
        placeholder=placeholder,
        value=value,
        className=className,
    )


def create_grid(
    rows: int,
    cols: int,
    items: List[Any],
) -> List[dbc.Row]:
    """Arrange components into a responsive (rows × cols) grid.

    Missing items are filled with empty Div placeholders. Components are
    wrapped in responsive Bootstrap columns that behave well on different
    screen sizes.

    Parameters:
        rows: Number of grid rows.
        cols: Number of grid columns.
        items: List of components to place in the grid.

    Returns:
        A list of `dbc.Row` elements containing the structured grid.
    """
    total_cells = rows * cols

    # Ensure the item list has exactly the required number of elements
    if len(items) < total_cells:
        # Fill with empty components
        items = items + [html.Div()] * (total_cells - len(items))
    else:
        # Extra elements are discarded
        items = items[:total_cells]

    # List that will contain the rows components
    grid: List[dbc.Row] = []
    # Iterator
    idx = 0

    for _ in range(rows):
        row_container: List[dbc.Col] = []

        for _ in range(cols):
            child = items[idx]

            # xs=12 : full width on small screens
            # md=12 // cols : equal split on medium+ screens
            # This ensures a clean, adaptive layout without manual tuning.
            row_container.append(dbc.Col(child, xs=12, md=12 // cols))

            idx += 1

        grid.append(dbc.Row(row_container))

    return grid
