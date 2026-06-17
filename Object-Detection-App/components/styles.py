"""Shared visual styling for the Streamlit application."""

import streamlit as st


def apply_app_styles():
    """Applies the application's custom CSS styling."""
    st.markdown(
        """
        <style>
            :root {
                --app-accent: #4f7cff;
                --app-accent-soft: rgba(79, 124, 255, 0.12);
                --app-border: rgba(128, 128, 128, 0.22);
                --app-muted: rgba(128, 128, 128, 0.78);
            }

            .block-container {
                max-width: 1500px;
                padding-top: 4.5rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid var(--app-border);
            }

            .app-hero {
                padding: 1.35rem 1.5rem;
                margin-bottom: 1.25rem;
                border: 1px solid var(--app-border);
                border-radius: 18px;
                overflow: hidden;
                background: radial-gradient(
                    circle at top right,
                    var(--app-accent-soft),
                    transparent 45%
                );
            }

            .app-hero h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.15;
            }

            .app-hero p {
                margin: 0.55rem 0 0 0;
                color: var(--app-muted);
                font-size: 1rem;
            }

            .status-pill {
                display: inline-block;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                background: var(--app-accent-soft);
                font-size: 0.8rem;
                font-weight: 600;
            }

            div[data-testid="stMetric"] {
                border: 1px solid var(--app-border);
                border-radius: 14px;
                padding: 0.9rem 1rem;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 10px;
                min-height: 2.55rem;
                font-weight: 600;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--app-border);
                border-radius: 12px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
