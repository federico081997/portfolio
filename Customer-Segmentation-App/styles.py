import streamlit as st


def inject_custom_css():
    """
    Injects custom CSS to elevate the UI into a polished dashboard.
    Hides default Streamlit branding, optimizes margins, and styles
    the metric cards with a modern blue aesthetic.
    """
    st.markdown(
        """
    <style>
        /* Hide default Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Optimize top padding for a cleaner look */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Style metric cards for a professional, dashboard-like appearance */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3b82f6; /* Matching Primary Blue accent */
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }

        /* Add a subtle lift effect when hovering over metrics */
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
        }

        /* Ensure headers match the dark slate text color */
        h1, h2, h3 {
            color: #0f172a !important;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }

        /* Clean up the dataframe rendering */
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            overflow: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
