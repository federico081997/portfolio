import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --primary-blue: #2563EB;
            --primary-blue-dark: #1E40AF;
            --primary-blue-light: #DBEAFE;
            --primary-blue-soft: #EFF6FF;
            --text-dark: #1F2937;
            --muted-text: #6B7280;
            --border-blue: #93C5FD;
            --surface-white: #FFFFFF;
        }

        html, body, [class*="css"] {
            color: var(--text-dark);
        }

        .stApp {
            background-color: #F8FAFC;
        }

        section.main > div {
            max-width: 100%;
            overflow-x: hidden;
        }

        section[data-testid="stSidebar"] {
            background-color: var(--primary-blue-soft);
            border-right: 1px solid var(--border-blue);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
            color: var(--text-dark) !important;
        }

        div.stButton > button {
            background-color: var(--primary-blue) !important;
            color: white !important;
            border: 1px solid var(--primary-blue) !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out !important;
        }

        div.stButton > button:hover {
            background-color: var(--primary-blue-dark) !important;
            border-color: var(--primary-blue-dark) !important;
            color: white !important;
            transform: translateY(-1px);
        }

        div[data-baseweb="select"] > div {
            border-color: var(--border-blue) !important;
            border-radius: 10px !important;
            background-color: white !important;
        }

        span[data-baseweb="tag"] {
            background-color: var(--primary-blue-light) !important;
            color: var(--primary-blue-dark) !important;
            border: 1px solid var(--border-blue) !important;
            border-radius: 8px !important;
        }

        .phone-card-wrapper {
            padding: 0.25rem 0.25rem 2rem 0.25rem;
        }

        .phone-card-title {
            height: 78px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: var(--text-dark);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.75rem;
        }

        .phone-image-box {
            height: 310px;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: transparent;
            margin-bottom: 1.25rem;
        }

        .phone-image-box img {
            max-height: 285px;
            max-width: 100%;
            object-fit: contain;
            border-radius: 12px;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #DBEAFE;
            border-radius: 12px;
            padding: 1rem;
            min-height: 135px;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--primary-blue-dark) !important;
            font-weight: 600;
        }

        details {
            border: 1px solid #BFDBFE !important;
            border-radius: 10px !important;
            background-color: white !important;
            margin-top: 1rem;
        }

        details summary {
            color: var(--primary-blue-dark) !important;
            font-weight: 600 !important;
        }

        div[data-testid="stPlotlyChart"] {
            border: 1px solid #DBEAFE;
            border-radius: 12px;
            padding: 0.5rem;
            background-color: white;
            overflow-x: auto;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #DBEAFE;
            border-radius: 12px;
            background-color: white;
        }

        h1, h2, h3, h4 {
            color: var(--text-dark);
        }

        .section-subtitle {
            color: var(--muted-text);
            font-size: 1rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
