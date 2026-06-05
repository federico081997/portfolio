"""
Authentication Module

Provides a secure, session-state-based login mechanism.
Utilizes Python's native hmac library for secure string comparison to
prevent timing attacks. Credentials are read securely from Streamlit secrets.
"""

import streamlit as st
import hmac


def check_password() -> bool:
    """
    Renders a login form and verifies credentials.
    Returns `True` if the user is authenticated, `False` otherwise.
    """

    def password_entered():
        """Callback to check the password against secrets."""
        # Retrieve secure credentials from .streamlit/secrets.toml
        valid_username = st.secrets["credentials"]["username"]
        valid_password = st.secrets["credentials"]["password"]

        # Use hmac.compare_digest to prevent timing attacks
        is_user_correct = hmac.compare_digest(
            st.session_state["username"], valid_username
        )
        is_pass_correct = hmac.compare_digest(
            st.session_state["password"], valid_password
        )

        if is_user_correct and is_pass_correct:
            st.session_state["password_correct"] = True
            # Delete credentials from session state for security
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # 1. If already authenticated, return True immediately
    if st.session_state.get("password_correct", False):
        return True

    # 2. If not authenticated, render the Login UI
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center;'>🔒 Secure Access</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>Please log in to access the ML Studio.</p>",
        unsafe_allow_html=True,
    )

    # Center the login form using columns
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")

            # The on_click triggers the validation logic before the page reruns
            submit = st.form_submit_button(
                "Log In", use_container_width=True, on_click=password_entered
            )

        if (
            "password_correct" in st.session_state
            and not st.session_state["password_correct"]
        ):
            st.error("🚫 Invalid username or password.")

    return False
