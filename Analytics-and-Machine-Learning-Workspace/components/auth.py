"""Authentication utilities for the Streamlit application.

This module provides a session-state-based login mechanism for protecting the
Analytics and Machine Learning Workspace. User credentials are loaded from Streamlit
secrets and compared using ``hmac.compare_digest`` to reduce exposure to
timing-based string comparison attacks.

Authentication status is stored in ``st.session_state`` so that users remain
authenticated across Streamlit reruns during the active session.
"""

import hmac

import streamlit as st


def check_password() -> bool:
    """Render the login form and validate the user's credentials.

    The function displays a centered login form when the user is not already
    authenticated. Submitted credentials are compared against the values stored
    in Streamlit secrets. If authentication succeeds, the credential fields are
    removed from session state and the authenticated status is preserved.

    Returns:
        True if the user is authenticated; otherwise, False.
    """

    def password_entered() -> None:
        """Validate submitted credentials against Streamlit secrets."""
        valid_username = st.secrets["credentials"]["username"]
        valid_password = st.secrets["credentials"]["password"]

        is_user_correct = hmac.compare_digest(
            st.session_state["username"],
            valid_username,
        )
        is_pass_correct = hmac.compare_digest(
            st.session_state["password"],
            valid_password,
        )

        if is_user_correct and is_pass_correct:
            st.session_state["password_correct"] = True
            del st.session_state["username"]
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center;'>🔒 Secure Access</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<p style='text-align: center; color: gray;'>"
            "Please log in to access the ML Studio."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    _, form_column, _ = st.columns([1, 1.5, 1])

    with form_column:
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button(
                "Log In",
                type="primary",
                width="stretch",
                on_click=password_entered,
            )

        if (
            "password_correct" in st.session_state
            and not st.session_state["password_correct"]
        ):
            st.error("❌ Invalid username or password.")

    return False
