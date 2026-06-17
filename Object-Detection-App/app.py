"""
Main entry point for the Streamlit object-detection application.

The page layout, controls, and mode-specific logic live in separate component
modules so this file remains intentionally small.
"""

import streamlit as st

from components.camera_mode import render_camera_mode
from components.image_mode import render_image_mode
from components.shared import (
    initialize_session_state,
    render_app_header,
    reset_camera_state,
)
from components.sidebar import render_sidebar
from components.styles import apply_app_styles
from components.video_mode import render_video_mode

# Configure the Streamlit page before rendering other interface elements.
st.set_page_config(
    page_title="VisionTrack Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply the application's custom CSS styles.
apply_app_styles()

# Create the session-state values used by all application modes.
initialize_session_state()

# Display the application title and description.
render_app_header()

# Render the sidebar and collect the selected configuration.
app_config = render_sidebar()

# Release the webcam and clear camera-specific state whenever the user
# switches away from live-camera mode.
if app_config["mode"] != "Live camera":
    reset_camera_state(release_capture=True)

# Render the workflow selected in the sidebar.
if app_config["mode"] == "Upload photo":
    render_image_mode(app_config)

elif app_config["mode"] == "Upload video":
    render_video_mode(app_config)

else:
    render_camera_mode(app_config)
