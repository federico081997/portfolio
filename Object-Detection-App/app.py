import streamlit as st

# Import our custom modular components
from ui.sidebar import render_sidebar
from ui.views import render_image_view, render_video_view
from models.detector import load_model

def main():
    """
    Main execution entry point for the Streamlit application.
    Initializes the UI, loads the machine learning model, and routes user
    interactions to the appropriate component views.
    """
    # 1. Global Page Configuration (Must be the first Streamlit command)
    st.set_page_config(
        page_title="Object Detection & Tracking",
        page_icon="👁️",
        layout="wide"
    )

    # 2. Render the sidebar control panel and capture the user's configurations
    settings = render_sidebar()

    # 3. Load the ML model
    # Because of our @st.cache_resource decorator in detector.py, this only
    # executes once. On subsequent interactions, it instantly returns the cached model.
    with st.spinner("Loading AI Model Weights..."):
        model = load_model()

    # 4. Application Routing
    # Direct the data flow based on the user's selected activity
    if settings["activity"] == "Upload Image":
        render_image_view(model, settings)

    elif settings["activity"] == "Upload Video":
        render_video_view(model, settings)

    elif settings["activity"] == "Use Live Camera":
        # Placeholder for our highly complex WebRTC implementation
        st.warning("Live Camera module is pending integration. Please use Image or Video modes.")
        st.info("Check back once src/utils/webrtc_stream.py is complete.")

if __name__ == "__main__":
    main()
