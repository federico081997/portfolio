"""Sidebar controls shared by image, video, and camera modes."""

from pathlib import Path

import cv2
import streamlit as st
import torch

from core.detector import DEFAULT_MODEL_PATH, MODEL_DIRECTORY
from components.shared import get_model_and_classes


def render_sidebar():
    """
    Renders navigation and shared inference and drawing controls.

    Returns:
        Dictionary containing the selected model, mode, and configuration values.
    """
    camera_running = st.session_state.get("camera_running", False)

    # Place all controls inside the Streamlit sidebar.
    with st.sidebar:
        # Display the application name at the top of the sidebar.
        st.markdown("## VisionTrack Studio")

        # Let the user choose which type of input will be processed.
        mode = st.radio(
            "Input Mode",
            ["Upload photo", "Upload video", "Live camera"],
            captions=[
                "Detect objects in one image.",
                "Track and analyse an uploaded video.",
                "Run local webcam inference.",
            ],
        )

        lock_camera_inference = mode == "Live camera" and camera_running

        # Separate navigation from the inference settings.
        st.divider()

        # Group model and inference controls inside an expanded section.
        st.markdown("**Model and Inference**")
        # Define the YOLO26 object-detection models available to the user.
        model_options = {
            "YOLO26 Nano": MODEL_DIRECTORY / "yolo26n.pt",
            "YOLO26 Small": MODEL_DIRECTORY / "yolo26s.pt",
            "YOLO26 Medium": MODEL_DIRECTORY / "yolo26m.pt",
            "YOLO26 Large": MODEL_DIRECTORY / "yolo26l.pt",
            "YOLO26 Extra Large": MODEL_DIRECTORY / "yolo26x.pt",
        }

        # Determine which dropdown option matches the application's default model.
        default_model_label = next(
            (
                label
                for label, path in model_options.items()
                if path == str(DEFAULT_MODEL_PATH)
            ),
            "YOLO26 Nano",
        )

        # Let the user choose from supported YOLO26 model sizes.
        selected_model_label = st.selectbox(
            "YOLO Model",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(default_model_label),
            help=(
                "Nano is the fastest and lightest option. Larger models "
                "generally require more memory and processing time."
            ),
            disabled=lock_camera_inference,
        )

        # Retrieve the model filename associated with the selected label.
        model_path = model_options[selected_model_label]

        # Check whether the installed PyTorch version can access a CUDA GPU.
        cuda_available = torch.cuda.is_available()

        # Always allow automatic selection and explicit CPU inference.
        device_options = [
            "Automatic",
            "CPU",
        ]

        # Show the GPU option only when CUDA is available.
        if cuda_available:
            device_options.append("GPU 0")

        # Let the user select from the devices available on the current system.
        device_label = st.selectbox(
            "Inference Device",
            options=device_options,
            disabled=lock_camera_inference,
        )

        # Convert readable interface labels into Ultralytics device values.
        device_map = {
            "Automatic": None,
            "CPU": "cpu",
            "GPU 0": "0",
        }

        # Set the minimum confidence required for a detection to be kept.
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            step=0.01,
            disabled=lock_camera_inference,
        )

        # Set the IoU threshold used during non-maximum suppression.
        # This controls how strongly overlapping boxes are removed.
        iou_threshold = st.slider(
            "Non-Maximum Suppression Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.45,
            step=0.01,
            disabled=lock_camera_inference,
        )

        # Select the image size used internally by YOLO during inference.
        # Larger values may improve small-object detection but require
        # more processing time and memory.
        image_size = st.select_slider(
            "Inference Image Size",
            options=[320, 416, 512, 640, 768, 960, 1280],
            value=640,
            disabled=lock_camera_inference,
        )

        # Limit the number of detections returned for one image or frame.
        max_detections = st.number_input(
            "Maximum Detections",
            min_value=1,
            max_value=2000,
            value=300,
            step=10,
            disabled=lock_camera_inference,
        )

        # Define default values before attempting to load the model.
        # These values allow the rest of the interface to render even when
        # the model cannot be loaded.
        model = None
        class_names = []
        model_error = None

        try:
            # Load the model through the cached loading function and retrieve
            # the class labels supported by that model.
            with st.spinner("Loading model..."):
                model, class_names = get_model_and_classes(model_path)

            # Confirm that the model loaded successfully.
            st.success(f"Model ready · {len(class_names)} classes")

        except Exception as error:
            # Store the error message so the main application can prevent
            # processing and display additional information when needed.
            model_error = str(error)

            st.error("The selected model could not be loaded.")

        # Allow the user to restrict inference results to selected classes.
        # An empty selection means that every detected class is retained.
        selected_classes = st.multiselect(
            "Classes to Retain",
            options=class_names,
            default=[],
            placeholder="Leave empty to keep all classes",
            # Disable the control when no model classes are available.
            disabled=not class_names or lock_camera_inference,
        )

        # Separate inference controls from drawing controls.
        st.divider()

        # Group bounding-box appearance settings.
        st.markdown("**Bounding Boxes**")
        # Select how each detected object's bounding box is drawn.
        box_style = st.selectbox(
            "Box Style",
            ["Standard", "Filled", "Corner"],
        )

        box_style_map = {
            "Standard": "standard",
            "Filled": "filled",
            "Corner": "corner",
        }

        # Select the bounding-box color using Streamlit's color picker.
        box_color = st.color_picker(
            "Box Color",
            "#00ff00",
        )

        # Control the thickness of bounding-box lines.
        box_thickness = st.slider(
            "Box Thickness",
            min_value=1,
            max_value=10,
            value=2,
        )

        # Control the opacity of the box fill. This setting only applies
        # when the filled bounding-box style is selected.
        alpha = st.slider(
            "Filled-Box Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.18,
            step=0.01,
            disabled=box_style_map[box_style] != "filled",
        )

        st.divider()

        # Group label and text appearance settings.
        st.markdown("**Labels**")
        # Control which pieces of information are included in each label.
        show_label = st.checkbox(
            "Show Class Name",
            value=True,
        )

        show_confidence = st.checkbox(
            "Show Confidence",
            value=True,
        )

        # Control whether a solid background is drawn behind label text.
        text_background = st.checkbox(
            "Show Text Background",
            value=True,
        )

        # Select the color used behind the label text.
        label_background_color = st.color_picker(
            "Text Background Color",
            "#00ff00",
            disabled=not text_background,
        )

        # Select the color used for the label text.
        text_color = st.color_picker(
            "Text Color",
            "#ffffff",
        )

        # Control the OpenCV font scale used for labels.
        text_size = st.slider(
            "Text Size",
            min_value=0.3,
            max_value=2.0,
            value=0.6,
            step=0.05,
        )

        # Control the thickness of the label text strokes.
        text_thickness = st.slider(
            "Text Thickness",
            min_value=1,
            max_value=6,
            value=1,
        )

        # Select automatic label placement or a fixed position relative
        # to each object's bounding box.
        text_position = st.selectbox(
            "Text Position",
            [
                "Automatic",
                "Above left",
                "Above right",
                "Below left",
                "Below right",
                "Inside top left",
                "Inside top right",
            ],
        )

        text_position_map = {
            "Automatic": "automatic",
            "Above left": "above left",
            "Above right": "above right",
            "Below left": "below left",
            "Below right": "below right",
            "Inside top left": "inside top left",
            "Inside top right": "inside top right",
        }

        # Present readable font names to the user.
        font_name = st.selectbox(
            "Font",
            ["Complex", "Simplex", "Duplex", "Triplex"],
        )

        # Convert the selected font name into the corresponding OpenCV
        # font constant required by cv2.putText().
        font_map = {
            "Complex": cv2.FONT_HERSHEY_COMPLEX,
            "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
            "Duplex": cv2.FONT_HERSHEY_DUPLEX,
            "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
        }

        st.divider()

        # Show a compact summary of the active runtime configuration.
        st.markdown("**Runtime Details**")
        # Path(...).name displays only the model filename when a complete
        # path was entered. The original value is used as a fallback.
        st.caption(f"Model: {Path(model_path).name or model_path}")
        st.caption(f"Device: {device_label}")
        st.caption(f"Mode: {mode}")

    # Return all selected values in one configuration dictionary. The image,
    # video, and camera interfaces can reuse this shared configuration.
    return {
        "mode": mode,
        "model": model,
        "model_path": model_path,
        "model_error": model_error,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "image_size": image_size,
        "device": device_map[device_label],
        "max_detections": int(max_detections),
        "selected_classes": selected_classes or None,
        "box_color": box_color,
        "label_background_color": label_background_color,
        "text_color": text_color,
        "box_thickness": box_thickness,
        "text_size": text_size,
        "text_thickness": text_thickness,
        "alpha": alpha,
        "show_label": show_label,
        "show_confidence": show_confidence,
        "text_background": text_background,
        "box_style": box_style_map[box_style],
        "text_font": font_map[font_name],
        "text_position": text_position_map[text_position],
    }
