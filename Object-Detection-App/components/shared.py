"""Shared Streamlit state, model, metrics, and export helpers."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.detector import get_model_class_names, load_model


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path):
    """
    Loads and caches a YOLO model.

    Args:
        model_path: Path or name of the YOLO model.

    Returns:
        Loaded YOLO model.
    """
    # Convert Path objects into strings because the model-loading function
    # expects a filesystem path or model name in string format.
    model_path = str(model_path)

    # Load and return the model. Streamlit stores the returned model in its
    # resource cache and reuses it when this function is called again with
    # the same model path.
    return load_model(model_path)


def initialize_session_state():
    """
    Initializes persistent state used by all application modes.
    """
    # Define the default values required by the image, video, and live-camera
    # workflows. These values remain available across Streamlit reruns.
    defaults = {
        # Stores the most recent image-processing result.
        "image_result": None,
        # Stores the most recent video-processing result.
        "video_result": None,
        # Stores the most recent processed live-camera frame and its results.
        "camera_result": None,
        # Stores the active OpenCV camera capture object.
        "camera_capture": None,
        # Indicates whether live-camera processing is currently active.
        "camera_running": False,
        # Stores object movement trails using track IDs as dictionary keys.
        "camera_trails": {},
        # Stores label positions from the previous camera frame.
        # These positions help reduce label movement between frames.
        "camera_label_positions": None,
        # Stores the number of the current camera frame.
        "camera_frame_number": 0,
    }

    # Add each default value only when the corresponding session-state key
    # does not already exist. This prevents existing results and camera state
    # from being reset whenever Streamlit reruns the application.
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_app_header():
    """
    Renders the application title and description.
    """
    # Display a custom HTML hero section at the top of the application.
    # The "app-hero" class is styled in the project's CSS file.
    st.markdown(
        """
        <div class="app-hero">
            <h1>🎯 VisionTrack Studio</h1>
            <p>
                Detect, track, inspect, and export objects from images,
                videos, and live camera streams.
            </p>
        </div>
        """,
        # Allow Streamlit to render the HTML elements and CSS class instead
        # of displaying the markup as plain text.
        unsafe_allow_html=True,
    )


def get_model_and_classes(model_path):
    """
    Loads the selected model and retrieves its class labels.

    Args:
        model_path: Path or name of the YOLO model.

    Returns:
        Loaded YOLO model and list of class names.
    """
    # Load the YOLO model through the cached model-loading function.
    # Streamlit reuses the same model when the model path has not changed.
    model = get_cached_model(model_path)

    # Retrieve the class labels supported by the loaded model, such as
    # "person", "car", "dog", and "laptop".
    class_names = get_model_class_names(model=model)

    # Return both values because the processing functions require the model,
    # while the interface uses the class names for selection and filtering.
    return model, class_names


def reset_camera_state(release_capture=True):
    """
    Resets camera tracking and frame state.

    Args:
        release_capture: Whether to release and remove the camera capture.
    """
    # Retrieve the current OpenCV camera capture object from session state.
    capture = st.session_state.get("camera_capture")

    # Release the physical camera only when requested and when an active
    # capture object exists.
    if release_capture and capture is not None:
        try:
            # Free the camera resource so it can be used by other applications
            # or opened again later by this application.
            capture.release()
        except Exception:
            # Prevent a camera-release failure from stopping the Streamlit app.
            # This can happen when the camera was already closed or disconnected.
            pass

        # Remove the released capture object from session state.
        st.session_state["camera_capture"] = None

    # Stop the live-camera processing loop.
    st.session_state["camera_running"] = False

    # Remove all stored movement trails from previous tracked objects.
    st.session_state["camera_trails"] = {}

    # Clear previous label positions so the next camera session calculates
    # label placement from the beginning.
    st.session_state["camera_label_positions"] = None

    # Reset frame numbering for the next camera session.
    st.session_state["camera_frame_number"] = 0

    # Remove the most recent camera-processing result.
    st.session_state["camera_result"] = None


def render_detection_metrics(summary, video_mode=False):
    """
    Renders the primary detection metrics.

    Args:
        summary: Dictionary containing detection summary statistics.
        video_mode: Whether to display video-specific metrics.
    """
    # Do not render the metric section when no summary data is available.
    if not summary:
        return

    if video_mode:
        # Video mode includes the number of processed frames, so five
        # equally sized metric columns are required.
        columns = st.columns(5)

        # Display how many video frames were processed by the detection model.
        columns[0].metric(
            "Processed frames",
            summary.get("processed_frames", 0),
        )

        # Display the total number of frame-level detections. The same physical
        # object may appear in several frames and therefore be counted more than once.
        columns[1].metric(
            "Detections",
            summary.get("total_detections", 0),
        )

        # Display the number of different object classes found in the video.
        columns[2].metric(
            "Unique classes",
            summary.get("unique_classes", 0),
        )

        # Display the mean confidence score across all video detections.
        columns[3].metric(
            "Average confidence",
            f'{summary.get("average_confidence", 0.0):.2f}',
        )

        # Display the class with the highest number of detection occurrences.
        # Use "None" when the video contains no detections.
        columns[4].metric(
            "Most common class",
            summary.get("most_common_class") or "None",
        )

    else:
        # Image and live-camera results do not require a processed-frame
        # metric, so four equally sized metric columns are sufficient.
        columns = st.columns(4)

        # Display the total number of objects detected in the image or frame.
        columns[0].metric(
            "Detections",
            summary.get("total_detections", 0),
        )

        # Display the number of different object classes detected.
        columns[1].metric(
            "Unique classes",
            summary.get("unique_classes", 0),
        )

        # Display the average confidence score across all detections.
        columns[2].metric(
            "Average confidence",
            f'{summary.get("average_confidence", 0.0):.2f}',
        )

        # Display the most frequently detected object class.
        # Use "None" when no objects were detected.
        columns[3].metric(
            "Most common class",
            summary.get("most_common_class") or "None",
        )


def dataframe_to_csv_bytes(dataframe):
    """
    Converts a DataFrame into downloadable UTF-8 CSV bytes.

    Args:
        dataframe: DataFrame containing the data to export.

    Returns:
        CSV content encoded as bytes.
    """
    # Replace a missing DataFrame with an empty one so the export operation
    # still returns valid CSV content instead of raising an error.
    if dataframe is None:
        dataframe = pd.DataFrame()

    # Convert the DataFrame into CSV text without including the pandas index.
    csv_text = dataframe.to_csv(index=False)

    # Encode the CSV text as UTF-8 bytes because Streamlit download buttons
    # expect binary data when downloading generated file content.
    return csv_text.encode("utf-8")


def object_to_json_bytes(value):
    """
    Converts a JSON-compatible object into downloadable UTF-8 bytes.

    Args:
        value: Object containing the data to export.

    Returns:
        JSON content encoded as bytes.
    """
    # Convert the object into formatted JSON text. The indentation makes the
    # exported file easier to read.
    json_text = json.dumps(
        value,
        indent=2,
        default=str,
    )

    # Encode the JSON text as UTF-8 bytes so it can be passed directly to a
    # Streamlit download button.
    return json_text.encode("utf-8")


def read_binary_file(file_path):
    """
    Reads a file as bytes for Streamlit downloads and playback.

    Args:
        file_path: Path of the file to read.

    Returns:
        Complete file content as bytes.
    """
    # Convert strings and other path-like values into a Path object so the
    # file can be validated and read consistently.
    path = Path(file_path)

    # Confirm that the requested output file exists before attempting to read
    # it. This provides a clearer error than allowing the read operation to fail.
    if not path.exists():
        raise FileNotFoundError(f"The output file does not exist: {path}")

    # Ensure that the path points to a regular file rather than a directory.
    if not path.is_file():
        raise ValueError(f"The specified path is not a file: {path}")

    # Read and return the complete file in binary form. The returned bytes can
    # be passed directly to st.download_button(), st.video(), or similar tools.
    return path.read_bytes()


def render_class_chart(class_count_df, count_column=None):
    """
    Renders a class-frequency bar chart.

    Args:
        class_count_df: DataFrame containing class names and count values.
        count_column: Optional name of the column containing the counts.
    """
    # Stop early when no class-count data is available.
    if class_count_df is None or class_count_df.empty:
        st.info("No class counts are available.")
        return

    # Automatically determine which count column should be used when the
    # caller does not provide one explicitly.
    if count_column is None:
        possible_columns = [
            "count",
            "detection_count",
            "unique_object_count",
        ]

        # Select the first supported count column found in the DataFrame.
        count_column = next(
            (column for column in possible_columns if column in class_count_df.columns),
            None,
        )

    # Create a smaller DataFrame containing only the class names and values
    # needed by the bar chart.
    chart_df = class_count_df[["class_name", count_column]].copy()

    # Use class names as the chart index so Streamlit displays one bar
    # for each detected object class.
    chart_df = chart_df.set_index("class_name")

    # Render the class-frequency chart.
    st.bar_chart(chart_df)
