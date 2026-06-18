"""
Live camera processing utilities for the Streamlit object detection app.

This module opens and reads a local camera and processes individual camera
frames using the same detection, tracking, trail, and annotation functions
used by the image and video pipelines.

The module intentionally does not contain a continuous processing loop.
Streamlit should control the loop and preserve camera state using
st.session_state.
"""

import cv2
import sys

from core.detector import DEFAULT_MODEL_PATH, load_model
from core.image_processor import process_image
from utils.drawing import draw_detections, draw_object_trails


def open_camera(
    camera_index=0,
    width=None,
    height=None,
    fps=None,
    backend=None,
    buffer_size=1,
):
    """
    Opens and validates a local camera.

    Args:
        camera_index: Numeric index identifying the camera.
        width: Optional requested frame width.
        height: Optional requested frame height.
        fps: Optional requested camera frame rate.
        backend: Optional OpenCV video-capture backend.
        buffer_size: Optional capture buffer size used to reduce latency.

    Returns:
        Initialized OpenCV VideoCapture object.
    """
    camera_index = int(camera_index)

    # Select the appropriate default backend for the operating system
    # running the Python process.
    if backend is None:
        if sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2

        elif sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW

        else:
            backend = cv2.CAP_ANY

    if backend is None:
        capture = cv2.VideoCapture(camera_index)
    else:
        capture = cv2.VideoCapture(
            camera_index,
            int(backend),
        )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(f"Could not open camera with index {camera_index}.")

    # Request a specific frame width when supplied. Camera drivers may use
    # the closest supported value rather than the exact requested value.
    if width is not None:
        width = int(width)

        if width <= 0:
            capture.release()
            raise ValueError("width must be greater than zero.")

        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

    # Request a specific frame height when supplied.
    if height is not None:
        height = int(height)

        if height <= 0:
            capture.release()
            raise ValueError("height must be greater than zero.")

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height,
        )

    # Request a specific frame rate when supplied.
    if fps is not None:
        fps = float(fps)

        if fps <= 0:
            capture.release()
            raise ValueError("fps must be greater than zero.")

        capture.set(
            cv2.CAP_PROP_FPS,
            fps,
        )

    # A small capture buffer can reduce the delay between the real camera
    # view and the frame currently processed by the application.
    if buffer_size is not None:
        buffer_size = int(buffer_size)

        if buffer_size <= 0:
            capture.release()
            raise ValueError("buffer_size must be greater than zero.")

        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            buffer_size,
        )

    return capture


def get_camera_properties(capture):
    """
    Reads the current camera properties.

    Args:
        capture: OpenCV VideoCapture object.

    Returns:
        Dictionary containing width, height, and frame rate.
    """
    if capture is None or not capture.isOpened():
        raise ValueError("The camera is not open.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = float(capture.get(cv2.CAP_PROP_FPS))

    return {
        "width": width,
        "height": height,
        "fps": fps,
    }


def read_camera_frame(capture, mirror=False):
    """
    Reads one frame from an open camera.

    Args:
        capture: OpenCV VideoCapture object.
        mirror: Whether to flip the frame horizontally.

    Returns:
        Camera frame as a BGR NumPy array.
    """
    if capture is None or not capture.isOpened():
        raise ValueError("The camera is not open.")

    success, frame = capture.read()

    if not success or frame is None:
        raise RuntimeError("Could not read a frame from the camera.")

    # A mirrored frame usually feels more natural for webcam previews.
    if mirror:
        frame = cv2.flip(
            frame,
            1,
        )

    return frame


def release_camera(capture):
    """
    Releases an OpenCV camera safely.

    Args:
        capture: OpenCV VideoCapture object.
    """
    if capture is not None:
        capture.release()


def process_camera_frame(
    frame,
    model=None,
    model_path=DEFAULT_MODEL_PATH,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    image_size=640,
    device=None,
    max_detections=300,
    selected_classes=None,
    box_color="#00ff00",
    label_background_color="#00ff00",
    text_color="#ffffff",
    box_thickness=2,
    text_size=0.6,
    text_thickness=1,
    alpha=0.18,
    show_label=True,
    show_confidence=True,
    text_background=True,
    box_style="standard",
    text_font=cv2.FONT_HERSHEY_COMPLEX,
    text_position="automatic",
    previous_label_positions=None,
    trails=None,
    show_trails=False,
    trail_color="#ffff00",
    trail_thickness=2,
    max_trail_length=30,
    max_inactive_frames=90,
    tracking=True,
    persist=True,
    tracker="bytetrack.yaml",
    frame_number=None,
    camera_fps=None,
    source_name="Live Camera",
):
    """
    Processes one live camera frame.

    Args:
        frame: Camera frame as a BGR NumPy array.
        model: Existing loaded YOLO model.
        model_path: Path or name of the YOLO model.
        confidence_threshold: Minimum detection confidence.
        iou_threshold: IoU threshold used during non-maximum suppression.
        image_size: Image size used during inference.
        device: Device used for inference.
        max_detections: Maximum number of detections.
        selected_classes: Object classes to retain.
        box_color: Bounding-box color in hexadecimal format.
        label_background_color: Label background color in hexadecimal format.
        text_color: Label text color in hexadecimal format.
        box_thickness: Bounding-box line thickness.
        text_size: Scale applied to label text.
        text_thickness: Label text thickness.
        alpha: Opacity of filled bounding boxes.
        show_label: Whether to show object class names.
        show_confidence: Whether to show confidence scores.
        text_background: Whether to draw label backgrounds.
        box_style: Bounding-box drawing style.
        text_font: OpenCV font used for labels.
        text_position: Automatic or predefined label position.
        previous_label_positions: Previous positions indexed by track ID.
        trails: Existing movement histories indexed by track ID.
        show_trails: Whether to draw movement trails.
        trail_color: Trail color in hexadecimal format.
        trail_thickness: Trail line thickness in pixels.
        max_trail_length: Maximum number of positions retained per track.
        max_inactive_frames: Number of consecutive frames a missing track is
            retained before its trail is removed.
        tracking: Whether to use YOLO object tracking.
        persist: Whether tracking IDs persist between frames.
        tracker: Ultralytics tracker configuration.
        frame_number: Optional sequential camera-frame number.
        camera_fps: Optional camera frame rate used for timestamps.
        source_name: Name stored with detection records.

    Returns:
        Dictionary containing the annotated frame, detections, tables,
        summaries, trails, and current label-position state.
    """
    if frame is None:
        raise ValueError("No camera frame was provided.")

    # Convert frame and trail settings into the required numeric types.
    trail_thickness = int(trail_thickness)
    max_trail_length = int(max_trail_length)
    max_inactive_frames = int(max_inactive_frames)

    # Validate the optional camera FPS value.
    if camera_fps is not None:
        camera_fps = float(camera_fps)

        if camera_fps <= 0:
            camera_fps = None

    # Load the model only when the caller has not supplied an existing
    # model instance. A persistent model should normally be stored in
    # Streamlit session state or a cached resource.
    if model is None:
        model = load_model(model_path)

    # Copy the persisted trail records so the caller's state is not modified
    # unexpectedly inside this function.
    updated_trails = {}

    for track_id, trail_data in (trails or {}).items():
        if isinstance(trail_data, dict):
            points = list(trail_data.get("points", []))
            last_seen_frame = int(trail_data.get("last_seen_frame", frame_number))

        else:
            # Backward compatibility for session state created by the previous
            # version, where each value was only a list of points.
            points = list(trail_data)
            last_seen_frame = frame_number

        updated_trails[track_id] = {
            "points": points,
            "last_seen_frame": last_seen_frame,
        }

    # Run inference and tracking without drawing annotations. This allows
    # trails to be drawn first and bounding boxes and labels afterward.
    frame_result = process_image(
        image=frame,
        model=model,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        device=device,
        max_detections=max_detections,
        selected_classes=selected_classes,
        box_color=box_color,
        label_background_color=label_background_color,
        text_color=text_color,
        box_thickness=box_thickness,
        text_size=text_size,
        text_thickness=text_thickness,
        alpha=alpha,
        show_label=show_label,
        show_confidence=show_confidence,
        text_background=text_background,
        box_style=box_style,
        text_font=text_font,
        text_position=text_position,
        previous_label_positions=previous_label_positions,
        draw_annotations=False,
        tracking=tracking,
        persist=persist,
        tracker=tracker,
        frame_number=frame_number,
        video_fps=camera_fps,
        source_name=source_name,
    )

    # Store the IDs detected in the current frame. Only trails belonging
    # to currently visible objects are drawn.
    active_track_ids = set()

    # Add the current bounding-box center to each tracked object's history.
    for detection in frame_result["detections"]:
        track_id = detection.get("track_id")

        # Ordinary predictions and unconfirmed tracker detections may not
        # contain a valid tracking ID.
        if track_id is None:
            continue

        active_track_ids.add(track_id)

        center_x = (detection["x1"] + detection["x2"]) / 2

        center_y = (detection["y1"] + detection["y2"]) / 2

        if track_id not in updated_trails:
            updated_trails[track_id] = {
                "points": [],
                "last_seen_frame": frame_number,
            }

        # Add the current centre position.
        updated_trails[track_id]["points"].append((center_x, center_y))

        # Keep only the most recent trail positions.
        updated_trails[track_id]["points"] = updated_trails[track_id]["points"][
            -max_trail_length:
        ]

        # Mark this track as visible in the current frame.
        updated_trails[track_id]["last_seen_frame"] = frame_number

    # Remove tracks that have not appeared for too many consecutive frames.
    stale_track_ids = [
        track_id
        for track_id, trail_data in updated_trails.items()
        if (frame_number - trail_data["last_seen_frame"] > max_inactive_frames)
    ]

    for track_id in stale_track_ids:
        del updated_trails[track_id]

    # Start with the clean BGR frame returned by process_image().
    drawing_frame = frame_result["original_image"].copy()

    # A visible trail requires at least two points belonging to the same
    # persistent tracking ID.
    active_trails = {
        track_id: updated_trails[track_id]["points"]
        for track_id in active_track_ids
        if (track_id in updated_trails and len(updated_trails[track_id]["points"]) >= 2)
    }

    # Draw trails before boxes and labels so all annotations remain visible
    # above the movement paths.
    if show_trails and active_trails:
        drawing_frame = draw_object_trails(
            image=drawing_frame,
            trails=active_trails,
            trail_color=trail_color,
            trail_thickness=trail_thickness,
        )

    # Draw bounding boxes and labels after the trails.
    annotated_frame, current_label_positions = draw_detections(
        image=drawing_frame,
        detections=frame_result["detections"],
        box_color=box_color,
        label_background_color=label_background_color,
        text_color=text_color,
        box_thickness=box_thickness,
        text_size=text_size,
        text_thickness=text_thickness,
        alpha=alpha,
        show_label=show_label,
        show_confidence=show_confidence,
        text_background=text_background,
        box_style=box_style,
        text_font=text_font,
        text_position=text_position,
        previous_label_positions=previous_label_positions,
    )

    # Add camera-specific tracking information to the frame summary.
    summary = frame_result["summary"].copy()

    summary["active_tracks"] = len(active_track_ids)

    summary["stored_trails"] = len(updated_trails)

    summary["visible_trails"] = len(active_trails)

    # Return the updated state so Streamlit can preserve it for the next
    # camera frame.
    return {
        "original_frame": frame_result["original_image"].copy(),
        "annotated_frame": annotated_frame.copy(),
        "detections": frame_result["detections"],
        "detection_df": frame_result["detection_df"],
        "class_count_df": frame_result["class_count_df"],
        "summary": summary,
        "trails": updated_trails,
        "active_track_ids": active_track_ids,
        "active_trails": active_trails,
        "current_label_positions": current_label_positions,
        "frame_number": frame_result["frame_number"],
        "timestamp_seconds": frame_result["timestamp_seconds"],
    }
