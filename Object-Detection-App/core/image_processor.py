"""
Image and video-frame processing for object detection.

This file connects the detector and drawing modules. The same function is
used for uploaded images and individual frames from videos or live cameras.
"""

import cv2

from core.detector import (
    DEFAULT_MODEL_PATH,
    count_detections_by_class,
    detect_objects,
    detections_to_dataframe,
    prepare_image,
)
from utils.drawing import draw_detections


def process_image(
    image,
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
    draw_annotations=True,
    tracking=False,
    persist=False,
    tracker="bytetrack.yaml",
    frame_number=None,
    video_fps=None,
    source_name=None,
):
    """
    Processes an uploaded image or a single video or camera frame.

    Args:
        image: Uploaded image, PIL image, or NumPy image.
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
        box_style: Bounding-box style.
        text_font: OpenCV font used for labels.
        text_position: Automatic or predefined label position.
        previous_label_positions: Previous label positions indexed by track ID.
        draw_annotations: Whether to draw bounding boxes and labels.
        tracking: Whether to run object tracking instead of prediction.
        persist: Whether tracking IDs should persist between frames.
        tracker: Ultralytics tracker configuration.
        frame_number: Optional video or camera frame number.
        video_fps: Optional source video frame rate.
        source_name: Optional image or video name.

    Returns:
        Dictionary containing images, detections, tables, statistics,
        and the current label-position state.
    """
    # Convert interface values into the numeric types expected by the
    # inference and drawing functions.
    confidence_threshold = float(confidence_threshold)
    iou_threshold = float(iou_threshold)
    image_size = int(image_size)
    max_detections = int(max_detections)
    box_thickness = int(box_thickness)
    text_size = float(text_size)
    text_thickness = int(text_thickness)
    alpha = float(alpha)

    # Use the uploaded filename when available.
    if source_name is None:
        source_name = getattr(image, "name", None)

    # Convert the input into a BGR uint8 NumPy image.
    working_image = prepare_image(image)
    image_height, image_width = working_image.shape[:2]

    # Run object detection and return structured detection dictionaries.
    detections = detect_objects(
        image=working_image,
        model=model,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        device=device,
        max_detections=max_detections,
        selected_classes=selected_classes,
        tracking=tracking,
        persist=persist,
        tracker=tracker,
    )

    # Calculate a timestamp only when valid video-frame information is present.
    timestamp_seconds = None

    if frame_number is not None and video_fps is not None and video_fps > 0:
        timestamp_seconds = frame_number / video_fps

    # Add optional source and frame metadata without modifying the original
    # detection dictionaries.
    detections_with_metadata = []

    for detection in detections:
        detection_copy = detection.copy()

        if source_name is not None:
            detection_copy["source_name"] = str(source_name)

        if frame_number is not None:
            detection_copy["frame_number"] = frame_number

        if timestamp_seconds is not None:
            detection_copy["timestamp_seconds"] = round(
                timestamp_seconds,
                3,
            )

        detections_with_metadata.append(detection_copy)

    # Preserve the previous label-position state when annotations are disabled.
    # No new positions are calculated until draw_detections() is called.
    if previous_label_positions is None:
        current_label_positions = {}
    else:
        current_label_positions = previous_label_positions.copy()

    # Draw bounding boxes and labels only when requested. Video processing can
    # disable this step, draw trails first, and then add annotations afterward.
    if draw_annotations:
        annotated_output, current_label_positions = draw_detections(
            image=working_image,
            detections=detections_with_metadata,
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
    else:
        # Return an annotated copy when drawing is handled by another function
        # such as process_video().
        annotated_output = working_image.copy()

    # Convert the structured detections into tables used by the interface,
    # analytics components, search, and export operations.
    detection_df = detections_to_dataframe(detections_with_metadata)
    class_count_df = count_detections_by_class(detections_with_metadata)

    # Calculate summary statistics for the processed image or frame.
    total_detections = len(detections_with_metadata)

    if total_detections == 0:
        average_confidence = 0.0
        unique_classes = 0
        most_common_class = None
    else:
        average_confidence = (
            sum(detection["confidence"] for detection in detections_with_metadata)
            / total_detections
        )

        unique_classes = len(
            set(detection["class_name"] for detection in detections_with_metadata)
        )

        most_common_class = class_count_df.iloc[0]["class_name"]

    # Collect the main image and detection statistics.
    summary = {
        "source_name": source_name,
        "frame_number": frame_number,
        "timestamp_seconds": timestamp_seconds,
        "image_width": image_width,
        "image_height": image_height,
        "total_detections": total_detections,
        "unique_classes": unique_classes,
        "average_confidence": round(average_confidence, 2),
        "most_common_class": most_common_class,
    }

    # Return all outputs needed by Streamlit and subsequent video frames.
    return {
        "original_image": working_image.copy(),
        "annotated_image": annotated_output.copy(),
        "detections": detections_with_metadata,
        "detection_df": detection_df,
        "class_count_df": class_count_df,
        "summary": summary,
        "current_label_positions": current_label_positions,
        "frame_number": frame_number,
        "timestamp_seconds": timestamp_seconds,
    }
