"""
Object detection engine for the Streamlit app.

This file loads a YOLO model, runs object detection, and returns clean
detection results that can be reused by image, video, and live camera modes.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import cv2

# Get the project root directory.
# detector.py is inside core/, so parent.parent points to the project root.
BASE_DIR = Path(__file__).resolve().parent.parent

# Store all YOLO model weights inside the models directory.
MODEL_DIRECTORY = BASE_DIR / "models"

# Create the directory if it does not already exist.
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Set the model loaded by default when the application starts.
DEFAULT_MODEL_PATH = MODEL_DIRECTORY / "yolo26x.pt"


def load_model(model_path=DEFAULT_MODEL_PATH):
    """
    Loads a new YOLO model instance.

    Args:
        model_path: Path or name of the YOLO model.

    Returns:
        Newly loaded YOLO model.
    """
    return YOLO(str(model_path))


def prepare_image(image):
    """
    Converts an input image into a uint8 NumPy BGR image.

    Args:
        image: PIL image, NumPy image, or uploaded image file.

    Returns:
        BGR image as a NumPy array.
    """
    # Ensure an image was provided
    if image is None:
        raise ValueError("No image was provided.")

    # Handle images that are already represented as NumPy arrays.
    if isinstance(image, np.ndarray):
        # Raise an error when the image is empty
        if image.size == 0:
            raise ValueError("The image is empty")

        # Convert a two-dimensional grayscale image into three BGR channels.
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        elif image.ndim == 3:
            num_channels = image.shape[2]

            # Convert a one-channel array into a standard BGR image.
            if num_channels == 1:
                image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

            # Accept standard three-channel BGR images unchanged.
            elif num_channels == 3:
                pass

            # Remove the alpha channel from a BGRA image.
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # Accept standard three-channel BGR images unchanged.
            else:
                raise ValueError("The image must contain 1, 3, or 4 channels.")

        else:
            raise ValueError("The image must have 2 or 3 dimensions.")

        # Validate and scale floating-point images.
        if np.issubdtype(image.dtype, np.floating):
            # NaN and infinite values cannot be converted into valid pixels.
            if not np.isfinite(image).all():
                raise ValueError("The image contains NaN or infinite values.")

            image_min = float(image.min())
            image_max = float(image.max())

            # Convert normalized pixel values from [0, 1] to [0, 255].
            if image_min >= 0.0 and image_max <= 1.0:
                image = image * 255.0

            # Floating-point images outside [0, 1] must use [0, 255].
            elif image_min < 0.0 or image_max > 255.0:
                raise ValueError(
                    "Floating-point image values must be in the range "
                    "[0, 1] or [0, 255]"
                )

        # Clip invalid integer ranges, round floating-point values, and
        # convert to the standard OpenCV image data type.
        image = np.clip(image, 0, 255)
        image = np.rint(image).astype(np.uint8)

        # Return a contiguous array suitable for OpenCV operations.
        return np.ascontiguousarray(image)

    # PIL images use RGB order, so convert them explicitly to BGR.
    if isinstance(image, Image.Image):
        rgb_image = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Treat any remaining input as a path or file-like object.
    try:
        # PIL reads the input in RGB order.
        rgb_image = np.array(Image.open(image).convert("RGB"))

        # Convert RGB into the BGR order expected by OpenCV.
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    except Exception as error:
        raise ValueError("Could not read the input image.") from error


def run_model(
    model,
    image,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    image_size=640,
    device=None,
    max_detections=300,
    tracking=False,
    persist=False,
    tracker="bytetrack.yaml",
):
    """
    Runs YOLO inference on one image.

    Args:
        model: Loaded YOLO model.
        image: Input image.
        confidence_threshold: Minimum confidence score.
        iou_threshold: IoU threshold used by non-maximum suppression.
        image_size: Image size used by the model.
        device: Device used for inference.
        max_detections: Maximum number of detections.
        tracking: Whether to run object tracking instead of prediction.
        persist: Whether tracking IDs should persist between frames.
        tracker: Ultralytics tracker configuration.

    Returns:
        Raw YOLO result for one image.
    """
    # Ensure that a loaded YOLO model is available before running inference.
    if model is None:
        raise ValueError("No YOLO model was provided.")

    # Convert the input into a valid RGB uint8 NumPy image.
    prepared_image = prepare_image(image)

    # Collect the inference settings passed to YOLO's predict method.
    inference_kwargs = {
        "source": prepared_image,
        "conf": confidence_threshold,
        "iou": iou_threshold,
        "imgsz": image_size,
        "max_det": max_detections,
        "verbose": False,
    }

    # Specify the inference device only when the caller provides one.
    # Otherwise, YOLO automatically selects an available device.
    if device:
        inference_kwargs["device"] = device

    # Use tracking for sequential video frames so detections receive
    # persistent object IDs.
    if tracking:
        inference_kwargs["persist"] = persist

        if tracker:
            inference_kwargs["tracker"] = tracker

        results = model.track(**inference_kwargs)

    # Use ordinary prediction for independent images.
    else:
        results = model.predict(**inference_kwargs)

    # Return None if the model does not produce any result objects.
    if not results:
        return None

    # Only one image was provided, so return its corresponding result.
    return results[0]


def extract_detections(result):
    """
    Extracts clean detection data from a YOLO result.

    Args:
        result: Raw YOLO result for one image.

    Returns:
        List of detection dictionaries.
    """
    # Store the processed information for each detected object.
    detections = []

    # Return an empty list when no result or bounding-box data is available.
    if result is None or result.boxes is None:
        return detections

    # Retrieve the mapping between numeric class IDs and class names.
    class_names = result.names

    # Process each bounding box predicted by the model.
    for box in result.boxes:
        # Extract the predicted class ID and confidence score.
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Extract the tracking id of the detection box
        track_id = int(box.id[0].item()) if box.id is not None else None

        # Extract the bounding-box corner coordinates in xyxy format:
        # top-left (x1, y1) and bottom-right (x2, y2).
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Calculate the bounding-box dimensions and area in pixels.
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Store the cleaned detection values in a consistent dictionary format.
        detections.append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "confidence": round(confidence, 2),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "box_width": round(float(width), 2),
                "box_height": round(float(height), 2),
                "box_area": round(float(area), 2),
            }
        )

    return detections


def filter_detections(detections, selected_classes=None):
    """
    Filters detections by class name.

    Args:
        detections: List of detection dictionaries.
        selected_classes: Classes to keep.

    Returns:
        Filtered list of detection dictionaries.
    """
    # Begin with all available detections.
    filtered = detections

    # Filter by class only when one or more class names are provided.
    if selected_classes:
        # Normalize the selected class names so matching is case-insensitive.
        selected_classes = [class_name.lower() for class_name in selected_classes]
        # Keep detections whose class name matches one of the selected classes.
        filtered = [
            detection
            for detection in filtered
            if detection["class_name"].lower() in selected_classes
        ]

    # Return the detections that satisfy all active filters.
    return filtered


def detections_to_dataframe(detections):
    """
    Converts detections into a pandas DataFrame.

    Args:
        detections: List of detection dictionaries.

    Returns:
        DataFrame with detection results.
    """
    # Return an empty DataFrame when no detections are available.
    if not detections:
        return pd.DataFrame()

    # Pandas creates one column for every key found in the dictionaries.
    detection_df = pd.DataFrame(detections)

    # Keep video tracking IDs as integers while still allowing missing values
    # for images and non-tracked inference.
    detection_df["track_id"] = detection_df["track_id"].astype("Int64")

    if "frame_number" in detection_df.columns:
        detection_df["frame_number"] = detection_df["frame_number"].astype("Int64")

    # Return the detection dataframe
    return detection_df


def detect_objects(
    image,
    model_path=DEFAULT_MODEL_PATH,
    model=None,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    image_size=640,
    device=None,
    max_detections=300,
    selected_classes=None,
    tracking=False,
    persist=False,
    tracker="bytetrack.yaml",
):
    """
    Runs object detection and returns clean results.

    Args:
        image: Input image.
        model_path: Path or name of the YOLO model.
        model: Existing loaded YOLO model.
        confidence_threshold: Minimum confidence score.
        iou_threshold: IoU threshold used by non-maximum suppression.
        image_size: Image size used by the model.
        device: Device used for inference.
        max_detections: Maximum number of detections.
        selected_classes: Classes to keep.
        tracking: Whether to run object tracking instead of prediction.
        persist: Whether tracking IDs should persist between frames.
        tracker: Ultralytics tracker configuration.

    Returns:
        Detection results as a list.
    """
    # Load the model only when an existing model instance is not provided.
    # Passing a previously loaded model avoids loading it again for every image.
    if model is None:
        model = load_model(model_path)

    # Run YOLO inference using the requested confidence, IoU, image-size,
    # device, and maximum-detection settings.
    result = run_model(
        model=model,
        image=image,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        device=device,
        max_detections=max_detections,
        tracking=tracking,
        persist=persist,
        tracker=tracker,
    )

    # Convert the raw YOLO result into a list of structured detection
    # dictionaries containing class, confidence, and bounding-box data.
    detections = extract_detections(result)

    # Apply any requested class filter to the extracted results.
    detections = filter_detections(
        detections,
        selected_classes=selected_classes,
    )

    # Return the default list-of-dictionaries representation.
    return detections


def get_model_class_names(model_path=DEFAULT_MODEL_PATH, model=None):
    """
    Gets the class names used by the model.

    Args:
        model_path: Path or name of the YOLO model.
        model: Existing loaded YOLO model.

    Returns:
        List of class names.
    """
    # Load the model only when an existing model instance is not provided.
    # Reusing a loaded model avoids unnecessary loading on repeated calls.
    if model is None:
        model = load_model(model_path)

    # YOLO commonly stores class names as a dictionary that maps each
    # numeric class ID to its corresponding class label.
    names = model.names

    # Extract only the labels when the class names are stored as a dictionary.
    if isinstance(names, dict):
        return list(names.values())

    # Convert other iterable formats, such as a tuple or list, into a list.
    return list(names)


def count_detections_by_class(detections):
    """
    Counts detections for each class.

    Args:
        detections: List of detection dictionaries.

    Returns:
        DataFrame with object counts.
    """
    # Return an empty DataFrame with a consistent structure when no
    # detections are available.
    if not detections:
        return pd.DataFrame(columns=["class_name", "count"])

    # Convert the detection dictionaries into a DataFrame so the class
    # labels can be counted using pandas.
    detection_df = detections_to_dataframe(detections)

    # Count occurrences of each class, move the class names from the index
    # into a regular column, and name the frequency column "count".
    return (
        detection_df["class_name"]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )
