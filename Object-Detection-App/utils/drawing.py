"""
Drawing utilities for object detection results.

This file contains helper functions for drawing bounding boxes, labels,
confidence scores, track IDs, and simple visual styles on images or frames.
"""

import cv2
import numpy as np

from core.detector import prepare_image


def hex_to_bgr(hex_color):
    """
    Converts a hexadecimal RGB color into a BGR tuple.

    Args:
        hex_color: Color in hexadecimal format.

    Returns:
        BGR color tuple.
    """
    # Ensure the input is a string before applying string operations.
    if not isinstance(hex_color, str):
        raise ValueError("Hex color must be a string.")

    # Remove surrounding whitespace.
    hex_color = hex_color.strip()

    # Remove one optional leading "#" character.
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]

    # A complete hexadecimal color must contain six characters: RRGGBB.
    if len(hex_color) != 6:
        raise ValueError(
            "Hex color must contain exactly 6 characters in RRGGBB format."
        )

    try:
        # Convert the red, green, and blue hexadecimal pairs to integers.
        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)

    except ValueError as error:
        raise ValueError(
            "Hex color can only contain characters from 0-9 and A-F."
        ) from error

    # OpenCV uses blue, green, and red channel order.
    return blue, green, red


def build_label_text(detection, show_label=True, show_confidence=True):
    """
    Builds the text displayed next to a bounding box.

    Args:
        detection: Detection dictionary.
        show_label: Whether to show the object class name.
        show_confidence: Whether to show the confidence score.

    Returns:
        Label text.
    """
    # Store the individual components of the final label.
    label_parts = []

    # Add the object class name when labels are enabled.
    if show_label:
        label_parts.append(str(detection.get("class_name")))

    # Add the confidence score rounded to two decimal places.
    if show_confidence:
        label_parts.append(f"{float(detection.get('confidence')):.2f}")

    # Combine the available parts into one label string.
    return " | ".join(label_parts)


def draw_corner_box(
    image,
    x1,
    y1,
    x2,
    y2,
    box_color,
    box_thickness,
):
    """
    Draws a corner-style bounding box.

    Args:
        image: Image where the box is drawn.
        x1: Left box coordinate.
        y1: Top box coordinate.
        x2: Right box coordinate.
        y2: Bottom box coordinate.
        box_color: Box color.
        box_thickness: Box line thickness.
    """
    # Calculate the box width and length
    box_width = x2 - x1
    box_height = y2 - y1

    # Use 25% of the smallest box dimension for each corner segment.
    corner_length = int(0.25 * min(box_width, box_height))

    # Keep the corner visible without allowing it to exceed the box.
    corner_length = max(1, corner_length)
    corner_length = min(corner_length, box_width, box_height)

    # Top-left corner
    cv2.line(
        image,
        (x1, y1),
        (x1 + corner_length, y1),
        box_color,
        box_thickness,
    )
    cv2.line(
        image,
        (x1, y1),
        (x1, y1 + corner_length),
        box_color,
        box_thickness,
    )

    # Top-right corner
    cv2.line(
        image,
        (x2 - corner_length, y1),
        (x2, y1),
        box_color,
        box_thickness,
    )
    cv2.line(
        image,
        (x2, y1),
        (x2, y1 + corner_length),
        box_color,
        box_thickness,
    )

    # Bottom-left corner
    cv2.line(
        image,
        (x1, y2),
        (x1 + corner_length, y2),
        box_color,
        box_thickness,
    )
    cv2.line(
        image,
        (x1, y2 - corner_length),
        (x1, y2),
        box_color,
        box_thickness,
    )

    # Bottom-right corner
    cv2.line(
        image,
        (x2 - corner_length, y2),
        (x2, y2),
        box_color,
        box_thickness,
    )
    cv2.line(
        image,
        (x2, y2 - corner_length),
        (x2, y2),
        box_color,
        box_thickness,
    )


def draw_filled_box(image, x1, y1, x2, y2, box_color, box_thickness, alpha=0.18):
    """
    Draws a semi-transparent filled bounding box.

    Args:
        image: Image where the box is drawn.
        x1: Left box coordinate.
        y1: Top box coordinate.
        x2: Right box coordinate.
        y2: Bottom box coordinate.
        box_color: Box color.
        box_thickness: Box line thickness.
        alpha: Transparency level.
    """
    # Create a temporary copy where the solid filled rectangle is drawn.
    overlay = image.copy()

    # Draw a completely filled rectangle on the temporary overlay.
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)

    # Blend the colored overlay with the original image and store
    # the result directly back into the original image array.
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

    # Draw a solid border around the semi-transparent filled region.
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)


def _calculate_intersection_box(rectangle_a, rectangle_b):
    """
    Calculates the intersection area between two axis-aligned rectangles.

    Args:
        rectangle_a: First rectangle in (x1, y1, x2, y2) format.
        rectangle_b: Second rectangle in (x1, y1, x2, y2) format.

    Returns:
        Intersection area in square pixels.
    """
    # Unpack the top-left and bottom-right coordinates of both rectangles.
    ax1, ay1, ax2, ay2 = rectangle_a
    bx1, by1, bx2, by2 = rectangle_b

    # The intersection begins at the larger left coordinate and ends at
    # the smaller right coordinate.
    intersection_width = max(
        0,
        min(ax2, bx2) - max(ax1, bx1),
    )

    # The intersection begins at the larger top coordinate and ends at
    # the smaller bottom coordinate.
    intersection_height = max(
        0,
        min(ay2, by2) - max(ay1, by1),
    )

    # A width or height of zero means the rectangles do not overlap.
    return intersection_width * intersection_height


def calculate_automatic_label_positions(
    image,
    detections,
    previous_label_positions=None,
    font=cv2.FONT_HERSHEY_COMPLEX,
    font_scale=0.6,
    font_thickness=1,
    show_label=True,
    show_confidence=True,
):
    """
    Calculates an appropriate label position for each detection.

    Args:
        image: Image or video frame used to determine the available space.
        detections: List of detection dictionaries.
        previous_label_positions: Previous position names indexed by track ID.
        font: OpenCV font used to calculate the text dimensions.
        font_scale: Scale applied to the label text.
        font_thickness: Thickness of the label text.
        show_label: Whether to include the class name in the label.
        show_confidence: Whether to include the confidence score in the label.

    Returns:
        List of label placement dictionaries.
        Dictionary of selected position names indexed by track ID.
    """
    # Use an empty position history when processing an image or the first
    # frame of a video.
    if previous_label_positions is None:
        previous_label_positions = {}

    # Retrieve the frame dimensions used to keep every label visible.
    image_height, image_width = image.shape[:2]

    # Convert all detection coordinates to integer pixel coordinates once,
    # avoiding repeated conversions during candidate evaluation.
    all_boxes = [
        (
            int(round(detection["x1"])),
            int(round(detection["y1"])),
            int(round(detection["x2"])),
            int(round(detection["y2"])),
        )
        for detection in detections
    ]

    # Store label rectangles already selected in the current frame so that
    # later labels can avoid overlapping them.
    occupied_label_rectangles = []

    # Store the complete drawing information for every detection label.
    label_placements = []

    # Store each tracked object's selected relative position so it can be
    # reused as a preference in the next video frame.
    current_label_positions = {}

    # Calculate a label placement independently for every detected object.
    for detection_index, detection in enumerate(detections):
        box = all_boxes[detection_index]
        x1, y1, x2, y2 = box

        # Build the visible label from the selected class-name and confidence
        # display options.
        label = build_label_text(
            detection, show_label=show_label, show_confidence=show_confidence
        )

        # Do not create a placement when both label components are disabled.
        if not label:
            continue

        # Measure the text so the surrounding label rectangle can be sized
        # before candidate positions are evaluated.
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Scale the internal spacing with the text height while keeping it
        # within a reasonable minimum and maximum range.
        padding = max(
            1,
            min(round(text_height * 0.15), 12),
        )

        # Add padding around the text and include the OpenCV text baseline.
        label_width = text_width + 2 * padding
        label_height = text_height + baseline + 2 * padding

        # A label larger than the frame cannot be made fully visible through
        # repositioning alone.
        if label_width > image_width or label_height > image_height:
            raise ValueError(
                f"The label '{label}' is larger than the image. "
                "Reduce the text size or thickness."
            )

        # Define candidate positions using the top-left corner of the label
        # rectangle as the anchor point. Their order also defines preference:
        # positions near and outside the box are considered before positions
        # that place the label inside the detected object.
        candidates = [
            (
                "above left",
                x1,
                y1 - label_height,
            ),
            (
                "above right",
                x2 - label_width,
                y1 - label_height,
            ),
            (
                "below left",
                x1,
                y2,
            ),
            (
                "below right",
                x2 - label_width,
                y2,
            ),
            (
                "outside right top",
                x2,
                y1,
            ),
            (
                "outside left top",
                x1 - label_width,
                y1,
            ),
            (
                "outside right bottom",
                x2,
                y2 - label_height,
            ),
            (
                "outside left bottom",
                x1 - label_width,
                y2 - label_height,
            ),
            (
                "inside top left",
                x1,
                y1,
            ),
            (
                "inside top right",
                x2 - label_width,
                y1,
            ),
            (
                "inside bottom left",
                x1,
                y2 - label_height,
            ),
            (
                "inside bottom right",
                x2 - label_width,
                y2 - label_height,
            ),
        ]

        # Exclude the current object's box when measuring overlap with other
        # detected objects.
        other_boxes = [
            other_box
            for other_index, other_box in enumerate(all_boxes)
            if other_index != detection_index
        ]

        # Retrieve the track ID when object tracking is active.
        track_id = detection.get("track_id")

        # Recover the position used in the previous frame so that tracked
        # labels do not switch sides unnecessarily.
        previous_position_name = None
        if track_id is not None:
            previous_position_name = previous_label_positions.get(track_id)

        # Prevent division by zero for extremely small or invalid rectangles.
        label_area = max(1, label_width * label_height)
        own_box_area = max(1, (x2 - x1) * (y2 - y1))

        # Lower scores indicate better candidate positions.
        best_score = float("inf")
        best_rectangle = None
        best_position_name = None

        # Evaluate every possible label position and retain the candidate
        # with the lowest total penalty.
        for priority, (position_name, candidate_x, candidate_y) in enumerate(
            candidates
        ):
            # Shift the candidate just enough to keep the complete label
            # rectangle inside the image boundaries.
            clamped_x = max(0, min(candidate_x, image_width - label_width))
            clamped_y = max(0, min(candidate_y, image_height - label_height))

            # Represent the visible candidate as an xyxy rectangle.
            candidate_rectangle = (
                clamped_x,
                clamped_y,
                clamped_x + label_width,
                clamped_y + label_height,
            )

            # Penalize overlap with labels already placed in this frame.
            # Division by label area converts the total intersection area
            # into a proportion of the candidate label.
            occupied_label_overlap = (
                sum(
                    _calculate_intersection_box(
                        candidate_rectangle,
                        occupied_rectangle,
                    )
                    for occupied_rectangle in occupied_label_rectangles
                )
                / label_area
            )

            # Penalize labels that cover bounding boxes belonging to other
            # detected objects.
            other_box_overlap = (
                sum(
                    _calculate_intersection_box(
                        candidate_rectangle,
                        other_box,
                    )
                    for other_box in other_boxes
                )
                / label_area
            )

            # Penalize the proportion of the current object's bounding box
            # covered by its own label.
            own_box_coverage = (
                _calculate_intersection_box(candidate_rectangle, box) / own_box_area
            )

            # Penalize candidates that must move significantly from their
            # intended location to remain inside the frame.
            boundary_movement = (
                abs(clamped_x - candidate_x) / label_width
                + abs(candidate_y - clamped_y) / label_height
            )

            # In videos, discourage tracked labels from switching relative
            # positions unless another candidate is substantially better.
            position_change_penalty = 0
            if (
                previous_position_name is not None
                and position_name != previous_position_name
            ):
                position_change_penalty = 1

            # Combine the penalties using larger weights for more undesirable
            # effects. The small priority term breaks ties in favour of
            # candidates appearing earlier in the candidate list.
            score = (
                occupied_label_overlap * 1000
                + other_box_overlap * 250
                + own_box_coverage * 300
                + boundary_movement * 50
                + position_change_penalty * 40
                + priority * 0.01
            )

            # Keep the candidate with the lowest penalty score.
            if score < best_score:
                best_score = score
                best_rectangle = candidate_rectangle
                best_position_name = position_name

        # Convert the selected label rectangle into the baseline-based text
        # origin expected by cv2.putText().
        label_x1, label_y1, _, _ = best_rectangle
        text_origin = (label_x1 + padding, label_y1 + padding + text_height)

        # Store all information required to draw the label later.
        label_placements.append(
            {
                "label": label,
                "box": box,
                "label_rectangle": best_rectangle,
                "position_name": best_position_name,
                "text_origin": text_origin,
                "text_size": (text_width, text_height),
                "baseline": baseline,
            }
        )

        # Reserve the selected rectangle so labels processed later can avoid it.
        occupied_label_rectangles.append(best_rectangle)

        # Preserve the selected relative position for the object's next frame.
        if track_id is not None:
            current_label_positions[track_id] = best_position_name

    # Return both the drawing instructions and the updated tracking history.
    return label_placements, current_label_positions


def calculate_predefined_label_positions(
    image,
    detections,
    text_position,
    font=cv2.FONT_HERSHEY_COMPLEX,
    font_scale=0.6,
    font_thickness=1,
    show_label=True,
    show_confidence=True,
):
    """
    Calculates a predefined label position for each detection.

    Args:
        image: Image or video frame used to determine the available space.
        detections: List of detection dictionaries.
        text_position: Predefined position used for every label.
        font: OpenCV font used to measure and draw the text.
        font_scale: Scale applied to the label text.
        font_thickness: Thickness of the label text.
        show_label: Whether to include the class name in the label.
        show_confidence: Whether to include the confidence score in the label.

    Returns:
        List of label placement dictionaries.
        Dictionary of selected position names indexed by track ID.
    """
    # Retrieve the frame dimensions so labels can be constrained within
    # the visible image boundaries.
    image_height, image_width = image.shape[:2]

    # Store the complete drawing information for each visible label.
    label_placements = []

    # Store the selected position for tracked objects. This keeps the return
    # format consistent with the automatic placement function.
    current_label_positions = {}

    # Calculate the same predefined relative position for every detection.
    for detection in detections:
        # Build the label using the selected class-name and confidence options.
        label = build_label_text(
            detection, show_label=show_label, show_confidence=show_confidence
        )

        # Do not create a placement when both label components are disabled.
        if not label:
            continue

        # Convert floating-point bounding-box coordinates into image pixels.
        x1 = int(round(detection["x1"]))
        y1 = int(round(detection["y1"]))
        x2 = int(round(detection["x2"]))
        y2 = int(round(detection["y2"]))

        box = (x1, y1, x2, y2)

        # Measure the text with the same font settings that will later be
        # passed to cv2.putText().
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness,
        )

        # Scale the internal spacing with the text height while keeping it
        # within a reasonable minimum and maximum range.
        padding = max(
            1,
            min(round(text_height * 0.15), 12),
        )

        # Calculate the complete label rectangle, including text padding
        # and the additional space required by the OpenCV text baseline.
        label_width = text_width + 2 * padding
        label_height = text_height + 2 * padding + baseline

        # Repositioning cannot keep a label visible when the label itself
        # is larger than the complete frame.
        if label_width > image_width or label_height > image_height:
            raise ValueError(
                f"The label '{label}' is larger than the image. "
                "Reduce the text size or thickness."
            )

        # Define the intended top-left coordinate of the label rectangle
        # for each supported position relative to the detection box.
        position_coordinates = {
            "above left": (
                x1,
                y1 - label_height,
            ),
            "above right": (
                x2 - label_width,
                y1 - label_height,
            ),
            "below left": (
                x1,
                y2,
            ),
            "below right": (
                x2 - label_width,
                y2,
            ),
            "inside top left": (
                x1,
                y1,
            ),
            "inside top right": (
                x2 - label_width,
                y1,
            ),
        }

        candidate_x, candidate_y = position_coordinates[text_position]

        # Shift the proposed label only when necessary to keep the complete
        # rectangle within the frame.
        label_x1 = max(
            0,
            min(candidate_x, image_width - label_width),
        )
        label_y1 = max(
            0,
            min(candidate_y, image_height - label_height),
        )

        # Calculate the remaining rectangle coordinates from its clamped
        # top-left corner and measured dimensions.
        label_x2 = label_x1 + label_width
        label_y2 = label_y1 + label_height

        label_rectangle = (
            label_x1,
            label_y1,
            label_x2,
            label_y2,
        )

        # cv2.putText() expects the text baseline origin, not the top-left
        # corner of the text or label rectangle.
        text_origin = (
            label_x1 + padding,
            label_y1 + padding + text_height,
        )

        # A track ID is normally available for video tracking but may be
        # absent for detections obtained from a single image.
        track_id = detection.get("track_id")

        # Preserve the selected position for tracked objects so this function
        # has the same output structure as automatic placement.
        if track_id is not None:
            current_label_positions[track_id] = text_position

        # Store everything required by the separate label-drawing function.
        label_placements.append(
            {
                "label": label,
                "box": box,
                "label_rectangle": label_rectangle,
                "position_name": text_position,
                "text_origin": text_origin,
                "text_size": (text_width, text_height),
                "baseline": baseline,
            }
        )

    # Return the drawing instructions and the position mapping for tracked objects.
    return label_placements, current_label_positions


def draw_labels(
    image,
    detections,
    text_color=(0, 255, 0),
    box_color=(255, 255, 255),
    text_size=0.6,
    text_thickness=1,
    text_background=True,
    text_font=cv2.FONT_HERSHEY_COMPLEX,
    text_position="automatic",
    previous_label_positions=None,
    show_label=True,
    show_confidence=True,
):
    """
    Calculates and draws labels for all detections.

    Args:
        image: Image or video frame where labels are drawn.
        detections: List of detection dictionaries.
        text_color: Text color in BGR format.
        box_color: Label background color in BGR format.
        text_size: Scale applied to the label text.
        text_thickness: Thickness of the label text.
        text_background: Whether to draw a filled background behind the text.
        text_font: OpenCV font used to measure and draw the text.
        text_position: Automatic or predefined label position.
        previous_label_positions: Previous position names indexed by track ID.
        show_label: Whether to include the class name in the label.
        show_confidence: Whether to include the confidence score in the label.

    Returns:
        Image containing the drawn labels.
        Dictionary of updated position names indexed by track ID.
    """
    # Use an empty position history when processing an image or the first
    # frame of a tracked video.
    if previous_label_positions is None:
        previous_label_positions = {}

    # Use the scoring-based placement method when automatic positioning
    # is selected.
    if text_position == "automatic":
        label_placements, current_label_positions = calculate_automatic_label_positions(
            image=image,
            detections=detections,
            previous_label_positions=previous_label_positions,
            font=text_font,
            font_scale=text_size,
            font_thickness=text_thickness,
            show_label=show_label,
            show_confidence=show_confidence,
        )
    # Otherwise, place every label using the selected fixed position.
    else:
        label_placements, current_label_positions = (
            calculate_predefined_label_positions(
                image=image,
                detections=detections,
                text_position=text_position,
                font=text_font,
                font_scale=text_size,
                font_thickness=text_thickness,
                show_label=show_label,
                show_confidence=show_confidence,
            )
        )

    # Draw each label using the coordinates calculated by the selected
    # placement function.
    for placement in label_placements:
        label = placement["label"]

        # Skip empty labels to avoid drawing a label with an empty text.
        if not label:
            continue

        # Retrieve the rectangle used for the optional label background.
        label_x1, label_y1, label_x2, label_y2 = placement["label_rectangle"]

        # Retrieve the baseline origin expected by cv2.putText().
        text_x, text_y = placement["text_origin"]

        # Draw a filled rectangle behind the text when label backgrounds
        # are enabled.
        if text_background:
            cv2.rectangle(
                image,
                (label_x1, label_y1),
                (label_x2, label_y2),
                box_color,
                -1,
            )

        # Draw the label text using the same font settings used when its
        # dimensions and position were calculated.
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            text_font,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

    # Return the modified image and the selected positions that can be
    # reused when processing the next video frame.
    return image, current_label_positions


def draw_detections(
    image,
    detections,
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
):
    """
    Draws bounding boxes and labels for all valid detections.

    Args:
        image: PIL image, NumPy image, or uploaded image file.
        detections: List of detection dictionaries.
        box_color: Bounding-box color as a hexadecimal string.
        label_background_color: Label background color as a hexadecimal string.
        text_color: Label text color as a hexadecimal string.
        box_thickness: Bounding-box line thickness.
        text_size: Scale applied to the label text.
        text_thickness: Thickness of the label text.
        alpha: Opacity of the filled bounding-box interior.
        show_label: Whether to include class names in labels.
        show_confidence: Whether to include confidence scores in labels.
        text_background: Whether to draw a filled background behind each label.
        box_style: Bounding-box style, such as standard, corner, or filled.
        text_font: OpenCV font used to measure and draw labels.
        text_position: Automatic or predefined label position.
        previous_label_positions: Previous position names indexed by track ID.

    Returns:
        Annotated image as a NumPy array.
        Dictionary of current label position names indexed by track ID.
    """
    # Convert the input into the standard NumPy image format expected by
    # the drawing functions.
    output_image = prepare_image(image)

    # Convert hexadecimal colors selected in Streamlit into the BGR format
    # expected by OpenCV drawing functions.
    bgr_box_color = hex_to_bgr(box_color)
    bgr_label_background_color = hex_to_bgr(label_background_color)
    bgr_text_color = hex_to_bgr(text_color)

    # Use an empty position history when processing a single image or the
    # first frame of a tracked video.
    if previous_label_positions is None:
        previous_label_positions = {}

    # Return the prepared image immediately when no objects were detected.
    # Copying the position dictionary prevents accidental modification of
    # the caller's original tracking state.
    if not detections:
        return output_image, previous_label_positions.copy()

    # Store only detections with valid positive-area bounding boxes. These
    # detections will be used for both box drawing and label placement.
    valid_detections = []

    # Draw the bounding box associated with each valid detection.
    for detection in detections:
        # Convert floating-point model coordinates into integer image pixels.
        x1 = int(round(detection["x1"]))
        y1 = int(round(detection["y1"]))
        x2 = int(round(detection["x2"]))
        y2 = int(round(detection["y2"]))

        # Ignore reversed or zero-area bounding boxes because they cannot be
        # drawn or used reliably for label placement.
        if x2 <= x1 or y2 <= y1:
            continue

        valid_detections.append(detection)

        # Draw a corner-only bounding box when the corner style is selected.
        if box_style == "corner":
            draw_corner_box(
                output_image,
                x1,
                y1,
                x2,
                y2,
                bgr_box_color,
                box_thickness,
            )

        # Draw a semi-transparent interior together with the box outline.
        elif box_style == "filled":
            draw_filled_box(
                output_image,
                x1,
                y1,
                x2,
                y2,
                bgr_box_color,
                box_thickness,
                alpha,
            )

        # Use a standard rectangular outline for the "standard" style.
        else:
            cv2.rectangle(
                output_image, (x1, y1), (x2, y2), bgr_box_color, box_thickness
            )

    # Calculate and draw all labels in one operation. Processing labels
    # together allows automatic placement to account for every detection box,
    # previously positioned labels, and tracked positions from the prior frame.
    output_image, current_label_positions = draw_labels(
        image=output_image,
        detections=valid_detections,
        text_color=bgr_text_color,
        box_color=bgr_label_background_color,
        text_size=text_size,
        text_thickness=text_thickness,
        text_background=text_background,
        text_font=text_font,
        text_position=text_position,
        previous_label_positions=previous_label_positions,
        show_label=show_label,
        show_confidence=show_confidence,
    )

    return output_image, current_label_positions


def draw_object_trails(
    image,
    trails,
    trail_color="#ffff00",
    trail_thickness=2,
):
    """
    Draws movement trails for tracked objects.

    Args:
        image: PIL image, NumPy image, or uploaded image file.
        trails: Dictionary mapping track IDs to sequences of (x, y) points.
        trail_color: Trail color as a hexadecimal string.
        trail_thickness: Trail line thickness in pixels.

    Returns:
        Image containing the object trails as a NumPy array.
    """
    # Convert the input into the standard NumPy image format expected by
    # OpenCV drawing functions.
    output_image = prepare_image(image)

    # Convert the hexadecimal color into the BGR format expected by OpenCV.
    bgr_trail_color = hex_to_bgr(trail_color)

    # Return the prepared image unchanged when no tracking history is available.
    if not trails:
        return output_image

    # Retrieve the image dimensions used to constrain trail coordinates.
    # NumPy image shapes follow the order (height, width, channels).
    image_height, image_width = output_image.shape[:2]

    # Process the recorded path belonging to each tracked object.
    for points in trails.values():
        # A visible line requires at least two recorded positions.
        if points is None or len(points) < 2:
            continue

        # Store only valid points converted into image pixel coordinates.
        valid_points = []

        for point in points:
            # Ignore missing or malformed coordinates.
            if point is None or len(point) != 2:
                continue

            point_x, point_y = point

            # Convert potentially floating-point tracking coordinates into
            # integer pixel locations.
            point_x = int(round(point_x))
            point_y = int(round(point_y))

            # Constrain each point to the visible image boundaries.
            point_x = max(0, min(point_x, image_width - 1))
            point_y = max(0, min(point_y, image_height - 1))

            valid_points.append((point_x, point_y))

        # Validation may remove malformed points, so confirm that at least
        # two usable coordinates remain before constructing the path.
        if len(valid_points) < 2:
            continue

        # OpenCV expects polyline coordinates with shape
        # (number_of_points, 1, 2).
        trail_array = np.asarray(
            valid_points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        # Draw one open, anti-aliased path connecting the object's positions
        # in chronological order.
        cv2.polylines(
            output_image,
            [trail_array],
            isClosed=False,
            color=bgr_trail_color,
            thickness=trail_thickness,
            lineType=cv2.LINE_AA,
        )

    # Return the image after drawing all available object trails.
    return output_image
