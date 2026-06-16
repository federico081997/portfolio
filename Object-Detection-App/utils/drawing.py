"""
Drawing utilities for object detection results.

This file contains helper functions for drawing bounding boxes, labels,
confidence scores, track IDs, and simple visual styles on images or frames.
"""

import cv2
import numpy as np
from PIL import Image

from core.detector import prepare_image


def hex_to_bgr(hex_color):
    """
    Converts a hex color into a BGR tuple, required by the CV2 module.

    Args:
        hex_color: Color in hex format.

    Returns:
        BGR color tuple.
    """
    # Ensure the input is a string before applying string operations.
    if not isinstance(hex_color, str):
        raise ValueError("Hex color must be a string.")

    # Remove surrounding whitespace and the optional leading "#" character.
    hex_color = hex_color.strip().replace("#", "")

    # A full RGB hex color must contain exactly six characters: RRGGBB.
    if len(hex_color) != 6:
        raise ValueError("Hex color must contain 6 characters.")

    # Split the string into red, green, and blue pairs, then convert
    # each pair from hexadecimal base 16 into a decimal integer.
    rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Reverse the tuple to obtain the BGR format
    return rgb_color[::-1]


def clip_box_to_image(x1, y1, x2, y2, image_width, image_height):
    """
    Keeps bounding box coordinates inside the image.

    Args:
        x1: Left box coordinate.
        y1: Top box coordinate.
        x2: Right box coordinate.
        y2: Bottom box coordinate.
        image_width: Width of the image.
        image_height: Height of the image.

    Returns:
        Clipped bounding box coordinates.
    """
    # Limit the left x-coordinate to the valid horizontal pixel range.
    x1 = max(0, min(x1, image_width - 1))

    # Limit the top y-coordinate to the valid vertical pixel range.
    y1 = max(0, min(y1, image_height - 1))

    # Limit the right x-coordinate to the valid horizontal pixel range.
    x2 = max(0, min(x2, image_width - 1))

    # Limit the bottom y-coordinate to the valid vertical pixel range.
    y2 = max(0, min(y2, image_height - 1))

    # Return the corrected coordinates in x1, y1, x2, y2 order.
    return x1, y1, x2, y2


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
        label_parts.append(f"{detection.get('confidence'):.2f}")

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
    ax1, ax2, ay1, ay2 = rectangle_a
    bx1, bx2, by1, by2 = rectangle_b

    intersection_width = max(0, min(ax2, bx2) - min(ax1, bx1))

    intersection_height = max(0, min(ay2, by2) - min(ay1, by1))

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


# def draw_single_detection(
#     image,
#     detection,
#     box_color=(0, 255, 0),
#     text_color=(255, 255, 255),
#     box_thickness=2,
#     text_size=1.0,
#     text_thickness=2,
#     show_label=True,
#     show_confidence=True,
#     show_track_id=False,
#     show_box=True,
#     text_background=True,
#     box_style="standard",
#     text_position="above",
# ):
#     """
#     Draws one detection on an image.

#     Args:
#         image: Image where the detection is drawn.
#         detection: Detection dictionary.
#         box_color: Bounding box color.
#         text_color: Label text color.
#         box_thickness: Bounding box thickness.
#         text_size: Label text size.
#         text_thickness: Label text thickness.
#         show_label: Whether to show the class name.
#         show_confidence: Whether to show the confidence score.
#         show_track_id: Whether to show the tracking ID.
#         show_box: Whether to draw the bounding box.
#         text_background: Whether to draw a text background.
#         box_style: Bounding box style.
#         text_position: Position of the label.
#     """
#     image_height, image_width = image.shape[:2]

#     x1, y1, x2, y2 = get_box_coordinates(detection)
#     x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, image_width, image_height)

#     if x2 <= x1 or y2 <= y1:
#         return

#     if show_box:
#         if box_style == "corner":
#             draw_corner_box(image, x1, y1, x2, y2, box_color, box_thickness)
#         elif box_style == "filled":
#             draw_filled_box(image, x1, y1, x2, y2, box_color, box_thickness)
#         else:
#             draw_standard_box(image, x1, y1, x2, y2, box_color, box_thickness)

#     label_text = build_label_text(
#         detection,
#         show_label=show_label,
#         show_confidence=show_confidence,
#         show_track_id=show_track_id,
#     )

#     draw_label(
#         image=image,
#         label_text=label_text,
#         x1=x1,
#         y1=y1,
#         x2=x2,
#         y2=y2,
#         text_color=text_color,
#         box_color=box_color,
#         text_size=text_size,
#         text_thickness=text_thickness,
#         text_background=text_background,
#         text_position=text_position,
#     )


# def draw_detections(
#     image,
#     detections,
#     box_color="#00ff00",
#     text_color="#ffffff",
#     box_thickness=2,
#     text_size=1.0,
#     text_thickness=2,
#     show_label=True,
#     show_confidence=True,
#     show_track_id=False,
#     show_box=True,
#     text_background=True,
#     box_style="standard",
#     text_position="above",
# ):
#     """
#     Draws all detections on an image.

#     Args:
#         image: PIL image or NumPy image.
#         detections: List of detection dictionaries.
#         box_color: Bounding box color.
#         text_color: Label text color.
#         box_thickness: Bounding box thickness.
#         text_size: Label text size.
#         text_thickness: Label text thickness.
#         show_label: Whether to show class names.
#         show_confidence: Whether to show confidence scores.
#         show_track_id: Whether to show tracking IDs.
#         show_box: Whether to draw bounding boxes.
#         text_background: Whether to draw text backgrounds.
#         box_style: Bounding box style.
#         text_position: Position of labels.

#     Returns:
#         Annotated RGB image as a NumPy array.
#     """
#     output_image = prepare_image_for_drawing(image)

#     box_color = normalize_color(box_color, default=(0, 255, 0))
#     text_color = normalize_color(text_color, default=(255, 255, 255))

#     box_thickness = int(box_thickness)
#     text_thickness = int(text_thickness)
#     text_size = float(text_size)

#     if detections is None:
#         return output_image

#     for detection in detections:
#         draw_single_detection(
#             image=output_image,
#             detection=detection,
#             box_color=box_color,
#             text_color=text_color,
#             box_thickness=box_thickness,
#             text_size=text_size,
#             text_thickness=text_thickness,
#             show_label=show_label,
#             show_confidence=show_confidence,
#             show_track_id=show_track_id,
#             show_box=show_box,
#             text_background=text_background,
#             box_style=box_style,
#             text_position=text_position,
#         )

#     return output_image


# def draw_object_trails(image, trails, trail_color="#ffff00", trail_thickness=2):
#     """
#     Draws object movement trails.

#     Args:
#         image: PIL image or NumPy image.
#         trails: Dictionary of track IDs and point lists.
#         trail_color: Trail line color.
#         trail_thickness: Trail line thickness.

#     Returns:
#         Image with trails as a NumPy array.
#     """
#     output_image = prepare_image_for_drawing(image)
#     trail_color = normalize_color(trail_color, default=(255, 255, 0))
#     trail_thickness = int(trail_thickness)

#     if not trails:
#         return output_image

#     for points in trails.values():
#         if len(points) < 2:
#             continue

#         for index in range(1, len(points)):
#             start_point = tuple(int(value) for value in points[index - 1])
#             end_point = tuple(int(value) for value in points[index])

#             cv2.line(
#                 output_image,
#                 start_point,
#                 end_point,
#                 trail_color,
#                 trail_thickness,
#             )

#     return output_image


# def draw_counting_line(
#     image,
#     line_position,
#     orientation="horizontal",
#     line_color="#ff0000",
#     line_thickness=2,
# ):
#     """
#     Draws a counting line on an image.

#     Args:
#         image: PIL image or NumPy image.
#         line_position: Position of the counting line.
#         orientation: Line orientation.
#         line_color: Line color.
#         line_thickness: Line thickness.

#     Returns:
#         Image with a counting line as a NumPy array.
#     """
#     output_image = prepare_image_for_drawing(image)
#     image_height, image_width = output_image.shape[:2]

#     line_color = normalize_color(line_color, default=(255, 0, 0))
#     line_thickness = int(line_thickness)

#     if orientation == "vertical":
#         x = int(line_position)
#         x = max(0, min(x, image_width - 1))
#         start_point = (x, 0)
#         end_point = (x, image_height)
#     else:
#         y = int(line_position)
#         y = max(0, min(y, image_height - 1))
#         start_point = (0, y)
#         end_point = (image_width, y)

#     cv2.line(
#         output_image,
#         start_point,
#         end_point,
#         line_color,
#         line_thickness,
#     )

#     return output_image


# def get_detection_center(detection):
#     """
#     Gets the center point of a detection box.

#     Args:
#         detection: Detection dictionary.

#     Returns:
#         Center point as an x, y tuple.
#     """
#     x1, y1, x2, y2 = get_box_coordinates(detection)

#     center_x = int((x1 + x2) / 2)
#     center_y = int((y1 + y2) / 2)

#     return center_x, center_y


# def crop_detection(image, detection):
#     """
#     Crops one detected object from an image.

#     Args:
#         image: PIL image or NumPy image.
#         detection: Detection dictionary.

#     Returns:
#         Cropped object image.
#     """
#     prepared_image = prepare_image_for_drawing(image)
#     image_height, image_width = prepared_image.shape[:2]

#     x1, y1, x2, y2 = get_box_coordinates(detection)
#     x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, image_width, image_height)

#     if x2 <= x1 or y2 <= y1:
#         return None

#     return prepared_image[y1:y2, x1:x2].copy()


# def crop_all_detections(image, detections):
#     """
#     Crops all detected objects from an image.

#     Args:
#         image: PIL image or NumPy image.
#         detections: List of detection dictionaries.

#     Returns:
#         List of cropped object dictionaries.
#     """
#     crops = []

#     if not detections:
#         return crops

#     for index, detection in enumerate(detections):
#         crop = crop_detection(image, detection)

#         if crop is None:
#             continue

#         crops.append(
#             {
#                 "index": index,
#                 "class_name": detection.get("class_name", "object"),
#                 "confidence": detection.get("confidence"),
#                 "image": crop,
#             }
#         )

#     return crops
