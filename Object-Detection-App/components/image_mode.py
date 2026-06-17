"""Streamlit UI for uploaded-image object detection."""

import streamlit as st

from core.image_processor import process_image
from components.shared import (
    dataframe_to_csv_bytes,
    object_to_json_bytes,
    render_class_chart,
    render_detection_metrics,
)


def render_image_mode(config):
    """
    Renders the uploaded-image workflow.

    Args:
        config: Dictionary containing the model, inference settings,
            and drawing configuration selected in the sidebar.
    """
    # Display the title and a short description of the image-processing mode.
    st.markdown("## Upload photo")
    st.caption("Run object detection on a JPG, JPEG, PNG, BMP, or WebP image.")

    # Allow the user to upload one supported image file.
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )

    # Create two compact action columns and one empty column that provides
    # additional horizontal spacing.
    action_columns = st.columns([1, 1, 4])

    # Run detection only when an image has been uploaded and a valid model
    # has been loaded successfully.
    process_clicked = action_columns[0].button(
        "Run detection",
        type="primary",
        width="stretch",
        disabled=uploaded_image is None or config["model"] is None,
    )

    # Clear the previously stored image result without removing the uploaded
    # file from the uploader.
    if action_columns[1].button(
        "Clear result",
        width="stretch",
    ):
        st.session_state["image_result"] = None

    # Inform the user when image processing is unavailable because the model
    # could not be loaded.
    if config["model"] is None:
        st.warning("Load a valid YOLO model from the sidebar before processing.")

    # Process the uploaded image only after the user clicks the detection button.
    if process_clicked:
        try:
            # Display a loading indicator while inference and annotation are
            # being performed.
            with st.spinner("Processing image..."):
                result = process_image(
                    image=uploaded_image,
                    model=config["model"],
                    model_path=config["model_path"],
                    confidence_threshold=config["confidence_threshold"],
                    iou_threshold=config["iou_threshold"],
                    image_size=config["image_size"],
                    device=config["device"],
                    max_detections=config["max_detections"],
                    selected_classes=config["selected_classes"],
                    box_color=config["box_color"],
                    box_thickness=config["box_thickness"],
                    box_style=config["box_style"],
                    alpha=config["alpha"],
                    label_background_color=config["label_background_color"],
                    text_color=config["text_color"],
                    text_size=config["text_size"],
                    text_thickness=config["text_thickness"],
                    show_label=config["show_label"],
                    show_confidence=config["show_confidence"],
                    text_background=config["text_background"],
                    text_font=config["text_font"],
                    text_position=config["text_position"],
                    draw_annotations=True,
                    tracking=False,
                    source_name=uploaded_image.name,
                )

            # Save the complete result in session state so it remains available
            # when Streamlit reruns after tab changes or download-button clicks.
            st.session_state["image_result"] = result

            st.success("Image processing completed.")

        except Exception as error:
            # Display the complete exception and traceback to make development
            # and configuration errors easier to diagnose.
            st.exception(error)

    # Retrieve the latest image-processing result from persistent session state.
    result = st.session_state.get("image_result")

    # Stop rendering the result interface until an image has been processed.
    if not result:
        st.info("Upload an image and select **Run detection** to view results.")
        return

    # Display the main detection statistics above the result tabs.
    render_detection_metrics(result["summary"])

    # Separate the visual results, raw detections, analytics, and downloads
    # into dedicated interface tabs.
    preview_tab, detections_tab, analytics_tab, export_tab = st.tabs(
        ["Preview", "Detections", "Analytics", "Export"]
    )

    with preview_tab:
        # Display the original and annotated images next to each other.
        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown("### Original")

            # The image processor returns a BGR NumPy image, so the BGR channel
            # order must be specified when displaying it with Streamlit.
            st.image(
                result["original_image"],
                channels="BGR",
                width="stretch",
            )

        with right_column:
            st.markdown("### Annotated")

            # Display the processed image containing bounding boxes and labels.
            st.image(
                result["annotated_image"],
                channels="BGR",
                width="stretch",
            )

    with detections_tab:
        # Retrieve the structured detection table created by process_image().
        detection_df = result["detection_df"]

        # Show a message instead of an empty table when no objects were detected.
        if detection_df is None or detection_df.empty:
            st.info("No detections were produced.")
        else:
            # Display one row for every detected object.
            st.dataframe(
                detection_df,
                width="stretch",
                hide_index=True,
            )

    with analytics_tab:
        # Give the class-frequency chart more width than the summary panel.
        chart_column, table_column = st.columns([2, 1])

        with chart_column:
            st.markdown("### Detections by class")

            # Display the number of detected objects belonging to each class.
            render_class_chart(result["class_count_df"])

        with table_column:
            st.markdown("### Summary")

            # Display the image metadata and calculated detection statistics.
            st.json(result["summary"])

    with export_tab:
        st.markdown("### Export image results")

        # Place the CSV and JSON download buttons next to each other.
        download_columns = st.columns(2)

        # Convert the detection DataFrame into downloadable CSV bytes.
        download_columns[0].download_button(
            "Download detections CSV",
            data=dataframe_to_csv_bytes(result["detection_df"]),
            file_name="image_detections.csv",
            mime="text/csv",
            width="stretch",
        )

        # Convert the summary dictionary into downloadable JSON bytes.
        download_columns[1].download_button(
            "Download summary JSON",
            data=object_to_json_bytes(result["summary"]),
            file_name="image_summary.json",
            mime="application/json",
            width="stretch",
        )
