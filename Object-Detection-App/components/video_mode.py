"""Streamlit UI for uploaded-video tracking and analysis."""

from pathlib import Path
from functools import partial

import streamlit as st

from core.video_processor import process_video
from components.shared import (
    dataframe_to_csv_bytes,
    object_to_json_bytes,
    read_binary_file,
    render_class_chart,
    render_detection_metrics,
)


def update_video_progress(progress_bar, progress):
    """
    Updates the video-processing progress bar.

    Args:
        progress_bar: Streamlit progress-bar element.
        progress: Processing completion value between 0 and 1.
    """
    # Keep the progress value inside Streamlit's supported range.
    progress = max(0.0, min(float(progress), 1.0))

    # Update both the visual bar and its percentage description.
    progress_bar.progress(
        progress,
        text=f"Processing video · {progress:.0%}",
    )


def render_video_mode(config):
    """
    Renders the uploaded-video workflow.

    Args:
        config: Dictionary containing the model, inference settings,
            and drawing configuration selected in the sidebar.
    """
    # Display the video-mode title and a short explanation of the workflow.
    st.markdown("## Upload video")
    st.caption(
        "Track objects across frames, draw movement trails, and export "
        "video-level analytics."
    )

    # Allow the user to upload one supported video file.
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv", "m4v"],
        key="video_uploader",
    )

    # Divide the video-specific settings into three interface columns.
    settings_column, trails_column, output_column = st.columns(3)

    with settings_column:
        st.markdown("### Frame processing")

        # Process every source frame when the value is 1. Larger values skip
        # intermediate frames and reduce processing time.
        frame_step = st.number_input(
            "Process every nth frame",
            min_value=1,
            max_value=120,
            value=1,
            step=1,
        )

        # Limit the number of processed frames when testing long videos.
        # A value of 0 means that the complete video should be processed.
        max_processed_frames_value = st.number_input(
            "Maximum processed frames",
            min_value=0,
            max_value=100000,
            value=0,
            step=10,
            help="Use 0 to process the complete video.",
        )

        # Select the Ultralytics tracking configuration used to maintain
        # object IDs between frames.
        tracker = st.selectbox(
            "Tracker",
            ["bytetrack.yaml", "botsort.yaml"],
        )

    with trails_column:
        st.markdown("### Movement trails")

        # Control whether tracked object paths are drawn on the output frames.
        show_trails = st.checkbox(
            "Draw object trails",
            value=True,
            key="video_show_trails",
        )

        # Select the movement-trail color. Disable the control when trails
        # are not being drawn.
        trail_color = st.color_picker(
            "Trail color",
            "#ffff00",
            disabled=not show_trails,
        )

        # Control the thickness of movement-trail lines.
        trail_thickness = st.slider(
            "Trail thickness",
            min_value=1,
            max_value=12,
            value=3,
            disabled=not show_trails,
        )

        # Control how many historical center points are retained for each
        # tracked object.
        max_trail_length = st.slider(
            "Trail history length",
            min_value=2,
            max_value=300,
            value=50,
            disabled=not show_trails,
        )

    with output_column:
        st.markdown("### Output")

        # Let the user set an explicit output frame rate. A value of 0 tells
        # process_video() to calculate it from the source FPS and frame step.
        output_fps_value = st.number_input(
            "Output FPS",
            min_value=0.0,
            max_value=240.0,
            value=0.0,
            step=1.0,
            help=(
                "Use 0 to calculate output FPS automatically from the "
                "source FPS and frame step."
            ),
        )

        # Select the codec passed to OpenCV's VideoWriter.
        codec = st.selectbox(
            "Video codec",
            ["mp4v", "avc1"],
        )

        # Set the filename used when the annotated video is downloaded.
        custom_output_name = st.text_input(
            "Output filename",
            value="annotated_video.mp4",
        )

    # Create two compact action columns and one wider empty column.
    action_columns = st.columns([1, 1, 4])

    # Enable video processing only when a file and a valid model are available.
    run_clicked = action_columns[0].button(
        "Process video",
        type="primary",
        width="stretch",
        disabled=uploaded_video is None or config["model"] is None,
    )

    # Remove the previous result from session state.
    if action_columns[1].button(
        "Clear result",
        width="stretch",
    ):
        st.session_state["video_result"] = None

    # Warn the user when processing is unavailable because model loading failed.
    if config["model"] is None:
        st.warning("Load a valid YOLO model from the sidebar before processing.")

    # Create a progress indicator that will be updated by process_video().
    progress_bar = st.progress(
        0.0,
        text="Waiting to process a video.",
    )

    if run_clicked:
        # Convert 0 into None because process_video() interprets None as
        # processing the complete video.
        max_processed_frames = (
            None if max_processed_frames_value == 0 else int(max_processed_frames_value)
        )

        # Convert 0 FPS into None so the video processor calculates the
        # appropriate output frame rate automatically.
        output_fps = None if output_fps_value == 0 else float(output_fps_value)

        # Remove directory information from the user-provided filename.
        # This prevents the filename from being treated as a filesystem path.
        output_name = Path(custom_output_name).name or "annotated_video.mp4"

        # Add an MP4 extension when the user did not provide one.
        if not Path(output_name).suffix:
            output_name += ".mp4"

        try:
            # Display a spinner while the video is decoded, tracked,
            # annotated, analysed, and written to the output file.
            with st.spinner("Running detection, tracking, drawing, and analytics..."):
                result = process_video(
                    video_source=uploaded_video,
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
                    show_trails=show_trails,
                    trail_color=trail_color,
                    trail_thickness=trail_thickness,
                    max_trail_length=max_trail_length,
                    frame_step=int(frame_step),
                    output_fps=output_fps,
                    output_path=None,
                    codec=codec,
                    max_processed_frames=max_processed_frames,
                    progress_callback=partial(
                        update_video_progress,
                        progress_bar,
                    ),
                    tracker=tracker,
                    source_name=uploaded_video.name,
                )

            # Store the requested download filename separately from the
            # temporary output path created by the video processor.
            result["download_name"] = output_name

            # Save the complete result so it remains available across
            # Streamlit reruns and tab interactions.
            st.session_state["video_result"] = result

            # Ensure the progress indicator reaches completion.
            progress_bar.progress(
                1.0,
                text="Video processing completed.",
            )

            st.success("Video processing completed.")

        except Exception as error:
            # Remove the progress indicator and display the complete exception
            # when video processing fails.
            progress_bar.empty()
            st.exception(error)

    # Retrieve the most recently completed video-processing result.
    result = st.session_state.get("video_result")

    # Stop rendering the result interface until a video has been processed.
    if not result:
        st.info("Upload a video and select **Process video** to view results.")
        return

    # Display the main video-level detection statistics.
    render_detection_metrics(
        result["summary"],
        video_mode=True,
    )

    # Separate video output, detection analytics, tracking results,
    # frame-level summaries, and downloads into dedicated tabs.
    preview_tab, detections_tab, tracks_tab, frames_tab, export_tab = st.tabs(
        [
            "Preview",
            "Detection analytics",
            "Unique tracked objects",
            "Frame summaries",
            "Export",
        ]
    )

    with preview_tab:
        try:
            # Read the processed output video as bytes and display it using
            # Streamlit's video player.
            video_bytes = read_binary_file(result["output_video_path"])
            st.video(video_bytes)

        except Exception as error:
            # Display a clear message when the output file cannot be read.
            st.error(str(error))

        # Optionally display the last processed frame below the video.
        if result.get("last_annotated_frame") is not None:
            with st.expander("Last processed frame"):
                st.image(
                    result["last_annotated_frame"],
                    channels="BGR",
                    width="stretch",
                )

    with detections_tab:
        # Give the class-frequency chart more space than the JSON summary.
        chart_column, summary_column = st.columns([2, 1])

        with chart_column:
            st.markdown("### Detection occurrences by class")

            # The current video processor returns the detection count column
            # using the name "count".
            render_class_chart(
                result["class_count_df"],
                count_column="detection_count",
            )

        with summary_column:
            st.markdown("### Video summary")

            # Display source metadata and calculated video statistics.
            st.json(result["summary"])

        st.markdown("### Combined detection table")

        # Display one row for every frame-level detection occurrence.
        if result["detection_df"].empty:
            st.info("No detections were produced.")
        else:
            st.dataframe(
                result["detection_df"],
                width="stretch",
                hide_index=True,
            )

    with tracks_tab:
        # Retrieve the class-level counts of unique persistent track IDs.
        unique_objects_df = result["unique_objects_df"]

        if unique_objects_df is None or unique_objects_df.empty:
            # Unique-object counts cannot be calculated when the detector does
            # not return persistent tracking IDs.
            st.info(
                "No unique tracked objects were available. Confirm that "
                "tracking IDs are present and persistent across frames."
            )

        else:
            # Display the unique-track chart and its underlying table.
            chart_column, table_column = st.columns([2, 1])

            with chart_column:
                render_class_chart(
                    unique_objects_df,
                    count_column="unique_object_count",
                )

            with table_column:
                st.dataframe(
                    unique_objects_df,
                    width="stretch",
                    hide_index=True,
                )

    with frames_tab:
        # Retrieve the summary created for every processed frame.
        frame_summary_df = result["frame_summary_df"]

        if frame_summary_df is None or frame_summary_df.empty:
            st.info("No frame summaries are available.")
        else:
            st.dataframe(
                frame_summary_df,
                width="stretch",
                hide_index=True,
            )

    with export_tab:
        st.markdown("### Download processed outputs")

        try:
            # Read the processed video once and reuse its bytes for the
            # download button.
            video_bytes = read_binary_file(result["output_video_path"])

        except Exception:
            # Use empty bytes when the output video is unavailable so the
            # download button can be disabled safely.
            video_bytes = b""

        # First row contains the annotated video and summary report.
        first_row = st.columns(2)

        first_row[0].download_button(
            "Download annotated video",
            data=video_bytes,
            file_name=result.get(
                "download_name",
                "annotated_video.mp4",
            ),
            mime="video/mp4",
            width="stretch",
            disabled=not video_bytes,
        )

        first_row[1].download_button(
            "Download video summary JSON",
            data=object_to_json_bytes(result["summary"]),
            file_name="video_summary.json",
            mime="application/json",
            width="stretch",
        )

        # Second row contains the detailed CSV exports.
        second_row = st.columns(3)

        second_row[0].download_button(
            "Detection table CSV",
            data=dataframe_to_csv_bytes(result["detection_df"]),
            file_name="video_detections.csv",
            mime="text/csv",
            width="stretch",
        )

        second_row[1].download_button(
            "Unique objects CSV",
            data=dataframe_to_csv_bytes(result["unique_objects_df"]),
            file_name="unique_tracked_objects.csv",
            mime="text/csv",
            width="stretch",
        )

        second_row[2].download_button(
            "Frame summaries CSV",
            data=dataframe_to_csv_bytes(result["frame_summary_df"]),
            file_name="frame_summaries.csv",
            mime="text/csv",
            width="stretch",
        )
