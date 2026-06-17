"""Streamlit UI for local live-camera and browser-snapshot modes."""

import streamlit as st

from core.camera_processor import (
    get_camera_properties,
    open_camera,
    process_camera_frame,
    read_camera_frame,
)
from core.image_processor import process_image
from components.shared import (
    dataframe_to_csv_bytes,
    render_class_chart,
    render_detection_metrics,
    reset_camera_state,
)


def _open_local_camera(camera_index, width, height, fps):
    """
    Opens a local camera and stores it in Streamlit session state.

    Args:
        camera_index: Index of the local camera device.
        width: Requested camera frame width.
        height: Requested camera frame height.
        fps: Requested camera frame rate.
    """
    # Close any previously opened camera and clear all stored camera state
    # before creating a new capture object.
    reset_camera_state(release_capture=True)

    # Open the selected local camera using the requested resolution and FPS.
    # A buffer size of one reduces latency by avoiding a queue of old frames.
    capture = open_camera(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        buffer_size=1,
    )

    # Store the OpenCV capture object so it remains available across
    # Streamlit reruns.
    st.session_state["camera_capture"] = capture

    # Mark the camera workflow as active.
    st.session_state["camera_running"] = True


def _process_local_camera_frame(config, settings):
    """
    Reads and processes one local-camera frame.

    Args:
        config: Shared model, inference, and drawing configuration.
        settings: Camera-specific tracking and trail settings.

    Returns:
        Dictionary containing the processed frame and detection results.
    """
    # Retrieve the active OpenCV camera capture object from session state.
    capture = st.session_state.get("camera_capture")

    # A frame cannot be read until the camera has been opened successfully.
    if capture is None:
        raise RuntimeError("The local camera is not open.")

    # Read one frame from the camera. The frame can optionally be mirrored
    # so the preview behaves like a conventional webcam display.
    frame = read_camera_frame(
        capture,
        mirror=settings["mirror"],
    )

    # Process the current frame using YOLO tracking, annotation, movement
    # trails, and frame-level analytics.
    result = process_camera_frame(
        frame=frame,
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
        previous_label_positions=st.session_state["camera_label_positions"],
        trails=st.session_state["camera_trails"],
        show_trails=settings["show_trails"],
        trail_color=settings["trail_color"],
        trail_thickness=settings["trail_thickness"],
        max_trail_length=settings["max_trail_length"],
        tracking=True,
        persist=True,
        tracker=settings["tracker"],
        frame_number=st.session_state["camera_frame_number"],
        camera_fps=settings["fps"],
        source_name="Local camera",
    )

    # Store the updated object trails for use by the next camera frame.
    st.session_state["camera_trails"] = result["trails"]

    # Store the current label positions so the next frame can prefer the
    # same positions when they remain suitable.
    st.session_state["camera_label_positions"] = result["current_label_positions"]

    # Advance the frame counter after successfully processing the frame.
    st.session_state["camera_frame_number"] += 1

    # Store the complete result so it remains visible across Streamlit reruns.
    st.session_state["camera_result"] = result

    # Return the current result for immediate display.
    return result


def _render_camera_result(result):
    """
    Renders the current local-camera result.

    Args:
        result: Dictionary containing the processed camera-frame results.
    """
    # Display an instruction until at least one camera frame has been processed.
    if not result:
        st.info("Open the camera to begin live processing.")
        return

    # Display the current annotated camera frame.
    # The processing pipeline returns frames in BGR channel order.
    st.image(
        result["annotated_frame"],
        channels="BGR",
        width="stretch",
    )

    # Display the main detection statistics for the current frame.
    render_detection_metrics(result["summary"])

    # Separate detailed detections, class counts, and tracking diagnostics
    # into dedicated tabs.
    details_tab, counts_tab, diagnostics_tab = st.tabs(
        [
            "Current detections",
            "Class counts",
            "Tracking diagnostics",
        ]
    )

    with details_tab:
        # Retrieve the structured table containing detections from the
        # current camera frame.
        detection_df = result["detection_df"]

        if detection_df is None or detection_df.empty:
            st.info("No objects are currently detected.")

        else:
            # Display one row for each object detected in the current frame.
            st.dataframe(
                detection_df,
                width="stretch",
                hide_index=True,
            )

            # Allow the current frame's detection table to be downloaded.
            st.download_button(
                "Download current detections CSV",
                data=dataframe_to_csv_bytes(detection_df),
                file_name="camera_detections.csv",
                mime="text/csv",
            )

    with counts_tab:
        # Display the number of current detections belonging to each class.
        render_class_chart(result["class_count_df"])

    with diagnostics_tab:
        # Display low-level tracking information useful for debugging and
        # understanding persistent object IDs and trail histories.
        st.json(
            {
                # Sort track IDs so the output remains easy to inspect.
                "active_track_ids": sorted(result.get("active_track_ids", [])),
                # Store the number of retained trail points for every active
                # tracked object.
                "active_trail_lengths": {
                    str(track_id): len(points)
                    for track_id, points in result.get(
                        "active_trails",
                        {},
                    ).items()
                },
                # Include frame and timestamp metadata.
                "frame_number": result.get("frame_number"),
                "timestamp_seconds": result.get("timestamp_seconds"),
            }
        )


def _render_local_camera(config):
    """
    Renders local OpenCV webcam controls and preview.

    Args:
        config: Shared model, inference, and drawing configuration.
    """
    # Explain that OpenCV accesses the camera attached to the machine running
    # Streamlit, which may not be the user's browser device on remote hosting.
    st.info(
        "Local live camera mode accesses the webcam attached to the machine "
        "running Streamlit. On a remote server, use the browser snapshot tab."
    )

    # Create controls for selecting the camera and requesting its resolution
    # and frame rate.
    settings_columns = st.columns(4)

    # Select the physical camera device. Index zero is normally the default
    # integrated or primary webcam.
    camera_index = settings_columns[0].number_input(
        "Camera index",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    # Request the camera's frame width.
    width = settings_columns[1].selectbox(
        "Requested width",
        [640, 960, 1280, 1920],
        index=0,
    )

    # Request the camera's frame height.
    height = settings_columns[2].selectbox(
        "Requested height",
        [480, 540, 720, 1080],
        index=0,
    )

    # Request the frame rate used by the camera and timestamp calculations.
    fps = settings_columns[3].selectbox(
        "Requested FPS",
        [15, 24, 30, 60],
        index=2,
    )

    # Create separate sections for trail settings, tracking settings, and
    # the current camera status.
    trails_column, tracker_column, preview_column = st.columns(3)

    with trails_column:
        # Enable or disable movement trails for tracked objects.
        show_trails = st.checkbox(
            "Draw movement trails",
            value=True,
            key="camera_show_trails",
        )

        # Select the trail color. Disable the control when trails are hidden.
        trail_color = st.color_picker(
            "Camera trail color",
            "#ffff00",
            disabled=not show_trails,
        )

        # Set the movement-trail line thickness.
        trail_thickness = st.slider(
            "Camera trail thickness",
            min_value=1,
            max_value=12,
            value=3,
            disabled=not show_trails,
        )

        # Set the maximum number of historical center points retained for
        # each tracked object.
        max_trail_length = st.slider(
            "Camera trail history",
            min_value=2,
            max_value=300,
            value=50,
            disabled=not show_trails,
        )

    with tracker_column:
        # Select the Ultralytics tracking configuration.
        tracker = st.selectbox(
            "Camera tracker",
            ["bytetrack.yaml", "botsort.yaml"],
        )

        # Mirror camera frames horizontally before processing and display.
        mirror = st.checkbox(
            "Mirror preview",
            value=True,
        )

    with preview_column:
        st.markdown("### Camera status")

        # Display a visual indicator showing whether the camera loop is active.
        if st.session_state["camera_running"]:
            st.markdown(
                '<span class="status-pill">● Running</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill">○ Stopped</span>',
                unsafe_allow_html=True,
            )

        # Retrieve and display the actual camera properties reported by OpenCV.
        capture = st.session_state.get("camera_capture")

        if capture is not None:
            try:
                properties = get_camera_properties(capture)

                st.caption(
                    f'{properties["width"]} × '
                    f'{properties["height"]} · '
                    f'{properties["fps"]:.1f} FPS'
                )

            except Exception:
                # The preview should remain usable even if OpenCV cannot
                # report one or more camera properties.
                pass

    # Create controls for opening, stopping, and manually advancing the camera.
    button_columns = st.columns([1, 1, 1, 3])

    # Open the selected camera. Disable this action when no model is loaded.
    if button_columns[0].button(
        "Open camera",
        type="primary",
        width="stretch",
        disabled=config["model"] is None,
    ):
        try:
            _open_local_camera(
                camera_index=camera_index,
                width=width,
                height=height,
                fps=fps,
            )

            st.success("Camera opened.")

            # Trigger a rerun so the updated camera status and live fragment
            # are rendered immediately.
            st.rerun()

        except Exception as error:
            # Release any partially opened camera and clear camera state when
            # initialization fails.
            reset_camera_state(release_capture=True)
            st.exception(error)

    # Stop and release the active camera.
    if button_columns[1].button(
        "Stop camera",
        width="stretch",
    ):
        reset_camera_state(release_capture=True)
        st.rerun()

    # Allow one frame to be processed manually. This is useful for testing
    # and for Streamlit versions without fragment-based automatic refresh.
    process_one_frame = button_columns[2].button(
        "Process next frame",
        width="stretch",
        disabled=not st.session_state["camera_running"],
    )

    # Collect the camera-specific settings into one dictionary so they can
    # be passed to the frame-processing helper.
    settings = {
        "fps": fps,
        "mirror": mirror,
        "show_trails": show_trails,
        "trail_color": trail_color,
        "trail_thickness": trail_thickness,
        "max_trail_length": max_trail_length,
        "tracker": tracker,
    }

    if process_one_frame:
        try:
            # Read, process, and display exactly one frame.
            result = _process_local_camera_frame(
                config,
                settings,
            )

            _render_camera_result(result)

        except Exception as error:
            # Release the camera when frame reading or inference fails.
            reset_camera_state(release_capture=True)
            st.exception(error)

        # Prevent the automatic-refresh branch from also processing a frame
        # during the same Streamlit run.
        return

    # Streamlit fragments can rerun independently from the rest of the page,
    # which allows the camera preview to update without blocking all controls.
    if st.session_state["camera_running"] and hasattr(st, "fragment"):

        @st.fragment(run_every="150ms")
        def live_camera_fragment():
            """
            Processes and renders frames during automatic camera refresh.
            """
            # Stop immediately when the camera is no longer active.
            if not st.session_state["camera_running"]:
                return

            try:
                # Process the next available camera frame.
                result = _process_local_camera_frame(
                    config,
                    settings,
                )

                # Render the newly processed result inside the fragment.
                _render_camera_result(result)

            except Exception as error:
                # Release the camera and stop automatic processing after
                # a frame-reading or inference error.
                reset_camera_state(release_capture=True)
                st.exception(error)

        # Render and activate the automatically refreshing fragment.
        live_camera_fragment()

    elif st.session_state["camera_running"]:
        # Older Streamlit versions may not provide fragment-based automatic
        # refresh, so manual frame processing is required.
        st.warning(
            "Automatic live refresh is unavailable in this Streamlit version. "
            "Use **Process next frame**."
        )

        # Continue displaying the most recently processed frame.
        _render_camera_result(st.session_state.get("camera_result"))

    else:
        # When the camera is stopped, keep displaying the last stored result
        # until the camera state is fully reset or a new session begins.
        _render_camera_result(st.session_state.get("camera_result"))


def _render_browser_snapshot(config):
    """
    Renders browser-based camera snapshot processing.

    Args:
        config: Shared model, inference, and drawing configuration.
    """
    # Explain that this mode uses the browser camera but captures only a
    # single still image rather than a continuous video stream.
    st.info(
        "Browser snapshot mode uses the camera available to the browser. "
        "It captures one still image rather than a continuous stream."
    )

    # Let the browser request camera access and capture one image.
    snapshot = st.camera_input(
        "Capture a photo",
        key="browser_camera_snapshot",
    )

    # Stop until the user captures a photo.
    if snapshot is None:
        return

    # Process the captured snapshot only after the user explicitly requests it.
    if st.button(
        "Process captured photo",
        type="primary",
        disabled=config["model"] is None,
    ):
        try:
            with st.spinner("Processing captured photo..."):
                result = process_image(
                    # Image captured by the browser camera.
                    image=snapshot,
                    # Reuse the cached YOLO model.
                    model=config["model"],
                    model_path=config["model_path"],
                    # Shared model-inference settings.
                    confidence_threshold=config["confidence_threshold"],
                    iou_threshold=config["iou_threshold"],
                    image_size=config["image_size"],
                    device=config["device"],
                    max_detections=config["max_detections"],
                    selected_classes=config["selected_classes"],
                    # Bounding-box appearance settings.
                    box_color=config["box_color"],
                    box_thickness=config["box_thickness"],
                    box_style=config["box_style"],
                    alpha=config["alpha"],
                    # Label appearance settings.
                    label_background_color=config["label_background_color"],
                    text_color=config["text_color"],
                    text_size=config["text_size"],
                    text_thickness=config["text_thickness"],
                    show_label=config["show_label"],
                    show_confidence=config["show_confidence"],
                    text_background=config["text_background"],
                    text_font=config["text_font"],
                    text_position=config["text_position"],
                    # Draw the final annotations during image processing.
                    draw_annotations=True,
                    # Tracking is unnecessary for one captured image.
                    tracking=False,
                    # Store a readable source label in the result metadata.
                    source_name="Browser camera snapshot",
                )

            # Display the snapshot-level detection statistics.
            render_detection_metrics(result["summary"])

            # Show the original and annotated snapshots side by side.
            columns = st.columns(2)

            with columns[0]:
                st.markdown("### Original")

                st.image(
                    result["original_image"],
                    channels="BGR",
                    width="stretch",
                )

            with columns[1]:
                st.markdown("### Annotated")

                st.image(
                    result["annotated_image"],
                    channels="BGR",
                    width="stretch",
                )

            # Display the structured detection results.
            detection_df = result["detection_df"]

            if detection_df is None or detection_df.empty:
                st.info("No detections were produced.")
            else:
                st.dataframe(
                    detection_df,
                    width="stretch",
                    hide_index=True,
                )

        except Exception as error:
            # Display the complete exception to support debugging.
            st.exception(error)


def render_camera_mode(config):
    """
    Renders live-camera and browser-snapshot workflows.

    Args:
        config: Shared model, inference, and drawing configuration.
    """
    # Display the camera-mode heading and workflow description.
    st.markdown("## Live camera")
    st.caption(
        "Run object tracking on a local webcam or process a browser camera " "snapshot."
    )

    # Separate continuous local-camera processing from browser snapshot mode.
    local_tab, browser_tab = st.tabs(["Local live camera", "Browser snapshot"])

    with local_tab:
        # Render OpenCV-based continuous local camera processing.
        _render_local_camera(config)

    with browser_tab:
        # Render browser-based single-image capture and detection.
        _render_browser_snapshot(config)
