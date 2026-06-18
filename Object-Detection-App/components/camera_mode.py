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
    object_to_json_bytes,
    render_class_chart,
    render_detection_metrics,
    reset_camera_state,
    image_to_png_bytes,
)
from core.detector import load_model


def _build_camera_tracking_config(config, settings):
    """
    Builds the configuration that must remain fixed during tracking.

    Args:
        config: Shared inference configuration.
        settings: Camera-specific configuration.

    Returns:
        Dictionary containing tracking-sensitive settings.
    """
    return {
        "model_path": str(config["model_path"]),
        "device": config["device"],
        "confidence_threshold": float(config["confidence_threshold"]),
        "iou_threshold": float(config["iou_threshold"]),
        "image_size": int(config["image_size"]),
        "max_detections": int(config["max_detections"]),
        "selected_classes": tuple(sorted(config["selected_classes"] or [])),
        "tracker": settings["tracker"],
        "mirror": bool(settings["mirror"]),
    }


def _open_local_camera(
    camera_index,
    width,
    height,
    fps,
    model_path,
    tracker,
    config,
    settings,
):
    """
    Opens a local camera and stores it in Streamlit session state.

    Args:
        camera_index: Index of the local camera device.
        width: Requested camera frame width.
        height: Requested camera frame height.
        fps: Requested camera frame rate.
        model_path: Path to the selected YOLO model.
        tracker: Tracker configuration used by this camera session.
    """
    # Close any previously opened camera and clear all stored camera state
    # before creating a new capture object.
    reset_camera_state(release_capture=True)

    # Create a fresh YOLO model exclusively for this camera session.
    camera_model = load_model(model_path)

    capture = None

    try:
        # Open and validate the camera before loading the YOLO model.
        capture = open_camera(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps,
            buffer_size=1,
        )

    except RuntimeError as error:
        # Convert the low-level OpenCV error into an actionable message that
        # can be displayed cleanly by the Streamlit interface.
        raise RuntimeError(
            f"Could not open camera {camera_index} using "
            f"{width} × {height} at {fps} FPS.\n\n"
            "Please check that:\n"
            "- the selected camera index is correct;\n"
            "- no other application is using the camera;\n"
            "- the requested resolution and FPS are supported;\n"
            "- the camera is available to the operating system running Streamlit."
        ) from error

    # Preserve the capture and isolated model across Streamlit reruns.
    st.session_state["camera_capture"] = capture
    st.session_state["camera_model"] = camera_model
    st.session_state["camera_model_path"] = str(model_path)
    st.session_state["camera_tracker"] = tracker
    st.session_state["camera_tracking_config"] = _build_camera_tracking_config(
        config, settings
    )
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
    camera_model = st.session_state.get("camera_model")
    stored_config = st.session_state.get("camera_tracking_config")

    if capture is None:
        raise RuntimeError("The local camera is not open.")

    if camera_model is None:
        raise RuntimeError("The camera tracking model is not initialized.")

    if stored_config is None:
        raise RuntimeError("The camera tracking configuration is missing.")

    current_config = _build_camera_tracking_config(
        config,
        settings,
    )

    # A persistent tracker cannot safely continue after any tracking-sensitive
    # setting changes.
    if current_config != stored_config:
        reset_camera_state(release_capture=True)

        raise RuntimeError(
            "A tracking-sensitive camera setting changed. "
            "Open the camera again to start a new tracking session."
        )

    active_model_path = st.session_state.get("camera_model_path")
    active_tracker = st.session_state.get("camera_tracker")

    model_changed = active_model_path != str(config["model_path"])
    tracker_changed = active_tracker != settings["tracker"]

    if model_changed or tracker_changed:
        reset_camera_state(release_capture=True)

        raise RuntimeError(
            "The camera model or tracker was changed. "
            "Open the camera again to start a new tracking session."
        )

    # Read one frame from the camera. The frame can optionally be mirrored
    # so the preview behaves like a conventional webcam display.
    frame = read_camera_frame(
        capture,
        mirror=settings["mirror"],
    )

    # Retrieve the model dedicated to this camera session.
    camera_model = st.session_state.get("camera_model")

    if camera_model is None:
        raise RuntimeError(
            "The camera tracking model is not initialized. "
            "Stop and reopen the camera."
        )

    # Process the current frame using YOLO tracking, annotation, movement
    # trails, and frame-level analytics.
    result = process_camera_frame(
        frame=frame,
        model=camera_model,
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

    # Separate detections, class analytics, and exports into dedicated tabs.
    details_tab, counts_tab, export_tab = st.tabs(
        [
            "Current detections",
            "Class counts",
            "Export",
        ]
    )

    with details_tab:
        # Retrieve the structured detection table for the current frame.
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

    with counts_tab:
        # Give the class-frequency chart more width than the summary panel.
        chart_column, summary_column = st.columns([2, 1])

        with chart_column:
            st.markdown("### Detections by class")

            # Display the number of current detections for each class.
            render_class_chart(
                result["class_count_df"],
            )

        with summary_column:
            st.markdown("### Summary")

            # Display frame metadata and detection statistics.
            st.json(
                result["summary"],
            )

    with export_tab:
        st.markdown("### Export current camera frame")

        # Use the current frame number in exported filenames.
        frame_number = result.get("frame_number")

        if frame_number is None:
            frame_name = "camera_frame"
        else:
            frame_name = f"camera_frame_{int(frame_number):06d}"

        detection_df = result["detection_df"]

        # Place the image, CSV, and JSON download buttons side by side.
        download_columns = st.columns(3)

        download_columns[0].download_button(
            "Download annotated frame",
            data=image_to_png_bytes(
                result["annotated_frame"],
            ),
            file_name=f"{frame_name}_annotated.png",
            mime="image/png",
            width="stretch",
            key=f"download_camera_frame_{frame_name}",
        )

        download_columns[1].download_button(
            "Download detections CSV",
            data=dataframe_to_csv_bytes(
                detection_df,
            ),
            file_name=f"{frame_name}_detections.csv",
            mime="text/csv",
            width="stretch",
            key=f"download_camera_csv_{frame_name}",
        )

        download_columns[2].download_button(
            "Download summary JSON",
            data=object_to_json_bytes(
                result["summary"],
            ),
            file_name=f"{frame_name}_summary.json",
            mime="application/json",
            width="stretch",
            key=f"download_camera_summary_{frame_name}",
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

    camera_running = st.session_state["camera_running"]

    # Create controls for selecting the camera device, resolution, and frame rate.
    settings_columns = st.columns(4)

    # Select the physical camera device.
    # Index 0 is commonly the primary webcam, but other capture devices may use
    # indices such as 1, 2, or 3.
    camera_index = settings_columns[0].number_input(
        "Camera index",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        disabled=camera_running,
    )

    # Request the camera's frame width.
    width = settings_columns[1].selectbox(
        "Requested width",
        [576, 640, 960, 1280, 1920],
        index=3,
        disabled=camera_running,
    )

    # Request the camera's frame height.
    height = settings_columns[2].selectbox(
        "Requested height",
        [480, 540, 720, 1080],
        index=0,
        disabled=camera_running,
    )

    # Request the frame rate used by the camera and timestamp calculations.
    fps = settings_columns[3].selectbox(
        "Requested FPS",
        [15, 24, 30, 60],
        index=2,
        disabled=camera_running,
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
            disabled=camera_running,
        )

        # Mirror camera frames horizontally before processing and display.
        mirror = st.checkbox(
            "Mirror preview",
            value=True,
            disabled=camera_running,
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
                camera_index,
                width=width,
                height=height,
                fps=fps,
                model_path=config["model_path"],
                tracker=tracker,
                config=config,
                settings=settings,
            )

            st.success("Camera opened.")

            # Trigger a rerun so the updated camera status and live fragment
            # are rendered immediately.
            st.rerun()

        except Exception as error:
            # Release any partially opened camera and clear camera state when
            # initialization fails.
            reset_camera_state(release_capture=True)
            st.error(str(error))

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
        "Process frame",
        width="stretch",
        disabled=not st.session_state["camera_running"],
    )

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
            "Use **Process frame**."
        )

        # Continue displaying the most recently processed frame.
        _render_camera_result(st.session_state.get("camera_result"))

    else:
        # When the camera is stopped, keep displaying the last stored result
        # until the camera state is fully reset or a new session begins.
        _render_camera_result(st.session_state.get("camera_result"))


def _render_browser_snapshot_result(result):
    """
    Renders a processed browser-camera snapshot.

    Args:
        result: Dictionary returned by process_image().
    """
    if not result:
        st.info("Capture a photo and select **Process photo** " "to view results.")
        return

    # Display the main snapshot-level metrics.
    render_detection_metrics(result["summary"])

    # Separate the visual output, detections, analytics, and exports.
    preview_tab, detections_tab, analytics_tab, export_tab = st.tabs(
        [
            "Preview",
            "Detections",
            "Analytics",
            "Export",
        ]
    )

    with preview_tab:
        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown("### Original")

            st.image(
                result["original_image"],
                channels="BGR",
                width="stretch",
            )

        with right_column:
            st.markdown("### Annotated")

            st.image(
                result["annotated_image"],
                channels="BGR",
                width="stretch",
            )

    with detections_tab:
        detection_df = result["detection_df"]

        if detection_df is None or detection_df.empty:
            st.info("No detections were produced.")

        else:
            st.dataframe(
                detection_df,
                width="stretch",
                hide_index=True,
            )

    with analytics_tab:
        chart_column, summary_column = st.columns([2, 1])

        with chart_column:
            st.markdown("### Detections by class")

            render_class_chart(
                result["class_count_df"],
            )

        with summary_column:
            st.markdown("### Summary")

            st.json(
                result["summary"],
            )

    with export_tab:
        st.markdown("### Export snapshot results")

        # Place the annotated image, detections table, and summary downloads
        # next to each other.
        download_columns = st.columns(3)

        # Download the annotated browser-camera image as a PNG file.
        download_columns[0].download_button(
            "Download annotated photo",
            data=image_to_png_bytes(
                result["annotated_image"],
            ),
            file_name="browser_snapshot_annotated.png",
            mime="image/png",
            width="stretch",
            key="download_browser_snapshot_annotated",
        )

        # Download the current snapshot detections as CSV.
        download_columns[1].download_button(
            "Download detections CSV",
            data=dataframe_to_csv_bytes(
                result["detection_df"],
            ),
            file_name="browser_snapshot_detections.csv",
            mime="text/csv",
            width="stretch",
            key="download_browser_snapshot_csv",
        )

        # Download the snapshot summary as JSON.
        download_columns[2].download_button(
            "Download summary JSON",
            data=object_to_json_bytes(
                result["summary"],
            ),
            file_name="browser_snapshot_summary.json",
            mime="application/json",
            width="stretch",
            key="download_browser_snapshot_summary",
        )


def _render_browser_snapshot(config):
    """
    Renders browser-based camera snapshot processing.

    Args:
        config: Shared model, inference, and drawing configuration.
    """
    st.info(
        "Browser snapshot mode uses the camera available to the browser. "
        "It captures one still image rather than a continuous stream."
    )

    # Ask the browser for camera access and capture one still image.
    snapshot = st.camera_input(
        "Capture a photo",
        key="browser_camera_snapshot",
    )

    # Keep the processing and clearing actions available in a stable layout.
    action_columns = st.columns([1, 1, 4])

    process_clicked = action_columns[0].button(
        "Process photo",
        type="primary",
        width="stretch",
        disabled=(snapshot is None or config["model"] is None),
        key="process_browser_snapshot",
    )

    clear_clicked = action_columns[1].button(
        "Clear result",
        width="stretch",
        key="clear_browser_snapshot_result",
    )

    if clear_clicked:
        st.session_state["browser_snapshot_result"] = None

    if config["model"] is None:
        st.warning("Load a valid YOLO model from the sidebar before processing.")

    if process_clicked:
        try:
            with st.spinner("Processing captured photo..."):
                result = process_image(
                    image=snapshot,
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
                    source_name="Browser camera snapshot",
                )

            # Preserve the complete result across Streamlit reruns.
            st.session_state["browser_snapshot_result"] = result

            st.success("Browser snapshot processing completed.")

        except Exception as error:
            st.exception(error)

    # Retrieve and display the latest result independently of the button click.
    result = st.session_state.get("browser_snapshot_result")

    _render_browser_snapshot_result(result)


def render_camera_mode(config):
    """
    Renders the selected camera workflow.

    Args:
        config: Shared model, inference, and drawing configuration.
    """
    st.markdown("## Live camera")
    st.caption(
        "Run object tracking on a local webcam or process a browser " "camera snapshot."
    )

    # Unlike st.tabs(), a radio selector allows only the selected workflow
    # to be executed during the current Streamlit run.
    camera_workflow = st.radio(
        "Camera workflow",
        options=[
            "Local live camera",
            "Browser snapshot",
        ],
        horizontal=True,
        key="camera_workflow",
    )

    # Release the local webcam when the user switches to browser snapshot
    # mode. Otherwise, the physical camera can remain open even though the
    # local-camera interface and fragment are no longer being rendered.
    if camera_workflow == "Browser snapshot" and st.session_state.get(
        "camera_running", False
    ):
        reset_camera_state(release_capture=True)

    # Execute only the currently selected workflow.
    if camera_workflow == "Local live camera":
        _render_local_camera(config)

    else:
        _render_browser_snapshot(config)
