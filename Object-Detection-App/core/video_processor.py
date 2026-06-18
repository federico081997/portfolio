"""
Uploaded-video processing for object detection.

This file reads a video frame by frame, processes selected frames with the
image processor, writes an annotated video, and combines all detection data.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import pandas as pd

from core.detector import DEFAULT_MODEL_PATH, load_model
from core.image_processor import process_image
from utils.drawing import draw_detections, draw_object_trails


def save_uploaded_video(uploaded_video, output_directory=None):
    """
    Saves a Streamlit uploaded video to a temporary file.

    Args:
        uploaded_video: Video file uploaded through Streamlit.
        output_directory: Optional directory where the temporary file is created.

    Returns:
        Path to the saved temporary video file.
    """
    # Ensure that a video file was provided before attempting to save it.
    if uploaded_video is None:
        raise ValueError("No video was provided.")

    # Preserve the uploaded video's original file extension so OpenCV can
    # recognize the temporary file's container format. Use MP4 when the
    # uploaded filename does not contain an extension.
    original_name = uploaded_video.name
    suffix = Path(original_name).suffix.lower() or ".mp4"

    # Create the requested output directory, including any missing parent
    # directories. No directory is created when the default system temporary
    # directory is used.
    if output_directory is not None:
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Create a uniquely named temporary file. Setting delete=False keeps the
    # file available after it is closed so OpenCV can open it by path.
    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=output_directory,
    )

    try:
        # Write the complete uploaded video content to the temporary file.
        temporary_file.write(uploaded_video.getvalue())
    finally:
        # Always close the file, even if writing fails, so system resources
        # are released and other functions can access the saved path.
        temporary_file.close()

    # Return the filesystem path used by OpenCV or other video-processing tools.
    return temporary_file.name


def create_output_video_path(output_path=None, output_directory=None):
    """
    Creates a filesystem path for the annotated output video.

    Args:
        output_path: Optional complete path for the output video.
        output_directory: Optional directory used for a temporary output file.

    Returns:
        Path to the output video as a string.
    """
    # Use the complete output path supplied by the caller when available.
    if output_path is not None:
        output_path = Path(output_path)

        # Add the default MP4 extension when the provided output path has no file extension.
        if not output_path.suffix:
            output_path = output_path.with_suffix(".mp4")

        # Create the parent directory and any missing intermediate directories
        # before the video writer attempts to create the output file.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return str(output_path)

    # Create the requested output directory when the caller does not provide
    # a complete output path. When this is None, the operating system's
    # default temporary directory will be used.
    if output_directory is not None:
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Generate a unique MP4 file path. Setting delete=False keeps the file
    # available after it is closed so OpenCV can later write the video to it.
    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4",
        dir=output_directory,
    )

    # Close the temporary file immediately because cv2.VideoWriter() must
    # open and manage the output path itself.
    temporary_file.close()

    # Return the generated path for use by the video-processing pipeline.
    return temporary_file.name


def create_video_writer(output_path, width, height, fps, codec="mp4v"):
    """
    Creates and validates an OpenCV video writer.

    Args:
        output_path: Path where the output video will be saved.
        width: Output video width in pixels.
        height: Output video height in pixels.
        fps: Number of frames written per second.
        codec: Four-character code identifying the video codec.

    Returns:
        Initialized OpenCV VideoWriter object.
    """
    # Convert the four-character codec name, such as "mp4v", into the
    # integer codec identifier required by OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*codec)

    # Create the video writer using the requested output path, codec,
    # frame rate, and frame dimensions. Frames passed to writer.write()
    # must have the same width and height.
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    # Confirm that OpenCV successfully opened the output file. Failure may
    # indicate an unsupported codec, an incompatible file extension, or an
    # invalid or inaccessible output path.
    if not writer.isOpened():
        raise RuntimeError(
            "Could not create the output video. "
            "Try using the 'mp4v' codec and an .mp4 output file."
        )

    # Return the initialized writer so frames can be added with writer.write().
    return writer


def convert_video_to_h264(input_path, output_path):
    """
    Converts a video into a browser-compatible H.264 MP4 file.

    Args:
        input_path: Path to the intermediate OpenCV video.
        output_path: Path where the H.264 MP4 video is written.

    Returns:
        Path to the converted H.264 video.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"The intermediate video does not exist: {input_path}")

    if input_path.stat().st_size == 0:
        raise RuntimeError("The intermediate video is empty and cannot be converted.")

    # Find the FFmpeg executable installed on the current system.
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg is not installed or is not available on PATH. "
            "Install FFmpeg to create browser-compatible H.264 videos."
        )

    # Make sure the destination directory exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        ffmpeg_path,
        "-y",
        # Read the OpenCV-generated intermediate video.
        "-i",
        str(input_path),
        # The OpenCV processing pipeline does not preserve source audio.
        "-an",
        # Encode the video using H.264.
        "-c:v",
        "libx264",
        # Use a broadly supported pixel format for browser playback.
        "-pix_fmt",
        "yuv420p",
        # Reasonable balance between conversion speed and compression.
        "-preset",
        "fast",
        "-crf",
        "23",
        # Move MP4 metadata to the beginning of the file so playback can
        # begin before the complete file has been read.
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    completed_process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed_process.returncode != 0:
        # Remove any incomplete output produced by FFmpeg.
        output_path.unlink(missing_ok=True)

        error_message = (
            completed_process.stderr.strip() or "Unknown FFmpeg conversion error."
        )

        raise RuntimeError(
            "Could not convert the processed video to H.264.\n" f"{error_message}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg completed without creating a valid output video.")

    return str(output_path)


def combine_detection_tables(detection_tables):
    """
    Combines frame-level detection DataFrames into one video-level table.

    Args:
        detection_tables: List of detection DataFrames created for individual frames.

    Returns:
        Combined detection DataFrame, or an empty DataFrame when no valid
        frame-level tables are available.
    """
    # Return an empty DataFrame when no frame-level tables were provided.
    if not detection_tables:
        return pd.DataFrame()

    # Keep only existing, non-empty DataFrames. Empty frame tables can occur
    # when the model does not detect any objects in a particular frame.
    valid_tables = [
        table for table in detection_tables if table is not None and not table.empty
    ]

    # Return an empty DataFrame when none of the processed frames contain
    # detection records.
    if not valid_tables:
        return pd.DataFrame()

    # Stack all frame-level rows into one table and create a new continuous
    # index for the combined video detections.
    return pd.concat(valid_tables, ignore_index=True)


def calculate_video_detection_counts(detection_df):
    """
    Counts frame-level detection occurrences for each class.

    Args:
        detection_df: Combined DataFrame containing detections from all frames.

    Returns:
        DataFrame containing each class and its total detection occurrences.
    """
    # Return an empty table with a consistent structure when no detections exist.
    if detection_df is None or detection_df.empty:
        return pd.DataFrame(columns=["class_name", "detection_count"])

    # Count every frame-level detection. The same tracked object may therefore
    # be counted multiple times when it appears across several frames.
    return (
        detection_df["class_name"]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="detection_count")
        .sort_values(by="detection_count", ascending=False)
        .reset_index(drop=True)
    )


def calculate_unique_object_counts(detection_df):
    """
    Counts unique tracked objects for each class in a video.

    Args:
        detection_df: Combined DataFrame containing detections from all frames.

    Returns:
        DataFrame containing each class and its number of unique tracked objects.
    """
    # Return an empty table when no detections are available.
    if detection_df is None or detection_df.empty:
        return pd.DataFrame(columns=["class_name", "unique_object_count"])

    # Remove detections that do not have a valid tracking ID.
    tracked_detections = detection_df.dropna(subset=["track_id", "class_name"]).copy()

    if tracked_detections.empty:
        return pd.DataFrame(columns=["class_name", "unique_object_count"])

    # Assign each track to the class predicted most frequently for that track.
    # This reduces errors caused by occasional class changes between frames.
    track_classes = (
        tracked_detections.groupby("track_id")["class_name"]
        .agg(lambda values: values.mode().iloc[0])
        .reset_index()
    )

    # Count the number of distinct tracks assigned to each class.
    return (
        track_classes["class_name"]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="unique_object_count")
        .sort_values(
            by="unique_object_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def process_video(
    video_source,
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
    show_trails=False,
    trail_color="#ffff00",
    trail_thickness=2,
    max_trail_length=30,
    frame_step=1,
    output_fps=None,
    output_path=None,
    output_directory=None,
    codec="mp4v",
    max_processed_frames=None,
    progress_callback=None,
    tracker="bytetrack.yaml",
    source_name=None,
):
    """
    Processes a video and creates an annotated output video.

    Args:
        video_source: Video path or uploaded video file.
        model: Existing loaded YOLO model.
        model_path: Path or name of the YOLO model.
        confidence_threshold: Minimum detection confidence.
        iou_threshold: IoU threshold used during non-maximum suppression.
        image_size: Image size used during inference.
        device: Device used for inference.
        max_detections: Maximum number of detections per frame.
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
        show_trails: Whether to draw tracked-object movement trails.
        trail_color: Trail color in hexadecimal format.
        trail_thickness: Trail line thickness in pixels.
        max_trail_length: Maximum number of positions retained per track.
        frame_step: Processes every nth source frame.
        output_fps: Optional output video frame rate.
        output_path: Optional complete output video path.
        output_directory: Optional folder for temporary and output files.
        codec: Four-character output video codec.
        max_processed_frames: Optional limit on processed frames.
        progress_callback: Optional function receiving progress from 0 to 1.
        source_name: Optional name stored with detection results.

    Returns:
        Dictionary containing the output video, detections, tables,
        frame summaries, and video-level statistics.
    """
    # Validate the input source before attempting to open the video.
    if video_source is None:
        raise ValueError("No video source was provided.")

    # Convert frame and trail settings into the required numeric types.
    frame_step = int(frame_step)
    trail_thickness = int(trail_thickness)
    max_trail_length = int(max_trail_length)

    # Create a fresh model for this video sequence.
    # Its tracking state cannot leak into another video or camera session.
    video_model = load_model(model_path)

    # Uploaded Streamlit files must be saved temporarily because OpenCV
    # VideoCapture requires a filesystem path. Existing paths can be used
    # directly and must not be deleted after processing.
    input_is_temporary = not isinstance(video_source, (str, Path))

    if input_is_temporary:
        input_path = save_uploaded_video(
            video_source,
            output_directory=output_directory,
        )
    else:
        input_path = str(video_source)

    # Use the uploaded filename or path name unless the caller provides
    # a custom source name.
    if source_name is None:
        source_name = getattr(video_source, "name", None) or Path(input_path).name

    # Initialize OpenCV resources before entering the protected processing
    # block so they can always be released safely in the finally clause.
    capture = None
    writer = None

    try:
        # This is the final browser-compatible H.264 output path returned to
        # Streamlit and used by the download button.
        final_output_path = create_output_video_path(
            output_path=output_path,
            output_directory=output_directory,
        )

        # OpenCV first writes an intermediate MP4V video. It is converted to H.264
        # only after VideoWriter has been released.
        intermediate_output_path = create_output_video_path(
            output_path=None,
            output_directory=output_directory,
        )

        # Open the source video for sequential frame reading.
        capture = cv2.VideoCapture(input_path)

        if not capture.isOpened():
            raise ValueError("Could not open the input video.")

        # Read the source video's metadata from the capture object.
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Some video files do not expose a valid frame rate. Use a reasonable
        # fallback so timestamps and output timing can still be calculated.
        if source_fps <= 0:
            source_fps = 30.0

        # Estimate the full source duration from its frame count and frame rate.
        duration_seconds = total_source_frames / source_fps

        # A video writer cannot be created without valid frame dimensions.
        if video_width <= 0 or video_height <= 0:
            raise ValueError("The input video has invalid dimensions.")

        # When frames are skipped, reduce the output frame rate by the same
        # factor so the output video approximately preserves real-time speed.
        if output_fps is None:
            output_fps = source_fps / frame_step

        output_fps = float(output_fps)

        if output_fps <= 0:
            raise ValueError("output_fps must be greater than zero.")

        # Create the writer that receives each annotated BGR frame.
        writer = create_video_writer(
            output_path=intermediate_output_path,
            width=video_width,
            height=video_height,
            fps=output_fps,
            codec=codec,
        )

        # source_frame_index uses zero-based numbering, so the first frame
        # corresponds to timestamp 0.0 seconds.
        source_frame_index = 0
        processed_frame_count = 0

        # Store the previous positions of tracked labels to reduce visual
        # movement between consecutive processed frames.
        previous_label_positions = None

        # Store the center-point history of each tracked object.
        trails = {}

        # Collect per-frame and per-detection results for video-level analysis.
        frame_summaries = []
        detection_tables = []
        all_detections = []

        # Store the last annotated RGB frame for display in Streamlit.
        last_annotated_frame = None

        while True:
            # Read the next frame. OpenCV returns False when the video ends
            # or when another frame cannot be decoded.
            success, frame = capture.read()

            if not success:
                break

            # Process only every nth source frame according to frame_step.
            if source_frame_index % frame_step == 0:

                # Run detection, annotation, table creation, and frame-level
                # summary generation for the selected source frame.
                frame_result = process_image(
                    image=frame,
                    model=video_model,
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
                    draw_annotations=False,
                    tracking=True,
                    persist=True,
                    tracker=tracker,
                    previous_label_positions=previous_label_positions,
                    frame_number=source_frame_index,
                    video_fps=source_fps,
                    source_name=source_name,
                )

                # Store the track IDs present in the current frame so stale
                # trails belonging to absent objects are not drawn.
                active_track_ids = set()

                # Add the current center position of each tracked detection
                # to its movement history.
                for detection in frame_result["detections"]:
                    track_id = detection.get("track_id")

                    # Detections without a track ID cannot maintain a trail
                    # across multiple video frames.
                    if track_id is None:
                        continue

                    active_track_ids.add(track_id)

                    # Use the bounding-box center as the object's trail point.
                    center_x = (detection["x1"] + detection["x2"]) / 2

                    center_y = (detection["y1"] + detection["y2"]) / 2

                    # Create a new trail when the track first appears.
                    if track_id not in trails:
                        trails[track_id] = []

                    trails[track_id].append((center_x, center_y))

                    # Retain only the most recent points to prevent unbounded
                    # trail growth during long videos.
                    trails[track_id] = trails[track_id][-max_trail_length:]

                # Begin drawing from the original unannotated BGR frame.
                drawing_frame = frame_result["original_image"].copy()

                # Draw movement trails first so bounding boxes and labels are
                # subsequently rendered above the trail lines.
                if show_trails and active_track_ids:
                    active_trails = {
                        track_id: trails[track_id]
                        for track_id in active_track_ids
                        if track_id in trails
                    }

                    drawing_frame = draw_object_trails(
                        image=drawing_frame,
                        trails=active_trails,
                        trail_color=trail_color,
                        trail_thickness=trail_thickness,
                    )

                # Draw bounding boxes and labels after the trails so detection
                # annotations remain fully visible.
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

                # Preserve the final label positions for the next processed
                # frame rather than the unchanged values from process_image().
                previous_label_positions = current_label_positions

                # Collect frame-level and detection-level outputs for the
                # final combined tables and summary.
                frame_summaries.append(frame_result["summary"])
                detection_tables.append(frame_result["detection_df"])
                all_detections.extend(frame_result["detections"])

                # The annotated image is already in BGR format, so it can be written
                # directly to the OpenCV output video.
                writer.write(annotated_frame)

                # Store a copy of the most recently processed frame.
                last_annotated_frame = annotated_frame.copy()

                processed_frame_count += 1

                # Stop once the optional processed-frame limit is reached.
                if (
                    max_processed_frames is not None
                    and processed_frame_count >= max_processed_frames
                ):
                    break

                # Report progress using the current position in the complete
                # source video rather than the processed-frame count.
                if progress_callback is not None and total_source_frames > 0:
                    progress = min((source_frame_index + 1) / total_source_frames, 1.0)
                    progress_callback(progress)

            # Advance the zero-based source frame index after reading a frame.
            source_frame_index += 1

    finally:
        # Always release OpenCV resources, even when inference, annotation,
        # or video writing raises an exception.
        if capture is not None:
            capture.release()

        if writer is not None:
            writer.release()

        # Delete only temporary copies created for uploaded Streamlit files.
        # Existing source paths belong to the caller and must be preserved.
        if input_is_temporary:
            try:
                Path(input_path).unlink(missing_ok=True)
            except OSError:
                pass

    # An empty result usually indicates an unreadable video, an excessive
    # frame step, or a processing limit/configuration problem.
    if processed_frame_count == 0:
        raise ValueError("No video frames were processed.")

    try:
        # Convert the finalized OpenCV video into an H.264 MP4 that browsers
        # and Streamlit's video player can decode.
        output_path = convert_video_to_h264(
            input_path=intermediate_output_path,
            output_path=final_output_path,
        )

    finally:
        # The intermediate MP4V video is no longer needed after conversion.
        try:
            Path(intermediate_output_path).unlink(missing_ok=True)
        except OSError:
            pass

    # Ensure the interface displays complete progress after successful
    # processing, including processing stopped by max_processed_frames.
    if progress_callback is not None:
        progress_callback(1.0)

    # Combine all non-empty frame-level detection tables into one video-level
    # table.
    detection_df = combine_detection_tables(detection_tables)

    # Count frame-level detection occurrences for each class. The same object
    # may be counted once in every frame where it appears.
    class_count_df = calculate_video_detection_counts(detection_df)

    # Count distinct tracked objects for each class when track IDs are present.
    unique_objects_df = calculate_unique_object_counts(detection_df)

    # Convert the individual frame summaries into a single tabular structure.
    frame_summary_df = pd.DataFrame(frame_summaries)

    # total_detections represents detection occurrences across all processed
    # frames, not the number of unique physical objects.
    total_detections = len(all_detections)

    if total_detections == 0:
        unique_classes = 0
        average_confidence = 0.0
        most_common_class = None
    else:
        # Calculate statistics from the combined detection table.
        unique_classes = detection_df["class_name"].nunique()
        average_confidence = detection_df["confidence"].mean()

        # calculate_video_detection_counts() sorts counts in descending
        # order, so the first row contains the most frequently detected class.
        most_common_class = class_count_df.iloc[0]["class_name"]

    # Store video metadata and detection statistics for interface metrics
    # and exported reports.
    summary = {
        "source_name": source_name,
        "video_width": video_width,
        "video_height": video_height,
        "source_fps": round(source_fps, 2),
        "output_fps": round(output_fps, 2),
        "total_source_frames": total_source_frames,
        "processed_frames": processed_frame_count,
        "frame_step": frame_step,
        "duration_seconds": round(duration_seconds, 2),
        "total_detections": total_detections,
        "unique_classes": unique_classes,
        "average_confidence": round(float(average_confidence), 2),
        "most_common_class": most_common_class,
    }

    # Return the output video and every structure needed by the Streamlit
    # interface, analytics components, and export features.
    return {
        "output_video_path": output_path,
        "detections": all_detections,
        "detection_df": detection_df,
        "class_count_df": class_count_df,
        "unique_objects_df": unique_objects_df,
        "frame_summary_df": frame_summary_df,
        "summary": summary,
        "last_annotated_frame": last_annotated_frame,
    }
