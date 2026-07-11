#!/usr/bin/env python3
"""
Object tracking coordinator running Ultralytics YOLOv8 detectors and bytetrack/botsort config trackers on video streams.
Outputs an annotated video with bounding boxes and track labels.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Add workspace root to python path to import app modules if needed
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from shared.utils import setup_logger

logger = setup_logger("RunTracking")


def get_color(track_id: int) -> tuple[int, int, int]:
    """Generate a deterministic and visually distinct BGR color for a track ID."""
    np.random.seed(track_id)
    color = tuple(map(int, np.random.randint(0, 256, size=3)))
    # Ensure color is not too dark to be visible on dark backgrounds
    if sum(color) < 150:
        # Boost color brightness
        color = tuple(min(255, c + 100) for c in color)
    return color  # type: ignore


def draw_box(
    frame: np.ndarray, bbox: np.ndarray, label: str, color: tuple[int, int, int], thickness: int = 2
) -> None:
    """Draw a beautifully styled bounding box and label on the frame."""
    x1, y1, x2, y2 = map(int, bbox)

    # Draw main bounding box rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Font setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1

    # Get size of text
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Position text slightly above the box, or inside if too close to top edge
    text_y = y1 - 5 if y1 - 5 - text_h > 0 else y1 + text_h + 5
    text_x = x1

    # Bound check position coordinates
    text_y = max(0, min(frame.shape[0] - 5, text_y))
    text_x = max(0, min(frame.shape[1] - text_w - 5, text_x))

    # Draw background box for text to make it extremely readable
    cv2.rectangle(
        frame, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), color, -1
    )

    # Determine text color based on background brightness
    b, g, r = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    # Draw text label
    cv2.putText(
        frame, label, (text_x, text_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics YOLOv8 tracking on a video and output annotated results."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the annotated output video will be saved.",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--model_path",
        type=str,
        default="trained_models/yolov8s.pt",
        help="Path to YOLO model weights (default: yolov8s.pt).",
        dest="model_path",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration filename (e.g. bytetrack.yaml, botsort.yaml) or custom config YAML path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Object detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 2, 3, 5, 7],
        help="COCO class IDs to track. Default: [0, 2, 3, 5, 7] (person, car, motorcycle, bus, truck). Use empty string or None to track all.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum frames to process (-1 for full video). Useful for quick testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check model path
    if not os.path.exists(args.model_path) and args.model_path == "trained_models/yolov8s.pt":
        logger.info(
            f"YOLO model weight file {args.model_path} not found in path, will download automatically."
        )

    logger.info(f"Loading YOLO model from: {args.model_path}")
    model = YOLO(args.model_path)

    # Open input video
    if not os.path.exists(args.video_path):
        logger.error(f"Input video file not found: {args.video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {args.video_path}")
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up classes list (if empty, set classes=None to track all classes)
    classes = args.classes if args.classes else None

    # Determine frames to process
    frames_to_process = total_frames
    if args.max_frames > 0:
        frames_to_process = min(args.max_frames, total_frames)

    logger.info(f"Input video resolution: {width}x{height} @ {fps:.2f} FPS")
    logger.info(f"Total video frames: {total_frames}, processing: {frames_to_process}")
    logger.info(f"Tracker configuration: {args.tracker}")
    logger.info(f"Tracking classes: {classes}")

    # Initialize video writer
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output codec based on extension
    ext = Path(args.output_path).suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    try:
        # Initialize tqdm progress bar
        with tqdm(total=frames_to_process, desc="Tracking progress") as pbar:
            frame_idx = 0
            while cap.isOpened() and frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform tracking using YOLOv8
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    classes=classes,
                    verbose=False,
                )

                # Check results and annotate boxes
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        track_ids = result.boxes.id.int().cpu().numpy()
                        cls_ids = result.boxes.cls.int().cpu().numpy()

                        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                            class_name = result.names[int(cls_id)]
                            label = f"{class_name} - {track_id}"
                            color = get_color(int(track_id))
                            draw_box(frame, box, label, color)

                # Write the annotated frame to output video
                out.write(frame)

                frame_idx += 1
                pbar.update(1)

        logger.info(f"Tracking finished. Output video saved to: {args.output_path}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
