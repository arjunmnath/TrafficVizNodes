#!/usr/bin/env python3
"""Example demonstration script showing the manual 2-step tracking pipeline (Detector + Tracker)."""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the workspace root to Python path so we can import 'reid' package
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(workspace_root))

from reid.tracking.detector import Detector
from reid.tracking.tracker import Tracker


def main() -> None:
    # 1. Paths setup
    video_path = os.path.join(workspace_root, "input_vids", "clip1.mp4")
    model_path = os.path.join(workspace_root, "trained_models", "yolov8s.pt")
    config_path = os.path.join(workspace_root, "reid", "tracking", "config", "bytetrackx.yaml")
    output_video_path = os.path.join(workspace_root, "runs", "detect", "manual_tracking_output.mp4")

    # 2. Check inputs availability
    if not os.path.exists(video_path):
        print(f"[-] Video file not found: {video_path}")
        print("Please place a test video under input_vids/clip1.mp4 or adjust paths.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[-] YOLO weights not found at: {model_path}. Will let Ultralytics download model automatically.")
        model_path = "yolov8s.pt"

    print("[+] Initializing Detector class...")
    detector = Detector(model_path)

    print("[+] Initializing Tracker class with custom config...")
    tracker = Tracker(config_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Failed to open video file: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"[+] Processing {total_frames} frames from video: {video_path}...")
    print(f"    - Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"    - Output: {output_video_path}")

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Step 1: Detect objects in the current frame
            # Target COCO classes: 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
            dets = detector.detect(frame, conf=0.25, classes=[0, 2, 3, 5, 7])

            # Step 2: Extract or mock ReID feature vectors/embeddings for detections
            # In a real pipeline, these would be computed using an embedding extractor (e.g. CLIP, FastReID).
            # Here we mock 128-dimensional L2-normalized feature vectors for demonstration.
            num_dets = len(dets["boxes"])
            if num_dets > 0:
                raw_features = np.random.randn(num_dets, 128).astype(np.float32)
                # Normalize features to be on unit sphere (L2-normalization)
                norms = np.linalg.norm(raw_features, axis=1, keepdims=True)
                features = raw_features / np.maximum(norms, 1e-12)
            else:
                features = np.empty((0, 128), dtype=np.float32)

            # Step 3: Run the tracker update manually
            tracks = tracker.update(
                boxes=dets["boxes"],
                scores=dets["scores"],
                classes=dets["classes"],
                features=features
            )

            # Draw tracked boxes on the frame
            for track in tracks:
                x1, y1, x2, y2, track_id, score, class_id, _ = track
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)

                # Draw bounding box
                color = (0, 255, 0)  # Green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Annotate track ID
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"    - Processed frame {frame_idx}/{total_frames}...")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"[+] Tracking completed! Outputs written to: {output_video_path}")


if __name__ == "__main__":
    main()
