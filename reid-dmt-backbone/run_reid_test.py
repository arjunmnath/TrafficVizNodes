#!/usr/bin/env python3
"""
Test script to run vehicle/person ReID on two input videos using the DMT backbone.
Maintains a simple global registry tracking occurrences of unique identities.
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

dmt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if dmt_path not in sys.path:
    sys.path.insert(0, dmt_path)

from config import cfg
from model import make_model


class SimpleRegistry:
    def __init__(self, match_threshold=0.6):
        self.identities = {}  # global_id -> {"embedding": np.ndarray, "general_class": str, "occurrences": []}
        self.next_id = 1
        self.match_threshold = match_threshold

    def match_and_add(self, embedding, video_name, frame_num, timestamp, bbox, local_track_id, class_label):
        best_id = None
        best_sim = -1.0

        # Generalize classes to prevent mismatching vehicles with persons
        general_class = "person" if class_label == "person" else "vehicle"

        # L2-normalize query embedding for cosine similarity
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        for global_id, data in self.identities.items():
            if data["general_class"] != general_class:
                continue

            db_emb = data["embedding"]
            db_norm = db_emb / (np.linalg.norm(db_emb) + 1e-8)
            sim = float(np.dot(emb_norm, db_norm))

            if sim > best_sim:
                best_sim = sim
                best_id = global_id

        occurrence = {
            "video": video_name,
            "frame": frame_num,
            "timestamp_seconds": round(timestamp, 2),
            "bbox": [int(x) for x in bbox],
            "local_track_id": int(local_track_id),
            "class_label": class_label,
            "similarity": round(best_sim, 4) if best_id is not None else 1.0
        }

        if best_id is not None and best_sim >= self.match_threshold:
            self.identities[best_id]["occurrences"].append(occurrence)
            # Update the identity's representative embedding via EMA (alpha = 0.9)
            alpha = 0.9
            updated_emb = alpha * self.identities[best_id]["embedding"] + (1.0 - alpha) * embedding
            self.identities[best_id]["embedding"] = updated_emb
            return best_id, best_sim
        else:
            new_id = self.next_id
            self.next_id += 1
            self.identities[new_id] = {
                "embedding": embedding,
                "general_class": general_class,
                "occurrences": [occurrence]
            }
            return new_id, 1.0

    def get_results_summary(self):
        summary = []
        for global_id, data in self.identities.items():
            summary.append({
                "global_id": global_id,
                "general_class": data["general_class"],
                "occurrences": data["occurrences"]
            })
        return summary


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(p, base_dir):
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def parse_track_file(track_path):
    if not track_path or not os.path.exists(track_path):
        return None
    tracks = {}
    with open(track_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 6:
                try:
                    frame_num = int(parts[0])
                    track_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    
                    # Convert MOT bbox [x, y, w, h] to xyxy [x1, y1, x2, y2]
                    x1 = int(round(x))
                    y1 = int(round(y))
                    x2 = int(round(x + w))
                    y2 = int(round(y + h))
                    
                    if frame_num not in tracks:
                        tracks[frame_num] = []
                    tracks[frame_num].append({
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2]
                    })
                except ValueError:
                    continue
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Test ReID on two videos using DMT backbone")
    parser.add_argument("--video1", type=str, required=True, help="Path to first video file")
    parser.add_argument("--video2", type=str, required=True, help="Path to second video file")
    parser.add_argument("--tracks1", type=str, default="", help="Path to track annotation file for video 1 (optional)")
    parser.add_argument("--tracks2", type=str, default="", help="Path to track annotation file for video 2 (optional)")
    parser.add_argument("--config", type=str, default="AICITY2021_Track2_DMT/configs/stage1/101a_384.yml",
                        help="DMT config file path")
    parser.add_argument("--weights", type=str, default="trained_models/101a_384/v1/resnet101_ibn_a_2.pth",
                        help="Path to trained model weights checkpoint")
    parser.add_argument("--yolo_model", type=str, default="../yolov8s.pt",
                        help="Path to YOLOv8 model file")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="ReID matching threshold")
    parser.add_argument("--output", type=str, default="reid_test_results.json",
                        help="Output path for JSON summary of occurrences")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Maximum frames to process per video (0 for all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run ReID on (cpu, cuda, mps)")
    parser.add_argument("--sample_fps", type=float, default=0.0,
                        help="Sample FPS rate to reduce computational load (0.0 for full video FPS)")
    
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Videos and track paths are relative to current working directory where user runs the command
    video1_path = os.path.abspath(args.video1)
    video2_path = os.path.abspath(args.video2)
    tracks1_path = os.path.abspath(args.tracks1) if args.tracks1 else None
    tracks2_path = os.path.abspath(args.tracks2) if args.tracks2 else None

    # Model configs and weights are relative to script's directory (inside agent-working)
    config_path = resolve_path(args.config, script_dir)
    weights_path = resolve_path(args.weights, script_dir)
    yolo_path = resolve_path(args.yolo_model, script_dir)
    
    # Output path is relative to current working directory
    output_path = os.path.abspath(args.output)

    print("--- Configuring DMT model ---")
    print(f"Loading config: {config_path}")
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["TEST.WEIGHT", weights_path, "MODEL.PRETRAIN_CHOICE", "no"])
    cfg.freeze()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build model using config parameters
    print("Building model backbone...")
    model = make_model(cfg, num_class=0)
    print(f"Loading weights: {weights_path}")
    model.load_param(weights_path)
    model.to(device)
    model.eval()

    # Define standard validation transformations from DMT dataloader
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # Simple registry matching instances across videos
    registry = SimpleRegistry(match_threshold=args.threshold)

    # Class mappings for online tracker
    class_mapping = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    videos = [video1_path, video2_path]
    tracks_by_video = [tracks1_path, tracks2_path]

    for idx, video in enumerate(videos):
        print(f"\n--- Processing Video {idx+1}/{len(videos)}: {os.path.basename(video)} ---")
        if not os.path.exists(video):
            print(f"Error: Video file not found: {video}")
            continue

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Error: Failed to open video {video}")
            continue

        # Look for pre-computed track file
        track_path = tracks_by_video[idx]
        if not track_path:
            # Check default path under camera directory: mtsc/mtsc_tnt_mask_rcnn.txt
            video_dir = os.path.dirname(video)
            default_track = os.path.join(video_dir, "mtsc", "mtsc_tnt_mask_rcnn.txt")
            if os.path.exists(default_track):
                track_path = default_track

        video_tracks = parse_track_file(track_path)
        tracker = None

        if video_tracks:
            print(f"Loaded {sum(len(v) for v in video_tracks.values())} pre-computed tracks from: {track_path}")
        else:
            print("No pre-computed tracks file found. Falling back to YOLOv8 online tracker...")
            from ultralytics import YOLO
            tracker = YOLO(yolo_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = 0
        processed_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval for sampling
        frame_interval = 1
        if args.sample_fps > 0:
            frame_interval = max(1, int(round(fps / args.sample_fps)))
            print(f"Sampling video at {args.sample_fps} FPS (processing every {frame_interval} frames)")

        while True:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

            # Skip frame using fast grab if not in interval
            if frame_interval > 1 and frame_count % frame_interval != 0:
                if not cap.grab():
                    break
                frame_count += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            processed_count += 1
            timestamp = frame_count / fps

            # Print progress every 50 processed frames
            if processed_count % 50 == 0 or processed_count == 1:
                progress_pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"Processing frame {frame_count}/{total_frames} ({progress_pct:.1f}%) - Time: {timestamp:.2f}s")

            # Get track detections for the current frame
            if video_tracks:
                frame_dets = video_tracks.get(frame_count, [])
                if not frame_dets:
                    continue
                boxes = np.array([det["bbox"] for det in frame_dets])
                track_ids = np.array([det["track_id"] for det in frame_dets])
                cls_ids = np.array([2 for _ in frame_dets])  # Default to class_id 2 (car) for dataset tracks
            else:
                # Fallback to YOLO online tracking
                results = tracker.track(
                    frame,
                    classes=list(class_mapping.keys()),
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                if not results or len(results) == 0:
                    continue

                boxes_res = results[0].boxes
                if boxes_res is None or boxes_res.id is None:
                    continue

                boxes = boxes_res.xyxy.cpu().numpy()
                track_ids = boxes_res.id.int().cpu().numpy()
                cls_ids = boxes_res.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]
                # Skip invalid/very small crops
                if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                    continue

                # BGR -> RGB -> PIL
                img = Image.fromarray(crop[..., ::-1])
                img_t = val_transforms(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    if cfg.TEST.FLIP_FEATS == 'on':
                        f2 = model(img_t)
                        img_flip = torch.flip(img_t, [3])
                        f1 = model(img_flip)
                        feat = f2 + f1
                    else:
                        feat = model(img_t)

                    if cfg.TEST.FEAT_NORM == 'yes':
                        feat = torch.nn.functional.normalize(feat, p=2, dim=1)

                    embedding = feat.squeeze(0).cpu().numpy()

                class_label = class_mapping.get(int(cls_id), "unknown")

                # Match crop embedding with SimpleRegistry
                global_id, similarity = registry.match_and_add(
                    embedding=embedding,
                    video_name=os.path.basename(video),
                    frame_num=frame_count,
                    timestamp=timestamp,
                    bbox=[x1, y1, x2, y2],
                    local_track_id=track_id,
                    class_label=class_label
                )

        cap.release()
        print(f"Completed processing {os.path.basename(video)}: {frame_count} frames analyzed.")

    # Generate results summary
    summary = registry.get_results_summary()

    # Write summary report to output JSON
    print(f"\nSaving simple registry occurrences to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # Print a clean human-readable text report
    print("\n" + "="*50)
    print("           GLOBAL REID MATCHING REPORT           ")
    print("="*50)
    print(f"Total Unique Identities Found: {len(summary)}")
    print("-"*50)
    for idx, item in enumerate(summary):
        g_id = item["global_id"]
        g_class = item["general_class"]
        occs = item["occurrences"]
        
        # Count occurrences per video source
        vid_counts = {}
        for o in occs:
            vid_counts[o["video"]] = vid_counts.get(o["video"], 0) + 1

        source_info = ", ".join([f"{v}: {c} frames" for v, c in vid_counts.items()])
        print(f"ID {g_id:03d} ({g_class.upper()}): {len(occs)} total occurrences | {source_info}")
        
        # Print a few sample occurrences
        for i, o in enumerate(occs[:3]):
            print(f"   - Video: {o['video']} | Frame: {o['frame']:4d} | Time: {o['timestamp_seconds']:6.2f}s | Track ID: {o['local_track_id']} | Sim: {o['similarity']:.4f}")
        if len(occs) > 3:
            print(f"   - ... and {len(occs) - 3} more occurrences")
        print("-"*50)


if __name__ == "__main__":
    main()
