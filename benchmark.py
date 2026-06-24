#!/usr/bin/env python3
"""
Performance benchmarking tool for Evaluating ReID module in the CCTV System.
Runs edge tracking and central ReID matching on S06 dataset videos,
evaluates tracking metrics using the official dataset/eval/eval.py script,
and compares baseline tracking against ReID-enabled multi-camera tracking.
"""

import sys
import os
import time
import argparse
import re
import cv2
import numpy as np
import pandas as pd

# Monkey-patch np.asfarray for compatibility with NumPy 2.x (removed in 2.0 but used in motmetrics)
if not hasattr(np, "asfarray"):
    np.asfarray = lambda x, *args, **kwargs: np.asarray(x, *args, dtype=float, **kwargs)


# Add the project root and evaluation folder to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "dataset", "eval"))

from camera_node.tracker import YOLOTracker
from camera_node.reid import ReIDFeatureExtractor
from camera_node.attributes import AttributeExtractor
from reid_server.config import ServerConfig
from reid_server.global_registry import GlobalRegistry
from reid_server.matcher import Matcher
from shared.schemas import TrackEvent, Attributes
from eval import eval as aicity_eval, print_results as aicity_print_results


def parse_args():
    parser = argparse.ArgumentParser(description="ReID Module Evaluation Benchmark")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=[
            "dataset/test/S06/c041/vdo.avi",
            "dataset/test/S06/c042/vdo.avi",
        ],
        help="List of video files to run benchmark on",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=200,
        help="Number of frames to process from each video (-1 for full video)",
    )
    parser.add_argument(
        "--reid_model_name",
        type=str,
        default="resnet101_ibn_a",
        help="DMT backbone name (resnet101_ibn_a, resnext101_ibn_a, etc.)",
    )
    parser.add_argument(
        "--reid_model_path",
        type=str,
        default="agent-working/trained_models/101a_384/v1/resnet101_ibn_a_2.pth",
        help="Path to trained .pth weights",
    )
    parser.add_argument(
        "--reid_flip_augment",
        action="store_true",
        help="Enable horizontal flip TTA in ReID feature extraction",
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.55,
        help="ReID matching threshold (default: 0.55)",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8s.pt",
        help="Path to YOLOv8 model file",
    )
    return parser.parse_args()


def get_camera_id_from_path(video_path):
    """Extract integer camera ID from directory name (e.g. c041 -> 41)."""
    match = re.search(r"c0(\d+)", video_path)
    if match:
        return int(match.group(1))
    # Fallback to hash of path if not matching pattern
    return abs(hash(video_path)) % 1000


def get_mtsc_path(video_path):
    """Find the corresponding MTSC tracking file path."""
    video_dir = os.path.dirname(video_path)
    return os.path.join(video_dir, "mtsc", "mtsc_tnt_mask_rcnn.txt")


def run_pipeline(args, videos_map):
    """Runs the YOLO, ReID, and attribute extraction on the videos, caching outputs."""
    print("\n=============================================")
    print(" 1. Running Tracking & ReID Feature Extraction")
    print("=============================================")
    
    # Initialize edge models
    print(f"Loading YOLOv8 trackers ({args.yolo_model})...")
    trackers = {cid: YOLOTracker(model_path=args.yolo_model, conf=0.4) for cid in videos_map}
    
    print(f"Loading ReID Feature Extractor ({args.reid_model_name})...")
    reid = ReIDFeatureExtractor(
        model_name=args.reid_model_name,
        model_path=args.reid_model_path,
        flip_augment=args.reid_flip_augment,
    )
    
    attributes = AttributeExtractor()
    
    # Open captures
    caps = {}
    fps_map = {}
    total_frames_map = {}
    for cid, path in videos_map.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Failed to open video {path}")
            sys.exit(1)
        caps[cid] = cap
        fps_map[cid] = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total_frames_map[cid] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Camera {cid}: {path} ({total_frames_map[cid]} frames, {fps_map[cid]} FPS)")
        
    # Determine frames count to run
    max_frames_available = min(total_frames_map.values())
    frames_to_process = args.num_frames
    if frames_to_process <= 0 or frames_to_process > max_frames_available:
        frames_to_process = max_frames_available
        
    print(f"Processing {frames_to_process} frames from each video concurrently...")
    
    # Cache to store extracted track features
    # Format: list of dicts containing bbox, embedding, attributes, frame_idx, camera_id
    features_cache = []
    
    t_start = time.time()
    t_det = 0.0
    t_reid = 0.0
    
    for frame_idx in range(1, frames_to_process + 1):
        if frame_idx % 20 == 0 or frame_idx == 1:
            print(f"  Processing frame {frame_idx}/{frames_to_process}...")
            
        for cid, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Simulate timestamp
            timestamp = frame_idx / fps_map[cid]
            video_pos_ms = frame_idx * (1000.0 / fps_map[cid])
            
            # 1. Detection and Tracking
            t_d_start = time.time()
            results = trackers[cid].track(frame)
            t_det += time.time() - t_d_start
            
            if results and results.boxes and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                cls_ids = results.boxes.cls.int().cpu().numpy()
                
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure bbox is within frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue
                        
                    class_label = "person" if cls_id == 0 else "vehicle"
                    
                    # We are evaluating vehicle ReID primarily, but track everything
                    t_r_start = time.time()
                    embedding = reid.extract(crop).tolist()
                    t_reid += time.time() - t_r_start
                    
                    color = attributes.extract_color(crop)
                    type_str = None
                    if class_label == "vehicle":
                        type_str = attributes.extract_type(int(cls_id))
                        
                    # Save event to cache
                    features_cache.append({
                        "camera_id": cid,
                        "track_id": int(track_id),
                        "frame_id": frame_idx,
                        "timestamp": timestamp,
                        "video_pos_ms": video_pos_ms,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], # width, height format
                        "class_label": class_label,
                        "embedding": embedding,
                        "attributes": {
                            "color": color,
                            "type": type_str
                        }
                    })
                    
    # Clean up captures
    for cap in caps.values():
        cap.release()
        
    t_total = time.time() - t_start
    total_processed_frames = frames_to_process * len(caps)
    print(f"\nInference completed in {t_total:.2f}s:")
    print(f"  Total processed frames: {total_processed_frames}")
    print(f"  YOLO Tracking:          {t_det:.2f}s ({total_processed_frames / t_det:.1f} FPS)")
    print(f"  ReID Embedding:         {t_reid:.2f}s ({len(features_cache) / t_reid:.1f} crops/sec)")
    print(f"  Total pipeline speed:   {total_processed_frames / t_total:.1f} FPS")
    print(f"  Total extracted crops:  {len(features_cache)}")
    
    return features_cache, frames_to_process


def evaluate_config(features_cache, videos_map, match_threshold, title):
    """Resolves identities with matching logic and formats the DataFrame."""
    print(f"\nResolving identities for configuration: {title} (threshold={match_threshold})...")
    
    # Initialize server matcher
    config = ServerConfig(match_threshold=match_threshold)
    registry = GlobalRegistry()
    matcher = Matcher(config, registry)
    
    pred_records = []
    
    # Sort events chronologically to simulate real-time stream
    sorted_events = sorted(features_cache, key=lambda e: (e["frame_id"], e["camera_id"]))
    
    for event_data in sorted_events:
        # Build event schema
        event = TrackEvent(
            camera_id=f"cam_{event_data['camera_id']}",
            track_id=event_data["track_id"],
            timestamp=event_data["timestamp"],
            video_pos_ms=event_data["video_pos_ms"],
            bbox=[
                event_data["bbox"][0],
                event_data["bbox"][1],
                event_data["bbox"][0] + event_data["bbox"][2],
                event_data["bbox"][1] + event_data["bbox"][3]
            ], # ReID matcher wants absolute x2, y2 format
            class_label=event_data["class_label"],
            embedding=event_data["embedding"],
            attributes=Attributes(
                color=event_data["attributes"]["color"],
                type=event_data["attributes"]["type"]
            )
        )
        
        # Match central identity
        global_id = matcher.match(event)
        
        # Save output in AI City format
        # CameraId, Id, FrameId, X, Y, Width, Height, Xworld, Yworld
        pred_records.append([
            event_data["camera_id"],
            global_id,
            event_data["frame_id"],
            event_data["bbox"][0],
            event_data["bbox"][1],
            event_data["bbox"][2],
            event_data["bbox"][3],
            -1.0,
            -1.0
        ])
        
    df_pred = pd.DataFrame(
        pred_records,
        columns=["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]
    )
    df_pred = df_pred.astype({
        "CameraId": int,
        "Id": int,
        "FrameId": int,
        "X": int,
        "Y": int,
        "Width": int,
        "Height": int
    })
    return df_pred


def load_ground_truth(videos_map, num_frames):
    """Loads reference ground truth from MTSC tracking baseline files."""
    print("\nLoading reference ground truth from MTSC baseline files...")
    gt_records = []
    for cid, path in videos_map.items():
        mtsc_path = get_mtsc_path(path)
        if not os.path.exists(mtsc_path):
            print(f"Error: MTSC ground truth file not found at {mtsc_path}")
            sys.exit(1)
            
        df_gt = pd.read_csv(mtsc_path, header=None)
        # Filter by frames
        df_gt = df_gt[df_gt[0] <= num_frames]
        
        for _, row in df_gt.iterrows():
            frame_id = int(row[0])
            track_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            
            # Prepend CameraId and append Xworld, Yworld
            gt_records.append([
                cid, track_id, frame_id, x, y, w, h, -1.0, -1.0
            ])
            
    df_gt_all = pd.DataFrame(
        gt_records,
        columns=["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]
    )
    df_gt_all = df_gt_all.astype({
        "CameraId": int,
        "Id": int,
        "FrameId": int,
        "X": int,
        "Y": int,
        "Width": int,
        "Height": int
    })
    print(f"  Loaded {len(df_gt_all)} ground truth records across cameras: {list(videos_map.keys())}")
    return df_gt_all


def main():
    args = parse_args()
    
    # Map video files to camera integers
    videos_map = {get_camera_id_from_path(p): p for p in args.videos}
    
    # Step 1: Run inference pipeline (YOLO + ReID)
    features_cache, processed_frames = run_pipeline(args, videos_map)
    
    # Step 2: Load Ground Truth from MTSC reference files
    df_gt = load_ground_truth(videos_map, processed_frames)
    
    # Step 3: Run ReID matching for Baseline (No matching / Threshold=1.0)
    df_pred_baseline = evaluate_config(
        features_cache, videos_map, match_threshold=1.0, title="Baseline (No Cross-Camera Match)"
    )
    
    # Step 4: Run ReID matching with configured Threshold
    df_pred_reid = evaluate_config(
        features_cache, videos_map, match_threshold=args.match_threshold, title="DMT ReID Enabled"
    )
    
    # Step 5: Perform MOT evaluation via eval.py
    print("\n=============================================")
    print(" 2. Evaluation Results (via dataset/eval/eval.py)")
    print("=============================================")
    
    summary_baseline = None
    summary_reid = None
    
    import traceback
    
    print("\n--- BASELINE (NO REID MATCHING) ---")
    try:
        summary_baseline = aicity_eval(
            df_gt.copy(),
            df_pred_baseline.copy(),
            roidir="dataset/test/S06",
            dstype=""
        )
        aicity_print_results(summary_baseline)
    except Exception as e:
        print(f"Error evaluating baseline: {e}")
        print("Note: Baseline tracking has no cross-camera matches, so all baseline trajectories are filtered out by the evaluator.")
        
    print("\n--- DMT REID ENABLED ---")
    try:
        summary_reid = aicity_eval(
            df_gt.copy(),
            df_pred_reid.copy(),
            roidir="dataset/test/S06",
            dstype=""
        )
        aicity_print_results(summary_reid)
    except Exception as e:
        print(f"Error evaluating ReID configuration: {e}")
        traceback.print_exc()
        
    # Step 6: Print comparative summary
    print("\n=============================================")
    print(" 3. Comparative Summary")
    print("=============================================")
    
    # Extract identity metrics from summaries
    def extract_metrics(summary):
        try:
            row = summary.iloc[-1]
            return {
                "idf1": row["idf1"] * 100,
                "idp": row["idp"] * 100,
                "idr": row["idr"] * 100,
                "idtp": int(row["idtp"]),
                "idfp": int(row["idfp"]),
                "idfn": int(row["idfn"])
            }
        except:
            return None
            
    m_base = extract_metrics(summary_baseline)
    m_reid = extract_metrics(summary_reid)
    
    if m_reid:
        if not m_base:
            m_base = {
                "idf1": 0.0,
                "idp": 0.0,
                "idr": 0.0,
                "idtp": 0,
                "idfp": 0,
                "idfn": df_gt["Id"].nunique()
            }
            
        print(f"{'Metric':<20} | {'ReID Enabled':<20}")
        print("-" * 78)
        for key in ["idf1", "idp", "idr"]:
            base_val = m_base[key]
            reid_val = m_reid[key]
            diff = reid_val - base_val
            print(f"{key.upper() + ' (%)':<20} | {reid_val:20.2f} ")
        for key in ["idtp", "idfp", "idfn"]:
            base_val = m_base[key]
            reid_val = m_reid[key]
            diff = reid_val - base_val
            print(f"{key.upper():<20} | {reid_val:20d}")
    else:
        print("Could not generate comparative summary because ReID evaluation failed.")
            
    base_identities = df_pred_baseline["Id"].nunique()
    reid_identities = df_pred_reid["Id"].nunique()
    print("\nGlobal Identity Statistics:")
    print(f"  Reference Ground Truth unique IDs: {df_gt['Id'].nunique()}")
    print(f"  Baseline resolved unique IDs:     {base_identities} (no cross-camera matching)")
    print(f"  ReID resolved unique IDs:         {reid_identities} (cross-camera matching active)")
    print(f"  Total merged cross-camera tracks:  {base_identities - reid_identities}")
    
    print("\nBenchmark Finished successfully!")


if __name__ == "__main__":
    main()
