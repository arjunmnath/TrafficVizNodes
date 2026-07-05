#!/usr/bin/env python3
"""
Test script to run vehicle/person ReID on two input videos using the DMT backbone.
Maintains a simple global registry tracking occurrences of unique identities.
Refactored to support both headless mode (for servers) and UI mode (for live monitoring).
"""

import os
import sys
import argparse

# Setup sys.path for DMT imports
dmt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if dmt_path not in sys.path:
    sys.path.insert(0, dmt_path)

from reid_pipeline import ReIDPipeline, resolve_path, get_device
from reid_ensemble_pipeline import EnsembleReIDPipeline
from reid_ui import RichUIListener, HeadlessUIListener


def main():
    parser = argparse.ArgumentParser(description="Test ReID on two videos using DMT backbone")
    parser.add_argument("--video1", type=str, default='input_vids/clip1.mp4', help="Path to first video file")
    parser.add_argument("--video2", type=str, default='input_vids/clip2.mp4', help="Path to second video file")
    parser.add_argument("--weights", type=str, default="trained_models/101a_384/v1/resnet101_ibn_a_2.pth",
                        help="Path to trained model weights checkpoint (for single model)")
    parser.add_argument("--yolo_model", type=str, default="yolov8s.pt",
                        help="Path to YOLOv8 model file")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="ReID matching threshold")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for JSON summary of occurrences")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Maximum frames to process per video (0 for all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run ReID on (cpu, cuda, mps, auto)")
    parser.add_argument("--sample_fps", type=float, default=5,
                        help="Sample FPS rate to reduce computational load (0.0 for full video FPS)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (no interactive terminal UI)")
    
    # Ensemble arguments
    parser.add_argument("--ensemble", action="store_true",
                        help="Run using the ensembled ReID pipeline instead of single model")
    parser.add_argument("--model_dir", type=str, default="trained_models",
                        help="Directory containing the ensembled models")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="Comma-separated paths to specific model checkpoints for the ensemble")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "mean"],
                        help="Embedding fusion method for ensemble (concat, mean)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Enable FP16 half-precision inference for ensemble")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16",
                        help="Disable FP16 half-precision inference for ensemble")
    
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    video1_path = os.path.abspath(args.video1)
    video2_path = os.path.abspath(args.video2)

    weights_path = resolve_path(args.weights, script_dir)
    yolo_path = resolve_path(args.yolo_model, script_dir)
    
    output_path = os.path.abspath(args.output)

    videos = [video1_path, video2_path]

    # Select appropriate listener based on mode
    if args.headless:
        listener = HeadlessUIListener(videos)
    else:
        listener = RichUIListener(videos)

    # Determine device
    device_to_show = args.device if args.device != "auto" else get_device()

    # Show configuration
    config_data = {
        "Video Sources": videos,
        "YOLO Model": yolo_path,
        "ReID Threshold": f"{args.threshold:.2f}",
        "Device": str(device_to_show),
        "Max Frames": str(args.max_frames) if args.max_frames > 0 else "All",
        "Sample FPS": str(args.sample_fps) if args.sample_fps > 0 else "Full FPS",
        "Output Path": output_path,
        "Pipeline Mode": "Ensemble" if args.ensemble else "Single Model"
    }

    if args.ensemble:
        model_paths = None
        if args.model_paths:
            model_paths = [resolve_path(p.strip(), script_dir) for p in args.model_paths.split(",")]
        model_dir = resolve_path(args.model_dir, script_dir)
        
        if args.model_paths:
            config_data["Ensemble Model Paths"] = args.model_paths
        else:
            config_data["Ensemble Model Dir"] = model_dir
        config_data["Ensemble Fusion"] = args.fusion
        config_data["FP16 Enabled"] = str(args.fp16)
    else:
        config_data["DMT Weights"] = weights_path

    listener.show_configuration(config_data)

    # Create and execute the pipeline
    if args.ensemble:
        pipeline = EnsembleReIDPipeline(
            model_dir=model_dir,
            model_paths=model_paths,
            yolo_path=yolo_path,
            threshold=args.threshold,
            device=args.device,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            output_path=output_path,
            fp16=args.fp16,
            fusion=args.fusion
        )
    else:
        pipeline = ReIDPipeline(
            weights_path=weights_path,
            yolo_path=yolo_path,
            threshold=args.threshold,
            device=args.device,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            output_path=output_path
        )

    pipeline.initialize(listener)
    pipeline.run(videos, listener)


if __name__ == "__main__":
    main()
