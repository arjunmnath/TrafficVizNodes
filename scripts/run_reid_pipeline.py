#!/usr/bin/env python3
"""
Pipeline runner to perform cross-video person and vehicle re-identification tracking using YOLOv8 detectors and ResNet features.
Maintains a simple global registry tracking occurrences of unique identities.
Supports both headless mode (for servers) and UI mode (for live monitoring).
"""

import os
import sys
import json
import argparse
import numpy as np

# Add workspace root to python path to import app modules
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from reid import (
    ReIDPipeline,
    SimpleRegistry,
    RichUIListener,
    HeadlessUIListener,
    resolve_path,
    VideoFeederStage,
    SamplerStage,
    YoloDetectionStage,
    SingleModelFeatureStage,
    EnsembleModelFeatureStage,
    TrackingStage,
)


def export_results(registry: SimpleRegistry, output_path: str) -> None:
    """Export the registry results to JSON and embeddings to NPZ, outside pipeline scope.

    Args:
        registry (SimpleRegistry): The global identity registry.
        output_path (str): Output path for JSON summary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    summary = registry.get_results_summary()
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    embeddings = registry.get_embeddings_dict()
    if embeddings:
        npz_path = os.path.splitext(output_path)[0] + ".npz"
        np.savez(npz_path, **embeddings)


def main():
    parser = argparse.ArgumentParser(description="Test ReID on two videos using DMT backbone")
    parser.add_argument(
        "--video1", type=str, required=True, help="Path to first video file"
    )
    parser.add_argument(
        "--video2", type=str, required=True, help="Path to second video file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="trained_models/101a_384/v1/resnet101_ibn_a_2.pth",
        help="Path to trained model weights checkpoint (for single model)",
    )
    parser.add_argument(
        "--yolo_model", type=str, default="trained_models/yolov8s.pt", help="Path to YOLOv8 model file"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="ReID matching threshold")
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for JSON summary of occurrences"
    )
    parser.add_argument(
        "--max_frames", type=int, default=0, help="Maximum frames to process per video (0 for all)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run ReID on (cpu, cuda, mps, auto)"
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=5,
        help="Sample FPS rate to reduce computational load (0.0 for full video FPS)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no interactive terminal UI)"
    )

    # Ensemble arguments
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run using the ensembled ReID pipeline instead of single model",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="trained_models",
        help="Directory containing the ensembled models",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        default=None,
        help="Comma-separated paths to specific model checkpoints for the ensemble",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=["concat", "mean"],
        help="Embedding fusion method for ensemble (concat, mean)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 half-precision inference for ensemble",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_false",
        dest="fp16",
        help="Disable FP16 half-precision inference for ensemble",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration filename (e.g. bytetrack.yaml, botsort.yaml) or custom config YAML path",
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))

    video1_path = os.path.abspath(args.video1)
    video2_path = os.path.abspath(args.video2)

    output_path = os.path.abspath(args.output)

    videos = [video1_path, video2_path]

    # Select appropriate listener based on mode
    if args.headless:
        listener = HeadlessUIListener(videos)
    else:
        listener = RichUIListener(videos)

    # Show configuration
    config_data = {
        "Video Sources": videos,
        "YOLO Model": args.yolo_model,
        "ReID Threshold": f"{args.threshold:.2f}",
        "Device": str(args.device),
        "Max Frames": str(args.max_frames) if args.max_frames > 0 else "All",
        "Sample FPS": str(args.sample_fps) if args.sample_fps > 0 else "Full FPS",
        "Output Path": output_path,
        "Pipeline Mode": "Ensemble" if args.ensemble else "Single Model",
        "YOLO Tracker": args.tracker,
    }

    if args.ensemble:
        model_paths = None
        if args.model_paths:
            model_paths = [resolve_path(p.strip(), workspace_root) for p in args.model_paths.split(",")]
        model_dir = resolve_path(args.model_dir, workspace_root)

        if args.model_paths:
            config_data["Ensemble Model Paths"] = args.model_paths
        else:
            config_data["Ensemble Model Dir"] = model_dir
        config_data["Ensemble Fusion"] = args.fusion
        config_data["FP16 Enabled"] = str(args.fp16)
    else:
        config_data["DMT Weights"] = args.weights

    listener.show_configuration(config_data)

    # Create the stages for the Pipeline pattern
    if args.ensemble:
        feature_stage = EnsembleModelFeatureStage(
            model_dir=model_dir,
            model_paths=model_paths,
            device=args.device,
            fp16=args.fp16,
            fusion=args.fusion,
        )
    else:
        feature_stage = SingleModelFeatureStage(
            weights_path=args.weights,
            device=args.device,
        )

    stages = [
        VideoFeederStage(),
        SamplerStage(sample_fps=args.sample_fps, time_based=False),
        YoloDetectionStage(yolo_path=args.yolo_model),
        feature_stage,
        TrackingStage(tracker_config=args.tracker)
    ]

    # Create a shared registry across all pipeline runs
    registry = SimpleRegistry(match_threshold=args.threshold)

    pipeline = ReIDPipeline(
        stages=stages,
        threshold=args.threshold,
        max_frames=args.max_frames,
        registry=registry,
    )

    feeder_stage = stages[0]  # VideoFeederStage

    pipeline.initialize(listener)

    for idx, video in enumerate(videos):
        feeder_stage.set_video_path(video)
        if listener:
            listener.current_video_idx = idx + 1
        pipeline.run(listener)

    # Export results outside the pipeline scope
    export_results(registry, output_path)
    if listener:
        listener.on_pipeline_end(registry, output_path)


if __name__ == "__main__":
    main()
