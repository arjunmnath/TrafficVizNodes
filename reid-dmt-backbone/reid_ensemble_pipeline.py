#!/usr/bin/env python3
"""
Ensembled ReID processing pipeline, utilizing the production-grade EnsembleReID
inference implementation from src/inference.
"""

import os
import sys
import torch
import numpy as np

# Setup sys.path for DMT imports
dmt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if dmt_path not in sys.path:
    sys.path.insert(0, dmt_path)

from inference import EnsembleReID
from reid_pipeline_base import BaseReIDPipeline, ReIDPipelineListener, get_device

class EnsembleReIDPipeline(BaseReIDPipeline):
    """Ensembled ReID Tracking Pipeline adapter using src/inference.EnsembleReID."""
    def __init__(
        self,
        model_dir="trained_models",
        model_paths=None,
        yolo_path="yolov8s.pt",
        threshold=0.8,
        device="cpu",
        max_frames=0,
        sample_fps=0.0,
        output_path="reid_test_results.json",
        fp16=True,
        fusion="concat"
    ):
        super().__init__(
            yolo_path=yolo_path,
            threshold=threshold,
            max_frames=max_frames,
            sample_fps=sample_fps,
            output_path=output_path
        )
        self.model_dir = model_dir
        self.model_paths = model_paths
        # Handle 'auto' device mapping
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        self.fusion = fusion
        self.ensemble = None

    def _initialize_extractor(self, listener: ReIDPipelineListener = None):
        if listener:
            listener.on_init_status("Loading and assembling ensembled models...")
        
        # Instantiate the EnsembleReID class from src/inference
        self.ensemble = EnsembleReID(
            model_dir=self.model_dir,
            model_paths=self.model_paths,
            device=self.device,
            fp16=self.fp16
        )

        if listener:
            listener.on_init_status(f"Loaded {len(self.ensemble.models)} ensembled models successfully.")

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract ensembled/fused embedding using EnsembleReID.extract."""
        # EnsembleReID.extract expects OpenCV BGR array directly if is_bgr=True
        # It handles conversion to PIL, transformation, running through all models, and fusing.
        feat_tensor = self.ensemble.extract(
            crop,
            is_bgr=True,
            return_dict=False,
            fusion=self.fusion
        )
        # Convert PyTorch Tensor to NumPy array
        return feat_tensor.cpu().numpy()
