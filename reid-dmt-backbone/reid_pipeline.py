#!/usr/bin/env python3
"""
Adapted ReID processing pipeline using the legacy single-model DMT backbone.
Inherits from BaseReIDPipeline and overrides extractor initialization and feature extraction.
"""

import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Setup sys.path for DMT imports
dmt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if dmt_path not in sys.path:
    sys.path.insert(0, dmt_path)

from model import make_model
from inference.loader import get_config_for_checkpoint
from inference.model_factory import build_model_from_config

# Import base classes and common functions
from reid_pipeline_base import (
    SimpleRegistry,
    ReIDPipelineListener,
    BaseReIDPipeline,
    get_device,
    is_valid_crop,
    resolve_path
)

# Re-expose base imports for backward compatibility with UI and testing runner
__all__ = [
    "SimpleRegistry",
    "ReIDPipelineListener",
    "BaseReIDPipeline",
    "get_device",
    "is_valid_crop",
    "resolve_path",
    "ReIDPipeline"
]

class ReIDPipeline(BaseReIDPipeline):
    """Adapter for the legacy single-model ReID pipeline."""
    def __init__(self, weights_path, yolo_path, threshold=0.8, device="cpu", max_frames=0, sample_fps=0.0, output_path="reid_test_results-v0.json"):
        super().__init__(
            yolo_path=yolo_path,
            threshold=threshold,
            max_frames=max_frames,
            sample_fps=sample_fps,
            output_path=output_path
        )
        self.weights_path = weights_path
        self.device = device if device != "auto" else get_device()
        
        self.model = None
        self.val_transforms = None
        self.inf_cfg = None

    def _initialize_extractor(self, listener: ReIDPipelineListener = None):
        if listener:
            listener.on_init_status("Loading configuration via InferenceConfig...")
            
        # Dynamically build and load InferenceConfig from weights_path
        self.inf_cfg = get_config_for_checkpoint(self.weights_path, device=self.device, fp16=False)
        # Enable flip_feats to match the default legacy behavior
        self.inf_cfg.flip_feats = True

        if listener:
            listener.on_init_status("Building model backbone and loading weights...")
        self.model = build_model_from_config(self.inf_cfg)

        # Define transformations matching 
        self.val_transforms = T.Compose([
            T.Resize(self.inf_cfg.image_size, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=self.inf_cfg.pixel_mean, std=self.inf_cfg.pixel_std)
        ])

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        # BGR -> RGB -> PIL
        img = Image.fromarray(crop[..., ::-1])
        img_t = self.val_transforms(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.inf_cfg.flip_feats:
                f2 = self.model(img_t)
                img_flip = torch.flip(img_t, [3])
                f1 = self.model(img_flip)
                feat = f2 + f1
            else:
                feat = self.model(img_t)

            # Feature normalization is standard for evaluation
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)

            embedding = feat.squeeze(0).cpu().numpy()
            
        return embedding
