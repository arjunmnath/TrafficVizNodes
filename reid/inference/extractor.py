import os
import torch
import numpy as np
from typing import Union, List, Any
from .config import InferenceConfig, EnsembleConfig
from .model_factory import build_model_from_config, build_ensemble_model
from .preprocessing import preprocess_images
from .ensemble import fuse_embeddings
from .utils import get_device


class EnsembleReID:
    """Production-grade inference pipeline supporting 3 ensembled models with mean centroid fusion."""

    def __init__(
        self,
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.device = device if device != "auto" else get_device(device)
        self.fp16 = fp16

        from reid.utils import resolve_model_weights

        # Config containing the 3 ensembled submodels
        self.config = EnsembleConfig(
            checkpoint_paths=[
                resolve_model_weights("resnet101_ibn_a_2.pth"),
                resolve_model_weights("resnet101_ibn_a_3.pth"),
                resolve_model_weights("resnext101_ibn_a_2.pth"),
            ],
            device=self.device,
            fp16=self.fp16,
        )

        # Build unified ensemble model
        self.model = build_ensemble_model(self.config)

        # Keep self.models as a list for compatibility (e.g. len(self.ensemble.models))
        self.models = self.model.submodels

    def extract(self, image: Any, is_bgr: bool = True) -> torch.Tensor:
        """Extract embeddings for a single image."""
        res = self.extract_batch([image], is_bgr=is_bgr)
        return res[0]

    def extract_batch(self, images: List[Any], is_bgr: bool = True) -> torch.Tensor:
        """Extract embeddings for a batch of images and fuse them using mean centroid fusion."""
        if not images:
            raise ValueError("No images provided for feature extraction.")

        # Preprocess the entire list of images once
        tensor_batch = preprocess_images(
            images=images,
            image_size=self.config.image_size,
            pixel_mean=self.config.pixel_mean,
            pixel_std=self.config.pixel_std,
            is_bgr=is_bgr,
        )

        device = torch.device(self.device)
        tensor_batch = tensor_batch.to(device)
        is_cuda = device.type == "cuda"

        # Batch extraction loop (split tensor_batch by config.batch_size)
        num_samples = tensor_batch.shape[0]
        batch_size = self.config.batch_size
        feats_list = []

        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                sub_batch = tensor_batch[start_idx:end_idx]

                if self.fp16 and is_cuda:
                    with torch.cuda.amp.autocast(enabled=True):
                        sub_feats = self.model(sub_batch)
                else:
                    sub_feats = self.model(sub_batch)

                feats_list.append(sub_feats)

        # Concatenate all batch features
        fused_features = torch.cat(feats_list, dim=0)
        return fused_features
