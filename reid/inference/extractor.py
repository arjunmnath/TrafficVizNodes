import torch
import numpy as np
from typing import Union, List, Dict, Any
from .config import InferenceConfig
from .loader import resolve_checkpoints_and_configs
from .model_factory import build_model_from_config
from .preprocessing import preprocess_images
from .ensemble import fuse_embeddings
from .utils import get_device


class EnsembleReID:
    """Production-grade inference pipeline supporting single/multiple/all models."""

    def __init__(
        self,
        model_dir: str = "trained_models",
        model_paths: Union[str, List[str]] = None,
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.device = device if device != "auto" else get_device(device)
        self.fp16 = fp16

        # Resolve configurations
        self.configs = resolve_checkpoints_and_configs(
            model_dir=model_dir, model_paths=model_paths, device=self.device, fp16=fp16
        )

        # Load all models
        self.models = []
        for cfg in self.configs:
            model = build_model_from_config(cfg)
            self.models.append(model)

    def _extract_for_model(
        self, model: torch.nn.Module, config: InferenceConfig, tensor_batch: torch.Tensor
    ) -> torch.Tensor:
        """Extracts features using a single model for a preprocessed batch of images."""
        device = torch.device(self.device)
        tensor_batch = tensor_batch.to(device)

        # Handle AMP if enabled
        is_cuda = device.type == "cuda"

        with torch.no_grad():
            if self.fp16 and is_cuda:
                with torch.cuda.amp.autocast(enabled=True):
                    if config.flip_feats:
                        f2 = model(tensor_batch)
                        # Flip horizontal: dimension 3 is width
                        inv_idx = torch.arange(
                            tensor_batch.size(3) - 1, -1, -1, device=device
                        ).long()
                        flipped_batch = tensor_batch.index_select(3, inv_idx)
                        f1 = model(flipped_batch)
                        feat = f2 + f1
                    else:
                        feat = model(tensor_batch)
            else:
                if config.flip_feats:
                    f2 = model(tensor_batch)
                    inv_idx = torch.arange(tensor_batch.size(3) - 1, -1, -1, device=device).long()
                    flipped_batch = tensor_batch.index_select(3, inv_idx)
                    f1 = model(flipped_batch)
                    feat = f2 + f1
                else:
                    feat = model(tensor_batch)

        # Normalize along channel (L2-norm)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat

    def extract(
        self, image: Any, is_bgr: bool = True, return_dict: bool = False, fusion: str = "concat"
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract embeddings for a single image.

        Args:
            image: PIL Image, OpenCV BGR, numpy array, torch Tensor, or filepath.
            is_bgr: If input is numpy array, whether it's in BGR format (typical for cv2.imread).
            return_dict: If True, returns a dict mapping checkpoint paths to their respective embeddings.
            fusion: If return_dict is False and multiple models exist, how to fuse the embeddings ('concat', 'mean').
        """
        # Simply calls extract_batch with a list containing the image, and index the first element
        res = self.extract_batch([image], is_bgr=is_bgr, return_dict=return_dict, fusion=fusion)
        if return_dict:
            return {k: v[0] for k, v in res.items()}
        return res[0]

    def extract_batch(
        self,
        images: List[Any],
        is_bgr: bool = True,
        return_dict: bool = False,
        fusion: str = "concat",
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract embeddings for a batch of images.

        Args:
            images: List/tuple of PIL Images, OpenCV BGR, numpy arrays, torch Tensors, or filepaths.
            is_bgr: If inputs are numpy arrays, whether they are in BGR format.
            return_dict: If True, returns a dict mapping checkpoint paths to their respective embeddings.
            fusion: If return_dict is False and multiple models exist, how to fuse the embeddings ('concat', 'mean').
        """
        if not images:
            raise ValueError("No images provided for feature extraction.")

        model_embeddings = {}

        for model, config in zip(self.models, self.configs):
            # Preprocess the entire list of images for this model's expected size and normalization parameters
            tensor_batch = preprocess_images(
                images=images,
                image_size=config.image_size,
                pixel_mean=config.pixel_mean,
                pixel_std=config.pixel_std,
                is_bgr=is_bgr,
            )

            # Batch extraction loop (split tensor_batch by config.batch_size)
            num_samples = tensor_batch.shape[0]
            batch_size = config.batch_size
            feats_list = []

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                sub_batch = tensor_batch[start_idx:end_idx]
                sub_feats = self._extract_for_model(model, config, sub_batch)
                feats_list.append(sub_feats)

            model_feats = torch.cat(feats_list, dim=0)
            model_embeddings[config.checkpoint_path] = model_feats

        if return_dict:
            return model_embeddings

        # Perform fusion if there are multiple models, or just return the single model's features
        if len(self.models) == 1:
            return list(model_embeddings.values())[0]

        # Fuse
        return fuse_embeddings(list(model_embeddings.values()), method=fusion)
