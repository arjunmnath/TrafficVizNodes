import os
from typing import Any, List, Optional
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, is_valid_crop, FrameData
from reid.inference.loader import get_config_for_checkpoint
from reid.inference.model_factory import build_model_from_config
from reid.inference.utils import get_device
from reid.inference import EnsembleReID


class SingleModelFeatureStage(PipelineStage):
    """Stage 2: Extracts ReID features using a legacy single model backbone."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        """Constructor.

        Args:
            weights_path (str): Path to single model weights checkpoint (.pth).
            device (str): Inference device (cpu, cuda, etc.).
        """
        self.weights_path = weights_path
        self.device = get_device(device)
        self.model = None
        self.val_transforms = None
        self.inf_cfg = None

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        if listener:
            listener.on_init_status("Loading single ReID model configuration...")
        self.inf_cfg = get_config_for_checkpoint(self.weights_path, device=self.device, fp16=False)
        self.inf_cfg.flip_feats = True

        if listener:
            listener.on_init_status("Building single ReID model backbone and loading weights...")
        self.model = build_model_from_config(self.inf_cfg)

        self.val_transforms = T.Compose([
            T.Resize(self.inf_cfg.image_size, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=self.inf_cfg.pixel_mean, std=self.inf_cfg.pixel_std),
        ])

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        if data.skip or data.end_of_stream:
            return data

        frame = data.frame
        boxes = data.boxes
        scores = data.scores
        classes = data.classes

        if boxes is None or len(boxes) == 0:
            data.features = np.empty((0, 0), dtype=np.float32)
            return data

        features = []
        valid_indices = []

        for idx, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if not is_valid_crop(box, frame.shape):
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

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

                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                embedding = feat.squeeze(0).cpu().numpy()

            features.append(embedding)
            valid_indices.append(idx)

        data.boxes = boxes[valid_indices]
        data.scores = scores[valid_indices]
        data.classes = classes[valid_indices]

        if features:
            data.features = np.stack(features, axis=0)
        else:
            data.features = np.empty((0, 0), dtype=np.float32)

        return data


class EnsembleModelFeatureStage(PipelineStage):
    """Stage 2: Extracts ReID features using an ensemble of models."""

    def __init__(
        self,
        model_dir: str = "trained_models",
        model_paths: Optional[List[str]] = None,
        device: str = "cpu",
        fp16: bool = True,
        fusion: str = "concat"
    ):
        """Constructor.

        Args:
            model_dir (str): Base directory containing ensembled models.
            model_paths (List[str], optional): Explicit list of model checkpoints.
            device (str): Inference device.
            fp16 (bool): Whether to enable half precision.
            fusion (str): Fusion method (concat or mean).
        """
        self.model_dir = model_dir
        self.model_paths = model_paths
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        self.fusion = fusion
        self.ensemble = None

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        if listener:
            listener.on_init_status("Loading and assembling ensembled ReID models...")
        self.ensemble = EnsembleReID(
            model_dir=self.model_dir,
            model_paths=self.model_paths,
            device=self.device,
            fp16=self.fp16,
        )
        if listener:
            listener.on_init_status(f"Loaded {len(self.ensemble.models)} ensembled models successfully.")

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        if data.skip or data.end_of_stream:
            return data

        frame = data.frame
        boxes = data.boxes
        scores = data.scores
        classes = data.classes

        if boxes is None or len(boxes) == 0:
            data.features = np.empty((0, 0), dtype=np.float32)
            return data

        features = []
        valid_indices = []

        for idx, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if not is_valid_crop(box, frame.shape):
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            feat_tensor = self.ensemble.extract(
                crop, is_bgr=True, return_dict=False, fusion=self.fusion
            )
            embedding = feat_tensor.cpu().numpy()
            features.append(embedding)
            valid_indices.append(idx)

        data.boxes = boxes[valid_indices]
        data.scores = scores[valid_indices]
        data.classes = classes[valid_indices]

        if features:
            data.features = np.stack(features, axis=0)
        else:
            data.features = np.empty((0, 0), dtype=np.float32)

        return data
