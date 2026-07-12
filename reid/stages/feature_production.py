import os
from typing import Any, List, Optional
import numpy as np
import torch

from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, has_minimum_roi_area, FrameData
from reid.inference import EnsembleReID


class FeatureStage(PipelineStage):
    """Stage 2: Extracts ReID features using the 3 ensemble models and returns their centroid."""

    def __init__(
        self,
        device: str = "cpu",
        fp16: bool = True,
    ):
        """Constructor.

        Args:
            device (str): Inference device.
            fp16 (bool): Whether to enable half precision.
        """
        self.device = (
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp16 = fp16
        self.ensemble = None

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        if listener:
            listener.on_init_status("Loading ensembled ReID models...")
        self.ensemble = EnsembleReID(
            device=self.device,
            fp16=self.fp16,
        )
        if listener:
            listener.on_init_status(
                f"Loaded {len(self.ensemble.models)} ensembled models successfully."
            )

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
        valid_crops = []
        valid_idxs = []

        for idx, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if not has_minimum_roi_area(box, frame.shape):
                features.append(None)
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            valid_crops.append(crop)
            valid_idxs.append(idx)
            features.append(None)  # Placeholder

        if len(valid_crops) > 0:
            embeddings_tensor = self.ensemble.extract_batch(valid_crops, is_bgr=True)
            embeddings = embeddings_tensor.cpu().numpy()
            for embed_idx, orig_idx in enumerate(valid_idxs):
                features[orig_idx] = embeddings[embed_idx]

        # Resolve missing feature dimensions
        valid_feat = next((f for f in features if f is not None), None)
        if valid_feat is not None:
            feat_dim = len(valid_feat)
        else:
            # All crops are invalid; run a dummy extraction to determine feature dimension
            dummy_crop = np.zeros((128, 64, 3), dtype=np.uint8)
            dummy_feat = self.ensemble.extract(dummy_crop, is_bgr=True)
            feat_dim = dummy_feat.shape[0]

        zeros = np.zeros(feat_dim, dtype=np.float32)
        features = [f if f is not None else zeros for f in features]

        data.features = np.stack(features, axis=0)

        return data
