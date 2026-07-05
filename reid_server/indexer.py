"""Image embedding for track event indexing using retrieval adapters."""

from __future__ import annotations

from typing import Optional

import numpy as np

from inference_node.retrieval.encoder import BaseRetrievalEncoder
from reid_server.crop_extractor import CropExtractor
from shared.utils import setup_logger


class RetrievalIndexer:
    """Builds embeddings from cropped object images at index time using a retrieval encoder."""

    def __init__(
        self,
        encoder: BaseRetrievalEncoder,
        crop_extractor: CropExtractor,
    ) -> None:
        self.logger = setup_logger("RetrievalIndexer")
        self.encoder = encoder
        self.crop_extractor = crop_extractor

    def embed_event(self, event_data: dict) -> Optional[np.ndarray]:
        camera_id = event_data.get("camera_id", "")
        video_pos_ms = event_data.get("video_pos_ms")
        bbox = event_data.get("bbox")

        if video_pos_ms is None or not bbox:
            self.logger.warning(f"Event missing video_pos_ms or bbox for camera {camera_id}")
            return None

        crop = self.crop_extractor.extract_crop(
            camera_id=camera_id,
            video_pos_ms=float(video_pos_ms),
            bbox=bbox,
        )
        if crop is None:
            return None

        return self.encoder.encode_image(crop)


# Compatibility alias
SigLIPIndexer = RetrievalIndexer
