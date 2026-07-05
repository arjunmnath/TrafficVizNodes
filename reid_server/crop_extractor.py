"""Extract cropped object images from configured video sources."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from shared.utils import setup_logger


class CropExtractor:
    """Loads video frames and returns bbox crops for indexing."""

    def __init__(self, video_sources: dict[str, str]) -> None:
        self.logger = setup_logger("CropExtractor")
        self.video_sources = video_sources

    def extract_crop(
        self,
        camera_id: str,
        video_pos_ms: float,
        bbox: list[float],
    ) -> Optional[np.ndarray]:
        path = self.video_sources.get(camera_id)
        if not path:
            self.logger.warning(f"No video source configured for camera {camera_id}")
            return None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {path}")
            return None

        cap.set(cv2.CAP_PROP_POS_MSEC, video_pos_ms)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self.logger.warning(f"Failed to read frame at {video_pos_ms:.0f}ms from {path}")
            return None

        if len(bbox) != 4:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
