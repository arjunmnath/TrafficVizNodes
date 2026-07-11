import os
from dataclasses import dataclass
from typing import Any, Optional, Dict
import numpy as np


class SimpleRegistry:
    # Forward declaration type hint helper
    pass


@dataclass
class FrameData:
    """Dataclass holding context payload data passed through the pipeline stages."""

    frame: Optional[np.ndarray] = None
    frame_count: int = 0
    feed_name: str = ""
    total_frames: int = 0
    timestamp: float = 0.0
    fps: float = 0.0

    # Routing execution state
    skip: bool = False
    end_of_stream: bool = False

    # Metadata for listeners/UI logging
    feed_idx: int = 1
    total_videos: int = 1
    elapsed_time: float = 0.0
    listener: Optional[Any] = None

    # Pipeline output fields populated by stages
    boxes: Optional[np.ndarray] = None  # shape (N, 4)
    scores: Optional[np.ndarray] = None  # shape (N,)
    classes: Optional[np.ndarray] = None  # shape (N,)
    features: Optional[np.ndarray] = None  # shape (N, D)
    tracks: Optional[np.ndarray] = None  # shape (M, 8)

    def __repr__(self):
        return f"FrameData(frame_count={self.frame_count}, feed_name={self.feed_name}, total_frames={self.total_frames}, timestamp={self.timestamp}, classes={self.classes}, features={self.features.shape if self.features is not None else None}, tracks={self.tracks.shape if self.tracks is not None else None})"


class ReIDPipelineListener:
    """Interface for listening to ReID pipeline execution events."""

    def on_init_start(self):
        pass

    def on_init_status(self, message: str):
        pass

    def on_init_end(self):
        pass

    def on_video_start(
        self, video_path: str, video_idx: int, total_videos: int, total_frames: int, fps: float
    ):
        pass

    def on_frame_processed(
        self,
        video_name: str,
        video_idx: int,
        total_videos: int,
        frame_count: int,
        total_frames: int,
        elapsed_time: float,
        fps: float,
        registry: "SimpleRegistry",
        log_message: str | None = None,
    ):
        pass

    def on_video_end(self, video_path: str, total_frames: int):
        pass

    def on_pipeline_end(self, registries: Dict[str, "SimpleRegistry"], output_path: str):
        pass

    def on_error(self, message: str):
        pass


def has_minimum_roi_area(bbox: np.ndarray, frame_shape: tuple, threshold: float = 3e-3) -> bool:
    """Validate if the crop bounding box meets size constraints.

    Args:
        bbox (np.ndarray): Bounding box coordinates [x1, y1, x2, y2].
        frame_shape (tuple): Shape of the image frame.

    Returns:
        bool: True if crop is valid.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    W, H, _ = frame_shape
    return (w * h) / (W * H) > threshold


def resolve_path(p: str, base_dir: str) -> str:
    """Resolve relative path to absolute path using base directory.

    Args:
        p (str): Input path.
        base_dir (str): Base directory.

    Returns:
        str: Resolved path.
    """
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))
