"""Shared types for the Florence reasoning stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image


@dataclass
class CandidateImage:
    """A retrieved candidate frame passed to the reasoning VLM."""

    camera_id: str
    camera_timestamp: float
    video_pos_ms: float
    track_id: int
    bbox: Optional[List[float]]
    frame: Image.Image
    retrieval_distance: float


@dataclass
class RankedResult:
    """A Florence-scored result returned to the orchestration layer."""

    camera_id: str
    camera_timestamp: float
    video_pos_ms: float
    track_id: int
    vlm_score: float
    vlm_explanation: str
    frame: Image.Image
