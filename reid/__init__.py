"""
ReID Package
Exposes ReID pipelines, listeners, registry, utilities, and stages.
"""

from .pipeline import ReIDPipeline
from .registry import SimpleRegistry
from .utils import ReIDPipelineListener, resolve_path
from .stages import (
    PipelineStage,
    VideoFeederStage,
    SamplerStage,
    YoloDetectionStage,
    SingleModelFeatureStage,
    EnsembleModelFeatureStage,
    TrackingStage,
)
from .ui import RichUIListener, HeadlessUIListener

__all__ = [
    "ReIDPipeline",
    "SimpleRegistry",
    "ReIDPipelineListener",
    "resolve_path",
    "PipelineStage",
    "VideoFeederStage",
    "SamplerStage",
    "YoloDetectionStage",
    "SingleModelFeatureStage",
    "EnsembleModelFeatureStage",
    "TrackingStage",
    "RichUIListener",
    "HeadlessUIListener",
]
