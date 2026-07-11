"""
reid/postprocessing/__init__.py
Exposes the postprocessing pipeline and all built-in stages.
"""

from .pipeline import PostProcessingPipeline, TerminatedTrack
from .base import PostProcessingStage
from .stages.trajectory_fusion import TrajectoryFusionStage

__all__ = [
    "PostProcessingPipeline",
    "TerminatedTrack",
    "PostProcessingStage",
    "TrajectoryFusionStage",
]
