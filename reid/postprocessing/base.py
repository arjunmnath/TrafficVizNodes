"""
reid/postprocessing/base.py
Abstract base class for all postprocessing stages.
"""

from abc import ABC, abstractmethod
from typing import Any


class PostProcessingStage(ABC):
    """Abstract base class for a single stage in the track postprocessing pipeline.

    Each stage receives a TerminatedTrack, modifies it in-place or returns a
    modified copy, and passes it to the next stage.
    """

    @abstractmethod
    def process(self, track: "TerminatedTrack") -> "TerminatedTrack":  # noqa: F821
        """Process a terminated track.

        Args:
            track: The TerminatedTrack being processed by this stage.

        Returns:
            The (optionally modified) TerminatedTrack.
        """
        ...
