from abc import ABC, abstractmethod
from typing import Any
from reid.utils import ReIDPipelineListener, FrameData


class PipelineStage(ABC):
    """Abstract base class representing a single processing stage in the ReID pipeline."""

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        """Initialize stage-specific resources (e.g. load weights).

        Args:
            listener (ReIDPipelineListener, optional): Listener for status updates.
        """
        pass

    @abstractmethod
    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        """Process input frame data and update state context.

        Args:
            data (FrameData): Dataclass context passed through pipeline stages.
            pipeline (ReIDPipeline): Active pipeline coordinator instance.

        Returns:
            FrameData: Updated dataclass context.
        """
        pass
