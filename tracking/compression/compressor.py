from typing import List, Tuple, Optional

from tracking.domain.interfaces import SegmentationStrategy, TrajectoryFitter
from tracking.domain.track import CompressedTrack
from tracking.compression.builder import CompressedTrackBuilder
from tracking.compression.segmentation import AdaptiveSegmentation


class TrajectoryCompressor:
    """Orchestrator class that runs the compression pipeline on track detections."""

    def __init__(
        self,
        segmentation_strategy: Optional[SegmentationStrategy] = None,
        fitter: Optional[TrajectoryFitter] = None,
    ):
        self.segmentation_strategy = segmentation_strategy or AdaptiveSegmentation()
        self.fitter = fitter

    def compress(
        self,
        track_id: int,
        camera_id: str,
        class_label: str,
        frames: List[int],
        timestamps: List[float],
        bboxes: List[Tuple[float, float, float, float]],
    ) -> CompressedTrack:
        """Run the compression pipeline on a sequence of raw track observations."""
        builder = CompressedTrackBuilder()
        builder.set_metadata(track_id=track_id, camera_id=camera_id, class_label=class_label)
        builder.add_observations(frames, timestamps, bboxes)

        builder.set_segmentation_strategy(self.segmentation_strategy)
        if self.fitter is not None:
            builder.set_fitter(self.fitter)

        return builder.build()
