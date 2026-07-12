from typing import Optional

from reid.postprocessing.base import PostProcessingStage
from reid.postprocessing.pipeline import TerminatedTrack
from tracking.compression.compressor import TrajectoryCompressor


class TrajectoryCompressionStage(PostProcessingStage):
    """Postprocessing stage that compresses the continuous trajectory of a terminated track.

    The compressed track is stored in ``track.compressed_track`` and ``track.extra["compressed_track"]``.
    """

    def __init__(self, compressor: Optional[TrajectoryCompressor] = None) -> None:
        """Constructor.

        Args:
            compressor: Optional TrajectoryCompressor instance. If None, a default one is created.
        """
        self.compressor = compressor or TrajectoryCompressor()

    def process(self, track: TerminatedTrack) -> TerminatedTrack:
        """Compress the track trajectory if history is present.

        Args:
            track: The TerminatedTrack entering this stage.

        Returns:
            The TerminatedTrack with compressed_track populated.
        """
        if track.history is None:
            return track

        frames = track.history.get("frames", [])
        timestamps = track.history.get("timestamps", [])
        bboxes = track.history.get("bboxes", [])

        if not frames or not timestamps or not bboxes:
            return track

        # Convert bboxes list-of-lists to list-of-tuples for type correctness
        bbox_tuples = [tuple(b) for b in bboxes]

        # Compress track observations
        compressed = self.compressor.compress(
            track_id=track.track_id,
            camera_id=track.feed_name,
            class_label=track.class_label,
            frames=frames,
            timestamps=timestamps,
            bboxes=bbox_tuples,
        )

        # Store the compressed track back to both the dedicated attribute and extra dict
        track.compressed_track = compressed
        track.extra["compressed_track"] = compressed

        return track

    def __repr__(self) -> str:
        return f"TrajectoryCompressionStage(compressor={self.compressor!r})"
