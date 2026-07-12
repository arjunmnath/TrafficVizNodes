from typing import List, Tuple
from tracking.domain.track import CompressedTrack


class BBoxReconstructor:
    """Reconstructs bounding boxes [x1, y1, x2, y2] from compressed track models."""

    @staticmethod
    def reconstruct(track: CompressedTrack, t: float) -> Tuple[float, float, float, float]:
        """Reconstruct bounding box [x1, y1, x2, y2] at timestamp t."""
        cx, cy = track.position(t)
        w = track.width(t)
        h = track.height(t)

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return x1, y1, x2, y2

    @classmethod
    def reconstruct_trajectory(
        cls, track: CompressedTrack, timestamps: List[float]
    ) -> List[Tuple[float, float, float, float]]:
        """Reconstruct a list of bounding boxes for a list of timestamps."""
        return [cls.reconstruct(track, t) for t in timestamps]
