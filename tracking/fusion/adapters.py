import math
from typing import Optional, Tuple
from tracking.domain.track import CompressedTrack


class FusionTrackAdapter:
    """Adapter utility to compare continuous CompressedTrack objects for multi-camera fusion."""

    @staticmethod
    def get_temporal_intersection(
        track_a: CompressedTrack, track_b: CompressedTrack
    ) -> Optional[Tuple[float, float]]:
        """Compute the temporal intersection interval [t_start, t_end] between two tracks.

        Returns None if there is no overlap.
        """
        start = max(track_a.trajectory.t0, track_b.trajectory.t0)
        end = min(track_a.trajectory.t1, track_b.trajectory.t1)

        if start <= end:
            return start, end
        return None

    @classmethod
    def compute_average_distance(
        cls,
        track_a: CompressedTrack,
        track_b: CompressedTrack,
        samples: int = 20,
    ) -> Optional[float]:
        """Compute average L2 distance between the two tracks during their temporal overlap."""
        interval = cls.get_temporal_intersection(track_a, track_b)
        if not interval:
            return None

        t_start, t_end = interval
        if t_start == t_end:
            pos_a = track_a.position(t_start)
            pos_b = track_b.position(t_start)
            return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

        import numpy as np

        times = np.linspace(t_start, t_end, samples)
        distances = []

        for t in times:
            pos_a = track_a.position(t)
            pos_b = track_b.position(t)
            dist = math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)
            distances.append(dist)

        return float(np.mean(distances))

    @classmethod
    def compute_velocity_alignment(
        cls,
        track_a: CompressedTrack,
        track_b: CompressedTrack,
        samples: int = 20,
    ) -> Optional[float]:
        """Compute average cosine similarity between velocity vectors during temporal overlap."""
        interval = cls.get_temporal_intersection(track_a, track_b)
        if not interval:
            return None

        t_start, t_end = interval
        if t_start == t_end:
            v_a = track_a.velocity(t_start)
            v_b = track_b.velocity(t_start)
            return cls._cosine_sim(v_a, v_b)

        import numpy as np

        times = np.linspace(t_start, t_end, samples)
        alignments = []

        for t in times:
            v_a = track_a.velocity(t)
            v_b = track_b.velocity(t)
            alignments.append(cls._cosine_sim(v_a, v_b))

        return float(np.mean(alignments))

    @staticmethod
    def _cosine_sim(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        return dot / (mag1 * mag2)
