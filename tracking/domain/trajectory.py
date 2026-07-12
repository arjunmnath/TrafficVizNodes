import math
from typing import List, Tuple

from tracking.domain.interfaces import TrajectoryModel, TrajectorySegment


class PiecewiseTrajectory(TrajectoryModel):
    """A trajectory composed of multiple sequential trajectory segments."""

    def __init__(self, segments: List[TrajectorySegment]):
        if not segments:
            raise ValueError("PiecewiseTrajectory must have at least one segment.")
        # Sort segments by start time
        self.segments = sorted(segments, key=lambda s: s.t0)

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.position(t)

    def _find_segment(self, t: float) -> TrajectorySegment:
        """Find the segment containing time t. Extrapolates to the closest boundary segment if out of range."""
        if t <= self.segments[0].t0:
            return self.segments[0]
        if t >= self.segments[-1].t1:
            return self.segments[-1]

        # Binary search or linear scan since segment counts are typically small
        for seg in self.segments:
            if seg.t0 <= t <= seg.t1:
                return seg
        # Fallback to the closest segment if there is a small gap due to float precision
        closest = min(self.segments, key=lambda s: min(abs(t - s.t0), abs(t - s.t1)))
        return closest

    def position(self, t: float) -> Tuple[float, float]:
        seg = self._find_segment(t)
        return seg.position(t)

    def velocity(self, t: float) -> Tuple[float, float]:
        seg = self._find_segment(t)
        return seg.velocity(t)

    def direction(self, t: float) -> float:
        """Evaluate the heading direction (radians, from -pi to pi) at time t.

        Direction is computed as arctan2(vy, vx) from velocity vector.
        """
        vx, vy = self.velocity(t)
        return math.atan2(vy, vx)

    def duration(self) -> float:
        return self.segments[-1].t1 - self.segments[0].t0

    @property
    def t0(self) -> float:
        return self.segments[0].t0

    @property
    def t1(self) -> float:
        return self.segments[-1].t1
