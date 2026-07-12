import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from tracking.domain.metadata import TrackMetadata
from tracking.domain.interfaces import SizeModel
from tracking.domain.trajectory import PiecewiseTrajectory


class TimeModel:
    """Handles mapping between frame numbers and timestamps, supporting variable FPS."""

    def __init__(self, frames: List[int], timestamps: List[float]):
        if len(frames) != len(timestamps):
            raise ValueError("Frames and timestamps lists must have the same length.")
        if not frames:
            raise ValueError("TimeModel must have at least one frame/timestamp entry.")

        # Ensure sorted order
        sorted_pairs = sorted(zip(frames, timestamps), key=lambda x: x[0])
        self.frames = [p[0] for p in sorted_pairs]
        self.timestamps = [p[1] for p in sorted_pairs]

    def frame_to_timestamp(self, frame: int) -> float:
        """Map frame index to timestamp via linear interpolation."""
        if len(self.frames) == 1:
            return self.timestamps[0]
        return float(np.interp(frame, self.frames, self.timestamps))

    def timestamp_to_frame(self, timestamp: float) -> int:
        """Map timestamp to frame index via linear interpolation."""
        if len(self.timestamps) == 1:
            return self.frames[0]
        return int(round(np.interp(timestamp, self.timestamps, self.frames)))

    def serialize(self) -> Dict[str, Any]:
        return {
            "frames": [int(f) for f in self.frames],
            "timestamps": [float(t) for t in self.timestamps],
        }


class Statistics:
    """Computes and holds aggregate statistics of a track's trajectory."""

    def __init__(
        self,
        avg_speed: float = 0.0,
        max_speed: float = 0.0,
        total_distance: float = 0.0,
        avg_acceleration: float = 0.0,
    ):
        self.avg_speed = avg_speed
        self.max_speed = max_speed
        self.total_distance = total_distance
        self.avg_acceleration = avg_acceleration

    @classmethod
    def compute(cls, trajectory: PiecewiseTrajectory) -> "Statistics":
        """Compute statistics by sampling the continuous trajectory."""
        t0, t1 = trajectory.t0, trajectory.t1
        if t1 <= t0:
            return cls()

        # Sample 100 points along the trajectory
        times = np.linspace(t0, t1, 100)
        speeds = []
        total_dist = 0.0
        prev_pos = None

        accel_mags = []
        prev_vel = None
        prev_t = None

        for t in times:
            pos = trajectory.position(t)
            vel = trajectory.velocity(t)
            speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            speeds.append(speed)

            if prev_pos is not None:
                total_dist += math.sqrt((pos[0] - prev_pos[0]) ** 2 + (pos[1] - prev_pos[1]) ** 2)
            prev_pos = pos

            if prev_vel is not None and prev_t is not None and t > prev_t:
                dt = t - prev_t
                ax = (vel[0] - prev_vel[0]) / dt
                ay = (vel[1] - prev_vel[1]) / dt
                accel_mags.append(math.sqrt(ax**2 + ay**2))

            prev_vel = vel
            prev_t = t

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        avg_accel = float(np.mean(accel_mags)) if accel_mags else 0.0

        return cls(
            avg_speed=avg_speed,
            max_speed=max_speed,
            total_distance=total_dist,
            avg_acceleration=avg_accel,
        )

    def serialize(self) -> Dict[str, Any]:
        return {
            "avg_speed": float(self.avg_speed),
            "max_speed": float(self.max_speed),
            "total_distance": float(self.total_distance),
            "avg_acceleration": float(self.avg_acceleration),
        }


class CompressedTrack:
    """Primary domain representation of a compressed trajectory track."""

    def __init__(
        self,
        metadata: TrackMetadata,
        time_model: TimeModel,
        size_model: SizeModel,
        trajectory: PiecewiseTrajectory,
        statistics: Optional[Statistics] = None,
    ):
        self.metadata = metadata
        self.time_model = time_model
        self.size_model = size_model
        self.trajectory = trajectory
        self.statistics = statistics or Statistics.compute(trajectory)

    def position(self, t: float) -> Tuple[float, float]:
        """Continuous 2D position (cx, cy) at timestamp t."""
        return self.trajectory.position(t)

    def velocity(self, t: float) -> Tuple[float, float]:
        """Continuous 2D velocity (vx, vy) at timestamp t."""
        return self.trajectory.velocity(t)

    def direction(self, t: float) -> float:
        """Continuous direction/heading (radians) at timestamp t."""
        return self.trajectory.direction(t)

    def width(self, t: float) -> float:
        """Continuous width at timestamp t."""
        return self.size_model.width(t)

    def height(self, t: float) -> float:
        """Continuous height at timestamp t."""
        return self.size_model.height(t)

    def acceleration(self, t: float) -> Tuple[float, float]:
        """Continuous 2D acceleration (ax, ay) at timestamp t (numerical derivative)."""
        dt = 1e-3
        v1 = self.velocity(t - dt)
        v2 = self.velocity(t + dt)
        return (v2[0] - v1[0]) / (2 * dt), (v2[1] - v1[1]) / (2 * dt)

    def heading(self, t: float) -> float:
        """Continuous heading angle (radians) at timestamp t (alias of direction)."""
        return self.direction(t)

    def curvature(self, t: float) -> float:
        """Continuous trajectory curvature at timestamp t."""
        vx, vy = self.velocity(t)
        ax, ay = self.acceleration(t)
        speed = math.sqrt(vx**2 + vy**2)
        if speed < 1e-6:
            return 0.0
        return abs(vx * ay - vy * ax) / (speed**3)
